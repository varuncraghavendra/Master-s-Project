#!/usr/bin/env python3
import time
import threading
import asyncio
import numpy as np
import zmq

import carb
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.timeline
import omni.usd
from omni.isaac.core.world import World
from omni.isaac.core.prims import XFormPrim

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from scipy.spatial.transform import Rotation
from pxr import Usd, UsdPhysics


# Map Isaac -> Etoile (so Sionna drone moves in the Etoile scene center)
SCENE_OFFSET = np.array([-115.0, 33.0, 0.0], dtype=float)
SCENE_SCALE  = 1.0

def isaac_to_etoile(p_isaac: np.ndarray) -> np.ndarray:
    return SCENE_OFFSET + SCENE_SCALE * p_isaac


def pick_moving_body_prim(stage: Usd.Stage, root_path="/World/quadrotor") -> str:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        carb.log_warn(f"[POSE] Root {root_path} invalid -> using root")
        return root_path

    base_link = None
    rigid = []

    for prim in Usd.PrimRange(root):
        if not prim.IsValid():
            continue
        p = prim.GetPath().pathString
        if "base_link" in p:
            base_link = p
        try:
            rb = UsdPhysics.RigidBodyAPI(prim)
            if rb and rb.GetRigidBodyEnabledAttr().IsValid():
                rigid.append(p)
        except Exception:
            pass

    if base_link:
        carb.log_warn(f"[POSE] Using base_link: {base_link}")
        return base_link
    if rigid:
        carb.log_warn(f"[POSE] Using rigidbody: {rigid[0]}")
        return rigid[0]

    carb.log_warn(f"[POSE] No rigid prim found -> using root")
    return root_path


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        vehicle_id = 0
        cfg = PX4MavlinkBackendConfig({
            "vehicle_id": vehicle_id,
            "connection_type": "tcpin",
            "connection_ip": "localhost",
            "connection_baseport": 4560,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe,
        })

        mcfg = MultirotorConfig()
        mcfg.backends = [PX4MavlinkBackend(cfg)]

        Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            vehicle_id,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0, 0, 0], degrees=True).as_quat(),
            config=mcfg,
        )

        self.world.reset()

        self.stage = omni.usd.get_context().get_stage()
        self.pose_path = pick_moving_body_prim(self.stage, "/World/quadrotor")
        self.pose_prim = XFormPrim(self.pose_path)

        self.stop_sim = False
        self._threads_started = False

        # ZMQ -> Sionna render server
        self.sionna_addr = "tcp://127.0.0.1:5555"

        # Cached GT updated in sim loop
        self._lock = threading.Lock()
        self._latest_gt = None         # Isaac coords
        self._latest_etoile = None     # Etoile coords
        self._latest_t = None
        self._seq = 0

        # match sionna render fps (2Hz)
        self._send_period_s = 0.5

        # MAVSDK candidates
        self._mavsdk_candidates = [
            "udpin://0.0.0.0:14540",
            "udpin://0.0.0.0:14550",
            "udpin://0.0.0.0:14541",
            "udpin://0.0.0.0:14551",
        ]

    def _update_gt_cache_mainthread(self):
        pos, _quat = self.pose_prim.get_world_pose()
        pos = np.array(pos, dtype=float).reshape(3)
        t_sim = float(self.timeline.get_current_time())

        pos_etoile = isaac_to_etoile(pos)

        with self._lock:
            self._latest_gt = pos
            self._latest_etoile = pos_etoile
            self._latest_t = t_sim

    def _start_threads(self):
        if self._threads_started:
            return
        self._threads_started = True
        threading.Thread(target=self._zmq_loop, daemon=True).start()
        threading.Thread(target=self._mavsdk_square_thread, daemon=True).start()

    def _zmq_loop(self):
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        sock.setsockopt(zmq.SNDTIMEO, 5000)
        sock.connect(self.sionna_addr)

        last_send_wall = 0.0

        while simulation_app.is_running() and not self.stop_sim:
            if (time.time() - last_send_wall) < self._send_period_s:
                time.sleep(0.01)
                continue
            last_send_wall = time.time()

            with self._lock:
                gt = None if self._latest_gt is None else self._latest_gt.copy()
                etoile = None if self._latest_etoile is None else self._latest_etoile.copy()
                t_sim = self._latest_t

            if gt is None or etoile is None or t_sim is None:
                continue

            self._seq += 1
            msg = {"seq": self._seq, "t_sim": float(t_sim), "tx_pos": etoile.tolist()}

            try:
                carb.log_warn(f"[ISAAC->SIONNA] seq={self._seq} t={t_sim:.2f} isaac={gt.tolist()} etoile={etoile.tolist()}")
                sock.send_json(msg)
                resp = sock.recv_json()
                w = resp.get("warning", None)
                if w:
                    carb.log_warn(f"[SIONNA] seq={resp.get('seq')} warning={w}")
            except Exception as e:
                carb.log_warn(f"[ISAAC] ZMQ error: {e}")
                time.sleep(0.2)

        # stop render server
        try:
            sock.send_json({"cmd": "stop"})
            _ = sock.recv_json()
        except Exception:
            pass

        sock.close()

    def _mavsdk_square_thread(self):
        try:
            asyncio.run(self._mavsdk_square_velocity_ned())
        except Exception as e:
            carb.log_error(f"[AUTO] crashed: {e}")
            self.stop_sim = True

    async def _wait_connected(self, drone, timeout_s=12):
        start = time.time()
        async for s in drone.core.connection_state():
            if s.is_connected:
                return True
            if time.time() - start > timeout_s:
                return False

    async def _connect_mavsdk(self):
        from mavsdk import System
        drone = System()
        for addr in self._mavsdk_candidates:
            try:
                await drone.connect(system_address=addr)
                if await self._wait_connected(drone):
                    carb.log_warn(f"[AUTO] MAVSDK connected on {addr}")
                    return drone
            except Exception:
                pass
        return None

    async def _send_ned_velocity_for(self, drone, vn, ve, vd, yaw_deg, duration_s, rate_hz=10):
        from mavsdk.offboard import VelocityNedYaw
        dt = 1.0 / rate_hz
        steps = int(duration_s * rate_hz)
        for _ in range(steps):
            await drone.offboard.set_velocity_ned(VelocityNedYaw(vn, ve, vd, yaw_deg))
            await asyncio.sleep(dt)

    async def _wait_altitude(self, drone, target_m=5.0, tol=0.3, timeout_s=25.0):
        """
        Uses NED telemetry: altitude = -down_m (approx).
        """
        start = time.time()
        async for pv in drone.telemetry.position_velocity_ned():
            alt = -float(pv.position.down_m)
            if alt >= (target_m - tol):
                return True
            if time.time() - start > timeout_s:
                return False

    async def _mavsdk_square_velocity_ned(self):
        from mavsdk.offboard import OffboardError, VelocityNedYaw

        drone = await self._connect_mavsdk()
        if drone is None:
            carb.log_error("[AUTO] MAVSDK connect failed")
            self.stop_sim = True
            return

        # allow arming without GPS in sim
        try:
            await drone.param.set_param_int("COM_ARM_WO_GPS", 1)
        except Exception:
            pass

        # TAKEOFF to 5m
        TAKEOFF_M = 5.0
        await drone.action.arm()
        await drone.action.set_takeoff_altitude(TAKEOFF_M)
        await drone.action.takeoff()

        ok_alt = await self._wait_altitude(drone, target_m=TAKEOFF_M, tol=0.35, timeout_s=30.0)
        carb.log_warn(f"[AUTO] takeoff altitude reached={ok_alt}")

        # Warm-up Offboard setpoints (required)
        for _ in range(20):
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(0.05)

        try:
            await drone.offboard.start()
        except OffboardError as e:
            carb.log_error(f"[AUTO] Offboard start failed: {e}")
            self.stop_sim = True
            return

        # 10x10 square in NED (world frame)
        SIDE = 10.0   # meters
        V = 1.0       # m/s
        T = SIDE / V  # seconds per side
        YAW = 0.0     # keep yaw

        carb.log_warn("[AUTO] Square 10x10: +North")
        await self._send_ned_velocity_for(drone, vn= V, ve=0.0, vd=0.0, yaw_deg=YAW, duration_s=T)

        carb.log_warn("[AUTO] Square 10x10: +East")
        await self._send_ned_velocity_for(drone, vn=0.0, ve= V, vd=0.0, yaw_deg=YAW, duration_s=T)

        carb.log_warn("[AUTO] Square 10x10: -North")
        await self._send_ned_velocity_for(drone, vn=-V, ve=0.0, vd=0.0, yaw_deg=YAW, duration_s=T)

        carb.log_warn("[AUTO] Square 10x10: -East")
        await self._send_ned_velocity_for(drone, vn=0.0, ve=-V, vd=0.0, yaw_deg=YAW, duration_s=T)

        # stop offboard
        try:
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, YAW))
            await asyncio.sleep(0.5)
            await drone.offboard.stop()
        except Exception:
            pass

        # LAND
        await drone.action.land()
        await asyncio.sleep(10.0)
        try:
            await drone.action.disarm()
        except Exception:
            pass

        carb.log_warn("[AUTO] done, stopping sim")
        self.stop_sim = True

    def run(self):
        self.timeline.play()
        warmup_steps = 300
        step = 0

        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
            step += 1

            self._update_gt_cache_mainthread()

            if step == warmup_steps:
                self._start_threads()

        self.timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    PegasusApp().run()
