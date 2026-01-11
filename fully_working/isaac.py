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

        # ZMQ
        self.sionna_addr = "tcp://127.0.0.1:5555"
        self.save_dir = "/home/varun"
        self.run_id = time.strftime("%Y%m%d_%H%M%S")

        # Cached GT updated in sim loop
        self._lock = threading.Lock()
        self._latest_gt = None
        self._latest_t = None
        self._seq = 0

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
        with self._lock:
            self._latest_gt = pos
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
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.setsockopt(zmq.SNDTIMEO, 2000)
        sock.connect(self.sionna_addr)

        last_send_wall = 0.0

        while simulation_app.is_running() and not self.stop_sim:
            if (time.time() - last_send_wall) < 0.10:
                time.sleep(0.01)
                continue
            last_send_wall = time.time()

            with self._lock:
                gt = None if self._latest_gt is None else self._latest_gt.copy()
                t_sim = self._latest_t
            if gt is None or t_sim is None:
                continue

            self._seq += 1
            msg = {"seq": self._seq, "t_sim": float(t_sim), "tx_pos": gt.tolist()}

            try:
                carb.log_warn(f"[ISAAC->SIONNA] send seq={self._seq} t={t_sim:.2f} pos={gt.tolist()}")
                sock.send_json(msg)
                resp = sock.recv_json()
                if not resp.get("ok", False):
                    carb.log_warn(f"[SIONNA->ISAAC] ERROR seq={resp.get('seq')} err={resp.get('error')}")
                else:
                    carb.log_warn(f"[SIONNA->ISAAC] ok seq={resp.get('seq')} err_m={resp.get('err_m')} fallback={resp.get('fallback_used')}")
            except Exception as e:
                carb.log_warn(f"[ISAAC] ZMQ error: {e}")
                time.sleep(0.2)

        # stop + report
        try:
            sock.send_json({"cmd": "stop", "save_dir": self.save_dir, "run_id": self.run_id})
            resp = sock.recv_json()
            carb.log_warn(f"[ISAAC] Sionna report: {resp}")
        except Exception as e:
            carb.log_warn(f"[ISAAC] stop/report error: {e}")

        sock.close()

    def _mavsdk_square_thread(self):
        try:
            asyncio.run(self._mavsdk_square_velocity())
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

    async def _send_velocity_for(self, drone, vx, vy, vz, yaw_rate, duration_s, rate_hz=10):
        from mavsdk.offboard import VelocityBodyYawspeed
        dt = 1.0 / rate_hz
        steps = int(duration_s * rate_hz)
        for _ in range(steps):
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx, vy, vz, yaw_rate))
            await asyncio.sleep(dt)

    async def _mavsdk_square_velocity(self):
        from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

        drone = await self._connect_mavsdk()
        if drone is None:
            carb.log_error("[AUTO] MAVSDK connect failed")
            self.stop_sim = True
            return

        try:
            await drone.param.set_param_int("COM_ARM_WO_GPS", 1)
        except Exception:
            pass

        await drone.action.arm()
        await drone.action.set_takeoff_altitude(2.5)
        await drone.action.takeoff()
        await asyncio.sleep(5.0)

        # Warm-up Offboard
        for _ in range(20):
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(0.05)

        try:
            await drone.offboard.start()
        except OffboardError as e:
            carb.log_error(f"[AUTO] Offboard start failed: {e}")
            self.stop_sim = True
            return

        # Fly 5x5 square using body-frame velocities (reliable)
        v = 1.0  # m/s
        T = 30.0  # seconds for 5m

        carb.log_warn("[AUTO] Square: +X (forward)")
        await self._send_velocity_for(drone, vx=v,  vy=0.0, vz=0.0, yaw_rate=0.0, duration_s=T)

        carb.log_warn("[AUTO] Square: +Y (right)")
        await self._send_velocity_for(drone, vx=0.0, vy=v,  vz=0.0, yaw_rate=0.0, duration_s=T)

        carb.log_warn("[AUTO] Square: -X (back)")
        await self._send_velocity_for(drone, vx=-v, vy=0.0, vz=0.0, yaw_rate=0.0, duration_s=T)

        carb.log_warn("[AUTO] Square: -Y (left)")
        await self._send_velocity_for(drone, vx=0.0, vy=-v, vz=0.0, yaw_rate=0.0, duration_s=T)

        # stop offboard
        try:
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(0.5)
            await drone.offboard.stop()
        except Exception:
            pass

        await drone.action.land()
        await asyncio.sleep(7.0)
        await drone.action.disarm()

        carb.log_warn("[AUTO] done, stopping sim")
        self.stop_sim = True

    def run(self):
        self.timeline.play()
        warmup_steps = 300
        step = 0

        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
            step += 1

            # update GT cache from MAIN THREAD
            self._update_gt_cache_mainthread()

            if step == warmup_steps:
                self._start_threads()

        self.timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    PegasusApp().run()
 
