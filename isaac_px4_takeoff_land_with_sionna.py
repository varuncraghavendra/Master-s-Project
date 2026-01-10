#!/usr/bin/env python3
import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.timeline
import omni.usd
from omni.isaac.core.world import World

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from scipy.spatial.transform import Rotation
from pxr import UsdGeom, Usd

import asyncio
import threading
import time
import math

try:
    import zmq
except Exception:
    zmq = None


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        # --- PX4 <-> sim link (Pegasus backend) ---
        vehicle_id = 0
        mavlink_baseport = 4560

        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": vehicle_id,
            "connection_type": "tcpin",
            "connection_ip": "localhost",
            "connection_baseport": mavlink_baseport,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe,
        })

        config_multirotor = MultirotorConfig()
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]

        Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            vehicle_id,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        self.world.reset()

        # --- Autonomy ---
        self.stop_sim = False
        self._autonomy_started = False

        self._mavsdk_candidates = [
            "udpin://0.0.0.0:14540",
            "udpin://0.0.0.0:14550",
            "udpin://0.0.0.0:14541",
            "udpin://0.0.0.0:14551",
        ]
        self._target_alt_m = 2.5

        # --- Time + home (NEW) ---
        self._sim_time = 0.0                 # fallback accumulator
        self._latest_sim_time = None
        self._home_pos_world = None          # set on first valid GT pose
        self._home_sent_to_sionna = False

        # --- Sionna bridge (NEW) ---
        self._sionna_enabled = True
        self._sionna_endpoint = "tcp://127.0.0.1:5555"
        self._sionna_rate_hz = 10.0

        self._pose_lock = threading.Lock()
        self._latest_gt_pos = None
        self._latest_sionna = None

        self._sionna_thread_started = False

    # ------------------ Ground truth pose from USD ------------------
    def _read_uav_gt_position(self):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/World/quadrotor")
        if not prim or not prim.IsValid():
            return None

        xform = UsdGeom.Xformable(prim)
        mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = mat.ExtractTranslation()
        return [float(t[0]), float(t[1]), float(t[2])]

    # ------------------ Sim time (Isaac timeline preferred) ------------------
    def _get_sim_time(self):
        # Try timeline time first
        try:
            t = float(self.timeline.get_current_time())
            return t
        except Exception:
            pass

        # Fallback: accumulate physics dt
        try:
            dt = float(self.world.get_physics_dt())
        except Exception:
            dt = 1.0 / 60.0
        self._sim_time += dt
        return self._sim_time

    # ------------------ Sionna ZMQ thread (NEW) ------------------
    def _start_sionna_thread(self):
        if self._sionna_thread_started or (not self._sionna_enabled):
            return
        self._sionna_thread_started = True

        if zmq is None:
            carb.log_warn("[SIONNA] pyzmq not installed in Isaac python. Install with ./python.sh -m pip install pyzmq")
            return

        def runner():
            ctx = zmq.Context()
            sock = ctx.socket(zmq.REQ)
            sock.connect(self._sionna_endpoint)
            sock.RCVTIMEO = 1000
            sock.SNDTIMEO = 1000

            carb.log_warn(f"[SIONNA] Connected to sensor server at {self._sionna_endpoint}")

            period = 1.0 / max(self._sionna_rate_hz, 1e-6)
            last_sent_t = -1e9

            while simulation_app.is_running() and not self.stop_sim:
                with self._pose_lock:
                    gt = None if self._latest_gt_pos is None else list(self._latest_gt_pos)
                    t_sim = self._latest_sim_time
                    home = self._home_pos_world

                # Send INIT once when home becomes available
                if (home is not None) and (not self._home_sent_to_sionna):
                    try:
                        sock.send_json({"type": "init", "home_pos_world": home})
                        ack = sock.recv_json()
                        if ack.get("ok", False):
                            self._home_sent_to_sionna = True
                            carb.log_warn(f"[SIONNA] Home sent to server: {home}")
                    except Exception as e:
                        carb.log_warn(f"[SIONNA] INIT failed: {e}")

                # Time-gated sending based on sim time
                if gt is not None and t_sim is not None and (t_sim - last_sent_t) >= period:
                    last_sent_t = t_sim
                    req = {"t_sim": float(t_sim), "tx_pos": gt}
                    try:
                        sock.send_json(req)
                        resp = sock.recv_json()
                        with self._pose_lock:
                            self._latest_sionna = resp
                    except Exception as e:
                        carb.log_warn(f"[SIONNA] request failed: {e}")

                time.sleep(0.001)  # avoid busy loop

        threading.Thread(target=runner, daemon=True).start()

    # ------------------ MAVSDK thread (unchanged) ------------------
    def _start_autonomy_thread(self):
        if self._autonomy_started:
            return
        self._autonomy_started = True

        def runner():
            try:
                asyncio.run(self._mavsdk_takeoff_land())
            except Exception as e:
                carb.log_error(f"[AUTO] MAVSDK thread crashed: {e}")
                self.stop_sim = True

        threading.Thread(target=runner, daemon=True).start()

    async def _with_retries(self, name, coro_fn, retries=10, delay_s=0.5):
        last = None
        for i in range(retries):
            try:
                return await coro_fn()
            except Exception as e:
                last = e
                carb.log_warn(f"[AUTO] {name} failed (try {i+1}/{retries}): {e}")
                await asyncio.sleep(delay_s)
        raise last

    async def _wait_connected(self, drone, timeout_s=20):
        start = time.time()
        async for s in drone.core.connection_state():
            if s.is_connected:
                return True
            if time.time() - start > timeout_s:
                return False

    async def _wait_in_air(self, drone, desired_in_air, timeout_s=60):
        start = time.time()
        async for ia in drone.telemetry.in_air():
            if ia == desired_in_air:
                return True
            if time.time() - start > timeout_s:
                return False

    async def _mavsdk_takeoff_land(self):
        try:
            from mavsdk import System
        except Exception as e:
            carb.log_error(
                "MAVSDK-Python not available in Isaac python.\n"
                "Install:\n"
                "  ./python.sh -m pip install mavsdk\n"
                f"Import error: {e}"
            )
            self.stop_sim = True
            return

        drone = System()

        connected = False
        last_err = None
        for addr in self._mavsdk_candidates:
            carb.log_warn(f"[AUTO] Trying MAVSDK connect at {addr} ...")
            try:
                await drone.connect(system_address=addr)
                connected = await self._wait_connected(drone, timeout_s=8)
                if connected:
                    carb.log_warn(f"[AUTO] MAVSDK connected using {addr}")
                    break
            except Exception as e:
                last_err = e
                carb.log_warn(f"[AUTO] Connect failed for {addr}: {e}")

        if not connected:
            carb.log_error(f"[AUTO] Could not connect MAVSDK to PX4. Last error: {last_err}")
            self.stop_sim = True
            return

        try:
            carb.log_warn("[AUTO] Setting COM_ARM_WO_GPS=1 ...")
            await self._with_retries(
                "set_param COM_ARM_WO_GPS",
                lambda: drone.param.set_param_int("COM_ARM_WO_GPS", 1),
                retries=5, delay_s=0.3
            )
        except Exception as e:
            carb.log_warn(f"[AUTO] Could not set COM_ARM_WO_GPS (continuing): {e}")

        await asyncio.sleep(1.0)

        carb.log_warn("[AUTO] Arming ...")
        await self._with_retries("arm", drone.action.arm, retries=12, delay_s=0.6)

        carb.log_warn(f"[AUTO] Setting takeoff altitude: {self._target_alt_m:.2f} m")
        await self._with_retries(
            "set_takeoff_altitude",
            lambda: drone.action.set_takeoff_altitude(self._target_alt_m),
            retries=6, delay_s=0.3
        )

        carb.log_warn("[AUTO] Takeoff ...")
        await self._with_retries("takeoff", drone.action.takeoff, retries=12, delay_s=0.6)

        ok = await self._wait_in_air(drone, desired_in_air=True, timeout_s=30)
        if not ok:
            carb.log_warn("[AUTO] Did not confirm in_air=True in time; continuing anyway.")

        carb.log_warn("[AUTO] Hover 5s ...")
        await asyncio.sleep(5)

        carb.log_warn("[AUTO] Land ...")
        await self._with_retries("land", drone.action.land, retries=12, delay_s=0.6)

        ok = await self._wait_in_air(drone, desired_in_air=False, timeout_s=60)
        if not ok:
            carb.log_warn("[AUTO] Did not confirm landed in time; disarming anyway.")

        carb.log_warn("[AUTO] Disarm ...")
        await self._with_retries("disarm", drone.action.disarm, retries=12, delay_s=0.6)

        carb.log_warn("[AUTO] Complete. Closing sim.")
        self.stop_sim = True

    # ------------------ Main loop ------------------
    def run(self):
        self.timeline.play()

        warmup_steps = 300
        step_count = 0

        # Start Sionna bridge early
        self._start_sionna_thread()

        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
            step_count += 1

            # Update sim time
            sim_time = self._get_sim_time()
            with self._pose_lock:
                self._latest_sim_time = sim_time

            # Update GT pose and set home on first valid GT
            gt = self._read_uav_gt_position()
            if gt is not None:
                with self._pose_lock:
                    self._latest_gt_pos = gt
                    if self._home_pos_world is None:
                        self._home_pos_world = gt
                        carb.log_warn(f"[HOME] Isaac home set (world): {self._home_pos_world}")

            # Start autonomy after warmup
            if step_count == warmup_steps:
                self._start_autonomy_thread()

            # Print tracking error ~1 Hz
            if step_count % 60 == 0:
                with self._pose_lock:
                    gt_now = None if self._latest_gt_pos is None else list(self._latest_gt_pos)
                    home = self._home_pos_world
                    meas = self._latest_sionna
                    t_now = self._latest_sim_time

                if gt_now and meas and meas.get("ok", False):
                    est = meas.get("est_pos_world", None)
                    if est is not None:
                        dx = est[0] - gt_now[0]
                        dy = est[1] - gt_now[1]
                        dz = est[2] - gt_now[2]
                        err_world = math.sqrt(dx*dx + dy*dy + dz*dz)

                        msg = f"[TRACK] t_sim={t_now:.2f}  |e_world|={err_world:.2f} m"

                        if home is not None:
                            gt_home = [gt_now[i] - home[i] for i in range(3)]
                            est_home = [est[i] - home[i] for i in range(3)]
                            ddx = est_home[0] - gt_home[0]
                            ddy = est_home[1] - gt_home[1]
                            ddz = est_home[2] - gt_home[2]
                            err_home = math.sqrt(ddx*ddx + ddy*ddy + ddz*ddz)
                            msg += f"  |e_home|={err_home:.2f} m"

                        carb.log_warn(msg)
                elif meas and (not meas.get("ok", True)):
                    carb.log_warn(f"[TRACK] t_sim={t_now:.2f}  Sionna error: {meas.get('error')}")

        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    app = PegasusApp()
    app.run()


if __name__ == "__main__":
    main()
