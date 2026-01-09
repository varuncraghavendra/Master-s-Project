#!/usr/bin/env python3
"""
Single vehicle Pegasus sim with PX4 backend + MAVSDK auto takeoff/land.

Fixes:
- Do NOT connect MAVSDK to PX4 simulator TCP lockstep port (4560). That port is PX4<->sim only.
- Use MAVSDK on UDP (udpin://...) and listen on a port PX4 is already sending to (14540 from your logs).
- Avoid "Address in use" on 14550 by not binding to it.
- Set COM_ARM_WO_GPS=1 to prevent arming blocked by missing GPS in sim.
"""

import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.timeline
from omni.isaac.core.world import World

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from scipy.spatial.transform import Rotation

import asyncio
import threading
import time


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        # --- PX4 <-> Simulator (Pegasus) link ---
        vehicle_id = 0
        mavlink_baseport = 4560  # PX4 "simulator_mavlink" TCP accepts here (vehicle_id added internally by backend)

        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": vehicle_id,

            # PX4 <-> sim lockstep link (DO NOT use this for MAVSDK)
            "connection_type": "tcpin",
            "connection_ip": "localhost",
            "connection_baseport": mavlink_baseport,

            # PX4 auto-launch
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe,  # set to 'iris' if PX4 < 1.14
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

        # From your logs: PX4 mavlink "Onboard" sends to remote port 14540
        # So we LISTEN there with MAVSDK.
        self._mavsdk_candidates = [
            "udpin://0.0.0.0:14540",  # best match for your setup
            "udpin://0.0.0.0:14550",  # fallback if you reconfigure PX4
            "udpin://0.0.0.0:14541",
            "udpin://0.0.0.0:14551",
        ]

        self._target_alt_m = 2.5

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
                "MAVSDK-Python is not available in this Python environment.\n"
                "Install it in Isaac Sim's python:\n"
                "  ./python.sh -m pip install mavsdk\n"
                f"Import error: {e}"
            )
            self.stop_sim = True
            return

        drone = System()

        # Try candidate endpoints until one works (fixes 'Address in use' / wrong port)
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

        # (Optional but very helpful in sim) allow arming without GPS
        # If GPS is already OK, this doesn’t hurt.
        try:
            carb.log_warn("[AUTO] Setting COM_ARM_WO_GPS=1 (sim convenience) ...")
            await self._with_retries(
                "set_param COM_ARM_WO_GPS",
                lambda: drone.param.set_param_int("COM_ARM_WO_GPS", 1),
                retries=5,
                delay_s=0.3
            )
        except Exception as e:
            carb.log_warn(f"[AUTO] Could not set COM_ARM_WO_GPS (continuing): {e}")

        # Give PX4 a moment after param change
        await asyncio.sleep(1.0)

        # Arm
        carb.log_warn("[AUTO] Arming ...")
        await self._with_retries("arm", drone.action.arm, retries=12, delay_s=0.6)

        # Set takeoff altitude + takeoff
        carb.log_warn(f"[AUTO] Setting takeoff altitude: {self._target_alt_m:.2f} m")
        await self._with_retries(
            "set_takeoff_altitude",
            lambda: drone.action.set_takeoff_altitude(self._target_alt_m),
            retries=6,
            delay_s=0.3
        )

        carb.log_warn("[AUTO] Takeoff ...")
        await self._with_retries("takeoff", drone.action.takeoff, retries=12, delay_s=0.6)

        # Wait until actually in air (more reliable than only watching altitude)
        ok = await self._wait_in_air(drone, desired_in_air=True, timeout_s=30)
        if not ok:
            carb.log_warn("[AUTO] Did not confirm in_air=True in time; continuing anyway.")

        # Climb monitor (best-effort)
        carb.log_warn("[AUTO] Monitoring altitude ...")
        start = time.time()
        async for pos in drone.telemetry.position():
            carb.log_warn(f"[AUTO] rel_alt={pos.relative_altitude_m:.2f} m")
            if pos.relative_altitude_m >= (self._target_alt_m - 0.2):
                break
            if time.time() - start > 30:
                carb.log_warn("[AUTO] Altitude timeout; moving to hover/land.")
                break

        carb.log_warn("[AUTO] Hover 5s ...")
        await asyncio.sleep(5)

        # Land
        carb.log_warn("[AUTO] Land ...")
        await self._with_retries("land", drone.action.land, retries=12, delay_s=0.6)

        ok = await self._wait_in_air(drone, desired_in_air=False, timeout_s=60)
        if not ok:
            carb.log_warn("[AUTO] Did not confirm landed (in_air=False) in time; disarming anyway.")

        carb.log_warn("[AUTO] Disarm ...")
        await self._with_retries("disarm", drone.action.disarm, retries=12, delay_s=0.6)

        carb.log_warn("[AUTO] Complete. Closing sim.")
        self.stop_sim = True

    def run(self):
        self.timeline.play()

        # Give PX4 + Pegasus time to boot, start mavlink instances, and settle
        warmup_steps = 300  # ~5 seconds at 60Hz

        step_count = 0
        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
            step_count += 1
            if step_count == warmup_steps:
                self._start_autonomy_thread()

        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    app = PegasusApp()
    app.run()


if __name__ == "__main__":
    main()

