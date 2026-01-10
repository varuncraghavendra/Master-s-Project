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
import os
import csv
import json
from datetime import datetime

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

        # PX4 backend
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

        cfg = MultirotorConfig()
        cfg.backends = [PX4MavlinkBackend(mavlink_config)]

        Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            vehicle_id,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=cfg,
        )
        self.world.reset()

        # MAVSDK
        self.stop_sim = False
        self._autonomy_started = False
        self._mavsdk_candidates = [
            "udpin://0.0.0.0:14540",
            "udpin://0.0.0.0:14550",
            "udpin://0.0.0.0:14541",
            "udpin://0.0.0.0:14551",
        ]
        self._target_alt_m = 2.5

        # Sim time
        self._sim_time_accum = 0.0
        self._latest_sim_time = None

        # Correct GT: detect rigid body prim under quadrotor
        self._uav_root_path = "/World/quadrotor"
        self._uav_body_path = None  # resolved at runtime
        self._home_pos_world = None

        # Shared state
        self._pose_lock = threading.Lock()
        self._latest_gt_pos = None
        self._latest_sionna = None

        # Sionna settings
        self._sionna_enabled = True
        self._sionna_endpoint = "tcp://127.0.0.1:5555"
        self._sionna_rate_hz = 2.0
        self._sionna_thread_started = False

        # Benchmark files
        self._bench_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._bench_dir = os.path.expanduser("~")
        self._csv_path = os.path.join(self._bench_dir, f"bench_{self._bench_tag}.csv")
        self._json_path = os.path.join(self._bench_dir, f"bench_{self._bench_tag}.json")
        self._summary_path = os.path.join(self._bench_dir, f"bench_summary_{self._bench_tag}.json")

        self._bench_records = []
        self._bench_written = False

    def _get_stage(self):
        return omni.usd.get_context().get_stage()

    def _resolve_uav_body_path(self):
        """Pick a moving rigid body prim under /World/quadrotor."""
        stage = self._get_stage()
        root = stage.GetPrimAtPath(self._uav_root_path)
        if not root or not root.IsValid():
            self._uav_body_path = self._uav_root_path
            return

        # Try physics API detection
        try:
            from pxr import UsdPhysics
            for prim in Usd.PrimRange(root):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    self._uav_body_path = prim.GetPath().pathString
                    carb.log_warn(f"[GT] Using rigid-body prim: {self._uav_body_path}")
                    return
        except Exception:
            pass

        # Fallback: name heuristics
        candidates = []
        for prim in Usd.PrimRange(root):
            name = prim.GetName().lower()
            if any(k in name for k in ("base_link", "body", "fmu", "chassis", "iris")):
                candidates.append(prim.GetPath().pathString)

        if candidates:
            self._uav_body_path = candidates[-1]
            carb.log_warn(f"[GT] Using heuristic prim: {self._uav_body_path}")
        else:
            self._uav_body_path = self._uav_root_path
            carb.log_warn(f"[GT] Fallback prim: {self._uav_body_path}")

    def _read_world_pos(self, prim_path):
        stage = self._get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        xform = UsdGeom.Xformable(prim)
        mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = mat.ExtractTranslation()
        return [float(t[0]), float(t[1]), float(t[2])]

    def _read_uav_gt_position(self):
        if self._uav_body_path is None:
            self._resolve_uav_body_path()
        return self._read_world_pos(self._uav_body_path)

    def _get_sim_time(self):
        try:
            return float(self.timeline.get_current_time())
        except Exception:
            pass
        try:
            dt = float(self.world.get_physics_dt())
        except Exception:
            dt = 1.0 / 60.0
        self._sim_time_accum += dt
        return self._sim_time_accum

    def _start_sionna_thread(self):
        if self._sionna_thread_started or (not self._sionna_enabled):
            return
        self._sionna_thread_started = True

        if zmq is None:
            carb.log_warn("[SIONNA] pyzmq missing. Install: ./python.sh -m pip install pyzmq")
            return

        def make_socket(ctx):
            s = ctx.socket(zmq.REQ)
            s.setsockopt(zmq.LINGER, 0)
            try:
                s.setsockopt(zmq.REQ_RELAXED, 1)
                s.setsockopt(zmq.REQ_CORRELATE, 1)
            except Exception:
                pass
            s.connect(self._sionna_endpoint)
            return s

        def runner():
            ctx = zmq.Context()
            sock = make_socket(ctx)
            poller = zmq.Poller()
            poller.register(sock, zmq.POLLIN)

            carb.log_warn(f"[SIONNA] Connected to {self._sionna_endpoint}")

            period = 1.0 / max(self._sionna_rate_hz, 1e-6)
            last_sent_t = -1e9

            while simulation_app.is_running() and not self.stop_sim:
                with self._pose_lock:
                    gt = None if self._latest_gt_pos is None else list(self._latest_gt_pos)
                    t_sim = self._latest_sim_time
                    home = self._home_pos_world

                if gt is None or t_sim is None:
                    time.sleep(0.01)
                    continue

                if (t_sim - last_sent_t) < period:
                    time.sleep(0.002)
                    continue
                last_sent_t = t_sim

                req = {"t_sim": float(t_sim), "tx_pos": gt}
                if home is not None:
                    req["home_pos_world"] = home

                t0 = time.perf_counter()
                try:
                    sock.send_json(req)
                    events = dict(poller.poll(5000))
                    if sock in events and events[sock] == zmq.POLLIN:
                        resp = sock.recv_json()
                        t1 = time.perf_counter()

                        # Save last response
                        with self._pose_lock:
                            self._latest_sionna = resp

                        # Build benchmark record with side-by-side XYZ + error
                        est = resp.get("est_pos_world", None)
                        ok = bool(resp.get("ok", False)) and (est is not None)
                        err = None
                        if ok:
                            dx = est[0] - gt[0]
                            dy = est[1] - gt[1]
                            dz = est[2] - gt[2]
                            err = math.sqrt(dx*dx + dy*dy + dz*dz)

                        rec = {
                            "t_sim": float(t_sim),
                            "gt_pos_world": gt,
                            "est_pos_world": est,
                            "err_m": err,
                            "ok": bool(resp.get("ok", False)),
                            "error_msg": resp.get("error", None),
                            "latency_ms": 1000.0 * (t1 - t0),
                            "t_solve_ms": resp.get("t_solve_ms", None),
                            "t_total_ms": resp.get("t_total_ms", None),
                            "n_valid_rx": resp.get("n_valid_rx", None),
                            "residual": resp.get("residual", None),
                            "conv_used": resp.get("conv_used", None),
                        }
                        with self._pose_lock:
                            self._bench_records.append(rec)

                    else:
                        carb.log_warn("[SIONNA] timeout -> reset socket")
                        poller.unregister(sock)
                        sock.close()
                        sock = make_socket(ctx)
                        poller.register(sock, zmq.POLLIN)

                except Exception as e:
                    carb.log_warn(f"[SIONNA] comm error ({e}) -> reset socket")
                    try:
                        poller.unregister(sock)
                    except Exception:
                        pass
                    try:
                        sock.close()
                    except Exception:
                        pass
                    sock = make_socket(ctx)
                    poller.register(sock, zmq.POLLIN)

                time.sleep(0.001)

        threading.Thread(target=runner, daemon=True).start()

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
            carb.log_error("Install mavsdk: ./python.sh -m pip install mavsdk\n" + str(e))
            self.stop_sim = True
            return

        drone = System()

        connected = False
        for addr in self._mavsdk_candidates:
            try:
                carb.log_warn(f"[AUTO] Trying MAVSDK connect at {addr} ...")
                await drone.connect(system_address=addr)
                connected = await self._wait_connected(drone, timeout_s=8)
                if connected:
                    carb.log_warn(f"[AUTO] MAVSDK connected using {addr}")
                    break
            except Exception:
                pass

        if not connected:
            carb.log_error("[AUTO] MAVSDK could not connect")
            self.stop_sim = True
            return

        try:
            await self._with_retries(
                "set_param COM_ARM_WO_GPS",
                lambda: drone.param.set_param_int("COM_ARM_WO_GPS", 1),
                retries=5, delay_s=0.3
            )
        except Exception:
            pass

        await asyncio.sleep(1.0)
        await self._with_retries("arm", drone.action.arm, retries=12, delay_s=0.6)
        await self._with_retries(
            "set_takeoff_altitude",
            lambda: drone.action.set_takeoff_altitude(self._target_alt_m),
            retries=6, delay_s=0.3
        )
        await self._with_retries("takeoff", drone.action.takeoff, retries=12, delay_s=0.6)
        await self._wait_in_air(drone, True, 30)

        carb.log_warn("[AUTO] Hover 15s for benchmarking ...")
        await asyncio.sleep(15)

        await self._with_retries("land", drone.action.land, retries=12, delay_s=0.6)
        await self._wait_in_air(drone, False, 60)
        await self._with_retries("disarm", drone.action.disarm, retries=12, delay_s=0.6)

        self.stop_sim = True

    def _write_outputs(self):
        if self._bench_written:
            return
        self._bench_written = True

        # Write CSV
        with open(self._csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "t_sim",
                "gt_x","gt_y","gt_z",
                "est_x","est_y","est_z",
                "err_m",
                "residual",
                "latency_ms",
                "t_solve_ms",
                "t_total_ms",
                "n_valid_rx",
                "conv_used",
                "ok",
                "error_msg",
            ])

            for r in self._bench_records:
                gt = r.get("gt_pos_world", [None,None,None])
                est = r.get("est_pos_world", [None,None,None]) if r.get("est_pos_world") is not None else [None,None,None]
                w.writerow([
                    r.get("t_sim"),
                    gt[0], gt[1], gt[2],
                    est[0], est[1], est[2],
                    r.get("err_m"),
                    r.get("residual"),
                    r.get("latency_ms"),
                    r.get("t_solve_ms"),
                    r.get("t_total_ms"),
                    r.get("n_valid_rx"),
                    r.get("conv_used"),
                    r.get("ok"),
                    r.get("error_msg"),
                ])

        # Write JSON (full)
        with open(self._json_path, "w") as f:
            json.dump(self._bench_records, f, indent=2)

        # Summary
        errs = [r["err_m"] for r in self._bench_records if r.get("err_m") is not None]
        def stats(a):
            if not a:
                return None
            a = sorted(a)
            n = len(a)
            mean = sum(a)/n
            rmse = math.sqrt(sum(x*x for x in a)/n)
            p50 = a[n//2]
            p95 = a[int(0.95*(n-1))]
            return {"count": n, "mean": mean, "rmse": rmse, "p50": p50, "p95": p95, "max": a[-1]}

        summary = {
            "csv": self._csv_path,
            "json": self._json_path,
            "samples": len(self._bench_records),
            "ok_samples": sum(1 for r in self._bench_records if r.get("ok")),
            "error_m": stats(errs),
        }
        with open(self._summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        carb.log_warn(f"[BENCH] Saved CSV: {self._csv_path}")
        carb.log_warn(f"[BENCH] Saved JSON: {self._json_path}")
        carb.log_warn(f"[BENCH] Saved Summary: {self._summary_path}")
        carb.log_warn(f"[BENCH] Summary: {summary}")

    def run(self):
        self.timeline.play()

        warmup_steps = 300
        step_count = 0

        self._start_sionna_thread()

        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
            step_count += 1

            t_sim = self._get_sim_time()
            with self._pose_lock:
                self._latest_sim_time = t_sim

            gt = self._read_uav_gt_position()
            if gt is not None:
                with self._pose_lock:
                    self._latest_gt_pos = gt
                    if self._home_pos_world is None:
                        self._home_pos_world = gt
                        carb.log_warn(f"[HOME] home_pos_world={self._home_pos_world}")

            if step_count == warmup_steps:
                self._start_autonomy_thread()

            # Optional live side-by-side print (1 Hz)
            if step_count % 60 == 0:
                with self._pose_lock:
                    resp = self._latest_sionna
                    gt_now = self._latest_gt_pos
                    t_now = self._latest_sim_time
                if resp and gt_now:
                    est = resp.get("est_pos_world", None)
                    if resp.get("ok", False) and est:
                        dx = est[0]-gt_now[0]; dy = est[1]-gt_now[1]; dz = est[2]-gt_now[2]
                        err = math.sqrt(dx*dx+dy*dy+dz*dz)
                        carb.log_warn(
                            f"[XYZ] t={t_now:.2f}  GT=({gt_now[0]:.2f},{gt_now[1]:.2f},{gt_now[2]:.2f})  "
                            f"EST=({est[0]:.2f},{est[1]:.2f},{est[2]:.2f})  err={err:.2f}m  "
                            f"resid={resp.get('residual', None)}  conv={resp.get('conv_used', None)}"
                        )
                    else:
                        carb.log_warn(f"[XYZ] t={t_now:.2f}  Sionna error: {resp.get('error')}")

        self._write_outputs()
        self.timeline.stop()
        simulation_app.close()


def main():
    PegasusApp().run()


if __name__ == "__main__":
    main()
