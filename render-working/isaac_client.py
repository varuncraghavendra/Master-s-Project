#!/usr/bin/env python3
import time
import threading
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
    def __init__(self, send_hz=10.0):
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

        # Spawn only. User controls manually.
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

        self.sionna_addr = "tcp://127.0.0.1:5555"

        self.send_hz = float(send_hz)
        self.send_period = 1.0 / max(self.send_hz, 1e-6)

        self._lock = threading.Lock()
        self._latest_gt = None
        self._latest_t = None
        self._seq = 0

        self._thread_started = False

    def _update_gt_cache_mainthread(self):
        pos, _quat = self.pose_prim.get_world_pose()
        pos = np.array(pos, dtype=float).reshape(3)
        t_sim = float(self.timeline.get_current_time())
        with self._lock:
            self._latest_gt = pos
            self._latest_t = t_sim

    def _start_thread(self):
        if self._thread_started:
            return
        self._thread_started = True
        threading.Thread(target=self._zmq_loop, daemon=True).start()

    def _new_req_socket(self):
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)

        # === KEY FIX ===
        # allow sending again even if a reply was missed
        sock.setsockopt(zmq.REQ_RELAXED, 1)
        sock.setsockopt(zmq.REQ_CORRELATE, 1)

        sock.setsockopt(zmq.RCVTIMEO, 4000)  # give server time
        sock.setsockopt(zmq.SNDTIMEO, 1000)
        sock.connect(self.sionna_addr)
        return sock

    def _zmq_loop(self):
        sock = self._new_req_socket()
        last_send_wall = 0.0

        while simulation_app.is_running() and not self.stop_sim:
            now = time.time()
            if (now - last_send_wall) < self.send_period:
                time.sleep(0.001)
                continue
            last_send_wall = now

            with self._lock:
                gt = None if self._latest_gt is None else self._latest_gt.copy()
                t_sim = self._latest_t
            if gt is None or t_sim is None:
                continue

            self._seq += 1
            msg = {"seq": self._seq, "t_sim": float(t_sim), "tx_pos": gt.tolist()}

            try:
                sock.send_json(msg)
                resp = sock.recv_json()

                warn = resp.get("warning", None)
                tri = resp.get("tri_pos", None)
                used = resp.get("used", 0)
                resid = resp.get("resid", None)

                if warn:
                    carb.log_warn(f"[SIONNA] seq={resp.get('seq')} warn={warn}")

                if (self._seq % 10) == 0:
                    if tri is None:
                        carb.log_warn(f"[TRI] seq={self._seq} triangulation unavailable used={used} resid={resid}")
                    else:
                        tri = np.array(tri, dtype=float)
                        err = tri - gt
                        carb.log_warn(
                            f"[TRI] seq={self._seq} used={used} resid={float(resid):.2e} "
                            f"GT=({gt[0]:.2f},{gt[1]:.2f},{gt[2]:.2f}) "
                            f"TRI=({tri[0]:.2f},{tri[1]:.2f},{tri[2]:.2f}) "
                            f"e=({err[0]:+.2f},{err[1]:+.2f},{err[2]:+.2f})"
                        )

            except Exception as e:
                # With REQ_RELAXED, we can just continue without recreating sockets constantly
                carb.log_warn(f"[ISAAC] ZMQ timeout/recv issue (continuing): {e}")
                time.sleep(0.05)

        try:
            sock.send_json({"cmd": "stop"})
            _ = sock.recv_json()
        except Exception:
            pass
        try:
            sock.close(0)
        except Exception:
            pass

    def run(self):
        self.timeline.play()
        warmup_steps = 150
        step = 0

        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
            step += 1
            self._update_gt_cache_mainthread()
            if step == warmup_steps:
                self._start_thread()

        self.timeline.stop()
        simulation_app.close()


if __name__ == "__main__":
    PegasusApp(send_hz=10.0).run()
