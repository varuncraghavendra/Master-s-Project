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
from omni.isaac.core.utils.prims import create_prim

from pegasus.simulator.params import SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Isaac world -> Etoile world mapping (your chosen offset)
SCENE_OFFSET = np.array([-115.0, 33.0, 0.0], dtype=float)
SCENE_SCALE  = 1.0

def isaac_to_sionna(pos_world: np.ndarray) -> np.ndarray:
    return SCENE_OFFSET + SCENE_SCALE * pos_world

class App:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        self.world.reset()
        self.stage = omni.usd.get_context().get_stage()

        # Track this if it exists; otherwise marker will be used
        self.quad_path = "/World/quadrotor"
        self.quad_prim = XFormPrim(self.quad_path)  # will be invalid if not created by your other script

        # Create a kinematic marker cube (always available)
        create_prim("/World/drone_marker", "Cube")
        self.marker = XFormPrim("/World/drone_marker")
        self.marker.set_world_pose([0.0, 0.0, 1.0], [1, 0, 0, 0])

        self.stop_sim = False
        self.sionna_addr = "tcp://127.0.0.1:5555"

        self._lock = threading.Lock()
        self._latest_scene = None
        self._latest_t = None
        self._seq = 0

        self.send_period_s = 0.5  # 2Hz

        # marker motion params (square)
        self.v = 0.8
        self.side = 5.0

    def _marker_square(self, t):
        # simple square param
        T = self.side / self.v
        period = 4 * T
        u = (t % period)

        x, y = 0.0, 0.0
        if u < T:
            x = self.v * u; y = 0.0
        elif u < 2*T:
            x = self.side; y = self.v * (u - T)
        elif u < 3*T:
            x = self.side - self.v * (u - 2*T); y = self.side
        else:
            x = 0.0; y = self.side - self.v * (u - 3*T)
        return np.array([x, y, 5.0], dtype=float)

    def _update_pose_cache_mainthread(self):
        t_sim = float(self.timeline.get_current_time())

        # Update marker pose (guarantees motion)
        p = self._marker_square(t_sim)
        self.marker.set_world_pose(p.tolist(), [1, 0, 0, 0])

        # Prefer quad if it exists; else marker
        try:
            qpos, _qquat = self.quad_prim.get_world_pose()
            qpos = np.array(qpos, dtype=float).reshape(3)
            if np.isfinite(qpos).all():
                pos_world = qpos
            else:
                pos_world = p
        except Exception:
            pos_world = p

        pos_scene = isaac_to_sionna(np.array(pos_world, dtype=float))
        with self._lock:
            self._latest_scene = pos_scene
            self._latest_t = t_sim

    def _zmq_loop(self):
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        sock.setsockopt(zmq.SNDTIMEO, 5000)
        sock.connect(self.sionna_addr)

        last_send = 0.0
        while simulation_app.is_running() and not self.stop_sim:
            if time.time() - last_send < self.send_period_s:
                time.sleep(0.01)
                continue
            last_send = time.time()

            with self._lock:
                pos_scene = None if self._latest_scene is None else self._latest_scene.copy()
                t_sim = self._latest_t

            if pos_scene is None or t_sim is None:
                continue

            self._seq += 1
            msg = {"seq": self._seq, "t_sim": float(t_sim), "tx_pos": pos_scene.tolist()}

            try:
                sock.send_json(msg)
                resp = sock.recv_json()
                warn = resp.get("warning", None)
                if warn:
                    carb.log_warn(f"[SIONNA] seq={resp.get('seq')} warning={warn}")
            except Exception as e:
                carb.log_warn(f"[ISAAC] ZMQ error: {e}")
                time.sleep(0.2)

        try:
            sock.send_json({"cmd": "stop"})
            _ = sock.recv_json()
        except Exception:
            pass
        sock.close()

    def run(self):
        self.timeline.play()

        # start ZMQ sender thread
        threading.Thread(target=self._zmq_loop, daemon=True).start()

        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
            self._update_pose_cache_mainthread()

        self.timeline.stop()
        simulation_app.close()

if __name__ == "__main__":
    App().run()
