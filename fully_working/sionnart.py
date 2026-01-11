#!/usr/bin/env python3
import os
import time
import json
import csv
import numpy as np
import zmq

# --- Matplotlib 3D live plot ---
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

C0 = 299_792_458.0


# ------------------------- helpers -------------------------
def to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.array(x)


def _find_path_axis(arr, num_paths):
    matches = [i for i, s in enumerate(arr.shape) if s == num_paths]
    return matches[-1] if matches else (arr.ndim - 1)


def _reduce_over_all_but_axis(arr, keep_axis, reducer=np.mean):
    arr = np.moveaxis(arr, keep_axis, -1)
    n = arr.shape[-1]
    flat = arr.reshape(-1, n)
    return reducer(flat, axis=0)


def pick_dominant_path_k_phi_theta_tau_amp(paths, a_np, tau_np, rx_idx):
    """Dominant-path extraction, returns (k,phi,theta,tau,amp). Raises if empty."""
    a_rx = np.asarray(a_np[rx_idx])
    tau_rx = np.asarray(tau_np[rx_idx])

    if a_rx.size == 0 or tau_rx.size == 0:
        raise RuntimeError("Empty CIR (no paths)")

    num_paths = int(np.max(tau_rx.shape)) if tau_rx.ndim > 0 else int(tau_rx.shape[0])
    if num_paths <= 0:
        raise RuntimeError("Invalid num_paths inferred")

    tau_axis = _find_path_axis(tau_rx, num_paths)
    a_axis = _find_path_axis(a_rx, num_paths)

    amp_per_path = _reduce_over_all_but_axis(np.abs(a_rx), a_axis, reducer=np.mean)
    k = int(np.argmax(amp_per_path))
    amp = float(amp_per_path[k])

    tau_per_path = _reduce_over_all_but_axis(tau_rx, tau_axis, reducer=np.mean)
    delay = float(tau_per_path[k])

    phi_r = np.asarray(to_numpy(paths.phi_r)[rx_idx])
    theta_r = np.asarray(to_numpy(paths.theta_r)[rx_idx])

    phi_axis = _find_path_axis(phi_r, num_paths)
    theta_axis = _find_path_axis(theta_r, num_paths)

    phi_per_path = _reduce_over_all_but_axis(phi_r, phi_axis, reducer=np.mean)
    theta_per_path = _reduce_over_all_but_axis(theta_r, theta_axis, reducer=np.mean)

    phi = float(phi_per_path[k])
    theta = float(theta_per_path[k])

    return k, phi, theta, delay, amp


def dir_from_phi_theta(phi, theta):
    """
    Practical default for SionnaRT:
      theta = zenith angle from +Z (rad)
      phi   = azimuth in XY from +X towards +Y (rad)
    """
    st, ct = np.sin(theta), np.cos(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    u = np.array([st * cp, st * sp, ct], dtype=float)
    return u / (np.linalg.norm(u) + 1e-12)


def triangulate_and_residual(rx_positions, ray_dirs, weights):
    """Weighted least squares intersection of lines."""
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)

    Ms, rs, ws = [], [], []
    for r, u, w in zip(rx_positions, ray_dirs, weights):
        r = np.asarray(r, dtype=float).reshape(3)
        u = np.asarray(u, dtype=float).reshape(3)
        u = u / (np.linalg.norm(u) + 1e-12)
        M = np.eye(3) - np.outer(u, u)

        A += w * M
        b += w * (M @ r)

        Ms.append(M); rs.append(r); ws.append(w)

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)

    resid = 0.0
    for M, r, w in zip(Ms, rs, ws):
        d = M @ (x - r)
        resid += w * float(d @ d)

    return x, resid


# ------------------------- live 3D plot -------------------------
class LivePlot3D:
    def __init__(self, rx_positions_map, axis_pad=40.0):
        self.rx_positions_map = {k: np.array(v, dtype=float).reshape(3) for k, v in rx_positions_map.items()}

        plt.ion()
        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")

        rx_pts = np.array(list(self.rx_positions_map.values()), dtype=float)
        self.ax.scatter(rx_pts[:, 0], rx_pts[:, 1], rx_pts[:, 2], marker="^", s=100, label="gNBs")
        for name, p in self.rx_positions_map.items():
            self.ax.text(p[0], p[1], p[2] + 0.5, name, fontsize=9)

        # Rename GT -> Quad
        self.quad_sc = self.ax.scatter([0], [0], [0], marker="o", s=70, label="Quad (Isaac)")
        self.est_sc  = self.ax.scatter([0], [0], [0], marker="x", s=90, label="EST (SionnaRT)")

        self.ray_lines = {}
        self.link_lines = {}
        for name, p in self.rx_positions_map.items():
            ln_ray, = self.ax.plot([p[0], p[0]], [p[1], p[1]], [p[2], p[2]], linewidth=2.2, alpha=0.85)
            ln_lnk, = self.ax.plot([p[0], p[0]], [p[1], p[1]], [p[2], p[2]], linestyle="--", linewidth=1.2, alpha=0.55)
            self.ray_lines[name] = ln_ray
            self.link_lines[name] = ln_lnk

        self.title = self.ax.set_title("SionnaRT Live (Quad vs Estimate)")
        self.info = self.fig.text(
            0.02, 0.98, "", va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )

        self.ax.legend(loc="lower left")

        self.ax.set_xlim(rx_pts[:, 0].min() - axis_pad, rx_pts[:, 0].max() + axis_pad)
        self.ax.set_ylim(rx_pts[:, 1].min() - axis_pad, rx_pts[:, 1].max() + axis_pad)
        self.ax.set_zlim(0.0, max(20.0, rx_pts[:, 2].max() + axis_pad))
        self.ax.view_init(elev=25, azim=45)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    @staticmethod
    def _set_scatter(sc, p):
        sc._offsets3d = ([p[0]], [p[1]], [p[2]])

    @staticmethod
    def _set_line(ln, p0, p1):
        ln.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        ln.set_3d_properties([p0[2], p1[2]])

    def update(self, t_sim, quad, est, err_m, resid, used_fallback, ray_endpoints):
        if quad is not None:
            self._set_scatter(self.quad_sc, quad)
        if est is not None:
            self._set_scatter(self.est_sc, est)

        for name, rx in self.rx_positions_map.items():
            if name in ray_endpoints:
                self._set_line(self.ray_lines[name], rx, ray_endpoints[name])
            else:
                self._set_line(self.ray_lines[name], rx, rx)

            if est is not None:
                self._set_line(self.link_lines[name], rx, est)
            else:
                self._set_line(self.link_lines[name], rx, rx)

        self.title.set_text(f"SionnaRT Live | t={t_sim:.2f}s | err={err_m:.2f}m | fallback={used_fallback}")

        msg = (
            f"Quad: ({quad[0]:.2f}, {quad[1]:.2f}, {quad[2]:.2f})\n"
            f"EST : ({est[0]:.2f}, {est[1]:.2f}, {est[2]:.2f})\n"
            f"err_m={err_m:.3f}\n"
            f"resid={resid:.3e}\n"
            f"fallback_used={used_fallback}"
        )
        self.info.set_text(msg)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


def main():
    BIND = "tcp://*:5555"

    # Use a LIGHT, reliable scene so every RX gets LoS often
    scene = load_scene(rt.scene.floor_wall, merge_shapes=True)
    scene.frequency = 28e9

    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
                                 pattern="tr38901", polarization="V")

    # gNBs far from the 5x5 square around (0..5,0..5)
    RX_CONFIG = [
        ("gNB_1", [-30.0, -30.0, 3.0]),
        ("gNB_2", [ 35.0, -30.0, 3.0]),
        ("gNB_3", [-30.0,  35.0, 3.0]),
        ("gNB_4", [ 35.0,  35.0, 3.0]),
    ]

    rx_names = []
    rx_positions_map = {}
    for name, pos in RX_CONFIG:
        scene.add(Receiver(name=name, position=pos))
        rx_names.append(name)
        rx_positions_map[name] = np.array(pos, dtype=float)

    tx = Transmitter(name="uav_tx", position=[0.0, 0.0, 2.0])
    scene.add(tx)

    solver = PathSolver()

    # ZMQ REP
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(BIND)
    print(f"[SIONNA] Listening on {BIND}")

    plotter = LivePlot3D(rx_positions_map)

    # Records for report
    records = []
    run_id_default = time.strftime("%Y%m%d_%H%M%S")

    while True:
        req = sock.recv_json()
        cmd = req.get("cmd", "measure")

        if cmd == "stop":
            save_dir = req.get("save_dir", os.getcwd())
            run_id = req.get("run_id", run_id_default)
            os.makedirs(save_dir, exist_ok=True)

            csv_path = os.path.join(save_dir, f"localization_{run_id}.csv")
            summary_path = os.path.join(save_dir, f"localization_{run_id}_summary.json")

            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["seq","t_sim","quad_x","quad_y","quad_z","est_x","est_y","est_z","err_m","residual","fallback_used","ok","error"])
                for r in records:
                    w.writerow([
                        r.get("seq"), r.get("t_sim"),
                        *(r.get("quad") or [None]*3),
                        *(r.get("est")  or [None]*3),
                        r.get("err_m"), r.get("resid"), r.get("fallback_used"),
                        r.get("ok"), r.get("error")
                    ])

            errs = [r["err_m"] for r in records if isinstance(r.get("err_m"), (int,float))]
            summary = {
                "run_id": run_id,
                "samples": len(records),
                "ok_samples": sum(1 for r in records if r.get("ok")),
                "mean_err": float(np.mean(errs)) if errs else None,
                "rmse_err": float(np.sqrt(np.mean(np.square(errs)))) if errs else None,
                "p95_err": float(np.percentile(errs, 95)) if errs else None,
                "csv_path": csv_path,
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            sock.send_json({"ok": True, "csv_path": csv_path, "summary_path": summary_path})
            break

        # measure
        seq = req.get("seq", None)
        t_sim = req.get("t_sim", None)
        quad_pos = req.get("tx_pos", None)

        if t_sim is None or quad_pos is None:
            sock.send_json({"ok": False, "seq": seq, "error": "Missing t_sim or tx_pos"})
            continue

        quad = np.array(quad_pos, dtype=float).reshape(3)

        try:
            tx.position = quad.tolist()

            # LoS only for reliability; you can increase depth later
            paths = solver(
                scene=scene,
                max_depth=0,
                los=True,
                specular_reflection=False,
                diffuse_reflection=False,
                refraction=False,
                synthetic_array=True,
                seed=1,
            )

            a_np, tau_np = paths.cir(normalize_delays=False, out_type="numpy")

            rx_positions = []
            ray_dirs = []
            weights = []
            fallback_used = False

            ray_endpoints = {}

            for rx_idx, rx_name in enumerate(rx_names):
                rx = rx_positions_map[rx_name]
                try:
                    k, phi, theta, delay, amp = pick_dominant_path_k_phi_theta_tau_amp(paths, a_np, tau_np, rx_idx)
                    u = dir_from_phi_theta(phi, theta)

                    # Ensure ray points roughly toward the quad (sign ambiguity fix)
                    if float(np.dot(u, (quad - rx))) < 0.0:
                        u = -u

                    w = max(amp, 1e-6)
                    tau = delay
                except Exception:
                    # fallback synthetic measurement (low weight)
                    fallback_used = True
                    u = (quad - rx)
                    u = u / (np.linalg.norm(u) + 1e-12)
                    tau = float(np.linalg.norm(quad - rx) / C0)
                    w = 1e-3

                rx_positions.append(rx)
                ray_dirs.append(u)
                weights.append(w)

                # ray endpoint: project to quad distance for visualization
                d = float(np.dot((quad - rx), u))
                d = max(d, 0.0)
                ray_endpoints[rx_name] = rx + d * u

            est, resid = triangulate_and_residual(rx_positions, ray_dirs, weights)

            err_m = float(np.linalg.norm(est - quad))

            plotter.update(float(t_sim), quad, est, err_m, resid, fallback_used, ray_endpoints)

            records.append({
                "seq": seq,
                "t_sim": float(t_sim),
                "quad": quad.tolist(),
                "est": est.tolist(),
                "err_m": err_m,
                "resid": float(resid),
                "fallback_used": bool(fallback_used),
                "ok": True,
                "error": None,
            })

            sock.send_json({
                "ok": True,
                "seq": seq,
                "t_sim": float(t_sim),
                "est_pos_world": est.tolist(),
                "err_m": err_m,
                "residual": float(resid),
                "fallback_used": bool(fallback_used),
            })

        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            records.append({
                "seq": seq,
                "t_sim": float(t_sim),
                "quad": quad.tolist(),
                "est": None,
                "err_m": None,
                "resid": None,
                "fallback_used": None,
                "ok": False,
                "error": msg,
            })
            sock.send_json({"ok": False, "seq": seq, "t_sim": float(t_sim), "error": msg})

    print("[SIONNA] Exiting")


if __name__ == "__main__":
    main()
