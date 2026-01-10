#!/usr/bin/env python3
import time
import numpy as np
import zmq

import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver


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
    """Return dominant path index k plus phi/theta/tau/amp (robust to tensor shapes)."""
    a_rx = np.asarray(a_np[rx_idx])
    tau_rx = np.asarray(tau_np[rx_idx])

    if tau_rx.size == 0 or a_rx.size == 0:
        raise RuntimeError("Empty CIR returned (no paths)")

    num_paths = int(np.max(tau_rx.shape)) if tau_rx.ndim > 0 else int(tau_rx.shape[0])
    if num_paths <= 0:
        raise RuntimeError("Invalid num_paths inferred")

    tau_path_axis = _find_path_axis(tau_rx, num_paths)
    a_path_axis = _find_path_axis(a_rx, num_paths)

    amp_per_path = _reduce_over_all_but_axis(np.abs(a_rx), a_path_axis, reducer=np.mean)
    k = int(np.argmax(amp_per_path))
    amp = float(amp_per_path[k])

    tau_per_path = _reduce_over_all_but_axis(tau_rx, tau_path_axis, reducer=np.mean)
    delay = float(tau_per_path[k])

    phi_r = np.asarray(to_numpy(paths.phi_r)[rx_idx])
    theta_r = np.asarray(to_numpy(paths.theta_r)[rx_idx])

    phi_path_axis = _find_path_axis(phi_r, num_paths)
    theta_path_axis = _find_path_axis(theta_r, num_paths)

    phi_per_path = _reduce_over_all_but_axis(phi_r, phi_path_axis, reducer=np.mean)
    theta_per_path = _reduce_over_all_but_axis(theta_r, theta_path_axis, reducer=np.mean)

    phi = float(phi_per_path[k])
    theta = float(theta_per_path[k])

    return k, phi, theta, delay, amp


def dir_from_phi_theta(phi, theta, mode):
    """
    Build a unit vector from (phi, theta) with multiple conventions.

    Base formulas:
      - zenith_xy : theta is zenith from +Z, phi from +X toward +Y
      - zenith_yx : same but swap x/y (phi from +Y)
      - elev_xy   : theta is elevation from XY plane
      - elev_yx   : elevation + swap x/y
    """
    # ensure float
    phi = float(phi)
    theta = float(theta)

    if mode.endswith("_deg"):
        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)

    base = mode.replace("_deg", "").replace("_rad", "")

    if base == "zenith_xy":
        st, ct = np.sin(theta), np.cos(theta)
        cp, sp = np.cos(phi), np.sin(phi)
        u = np.array([st * cp, st * sp, ct], dtype=float)
    elif base == "zenith_yx":
        st, ct = np.sin(theta), np.cos(theta)
        cp, sp = np.cos(phi), np.sin(phi)
        u = np.array([st * sp, st * cp, ct], dtype=float)
    elif base == "elev_xy":
        ce, se = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(phi), np.sin(phi)
        u = np.array([ce * cp, ce * sp, se], dtype=float)
    elif base == "elev_yx":
        ce, se = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(phi), np.sin(phi)
        u = np.array([ce * sp, ce * cp, se], dtype=float)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    n = np.linalg.norm(u)
    return u / (n + 1e-12)


def triangulate_and_residual(rx_positions, ray_dirs, weights):
    """
    Solve x minimizing Σ w ||(I - u u^T)(x - r)||^2.
    Return x, residual.
    """
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)

    Ms = []
    rs = []
    ws = []
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

    # residual
    resid = 0.0
    for M, r, w in zip(Ms, rs, ws):
        d = M @ (x - r)
        resid += w * float(d @ d)
    return x, resid


def main():
    BIND = "tcp://*:5555"

    # Scene (replace later with an Isaac-matched scene)
    scene = load_scene(rt.scene.munich, merge_shapes=True)
    scene.frequency = 28e9

    # Keyword-only PlanarArray (matches your build)
    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="tr38901", polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="dipole", polarization="cross",
    )

    # Receivers (make these match Isaac world frame)
    RX_CONFIG = [
        ("gNB_1", [ 20.0,  0.0, 1.5]),
        ("gNB_2", [-20.0,  0.0, 1.5]),
        ("gNB_3", [  0.0, 20.0, 1.5]),
        ("gNB_4", [  0.0,-20.0, 1.5]),
    ]
    rx_positions_map = {}
    rx_names = []
    for name, pos in RX_CONFIG:
        scene.add(Receiver(name=name, position=pos))
        rx_positions_map[name] = np.array(pos, dtype=float)
        rx_names.append(name)

    tx = Transmitter(name="uav_tx", position=[0.0, 0.0, 2.0])
    scene.add(tx)

    solver = PathSolver()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(BIND)
    print(f"[SIONNA_SENSOR] Listening on {BIND}")

    # Candidate conversion modes
    base_modes = ["zenith_xy", "zenith_yx", "elev_xy", "elev_yx"]
    modes = []
    for b in base_modes:
        modes += [b + "_rad", b + "_deg"]

    while True:
        req = sock.recv_json()
        t_sim = req.get("t_sim", None)
        tx_pos = req.get("tx_pos", None)
        home = req.get("home_pos_world", None)

        if t_sim is None or tx_pos is None:
            sock.send_json({"ok": False, "t_sim": t_sim, "error": "Missing t_sim or tx_pos"})
            continue

        t0 = time.perf_counter()
        try:
            tx.position = tx_pos

            t_s0 = time.perf_counter()
            paths = solver(
                scene=scene,
                max_depth=2,
                los=True,
                specular_reflection=True,
                diffuse_reflection=False,
                refraction=False,
                synthetic_array=True,
                seed=1,
            )
            t_s1 = time.perf_counter()

            a_np, tau_np = paths.cir(normalize_delays=False, out_type="numpy")

            # Extract dominant path parameters per RX
            per_rx_raw = {}
            valid_rxs = []
            for rx_idx, rx_name in enumerate(rx_names):
                try:
                    k, phi, theta, delay, amp = pick_dominant_path_k_phi_theta_tau_amp(paths, a_np, tau_np, rx_idx)
                    per_rx_raw[rx_name] = {
                        "valid": True,
                        "k": int(k),
                        "phi": float(phi),
                        "theta": float(theta),
                        "tau_s": float(delay),
                        "amp": float(amp),
                        "rx_pos_world": rx_positions_map[rx_name].tolist(),
                    }
                    valid_rxs.append(rx_name)
                except Exception as e_rx:
                    per_rx_raw[rx_name] = {"valid": False, "error": f"{type(e_rx).__name__}: {e_rx}"}

            if len(valid_rxs) < 2:
                sock.send_json({
                    "ok": False,
                    "t_sim": float(t_sim),
                    "error": "Not enough valid receivers (need >=2)",
                    "n_valid_rx": len(valid_rxs),
                    "t_solve_ms": 1000.0 * (t_s1 - t_s0),
                    "t_total_ms": 1000.0 * (time.perf_counter() - t0),
                    "a_shape": list(np.shape(a_np)),
                    "tau_shape": list(np.shape(tau_np)),
                    "per_rx": per_rx_raw,
                })
                continue

            # Try multiple angle conventions and pick the best residual
            best = None  # (resid, est, conv_used, rays_used)
            for mode in modes:
                for sign in [+1.0, -1.0]:
                    rx_positions = []
                    ray_dirs = []
                    weights = []

                    for rx_name in valid_rxs:
                        info = per_rx_raw[rx_name]
                        phi = info["phi"]
                        theta = info["theta"]
                        u = dir_from_phi_theta(phi, theta, mode) * sign

                        # Ray direction should point from RX toward TX
                        rx_positions.append(rx_positions_map[rx_name])
                        ray_dirs.append(u)
                        weights.append(max(info["amp"], 1e-6))

                    est, resid = triangulate_and_residual(rx_positions, ray_dirs, weights)

                    if (best is None) or (resid < best[0]):
                        best = (resid, est, f"{mode},sign={int(sign)}")

            resid, est_world, conv_used = best

            resp = {
                "ok": True,
                "t_sim": float(t_sim),
                "est_pos_world": est_world.tolist(),
                "residual": float(resid),
                "conv_used": conv_used,
                "n_valid_rx": len(valid_rxs),
                "a_shape": list(np.shape(a_np)),
                "tau_shape": list(np.shape(tau_np)),
                "t_solve_ms": 1000.0 * (t_s1 - t_s0),
                "t_total_ms": 1000.0 * (time.perf_counter() - t0),
                "per_rx": per_rx_raw,
            }

            if home is not None:
                h = np.array(home, dtype=float).reshape(3)
                resp["home_pos_world"] = home
                resp["est_pos_home"] = (est_world - h).tolist()

            sock.send_json(resp)

        except Exception as e:
            sock.send_json({
                "ok": False,
                "t_sim": float(t_sim) if t_sim is not None else None,
                "error": f"{type(e).__name__}: {e}",
                "t_total_ms": 1000.0 * (time.perf_counter() - t0),
            })


if __name__ == "__main__":
    main()
