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


def aoa_to_unit_vec(theta, phi):
    """
    Convert spherical angles (theta, phi) to a unit vector in global XYZ.
    We only need the line direction; sign doesn't matter for our LS triangulation.
    """
    st = np.sin(theta)
    ct = np.cos(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    return np.array([st * cp, st * sp, ct], dtype=float)


def triangulate_rays(rx_positions, ray_dirs, weights=None):
    """
    Least squares intersection of 3D lines:
      minimize Σ ||(I - u u^T)(x - r)||^2
    Note: invariant to u -> -u.
    """
    if weights is None:
        weights = [1.0] * len(rx_positions)

    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)

    for r, u, w in zip(rx_positions, ray_dirs, weights):
        r = np.asarray(r, dtype=float).reshape(3)
        u = np.asarray(u, dtype=float).reshape(3)
        u = u / (np.linalg.norm(u) + 1e-12)

        M = np.eye(3) - np.outer(u, u)
        A += w * M
        b += w * (M @ r)

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x


def pick_dominant_path_angles(paths, a_np, tau_np, rx_idx):
    """
    Pick dominant path by |a| for that receiver.
    a_np (CIR amplitudes) and tau_np (delays) come from paths.cir(out_type="numpy").

    a_np is typically shaped like:
      [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    tau_np:
      [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    """
    # Use first antenna and first TX indices (1x1 arrays makes this clean)
    a = a_np[rx_idx, 0, 0, 0, :, 0]
    tau = tau_np[rx_idx, 0, 0, 0, :]

    k = int(np.argmax(np.abs(a)))
    amp = float(np.abs(a[k]))
    delay = float(tau[k])

    # Try common indexing for angles
    phi_r = to_numpy(paths.phi_r)
    theta_r = to_numpy(paths.theta_r)

    # Most common shape: [rx, rx_ant, tx, tx_ant, path]
    try:
        phi = float(phi_r[rx_idx, 0, 0, 0, k])
        theta = float(theta_r[rx_idx, 0, 0, 0, k])
    except Exception:
        # fallback: [rx, path]
        phi = float(phi_r[rx_idx, k])
        theta = float(theta_r[rx_idx, k])

    return phi, theta, delay, amp


def main():
    BIND = "tcp://*:5555"

    # --- Scene ---
    # For bring-up use a built-in Sionna scene. Later replace with a Mitsuba XML matching Isaac.
    scene = load_scene(rt.scene.munich, merge_shapes=True)

    # 5G-ish carrier
    scene.frequency = 28e9  # Hz

    # Arrays (fast bring-up: 1x1, synthetic_array=True)
    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="tr38901", polarization="V"
    )
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="dipole", polarization="cross"
    )

    # --- Ground receivers (EDIT to match your Isaac Sim coordinates) ---
    RX_CONFIG = [
        ("gNB_1", [0.0,  0.0,  1.5]),
        ("gNB_2", [50.0, 0.0,  1.5]),
        ("gNB_3", [0.0,  50.0, 1.5]),
        ("gNB_4", [50.0, 50.0, 1.5]),
    ]

    rx_names = []
    for name, pos in RX_CONFIG:
        scene.add(Receiver(name=name, position=pos, display_radius=0.8))
        rx_names.append(name)

    # UAV transmitter
    tx = Transmitter(name="uav_tx", position=[0.0, 0.0, 2.0], display_radius=0.8)
    scene.add(tx)

    solver = PathSolver()

    # Optional: store home received from Isaac
    home_pos_world = None

    # ZMQ REP server
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(BIND)
    print(f"[SIONNA_SENSOR] Listening on {BIND}")

    # Measurement noise knobs (optional)
    aoa_noise_deg = 1.0
    aoa_noise = np.deg2rad(aoa_noise_deg)

    while True:
        req = sock.recv_json()

        # INIT message: Isaac tells us "home"
        if req.get("type") == "init":
            home_pos_world = req.get("home_pos_world", None)
            sock.send_json({"ok": True, "type": "init_ack", "home_pos_world": home_pos_world})
            continue

        # Regular measurement request
        tx_pos = req.get("tx_pos", None)
        t_sim = req.get("t_sim", None)

        if tx_pos is None or t_sim is None:
            sock.send_json({"ok": False, "t_sim": t_sim, "error": "Missing tx_pos or t_sim"})
            continue

        # Update UAV TX pose
        tx.position = tx_pos

        # Compute paths
        paths = solver(
            scene=scene,
            max_depth=2,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=False,
            synthetic_array=True,
            seed=1
        )

        # CIR for dominant path selection
        a_np, tau_np = paths.cir(normalize_delays=False, out_type="numpy")

        per_rx = {}
        rx_positions = []
        ray_dirs = []
        weights = []

        # IMPORTANT: We assume rx ordering in paths matches insertion order.
        # We iterate in the same order we created receivers.
        for rx_idx, rx_name in enumerate(rx_names):
            phi, theta, delay, amp = pick_dominant_path_angles(paths, a_np, tau_np, rx_idx=rx_idx)

            # Add synthetic noise
            phi_n = phi + np.random.normal(0.0, aoa_noise)
            theta_n = theta + np.random.normal(0.0, aoa_noise)

            u = aoa_to_unit_vec(theta_n, phi_n)
            r = np.array(scene.receivers[rx_name].position, dtype=float).reshape(3)

            rx_positions.append(r)
            ray_dirs.append(u)
            weights.append(max(amp, 1e-6))

            per_rx[rx_name] = {
                "rx_pos_world": r.tolist(),
                "phi_r_rad": float(phi_n),
                "theta_r_rad": float(theta_n),
                "tau_s": float(delay),
                "amp": float(amp),
            }

        if len(rx_positions) < 2:
            sock.send_json({"ok": False, "t_sim": t_sim, "error": "Need >=2 receivers", "per_rx": per_rx})
            continue

        est_world = triangulate_rays(rx_positions, ray_dirs, weights=weights)

        resp = {
            "ok": True,
            "t_sim": float(t_sim),          # echo Isaac sim time (synchronization)
            "est_pos_world": est_world.tolist(),
            "per_rx": per_rx,
        }

        # If we know home, also return home-relative estimate
        if home_pos_world is not None:
            h = np.array(home_pos_world, dtype=float).reshape(3)
            resp["home_pos_world"] = home_pos_world
            resp["est_pos_home"] = (est_world - h).tolist()

        sock.send_json(resp)


if __name__ == "__main__":
    main()
