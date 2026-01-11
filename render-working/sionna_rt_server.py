#!/usr/bin/env python3
"""
SionnaRT Etoile render server (stable) + real-time drone marker + real ray tracing (paths)
WITHOUT using scene.render(paths=...) (which is what triggers:
  TypeError: cannot unpack non-iterable NoneType object
in some SionnaRT installs).

What it does:
- Loads rt.scene.etoile
- Adds visible mesh markers for:
    * UAV (moving)
    * 4 gNB "antenna" markers (static)
- Adds actual Transmitter/Receivers (hidden) for ray tracing computation
- Every render tick:
    * computes paths = PathSolver(scene,...)
    * extracts dominant ray per RX (phi/theta/tau)
    * renders Etoile normally (no paths)
    * overlays 2D projected rays on the image using OpenCV (if available)

ZMQ protocol:
- Receives: {"seq": int, "t_sim": float, "tx_pos": [x,y,z]}  (Etoile/world coords)
- Returns : {"ok": bool, "seq": int, "t_sim": float, "warning": str|None}
"""

import os
import time
import argparse
import inspect
import warnings
import numpy as np
import zmq

# Make OpenCV windows more reliable on GNOME/Wayland
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# Silence noisy warnings
warnings.filterwarnings("ignore", message=r"Unable to import Axes3D.*")
warnings.filterwarnings("ignore", message=r"More than 20 figures have been opened.*")

# Force non-interactive backend (we capture figures -> images)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C0 = 299_792_458.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="tcp://*:5555")
    ap.add_argument("--freq_ghz", type=float, default=28.0)
    ap.add_argument("--variant", default="auto",
                    help="auto|llvm_ad_mono_polarized|cuda_ad_mono_polarized (avoid *_rgb)")
    ap.add_argument("--render_fps", type=float, default=2.0)
    ap.add_argument("--num_samples", type=int, default=32)
    ap.add_argument("--resolution", default="960x540")
    ap.add_argument("--max_depth", type=int, default=2, help="0=LoS only, 1-3 recommended")
    ap.add_argument("--viewer", choices=["auto", "opencv", "save"], default="auto")
    ap.add_argument("--save_dir", default=os.path.join(os.getcwd(), "sionna_renders"))
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--fov_deg", type=float, default=55.0, help="Camera FOV used for 3D->2D projection overlay")
    return ap.parse_args()


args = parse_args()

if os.path.basename(__file__) == "sionna.py":
    raise RuntimeError("Do NOT name this file sionna.py (it shadows the sionna package). Rename it.")

# -------- Mitsuba init BEFORE importing sionna.rt --------
import mitsuba as mi


def pick_variant(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd.strip() not in ["", "-1"]:
        return "cuda_ad_mono_polarized"
    return "llvm_ad_mono_polarized"


VARIANT = pick_variant(args.variant)
if "rgb" in VARIANT.lower():
    VARIANT = "llvm_ad_mono_polarized"

mi.set_variant(VARIANT)
print(f"[SIONNA] Mitsuba variant = {VARIANT}")
print(f"[SIONNA] CUDA_VISIBLE_DEVICES = '{os.environ.get('CUDA_VISIBLE_DEVICES','')}'")

# -------- SionnaRT imports --------
import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Camera, ITURadioMaterial, SceneObject
from sionna.rt import Transmitter, Receiver, PathSolver


def parse_resolution(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)


def to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.array(x)


def make_planar_array_iso():
    # Handle small API differences across SionnaRT releases
    attempts = [
        dict(num_rows=1, num_cols=1, pattern="iso", polarization="V"),
        dict(num_cols=1, num_rows=1, pattern="iso", polarization="V"),
        dict(num_rows=1, num_cols=1, pattern="tr38901", polarization="V"),
    ]
    last = None
    for kw in attempts:
        try:
            return PlanarArray(**kw)
        except Exception as e:
            last = e
    raise TypeError(f"Could not construct PlanarArray. Last error: {last}")


def make_camera(position, look_at):
    try:
        return Camera(position=position, look_at=look_at)
    except TypeError:
        return Camera(position, look_at)


def fig_to_rgb01_and_close(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    rgb01 = buf.astype(np.float32) / 255.0
    plt.close(fig)
    return rgb01


def render_scene_rgb01(scene, cam, resolution, num_samples):
    """
    Render Etoile WITHOUT paths overlay. Tries bitmap mode if supported; otherwise captures matplotlib fig.
    This avoids the SionnaRT bug in scene.render(paths=...).
    """
    w, h = resolution
    sig = inspect.signature(scene.render)
    params = sig.parameters

    # Candidate 1: return_bitmap if supported
    if "return_bitmap" in params:
        kw = {}
        if "camera" in params:
            kw["camera"] = cam
        if "resolution" in params:
            kw["resolution"] = (w, h)
        if "num_samples" in params:
            kw["num_samples"] = int(num_samples)
        kw["return_bitmap"] = True
        # keep show/display off if they exist
        if "show" in params:
            kw["show"] = False
        if "display" in params:
            kw["display"] = False

        out = scene.render(**kw)
        if hasattr(out, "convert"):
            bmp = out.convert(
                pixel_format=mi.Bitmap.PixelFormat.RGB,
                component_format=mi.Struct.Type.Float32,
                srgb_gamma=True,
            )
            arr = np.array(bmp, copy=False)
            if arr.ndim == 1:
                arr = arr.reshape(h, w, 3)
            if arr.shape[-1] > 3:
                arr = arr[..., :3]
            return arr.astype(np.float32)

    # Candidate 2: force show=True so a figure exists (Agg backend prevents GUI popping)
    kw = {}
    if "camera" in params:
        kw["camera"] = cam
    if "resolution" in params:
        kw["resolution"] = (w, h)
    if "num_samples" in params:
        kw["num_samples"] = int(num_samples)
    if "show" in params:
        kw["show"] = True
    if "display" in params:
        kw["display"] = False

    out = scene.render(**kw)

    # output may be (fig, ax), fig, or None (in some versions it draws current fig)
    if isinstance(out, (tuple, list)) and len(out) >= 1 and hasattr(out[0], "canvas"):
        return fig_to_rgb01_and_close(out[0])
    if hasattr(out, "canvas"):
        return fig_to_rgb01_and_close(out)

    if out is None:
        # try current figure
        fig = plt.gcf()
        if fig and hasattr(fig, "canvas"):
            return fig_to_rgb01_and_close(fig)

    raise TypeError(f"scene.render produced unsupported output: {type(out)}")


# ---------- Ray extraction helpers (dominant ray per RX) ----------
def _find_path_axis(arr, num_paths):
    matches = [i for i, s in enumerate(arr.shape) if s == num_paths]
    return matches[-1] if matches else (arr.ndim - 1)


def _reduce_over_all_but_axis(arr, keep_axis, reducer=np.mean):
    arr = np.moveaxis(arr, keep_axis, -1)
    n = arr.shape[-1]
    flat = arr.reshape(-1, n)
    return reducer(flat, axis=0)


def dir_from_phi_theta(phi, theta):
    # theta = zenith from +Z, phi = azimuth in XY from +X toward +Y
    st, ct = np.sin(theta), np.cos(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    u = np.array([st * cp, st * sp, ct], dtype=float)
    return u / (np.linalg.norm(u) + 1e-12)


def dominant_ray_from_paths(paths, a_np, tau_np, rx_idx, rx_pos, tx_pos):
    """
    Returns a unit direction u (RX->towards TX) and a weight (amp) and endpoint for visualization.
    If no path, returns LoS ray.
    """
    rx_pos = np.asarray(rx_pos, dtype=float).reshape(3)
    tx_pos = np.asarray(tx_pos, dtype=float).reshape(3)

    try:
        a_rx = np.asarray(a_np[rx_idx])
        tau_rx = np.asarray(tau_np[rx_idx])

        if a_rx.size == 0 or tau_rx.size == 0:
            raise RuntimeError("empty cir")

        # infer number of paths
        num_paths = int(np.max(tau_rx.shape)) if tau_rx.ndim > 0 else int(tau_rx.shape[0])
        if num_paths <= 0:
            raise RuntimeError("bad num_paths")

        tau_axis = _find_path_axis(tau_rx, num_paths)
        a_axis = _find_path_axis(a_rx, num_paths)

        amp_per_path = _reduce_over_all_but_axis(np.abs(a_rx), a_axis, reducer=np.mean)
        k = int(np.argmax(amp_per_path))
        amp = float(max(amp_per_path[k], 1e-6))

        tau_per_path = _reduce_over_all_but_axis(tau_rx, tau_axis, reducer=np.mean)
        delay = float(max(tau_per_path[k], 0.0))

        phi_r = np.asarray(to_numpy(paths.phi_r)[rx_idx])
        theta_r = np.asarray(to_numpy(paths.theta_r)[rx_idx])

        phi_axis = _find_path_axis(phi_r, num_paths)
        theta_axis = _find_path_axis(theta_r, num_paths)

        phi_per_path = _reduce_over_all_but_axis(phi_r, phi_axis, reducer=np.mean)
        theta_per_path = _reduce_over_all_but_axis(theta_r, theta_axis, reducer=np.mean)

        phi = float(phi_per_path[k])
        theta = float(theta_per_path[k])

        u = dir_from_phi_theta(phi, theta)

        # Flip if it points away from TX
        if float(np.dot(u, (tx_pos - rx_pos))) < 0.0:
            u = -u

        # Use projection distance to TX for a clean visible segment
        d = float(np.dot((tx_pos - rx_pos), u))
        d = max(d, 0.0)
        end = rx_pos + d * u

        return u, amp, end

    except Exception:
        # LoS fallback (still draws something)
        u = tx_pos - rx_pos
        u = u / (np.linalg.norm(u) + 1e-12)
        amp = 1e-3
        end = tx_pos.copy()
        return u, amp, end


# ---------- 3D -> 2D projection for ray overlay ----------
def make_camera_basis(cam_pos, look_at, up=np.array([0.0, 0.0, 1.0], dtype=float)):
    cam_pos = np.asarray(cam_pos, dtype=float)
    look_at = np.asarray(look_at, dtype=float)
    fwd = look_at - cam_pos
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    up2 = np.cross(right, fwd)
    up2 = up2 / (np.linalg.norm(up2) + 1e-12)
    return right, up2, fwd


def project_point(p_world, cam_pos, basis, w, h, fov_deg):
    right, up2, fwd = basis
    p = np.asarray(p_world, dtype=float) - np.asarray(cam_pos, dtype=float)

    x = float(np.dot(right, p))
    y = float(np.dot(up2, p))
    z = float(np.dot(fwd, p))

    if z <= 1e-3:
        return None

    f = 0.5 * w / np.tan(np.deg2rad(fov_deg) * 0.5)
    u = f * (x / z) + (w * 0.5)
    v = -f * (y / z) + (h * 0.5)

    if not np.isfinite(u) or not np.isfinite(v):
        return None

    return int(round(u)), int(round(v))


class Viewer:
    def __init__(self, mode: str, save_dir: str):
        self.mode = mode
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.frame_idx = 0

        self.use_cv2 = False
        self.cv2 = None
        if mode in ("auto", "opencv"):
            try:
                import cv2  # type: ignore
                self.cv2 = cv2
                self.use_cv2 = True
                self.cv2.namedWindow("SionnaRT Render", self.cv2.WINDOW_NORMAL)
            except Exception:
                self.use_cv2 = False

        if mode == "opencv" and not self.use_cv2:
            raise RuntimeError("viewer=opencv requested but OpenCV (cv2) is not available")

    def show(self, rgb01: np.ndarray, title: str, save_every: int):
        if rgb01 is None:
            return
        self.frame_idx += 1

        if self.use_cv2:
            bgr = (np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)[..., ::-1]
            self.cv2.imshow("SionnaRT Render", bgr)
            try:
                self.cv2.setWindowTitle("SionnaRT Render", title)
            except Exception:
                pass
            self.cv2.waitKey(1)
        else:
            if (self.frame_idx % max(1, save_every)) == 0:
                path = os.path.join(self.save_dir, f"frame_{self.frame_idx:06d}.png")
                mi.Bitmap(
                    rgb01,
                    pixel_format=mi.Bitmap.PixelFormat.RGB,
                    component_format=mi.Struct.Type.Float32
                ).write(path)
                print(f"[SIONNA] Saved {path}")

    def draw_line(self, rgb01, p0, p1, color_bgr=(0, 255, 255), thickness=2):
        if not self.use_cv2:
            return rgb01
        img = (np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
        bgr = img[..., ::-1].copy()
        self.cv2.line(bgr, p0, p1, color_bgr, thickness, lineType=self.cv2.LINE_AA)
        out = bgr[..., ::-1].astype(np.float32) / 255.0
        return out


def build_scene(freq_ghz: float):
    scene = load_scene(rt.scene.etoile, merge_shapes=True)
    scene.frequency = float(freq_ghz) * 1e9

    arr = make_planar_array_iso()
    scene.tx_array = arr
    scene.rx_array = arr

    # Keep camera same as your previously working view
    cam_pos = np.array([-360.0, 145.0, 400.0], dtype=float)
    look_at = np.array([-115.0, 33.0, 1.5], dtype=float)
    cam = make_camera(position=cam_pos.tolist(), look_at=look_at.tolist())

    # Center used for placing markers + radios
    cx, cy = -115.0, 33.0

    # Visible UAV marker
    uav_mat = ITURadioMaterial("uav-mat", "metal", thickness=0.01, color=(0.1, 0.9, 0.1))
    uav_obj = SceneObject(fname=rt.scene.low_poly_car, name="uav-marker", radio_material=uav_mat)

    # Visible gNB markers ("antennae")
    gnb_mat = ITURadioMaterial("gnb-mat", "metal", thickness=0.01, color=(0.95, 0.85, 0.1))
    gnb_objs = [
        SceneObject(fname=rt.scene.low_poly_car, name="gNB_1_marker", radio_material=gnb_mat),
        SceneObject(fname=rt.scene.low_poly_car, name="gNB_2_marker", radio_material=gnb_mat),
        SceneObject(fname=rt.scene.low_poly_car, name="gNB_3_marker", radio_material=gnb_mat),
        SceneObject(fname=rt.scene.low_poly_car, name="gNB_4_marker", radio_material=gnb_mat),
    ]

    # Add visible meshes first
    scene.edit(add=[uav_obj] + gnb_objs)

    uav_obj.scaling = 0.15
    uav_obj.position = mi.Point3f(cx, cy, 5.0)

    RX_CONFIG = [
        ("gNB_1", [cx - 220.0, cy - 220.0, 3.0]),
        ("gNB_2", [cx + 220.0, cy - 220.0, 3.0]),
        ("gNB_3", [cx - 220.0, cy + 220.0, 3.0]),
        ("gNB_4", [cx + 220.0, cy + 220.0, 3.0]),
    ]
    for obj, (_n, pos) in zip(gnb_objs, RX_CONFIG):
        obj.scaling = 0.10
        obj.position = mi.Point3f(float(pos[0]), float(pos[1]), float(pos[2]))

    # Add actual radios for ray tracing computation (hidden via display_radius=0)
    # Remove if already exists
    try:
        scene.remove("uav_tx")
    except Exception:
        pass
    tx = Transmitter("uav_tx", position=[cx, cy, 5.0], display_radius=0.0)
    scene.add(tx)

    rx_names = []
    rx_map = {}
    for name, pos in RX_CONFIG:
        try:
            scene.remove(name)
        except Exception:
            pass
        scene.add(Receiver(name, position=pos, display_radius=0.0))
        rx_names.append(name)
        rx_map[name] = np.array(pos, dtype=float)

    solver = PathSolver()
    return scene, cam, cam_pos, look_at, uav_obj, tx, rx_names, rx_map, solver


def main():
    W, H = parse_resolution(args.resolution)
    viewer = Viewer(args.viewer, args.save_dir)

    scene, cam, cam_pos, look_at, uav_obj, tx, rx_names, rx_map, solver = build_scene(args.freq_ghz)
    basis = make_camera_basis(cam_pos, look_at)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)
    print(f"[SIONNA] Listening on {args.bind}")

    render_period = 1.0 / max(args.render_fps, 1e-6)
    last_render = 0.0

    # Initial render (no paths overlay)
    try:
        rgb01 = render_scene_rgb01(scene, cam, (W, H), args.num_samples)
        viewer.show(rgb01, "Etoile (initial)", args.save_every)
    except Exception as e:
        print(f"[SIONNA] Initial render failed: {type(e).__name__}: {e}")

    # per-RX overlay colors
    rx_colors = {
        "gNB_1": (0, 255, 255),   # yellow
        "gNB_2": (255, 0, 255),   # magenta
        "gNB_3": (255, 255, 0),   # cyan-ish (BGR)
        "gNB_4": (0, 255, 0),     # green
    }

    while True:
        req = sock.recv_json()
        cmd = req.get("cmd", "measure")
        if cmd == "stop":
            sock.send_json({"ok": True})
            break

        seq = req.get("seq", None)
        t_sim = req.get("t_sim", None)
        tx_pos = req.get("tx_pos", None)

        warn = None
        if t_sim is None or tx_pos is None:
            sock.send_json({"ok": False, "seq": seq, "error": "Missing t_sim or tx_pos"})
            continue

        # Update UAV mesh + TX position
        try:
            p = np.array(tx_pos, dtype=float).reshape(3)
            if not np.isfinite(p).all():
                raise ValueError("Non-finite tx_pos")
            if p[2] < 1.0:
                p[2] = 1.0

            uav_obj.position = mi.Point3f(float(p[0]), float(p[1]), float(p[2]))
            tx.position = [float(p[0]), float(p[1]), float(p[2])]
        except Exception as e:
            warn = f"UAV update failed: {type(e).__name__}: {e}"
            p = np.array(tx.position, dtype=float)

        now = time.time()
        if (now - last_render) >= render_period:
            last_render = now

            # Compute ray tracing paths
            paths = None
            a_np = tau_np = None
            try:
                paths = solver(scene=scene, max_depth=int(args.max_depth))
                cir_out = paths.cir(normalize_delays=False, out_type="numpy")
                if isinstance(cir_out, (tuple, list)) and len(cir_out) >= 2:
                    a_np, tau_np = cir_out[0], cir_out[1]
                else:
                    raise TypeError("paths.cir() returned unexpected type")
            except Exception as e:
                warn = (warn + " | " if warn else "") + f"PathSolver/CIR failed: {type(e).__name__}: {e}"
                paths = None

            # Render base scene (stable)
            try:
                rgb01 = render_scene_rgb01(scene, cam, (W, H), args.num_samples)
            except Exception as e:
                warn = (warn + " | " if warn else "") + f"Render failed: {type(e).__name__}: {e}"
                print(f"[SIONNA] seq={seq} warning={warn}")
                sock.send_json({"ok": True, "seq": seq, "t_sim": float(t_sim), "warning": warn})
                continue

            # Overlay rays (dominant path per gNB)
            if (paths is not None) and (a_np is not None) and (tau_np is not None) and viewer.use_cv2:
                for rx_idx, rx_name in enumerate(rx_names):
                    rx_pos = rx_map[rx_name]
                    _u, amp, end = dominant_ray_from_paths(paths, a_np, tau_np, rx_idx, rx_pos, p)

                    p0 = project_point(rx_pos, cam_pos, basis, W, H, args.fov_deg)
                    p1 = project_point(end,    cam_pos, basis, W, H, args.fov_deg)
                    if (p0 is None) or (p1 is None):
                        continue

                    thickness = 2 if amp < 0.05 else 3
                    rgb01 = viewer.draw_line(
                        rgb01, p0, p1,
                        color_bgr=rx_colors.get(rx_name, (0, 255, 255)),
                        thickness=thickness
                    )

            title = f"Etoile | seq={seq} t={float(t_sim):.2f} depth={args.max_depth}"
            viewer.show(rgb01, title, args.save_every)

        sock.send_json({"ok": True, "seq": seq, "t_sim": float(t_sim), "warning": warn})

    print("[SIONNA] Exiting")


if __name__ == "__main__":
    main()
