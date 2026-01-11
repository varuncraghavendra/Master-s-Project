#!/usr/bin/env python3
"""
SionnaRT Etoile renderer + ZMQ position updates.

Fixes:
- Your Sionna 'scene.render()' creates matplotlib figures by default -> no image shown + open-figure leak.
- This script forces bitmap return when supported, otherwise converts matplotlib Figure to RGB,
  then ALWAYS closes the figure to avoid the "More than 20 figures opened" warning.
- Uses OpenCV window if available (viewer=auto/opencv), else saves PNGs.
"""

import os
import time
import argparse
import inspect
import warnings
import numpy as np
import zmq

# ---- reduce noisy warnings ----
warnings.filterwarnings("ignore", message=r"Unable to import Axes3D.*")
warnings.filterwarnings("ignore", message=r"More than 20 figures have been opened.*")

# Ensure Qt window works on GNOME Wayland setups (OpenCV highgui)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# Force non-interactive matplotlib backend BEFORE Sionna imports (Sionna may use matplotlib internally)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="tcp://*:5555")
    ap.add_argument("--freq_ghz", type=float, default=28.0)
    ap.add_argument("--variant", default="auto",
                    help="auto|llvm_ad_mono_polarized|cuda_ad_mono_polarized (never *_rgb)")
    ap.add_argument("--render_fps", type=float, default=2.0)
    ap.add_argument("--num_samples", type=int, default=32)
    ap.add_argument("--resolution", default="960x540")
    ap.add_argument("--viewer", choices=["auto", "opencv", "save"], default="auto")
    ap.add_argument("--save_dir", default=os.path.join(os.getcwd(), "sionna_renders"))
    ap.add_argument("--save_every", type=int, default=5)
    return ap.parse_args()


args = parse_args()

# Guard: don't shadow package
if os.path.basename(__file__) == "sionna.py":
    raise RuntimeError("Do NOT name this file sionna.py (it shadows the sionna package). Rename it.")


# ----------------------- Mitsuba/DrJit init BEFORE importing sionna.rt -----------------------
import drjit as dr
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

# ----------------------- Import SionnaRT -----------------------
import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Camera, ITURadioMaterial, SceneObject


def parse_resolution(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)


def make_planar_array():
    # Sionna versions differ; try a few constructor styles
    attempts = [
        dict(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
             pattern="tr38901", polarization="V"),
        dict(num_rows=1, num_cols=1, v_spacing=0.5, h_spacing=0.5,
             pattern="tr38901", polarization="V"),
        dict(num_rows=1, num_cols=1, element_spacing=(0.5, 0.5),
             pattern="tr38901", polarization="V"),
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


def figure_to_rgb01(fig):
    """Convert Matplotlib Figure to float32 RGB [0..1] and CLOSE it."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    rgb01 = buf.astype(np.float32) / 255.0
    plt.close(fig)
    return rgb01


def render_to_rgb01(scene, cam, resolution, num_samples):
    """
    Robust rendering:
    1) Try forcing return_bitmap=True if supported
    2) Try show=False if supported
    3) If output is a Matplotlib figure or (fig, ax), convert it to RGB and close it
    """
    w, h = resolution
    rsig = inspect.signature(scene.render)

    # Build kwargs candidates (filtered to supported params)
    candidates = []

    # Prefer returning bitmap (prevents matplotlib figure creation)
    base = {"camera": cam, "resolution": (w, h), "num_samples": int(num_samples)}
    for rb in (True,):
        kw = dict(base)
        if "return_bitmap" in rsig.parameters:
            kw["return_bitmap"] = rb
        if "show" in rsig.parameters:
            kw["show"] = False
        if "display" in rsig.parameters:
            kw["display"] = False
        candidates.append({k: v for k, v in kw.items() if k in rsig.parameters})

    # Fallback minimal
    kw = {"camera": cam}
    if "show" in rsig.parameters:
        kw["show"] = False
    if "display" in rsig.parameters:
        kw["display"] = False
    candidates.append({k: v for k, v in kw.items() if k in rsig.parameters})

    last_err = None
    for kw in candidates:
        try:
            out = scene.render(**kw)

            # Case A: Mitsuba bitmap-like
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
                return arr.astype(np.float32), None

            # Case B: numpy already
            if isinstance(out, np.ndarray):
                arr = out
                if arr.ndim == 3 and arr.shape[-1] >= 3:
                    return arr[..., :3].astype(np.float32), None

            # Case C: (fig, ax) or fig
            if isinstance(out, (tuple, list)) and len(out) >= 1 and hasattr(out[0], "canvas"):
                return figure_to_rgb01(out[0]), None
            if hasattr(out, "canvas"):
                return figure_to_rgb01(out), None

            # Unknown -> fail
            last_err = TypeError(f"Unknown render output type: {type(out)}")
        except Exception as e:
            last_err = e
            continue

    return None, last_err


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
                cv2.namedWindow("SionnaRT Render", cv2.WINDOW_NORMAL)
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
                mi.Bitmap(rgb01,
                          pixel_format=mi.Bitmap.PixelFormat.RGB,
                          component_format=mi.Struct.Type.Float32).write(path)
                print(f"[SIONNA] Saved {path}")


def build_scene(freq_ghz: float):
    scene = load_scene(rt.scene.etoile, merge_shapes=True)
    scene.frequency = float(freq_ghz) * 1e9

    arr = make_planar_array()
    scene.tx_array = arr
    scene.rx_array = arr

    cam = make_camera(position=[-360, 145, 400], look_at=[-115, 33, 1.5])

    # Materials
    uav_mat = ITURadioMaterial("uav-mat", "metal", thickness=0.01, color=(0.1, 0.9, 0.1))
    gnb_mat = ITURadioMaterial("gnb-mat", "metal", thickness=0.01, color=(0.9, 0.9, 0.1))

    # Create objects (properties set AFTER adding)
    uav_obj = SceneObject(fname=rt.scene.low_poly_car, name="uav-marker", radio_material=uav_mat)

    cx, cy = -115.0, 33.0
    RX_CONFIG = [
        ("gNB_1", [cx - 220.0, cy - 220.0, 1.5]),
        ("gNB_2", [cx + 220.0, cy - 220.0, 1.5]),
        ("gNB_3", [cx - 220.0, cy + 220.0, 1.5]),
        ("gNB_4", [cx + 220.0, cy + 220.0, 1.5]),
    ]
    gnb_objs = []
    for name, _pos in RX_CONFIG:
        gnb_objs.append(SceneObject(fname=rt.scene.low_poly_car, name=f"{name}-marker", radio_material=gnb_mat))

    # Add first (required by your version)
    scene.edit(add=[uav_obj] + gnb_objs)

    # Now set properties
    uav_obj.scaling = 0.15
    uav_obj.position = mi.Point3f(cx, cy, 5.0)

    for obj, (_name, pos) in zip(gnb_objs, RX_CONFIG):
        obj.scaling = 0.10
        obj.position = mi.Point3f(float(pos[0]), float(pos[1]), float(pos[2]))

    return scene, cam, uav_obj


def main():
    w, h = parse_resolution(args.resolution)
    viewer = Viewer(args.viewer, args.save_dir)

    scene, cam, uav_obj = build_scene(args.freq_ghz)

    # ZMQ REP
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)
    print(f"[SIONNA] Listening on {args.bind}")

    render_period = 1.0 / max(args.render_fps, 1e-6)
    last_render = 0.0

    # Initial render
    rgb, err = render_to_rgb01(scene, cam, (w, h), args.num_samples)
    if err:
        print(f"[SIONNA] Initial render failed: {type(err).__name__}: {err}")
    else:
        viewer.show(rgb, "Etoile (initial)", args.save_every)

    while True:
        req = sock.recv_json()
        cmd = req.get("cmd", "measure")
        if cmd == "stop":
            sock.send_json({"ok": True})
            break

        seq = req.get("seq")
        t_sim = req.get("t_sim")
        tx_pos = req.get("tx_pos")

        warn = None
        if t_sim is None or tx_pos is None:
            sock.send_json({"ok": False, "seq": seq, "error": "Missing t_sim or tx_pos"})
            continue

        # Update UAV marker position
        try:
            p = np.array(tx_pos, dtype=float).reshape(3)
            if p[2] < 1.0:
                p[2] = 1.0
            uav_obj.position = mi.Point3f(float(p[0]), float(p[1]), float(p[2]))
        except Exception as e:
            warn = f"UAV update failed: {type(e).__name__}: {e}"

        now = time.time()
        if (now - last_render) >= render_period:
            last_render = now
            rgb, err = render_to_rgb01(scene, cam, (w, h), args.num_samples)
            if err:
                warn = f"Render failed: {type(err).__name__}: {err}"
                print("[SIONNA] " + warn)
            else:
                title = f"Etoile | seq={seq} t={float(t_sim):.2f}"
                if warn:
                    title += " | WARN"
                viewer.show(rgb, title, args.save_every)

        sock.send_json({"ok": True, "seq": seq, "t_sim": float(t_sim), "warning": warn})

    print("[SIONNA] Exiting")


if __name__ == "__main__":
    main()
