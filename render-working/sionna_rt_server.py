#!/usr/bin/env python3
"""
Robust SionnaRT Etoile server:
- Black cube (drone marker) updates ASAP on every received pose
- gNB transmitters (visible markers) + drone receiver
- Ray tracing & triangulation run at --rt_fps (throttled) to keep server responsive
- Rendering runs at --render_fps (throttled)
- NEVER uses scene.render(paths=...) (avoids NoneType unpack bug)
- If PathSolver/compute_paths crashes -> LoS fallback rays so you still see rays and triangulation

REQ/REP:
Request: {"seq": int, "t_sim": float, "tx_pos": [x,y,z]}
Reply  : {"ok": bool, "seq": int, "t_sim": float, "tri_pos": [x,y,z]|None, "used": int, "resid": float, "warning": str|None}
"""

import os
import time
import argparse
import inspect
import warnings
import numpy as np
import zmq

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

warnings.filterwarnings("ignore", message=r"Unable to import Axes3D.*")
warnings.filterwarnings("ignore", message=r"More than 20 figures have been opened.*")
warnings.filterwarnings("ignore", message=r"The AST-transforming decorator @drjit.syntax.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mitsuba as mi


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="tcp://*:5555")
    ap.add_argument("--freq_ghz", type=float, default=28.0)

    ap.add_argument("--variant", default="auto",
                    help="auto|llvm_ad_mono_polarized|cuda_ad_mono_polarized|llvm_ad_rgb|llvm_rgb")
    ap.add_argument("--render", choices=["on", "off"], default="on")
    ap.add_argument("--viewer", choices=["auto", "opencv", "save", "off"], default="auto")
    ap.add_argument("--render_fps", type=float, default=10.0)
    ap.add_argument("--rt_fps", type=float, default=5.0, help="ray tracing + triangulation rate")
    ap.add_argument("--num_samples", type=int, default=32)
    ap.add_argument("--resolution", default="960x540")
    ap.add_argument("--max_depth", type=int, default=2)
    ap.add_argument("--num_rays_per_tx", type=int, default=20)

    ap.add_argument("--num_gnbs", type=int, default=4)
    ap.add_argument("--gnb_radius", type=float, default=220.0)

    ap.add_argument("--center", default="-115,33,1.5", help="Etoile center where camera looks: x,y,z")
    ap.add_argument("--auto_align", action="store_true", help="auto-align first received Isaac pose to center")
    ap.set_defaults(auto_align=True)

    ap.add_argument("--save_dir", default=os.path.join(os.getcwd(), "sionna_renders"))
    ap.add_argument("--save_every", type=int, default=1)

    ap.add_argument("--fov_deg", type=float, default=55.0)
    ap.add_argument("--print_every", type=int, default=10)
    return ap.parse_args()


def pick_variant(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd not in ["", "-1"]:
        return "cuda_ad_mono_polarized"
    return "llvm_ad_mono_polarized"


args = parse_args()

if os.path.basename(__file__) in ("sionna.py", "sionna_rt.py"):
    raise RuntimeError("Do NOT name this file sionna.py / sionna_rt.py (it shadows the package). Rename it.")

VARIANT = pick_variant(args.variant)
mi.set_variant(VARIANT)

print(f"[SIONNA] Mitsuba variant = {VARIANT}")
print(f"[SIONNA] CUDA_VISIBLE_DEVICES = '{os.environ.get('CUDA_VISIBLE_DEVICES','')}'")
print(f"[SIONNA] render={args.render} viewer={args.viewer} render_fps={args.render_fps} rt_fps={args.rt_fps} depth={args.max_depth}")

import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Camera, ITURadioMaterial, SceneObject
from sionna.rt import Transmitter, Receiver, PathSolver


def parse_resolution(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)


def parse_center(s: str):
    x, y, z = s.split(",")
    return float(x), float(y), float(z)


def f3(x):
    return float(np.asarray(x).item())


def make_planar_array_iso():
    attempts = [
        dict(num_rows=1, num_cols=1, pattern="iso", polarization="V"),
        dict(num_cols=1, num_rows=1, pattern="iso", polarization="V"),
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
    w, h = resolution
    sig = inspect.signature(scene.render)
    params = sig.parameters

    if "return_bitmap" in params:
        kw = {}
        if "camera" in params:
            kw["camera"] = cam
        if "resolution" in params:
            kw["resolution"] = (w, h)
        if "num_samples" in params:
            kw["num_samples"] = int(num_samples)
        kw["return_bitmap"] = True
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

    if isinstance(out, (tuple, list)) and len(out) >= 1 and hasattr(out[0], "canvas"):
        return fig_to_rgb01_and_close(out[0])
    if hasattr(out, "canvas"):
        return fig_to_rgb01_and_close(out)
    if out is None:
        fig = plt.gcf()
        if fig and hasattr(fig, "canvas"):
            return fig_to_rgb01_and_close(fig)

    raise TypeError(f"scene.render produced unsupported output: {type(out)}")


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
            raise RuntimeError("viewer=opencv requested but OpenCV not available")

    def show(self, rgb01: np.ndarray, title: str, save_every: int):
        if rgb01 is None or self.mode == "off":
            return
        self.frame_idx += 1

        if self.use_cv2 and self.mode in ("auto", "opencv"):
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

    def draw_polyline(self, rgb01, pts_2d, color_bgr=(0, 255, 0), thickness=2):
        if not self.use_cv2:
            return rgb01
        img = (np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
        bgr = img[..., ::-1].copy()
        for i in range(len(pts_2d) - 1):
            self.cv2.line(bgr, pts_2d[i], pts_2d[i + 1], color_bgr, thickness, lineType=self.cv2.LINE_AA)
        return (bgr[..., ::-1].astype(np.float32) / 255.0)


def write_cube_obj(path: str):
    cube = """# cube
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
v -0.5  0.5 -0.5
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5  0.5
f 1 2 3 4
f 5 6 7 8
f 1 5 8 4
f 2 6 7 3
f 4 3 7 8
f 1 2 6 5
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(cube)


def unit(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def dir_from_theta_phi(theta, phi):
    st, ct = np.sin(theta), np.cos(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    u = np.array([st * cp, st * sp, ct], dtype=float)
    return unit(u)


def triangulate_from_rays(tx_positions, dirs):
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    used = 0
    for p, u in zip(tx_positions, dirs):
        if u is None:
            continue
        p = np.asarray(p, dtype=float).reshape(3)
        u = unit(u)
        P = np.eye(3) - np.outer(u, u)
        A += P
        b += P @ p
        used += 1
    if used < 2:
        return None, float("inf"), used
    try:
        x = np.linalg.solve(A, b)
        resid = float(np.linalg.norm(A @ x - b))
        return x, resid, used
    except Exception:
        return None, float("inf"), used


def build_scene(freq_ghz: float):
    scene = load_scene(rt.scene.etoile, merge_shapes=True)
    scene.frequency = float(freq_ghz) * 1e9

    arr = make_planar_array_iso()
    scene.tx_array = arr
    scene.rx_array = arr

    # Camera that matches the working "top view" style
    cam_pos = np.array([-360.0, 145.0, 400.0], dtype=float)
    cx, cy, cz = parse_center(args.center)
    look_at = np.array([cx, cy, cz], dtype=float)
    cam = make_camera(position=cam_pos.tolist(), look_at=look_at.tolist())

    os.makedirs(args.save_dir, exist_ok=True)
    cube_path = os.path.join(args.save_dir, "drone_cube.obj")
    if not os.path.exists(cube_path):
        write_cube_obj(cube_path)

    # IMPORTANT: ITURadioMaterial signature is (name, itu_type, ...)
    drone_mat = ITURadioMaterial("drone-material", "metal", thickness=0.01, color=(0.02, 0.02, 0.02))
    gnb_mat = ITURadioMaterial("gnb-material", "metal", thickness=0.01, color=(0.10, 0.90, 0.10))

    drone_marker = SceneObject(fname=cube_path, name="drone_marker", radio_material=drone_mat)
    gnb_markers = [SceneObject(fname=rt.scene.sphere, name=f"gnb_marker_{i}", radio_material=gnb_mat)
                   for i in range(args.num_gnbs)]
    scene.edit(add=[drone_marker] + gnb_markers)

    drone_marker.scaling = 3.0
    drone_marker.position = mi.Point3f(float(cx), float(cy), float(5.0))

    # gNB positions on a ring
    thetas = np.linspace(0.0, 2.0 * np.pi, args.num_gnbs, endpoint=False)
    gnb_positions = []
    for i, th in enumerate(thetas):
        x = cx + args.gnb_radius * float(np.cos(th))
        y = cy + args.gnb_radius * float(np.sin(th))
        z = 3.0
        gnb_positions.append([float(x), float(y), float(z)])
        gnb_markers[i].scaling = 10.0
        gnb_markers[i].position = mi.Point3f(float(x), float(y), float(z))

    # Radios: gNB TX + drone RX
    for i in range(args.num_gnbs):
        name = f"gNB_{i}"
        try:
            scene.remove(name)
        except Exception:
            pass
        scene.add(Transmitter(name, position=[float(v) for v in gnb_positions[i]], display_radius=0.0))

    try:
        scene.remove("drone_rx")
    except Exception:
        pass
    drone_rx = Receiver("drone_rx", position=[float(cx), float(cy), 5.0], display_radius=0.0)
    scene.add(drone_rx)

    solver = PathSolver()
    return scene, cam, cam_pos, look_at, drone_marker, drone_rx, gnb_positions, solver


def compute_paths(scene, solver, max_depth: int):
    # Prefer scene.compute_paths if available (often more stable)
    if hasattr(scene, "compute_paths") and callable(getattr(scene, "compute_paths")):
        try:
            sig = inspect.signature(scene.compute_paths)
            kw = {}
            if "max_depth" in sig.parameters:
                kw["max_depth"] = int(max_depth)
            out = scene.compute_paths(**kw)
            return out
        except Exception:
            pass
    # Fallback to PathSolver
    return solver(scene=scene, max_depth=int(max_depth))


def extract_rays_or_fallback(paths, gnb_positions, drone_pos, max_rays_per_tx: int):
    """
    Returns:
      polylines_by_tx: list[list[np.ndarray]]  each polyline is list of 3D points
      dirs_for_triang: list[np.ndarray|None]   unit direction from each TX toward RX (dominant path)
      used_count: int
      warning: str|None
    """
    warning = None
    polylines_by_tx = [[] for _ in gnb_positions]
    dirs = [None for _ in gnb_positions]

    # Always have LoS fallback available
    def los_for_tx(i):
        txp = np.array(gnb_positions[i], dtype=float)
        rxp = np.array(drone_pos, dtype=float)
        poly = [txp, rxp]
        return poly, unit(rxp - txp)

    if paths is None:
        for i in range(len(gnb_positions)):
            poly, d = los_for_tx(i)
            polylines_by_tx[i].append(poly)
            dirs[i] = d
        return polylines_by_tx, dirs, len(gnb_positions), "paths=None -> LoS fallback"

    # Try to access valid/tau/phi_t/theta_t/vertices robustly
    try:
        valid = np.array(getattr(paths, "valid"))
        tau = np.array(getattr(paths, "tau"))
        phi_t = np.array(getattr(paths, "phi_t"))
        theta_t = np.array(getattr(paths, "theta_t"))
        verts = np.array(getattr(paths, "vertices", None))  # may be None/absent

        # Reduce antenna dims (we use single element arrays)
        while valid.ndim > 3:
            valid = valid[:, 0, ...]
        while tau.ndim > 3:
            tau = tau[:, 0, ...]
        while phi_t.ndim > 3:
            phi_t = phi_t[:, 0, ...]
        while theta_t.ndim > 3:
            theta_t = theta_t[:, 0, ...]
        if verts is not None:
            while verts.ndim > 5:
                verts = verts[:, :, 0, ...]

        # Expect [num_rx, num_tx, num_paths] with num_rx=1
        # If swapped, try transpose
        if valid.shape[0] != 1 and valid.shape[1] == 1:
            valid = np.transpose(valid, (1, 0, 2))
            tau = np.transpose(tau, (1, 0, 2))
            phi_t = np.transpose(phi_t, (1, 0, 2))
            theta_t = np.transpose(theta_t, (1, 0, 2))
            if verts is not None and verts.ndim == 5:
                # verts: [depth, rx, tx, path, 3] -> swap rx<->tx
                verts = np.transpose(verts, (0, 2, 1, 3, 4))

        num_tx = min(valid.shape[1], len(gnb_positions))

        used_any = 0
        for tx_i in range(num_tx):
            v = valid[0, tx_i, :].astype(bool)
            idx = np.where(v)[0]
            if idx.size == 0:
                # fallback LoS for this TX
                poly, d = los_for_tx(tx_i)
                polylines_by_tx[tx_i].append(poly)
                dirs[tx_i] = d
                used_any += 1
                continue

            # Sort by delay (shortest first)
            tt = tau[0, tx_i, idx]
            order = idx[np.argsort(tt)]
            order = order[:max(1, int(max_rays_per_tx))]

            # Dominant direction = shortest-delay path
            k0 = int(order[0])
            dirs[tx_i] = dir_from_theta_phi(theta_t[0, tx_i, k0], phi_t[0, tx_i, k0])
            used_any += 1

            txp = np.array(gnb_positions[tx_i], dtype=float)
            rxp = np.array(drone_pos, dtype=float)

            for k in order:
                pts = [txp.copy()]

                # Try to add bounce vertices if available
                if verts is not None and verts.size > 0:
                    try:
                        vv = np.array(verts[:, 0, tx_i, int(k), :], dtype=float)  # [depth, 3]
                        for j in range(vv.shape[0]):
                            pj = vv[j]
                            # ignore zero/invalid rows
                            if np.isfinite(pj).all() and np.linalg.norm(pj) > 1e-6:
                                pts.append(pj.copy())
                    except Exception:
                        pass

                pts.append(rxp.copy())
                polylines_by_tx[tx_i].append(pts)

        return polylines_by_tx, dirs, used_any, None

    except Exception as e:
        warning = f"extract failed: {type(e).__name__}: {e} -> LoS fallback"
        for i in range(len(gnb_positions)):
            poly, d = los_for_tx(i)
            polylines_by_tx[i].append(poly)
            dirs[i] = d
        return polylines_by_tx, dirs, len(gnb_positions), warning


def overlay_polylines(rgb01, polylines_by_tx, cam_pos, look_at, W, H, viewer):
    if rgb01 is None or not viewer.use_cv2:
        return rgb01
    basis = make_camera_basis(cam_pos, look_at)

    colors = [
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 128, 255),
        (255, 128, 0),
    ]

    for tx_i, polys in enumerate(polylines_by_tx):
        col = colors[tx_i % len(colors)]
        for pts3 in polys:
            pts2 = []
            ok = True
            for p3 in pts3:
                uv = project_point(p3, cam_pos, basis, W, H, args.fov_deg)
                if uv is None:
                    ok = False
                    break
                pts2.append(uv)
            if ok and len(pts2) >= 2:
                rgb01 = viewer.draw_polyline(rgb01, pts2, color_bgr=col, thickness=2)
    return rgb01


def main():
    W, H = parse_resolution(args.resolution)
    viewer = Viewer(args.viewer, args.save_dir)

    scene, cam, cam_pos, look_at, drone_marker, drone_rx, gnb_positions, solver = build_scene(args.freq_ghz)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)
    print(f"[SIONNA] Listening on {args.bind}")

    # State
    align_offset = None  # np.array([dx,dy,0])
    last_rt_wall = 0.0
    last_render_wall = 0.0
    rt_period = 0.0 if args.rt_fps <= 0 else (1.0 / max(args.rt_fps, 1e-6))
    render_period = 0.0 if args.render_fps <= 0 else (1.0 / max(args.render_fps, 1e-6))

    last_polylines = [[] for _ in gnb_positions]
    last_tri = None
    last_resid = float("inf")
    last_used = 0
    last_warn = None

    # Initial render
    if args.render == "on":
        try:
            rgb01 = render_scene_rgb01(scene, cam, (W, H), args.num_samples)
            viewer.show(rgb01, "Etoile (initial)", args.save_every)
        except Exception as e:
            print(f"[SIONNA] Initial render failed: {type(e).__name__}: {e}")

    while True:
        req = sock.recv_json()
        cmd = req.get("cmd", "measure")
        if cmd == "stop":
            sock.send_json({"ok": True})
            break

        seq = req.get("seq", None)
        t_sim = req.get("t_sim", None)
        pos_in = req.get("tx_pos", None)

        if t_sim is None or pos_in is None:
            sock.send_json({"ok": False, "seq": seq, "error": "Missing t_sim or tx_pos"})
            continue

        warning = None

        # === ASAP pose update ===
        p_in = np.array(pos_in, dtype=float).reshape(3)
        if not np.isfinite(p_in).all():
            p_in = np.array([0.0, 0.0, 1.0], dtype=float)
            warning = "Non-finite pose received"

        # Auto-align first received pose to scene center (so cube is visible)
        if args.auto_align and align_offset is None:
            cx, cy, _cz = parse_center(args.center)
            align_offset = np.array([cx - p_in[0], cy - p_in[1], 0.0], dtype=float)
            print(f"[SIONNA] Auto-align enabled. First pose {p_in.tolist()} -> offset {align_offset.tolist()}")

        if align_offset is None:
            align_offset = np.array([0.0, 0.0, 0.0], dtype=float)

        p = p_in + align_offset
        if p[2] < 0.5:
            p[2] = 0.5

        # Update visible cube + receiver immediately
        try:
            drone_marker.position = mi.Point3f(float(p[0]), float(p[1]), float(p[2]))
            drone_rx.position = [float(p[0]), float(p[1]), float(p[2])]
        except Exception as e:
            warning = (warning + " | " if warning else "") + f"pose apply failed: {type(e).__name__}: {e}"

        now = time.time()

        # === Ray tracing throttled ===
        if rt_period == 0.0 or (now - last_rt_wall) >= rt_period:
            last_rt_wall = now
            try:
                paths = compute_paths(scene, solver, args.max_depth)
                polylines, dirs, used_cnt, warn2 = extract_rays_or_fallback(
                    paths, gnb_positions, p, args.num_rays_per_tx
                )
                tri, resid, used = triangulate_from_rays(gnb_positions, dirs)

                last_polylines = polylines
                last_tri = tri
                last_resid = resid
                last_used = used
                last_warn = warn2

            except Exception as e:
                # Full fallback LoS rays
                last_warn = f"Path computation crashed: {type(e).__name__}: {e} -> LoS fallback"
                last_polylines = [[] for _ in gnb_positions]
                dirs = []
                for i in range(len(gnb_positions)):
                    txp = np.array(gnb_positions[i], dtype=float)
                    last_polylines[i].append([txp, p.copy()])
                    dirs.append(unit(p - txp))
                last_tri, last_resid, last_used = triangulate_from_rays(gnb_positions, dirs)

        # === Render throttled ===
        if args.render == "on" and (render_period == 0.0 or (now - last_render_wall) >= render_period):
            last_render_wall = now
            try:
                rgb01 = render_scene_rgb01(scene, cam, (W, H), args.num_samples)
                rgb01 = overlay_polylines(rgb01, last_polylines, cam_pos, look_at, W, H, viewer)
                viewer.show(
                    rgb01,
                    f"Etoile | seq={seq} t={float(t_sim):.2f} used={last_used} resid={last_resid:.2e}",
                    args.save_every
                )
            except Exception as e:
                warning = (warning + " | " if warning else "") + f"render failed: {type(e).__name__}: {e}"

        # Terminal log
        if seq is not None and args.print_every > 0 and (int(seq) % int(args.print_every) == 0):
            if last_tri is None:
                print(f"[SIONNA] seq={seq} t={float(t_sim):.2f} triangulation unavailable (used={last_used}) warn={last_warn}")
            else:
                err = last_tri - p
                print(f"[SIONNA] seq={seq} t={float(t_sim):.2f} "
                      f"GT=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) "
                      f"TRI=({last_tri[0]:.2f},{last_tri[1]:.2f},{last_tri[2]:.2f}) "
                      f"e=({err[0]:+.2f},{err[1]:+.2f},{err[2]:+.2f}) resid={last_resid:.3e} used={last_used} warn={last_warn}")

        # Reply quickly (never block here)
        final_warn = warning
        if last_warn:
            final_warn = (final_warn + " | " if final_warn else "") + last_warn

        sock.send_json({
            "ok": True,
            "seq": seq,
            "t_sim": float(t_sim),
            "tri_pos": None if last_tri is None else [float(last_tri[0]), float(last_tri[1]), float(last_tri[2])],
            "used": int(last_used),
            "resid": float(last_resid),
            "warning": final_warn
        })

    print("[SIONNA] Exiting")


if __name__ == "__main__":
    main()
