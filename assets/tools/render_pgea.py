"""Draw the generated URDFs (gripper drawing + FR5 with gripper) to PNG, straight from the URDF + meshes.

usage: python render_pgea.py [out_dir]
"""
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser("~"), "Desktop")


def load_stl(path):
    d = open(path, "rb").read()
    n = struct.unpack("<I", d[80:84])[0]
    a = np.frombuffer(d[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return a["v"].reshape(-1, 3, 3).astype(float)


def box_tris(size, center):
    sx, sy, sz = np.array(size) / 2
    v = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]) + center
    f = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([[v[a], v[b], v[c]] for q in f for a, b, c in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))])


def cyl_tris(r, length, center, n=32):
    t = np.linspace(0, 2 * np.pi, n + 1)
    tris = []
    for i in range(n):
        p = [np.array([r * math.cos(t[k]), r * math.sin(t[k]), 0]) for k in (i, i + 1)]
        lo, hi = np.array([0, 0, -length / 2]), np.array([0, 0, length / 2])
        tris += [[p[0] + lo, p[1] + lo, p[1] + hi], [p[0] + lo, p[1] + hi, p[0] + hi], [lo, p[0] + lo, p[1] + lo],
                 [hi, p[1] + hi, p[0] + hi]]
    return np.array(tris) + center


def rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_mat(ax, q):
    ax = np.array(ax) / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K


def T(R=np.eye(3), t=(0, 0, 0)):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def parse_vec(e, key, default="0 0 0"):
    return np.array([float(v) for v in (e.get(key) if e is not None and e.get(key) else default).split()])


def link_geoms(urdf_path):
    """{link: [(tris in link frame, rgba, name)]} from visual elements."""
    root = ET.parse(urdf_path).getroot()
    mats = {m.get("name"): parse_vec(m.find("color"), "rgba", "0.6 0.6 0.6 1") for m in root.iter("material")
            if m.find("color") is not None}
    out = {}
    for link in root.findall("link"):
        geoms = []
        for vis in link.findall("visual"):
            o = vis.find("origin")
            M = T(rpy_mat(*parse_vec(o, "rpy")), parse_vec(o, "xyz"))
            g = vis.find("geometry")[0]
            if g.tag == "mesh":
                tris = load_stl(os.path.normpath(os.path.join(os.path.dirname(urdf_path), g.get("filename"))))
            elif g.tag == "box":
                tris = box_tris(parse_vec(g, "size"), (0, 0, 0))
            elif g.tag == "cylinder":
                tris = cyl_tris(float(g.get("radius")), float(g.get("length")), (0, 0, 0))
            else:
                continue
            tris = tris @ M[:3, :3].T + M[:3, 3]
            m = vis.find("material")
            rgba = mats.get(m.get("name"), np.array([0.7, 0.7, 0.72, 1])) if m is not None else np.array([0.7, 0.7, 0.72, 1])
            geoms.append((tris, rgba, vis.get("name") or ""))
        out[link.get("name")] = geoms
    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")
        joints.append(dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                           child=j.find("child").get("link"), xyz=parse_vec(o, "xyz"), rpy=parse_vec(o, "rpy"),
                           axis=parse_vec(j.find("axis"), "xyz", "1 0 0")))
    return out, joints


def fk(joints, q):
    """World transform of every link. q: {joint name: value (rad or m)}."""
    child_of = {j["child"] for j in joints}
    roots = {j["parent"] for j in joints} - child_of
    W = {r: np.eye(4) for r in roots}
    pending = list(joints)
    while pending:
        for j in list(pending):
            if j["parent"] in W:
                M = W[j["parent"]] @ T(rpy_mat(*j["rpy"]), j["xyz"])
                v = q.get(j["name"], 0.0)
                if j["type"] in ("revolute", "continuous"):
                    M = M @ T(axis_mat(j["axis"], v))
                elif j["type"] == "prismatic":
                    M = M @ T(t=np.array(j["axis"]) * v)
                W[j["child"]] = M
                pending.remove(j)
    return W


def world_tris(geoms, W, only=None):
    res = []
    for link, gl in geoms.items():
        if link not in W or (only and link not in only):
            continue
        M = W[link]
        for tris, rgba, nm in gl:
            res.append((tris @ M[:3, :3].T + M[:3, 3], rgba))
    return res


def draw3d(ax, items, scale=1.0, alpha=1.0):
    pts = []
    for tris, rgba in items:
        t = tris * scale
        # simple shading from the face normal
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        shade = 0.55 + 0.45 * np.clip(n @ np.array([0.4, -0.5, 0.75]), 0, 1)
        col = np.clip(rgba[:3][None] * shade[:, None], 0, 1)
        pc = Poly3DCollection(t, facecolors=np.c_[col, np.full(len(t), alpha)], edgecolors="none")
        ax.add_collection3d(pc)
        pts.append(t.reshape(-1, 3))
    p = np.concatenate(pts)
    c, r = (p.max(0) + p.min(0)) / 2, (p.max(0) - p.min(0)).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))


def draw2d(ax, items, plane, scale=1000.0):
    """Orthographic silhouette: plane=(i, j) axes, e.g. (0, 2) = front view X-Z."""
    from matplotlib.collections import PolyCollection

    i, j = plane
    for tris, rgba in items:
        t = tris * scale
        ax.add_collection(PolyCollection(t[:, :, [i, j]], facecolors=rgba[:3] * 0.9 + 0.05, edgecolors=rgba[:3] * 0.5,
                                         linewidths=0.2))
    ax.set_aspect("equal")
    ax.autoscale_view()


def dim(ax, p0, p1, text, off=(0, 0), color="#c0392b", fs=9):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", color=color, lw=1))
    ax.text((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1], text, color=color, fontsize=fs,
            ha="center", va="center", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))


def gripper_figure(path):
    urdf = os.path.join(HERE, "pgea_description", "urdf", "pgea_100_40_o.urdf")
    geoms, joints = link_geoms(urdf)
    open_mm = 30.8  # 70 % (pre-grasp)
    W = fk(joints, {"finger_left_joint": open_mm / 2000, "finger_right_joint": open_mm / 2000})
    items = world_tris(geoms, W)
    # flip Z so the fingertips point down in the drawing (tool +Z = down)
    flip = [(t * np.array([1, 1, -1]), c) for t, c in items]
    cube = [(box_tris((0.025, 0.025, 0.025), (0, 0, -0.1595)), np.array([0.95, 0.75, 0.1, 1]))]

    fig = plt.figure(figsize=(17, 9.5))
    fig.suptitle("PGEA-100-40-O 그리퍼 URDF (pgea_100_40_o.urdf) · 손가락 70% 벌림(30.8 mm), 25 mm 큐브 (손가락 끝 = 바닥 위 12 mm)", fontsize=15)
    ax = fig.add_subplot(1, 3, 1)
    draw2d(ax, cube + flip, (0, 2))
    ax.set_title("앞에서 (X-Z, 손가락이 벌어지는 방향)", fontsize=11)
    ax.set_xlabel("X [mm] (공구 X축)")
    ax.set_ylabel("Z [mm] (플랜지 면 = 0, 아래로 +)")
    ax.set_yticks(range(0, -170, -20), [str(v) for v in range(0, 170, 20)])
    dim(ax, (-27.5, -96), (27.5, -96), "하우징 55")
    dim(ax, (-18.5, -55), (18.5, -55), "몸체 37 (55-2×9)")
    dim(ax, (-15.4, -164), (15.4, -164), "벌림 30.8 (70%)", off=(0, -4))
    dim(ax, (15.4, -128.5), (31.4, -128.5), "조 16", off=(14, 0), fs=8)
    dim(ax, (45, 0), (45, -160), "TCP 160", off=(9, 0))
    for z, t in ((10, "어댑터 0~10"), (100, "몸체 10~100"), (122, "하우징 100~122"), (127, "조 122~127"),
                 (160, "손가락 127~160 (33)")):
        ax.axhline(-z, color="#888", lw=0.4, ls="--")
        ax.text(-60, -z + 2.5, t, fontsize=8, color="#555")
    ax.text(-60, -152, "손가락 두께: 끝 3 → 끝에서\n15 mm 지점부터 15 (쐐기)", fontsize=8, color="#555")
    ax.plot(0, -160, "r+", ms=12, mew=2)
    ax.set_xlim(-65, 75)
    ax.set_ylim(-180, 8)

    ax = fig.add_subplot(1, 3, 2)
    draw2d(ax, cube + flip, (1, 2))
    ax.set_title("옆에서 (Y-Z)", fontsize=11)
    ax.set_xlabel("Y [mm]")
    ax.set_yticks(range(0, -170, -20), [str(v) for v in range(0, 170, 20)])
    dim(ax, (-20, -111), (20, -111), "깊이 40 (가정)")
    dim(ax, (-10, -150), (10, -150), "손가락 폭 20\n(가정)", off=(0, -10), fs=8)
    dim(ax, (-40, -5), (40, -5), "어댑터 Ø80 (가정)", off=(0, 7), fs=8)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-180, 8)

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    draw3d(ax, flip + cube, scale=1000)
    ax.view_init(elev=18, azim=-60)
    ax.set_title("3D", fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.text(0.01, 0.01, "실측(사진 11/21): TCP 160, 어댑터 10, 몸체 90·가로 37, 하우징 55×22, 조 16×5, 손가락 33 (15 두께 18 + 쐐기 15→끝 3), "
             "스트로크 0~44.   가정: 깊이(Y) 40, 손가락·조 폭 20, 어댑터 Ø80, 질량.   D435·브래킷 미포함(미측정).",
             fontsize=9, color="#333")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fr5_figure(path):
    urdf = os.path.join(HERE, "fr5", "fr5_pgea_v2.urdf")
    geoms, joints = link_geoms(urdf)
    deg = math.radians
    poses = [
        ("모든 관절 0°", {}),
        ("아래를 보는 자세 (j2 -90, j3 90, j4 -90, j5 -90)",
         {"j1": 0, "j2": deg(-90), "j3": deg(90), "j4": deg(-90), "j5": deg(-90), "j6": 0}),
    ]
    fig = plt.figure(figsize=(17, 9))
    fig.suptitle("FR5 + PGEA-100-40-O (fr5_pgea_v2.urdf) · 그리퍼는 손목 플랜지(wrist3_link z 0.100)에 장착, 손가락 닫힘", fontsize=14)
    for k, (title, q) in enumerate(poses):
        q = {**q, "finger_left_joint": 0.0, "finger_right_joint": 0.0}
        W = fk(joints, q)
        items = world_tris(geoms, W)
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        draw3d(ax, items, scale=1.0)
        tcp = W["tcp"][:3, 3]
        ax.scatter(*tcp, color="red", s=30)
        ax.text(*tcp, f"  TCP\n  ({tcp[0]*1000:.0f}, {tcp[1]*1000:.0f}, {tcp[2]*1000:.0f}) mm", color="red", fontsize=8)
        ax.view_init(elev=20, azim=-50)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
    # close-up of the wrist + gripper in the second pose
    W = fk(joints, {"j2": deg(-90), "j3": deg(90), "j4": deg(-90), "j5": deg(-90)})
    only = {"wrist2_link", "wrist3_link", "pgea_base", "finger_left", "finger_right"}
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    draw3d(ax, world_tris(geoms, W, only), scale=1.0)
    f = W["pgea_base"][:3, 3]
    t = W["tcp"][:3, 3]
    ax.plot(*np.c_[f, t], color="red", lw=1.5)
    ax.text(*((f + t) / 2), f"  플랜지→TCP {np.linalg.norm(t - f)*1000:.0f} mm", color="red", fontsize=9)
    ax.view_init(elev=10, azim=-40)
    ax.set_title("손목 확대", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    gripper_figure(os.path.join(OUT, "PGEA_그리퍼_URDF.png"))
    fr5_figure(os.path.join(OUT, "FR5_PGEA_URDF.png"))
    print("saved to", OUT)
