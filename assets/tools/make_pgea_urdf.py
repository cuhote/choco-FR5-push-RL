"""Build the PGEA-100-40-O URDF from the measured dimensions and attach it to the FR5.

Outputs
  assets/pgea_description/urdf/pgea_100_40_o.urdf   gripper alone, root = flange face (tool frame)
  assets/pgea_description/meshes/finger_{left,right}.stl
  assets/fr5/fr5_pgea_v2.urdf                        FR5 + this gripper (flange = wrist3_link z 0.100)

Tool frame (FR5 tool 1): origin at the flange face, +Z toward the fingertips, fingers open along +-X.
Measured (photo 2026-11-21, design 5.2): adapter ring 10, body 90 (width 37 = housing 55 - 2 x 9 overhang),
housing 55 x 22, jaw 16 x 5, finger 33 (15 thick for 18 mm, wedge 15 -> 3 mm tip), TCP z 160, stroke 0~44 mm.
Assumed (not measured yet): body/housing depth Y 40, adapter ring dia 80 (= FR5 flange), finger/jaw width Y 20,
masses. D435 + bracket NOT included (camera not mounted yet).

usage: python make_pgea_urdf.py
"""
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
MM = 1e-3

ADAPTER_R, ADAPTER_Z = 40.0, (0.0, 10.0)
BODY = (55.0, 23.0, 10.0, 100.0)  # x, y, z0, z1  (2026-11-23: x = housing 73 - 2 x 9 overhang, y assumed = housing)
HOUSING = (73.0, 23.0, 100.0, 122.0)  # measured 2026-11-23: 73 x 23 (fixed grey housing)
JAW_X, JAW_H = 16.0, 5.0  # per jaw, z 122~127
FINGER_Z0, FINGER_LEN, FULL_T, TIP_T, WEDGE = 127.0, 33.0, 11.0, 3.0, 14.0
THIN_T = 8.0  # measured 2026-11-23: thin tip section 14 mm, 8 -> 3 mm; above it (screw height) 11 mm incl. screw head
FINGER_W = 14.0  # measured 2026-11-23
JAW_W = 6.0  # measured 2026-11-23 (jaw block depth Y)
TCP_Z = 160.0
STROKE = 44.0
FINGER_ORIGIN_Z = HOUSING[3]  # finger link frame at the housing bottom (jaw top), inner face at x = 0


Z_TIP = FINGER_Z0 + FINGER_LEN - FINGER_ORIGIN_Z  # 38 (finger link frame, mm)
Z_WEDGE = Z_TIP - WEDGE  # 24


def finger_prism(side):
    """Jaw + finger outline (visual) in the finger link frame (mm). side=+1 for the +X finger.
    Star-shaped from (0, 0), so the fan triangulation in write_stl is valid."""
    s = side
    prof = [(0, 0), (s * JAW_X, 0), (s * JAW_X, JAW_H), (s * FULL_T, JAW_H), (s * FULL_T, Z_WEDGE),
            (s * THIN_T, Z_WEDGE), (s * TIP_T, Z_TIP), (0, Z_TIP)]
    if s < 0:
        prof = prof[::-1]
    return prof


def tip_prism(side):
    """Thin tapered tip (collision, convex): 8 -> 3 mm over the last 14 mm."""
    s = side
    prof = [(0, Z_WEDGE), (s * THIN_T, Z_WEDGE), (s * TIP_T, Z_TIP), (0, Z_TIP)]
    if s < 0:
        prof = prof[::-1]
    return prof


def write_stl(path, prof):
    """Extrude an XZ profile along Y (+-FINGER_W/2) into a closed binary STL (metres)."""
    hw = FINGER_W / 2
    n = len(prof)
    front = [(x * MM, -hw * MM, z * MM) for x, z in prof]
    back = [(x * MM, hw * MM, z * MM) for x, z in prof]
    tris = []
    for i in range(1, n - 1):  # fan; profile is convex
        tris.append((front[0], front[i], front[i + 1]))
        tris.append((back[0], back[i + 1], back[i]))
    for i in range(n):
        j = (i + 1) % n
        tris.append((front[i], back[j], front[j]))
        tris.append((front[i], back[i], back[j]))
    with open(path, "wb") as f:
        f.write(b"pgea finger".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            ux, uy, uz = (b[k] - a[k] for k in range(3))
            vx, vy, vz = (c[k] - a[k] for k in range(3))
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            ln = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
            f.write(struct.pack("<3f", nx / ln, ny / ln, nz / ln))
            for p in (a, b, c):
                f.write(struct.pack("<3f", *p))
            f.write(b"\0\0")


def box_inertia(m, sx, sy, sz):
    return m * (sy * sy + sz * sz) / 12, m * (sx * sx + sz * sz) / 12, m * (sx * sx + sy * sy) / 12


def box_xml(name, spec, material):
    sx, sy, z0, z1 = spec
    o = f'<origin xyz="0 0 {(z0 + z1) / 2 * MM:.4f}"/>'
    g = f'<geometry><box size="{sx * MM:.4f} {sy * MM:.4f} {(z1 - z0) * MM:.4f}"/></geometry>'
    return (f'    <visual name="{name}">{o}{g}<material name="{material}"/></visual>\n'
            f'    <collision name="{name}">{o}{g}</collision>\n')


def gripper_links(mesh_prefix, parent, mount_xyz):
    za = (ADAPTER_Z[0] + ADAPTER_Z[1]) / 2 * MM
    ixx, iyy, izz = box_inertia(0.55, 0.055, 0.040, 0.122)
    fx, fy, fz = box_inertia(0.03, 0.016, 0.020, 0.038)
    lim = STROKE / 2 * MM
    x = f"""
  <!-- ===== DH Robotics PGEA-100-40-O (measured 2026-11-21; see make_pgea_urdf.py for assumed values) ===== -->
  <material name="pgea_black"><color rgba="0.12 0.12 0.13 1"/></material>
  <material name="pgea_white"><color rgba="0.92 0.92 0.92 1"/></material>
  <material name="pgea_alu"><color rgba="0.75 0.75 0.78 1"/></material>
  <joint name="pgea_mount" type="fixed">
    <origin xyz="{mount_xyz}" rpy="0 0 0"/><parent link="{parent}"/><child link="pgea_base"/>
  </joint>
  <link name="pgea_base">
    <inertial><origin xyz="0 0 0.065"/><mass value="0.55"/>
      <inertia ixx="{ixx:.6g}" ixy="0" ixz="0" iyy="{iyy:.6g}" iyz="0" izz="{izz:.6g}"/></inertial>
    <visual name="adapter"><origin xyz="0 0 {za:.4f}"/><geometry><cylinder radius="{ADAPTER_R * MM:.4f}" length="{(ADAPTER_Z[1] - ADAPTER_Z[0]) * MM:.4f}"/></geometry><material name="pgea_black"/></visual>
    <collision name="adapter"><origin xyz="0 0 {za:.4f}"/><geometry><cylinder radius="{ADAPTER_R * MM:.4f}" length="{(ADAPTER_Z[1] - ADAPTER_Z[0]) * MM:.4f}"/></geometry></collision>
{box_xml("body", BODY, "pgea_black")}{box_xml("housing", HOUSING, "pgea_alu")}  </link>
"""
    for side, nm in ((1, "left"), (-1, "right")):
        cx = side * JAW_X / 2 * MM
        x += f"""  <link name="finger_{nm}">
    <inertial><origin xyz="{cx:.4f} 0 0.019"/><mass value="0.03"/>
      <inertia ixx="{fx:.6g}" ixy="0" ixz="0" iyy="{fy:.6g}" iyz="0" izz="{fz:.6g}"/></inertial>
    <visual><geometry><mesh filename="{mesh_prefix}finger_{nm}.stl"/></geometry><material name="pgea_white"/></visual>
    <collision name="jaw"><origin xyz="{side * JAW_X / 2 * MM:.4f} 0 {JAW_H / 2 * MM:.4f}"/><geometry><box size="{JAW_X * MM:.4f} {JAW_W * MM:.4f} {JAW_H * MM:.4f}"/></geometry></collision>
    <collision name="upper"><origin xyz="{side * FULL_T / 2 * MM:.4f} 0 {(JAW_H + Z_WEDGE) / 2 * MM:.4f}"/><geometry><box size="{FULL_T * MM:.4f} {FINGER_W * MM:.4f} {(Z_WEDGE - JAW_H) * MM:.4f}"/></geometry></collision>
    <collision name="tip"><geometry><mesh filename="{mesh_prefix}finger_{nm}_tip.stl"/></geometry></collision>
  </link>
  <joint name="finger_{nm}_joint" type="prismatic">
    <origin xyz="0 0 {FINGER_ORIGIN_Z * MM:.4f}" rpy="0 0 0"/><parent link="pgea_base"/><child link="finger_{nm}"/>
    <axis xyz="{side} 0 0"/><limit lower="0" upper="{lim:.4f}" effort="100" velocity="0.05"/>
  </joint>
"""
    x += f"""  <!-- TCP = fingertip centre, FR5 tool 1 (0, 0, 160) -->
  <link name="tcp"/>
  <joint name="tcp_joint" type="fixed">
    <origin xyz="0 0 {TCP_Z * MM:.4f}" rpy="0 0 0"/><parent link="pgea_base"/><child link="tcp"/>
  </joint>
  <!-- D435 + bracket: add a fixed camera link here once the mount is measured (design 5.2) -->
"""
    return x


def main():
    desc = os.path.join(HERE, "pgea_description")
    os.makedirs(os.path.join(desc, "urdf"), exist_ok=True)
    os.makedirs(os.path.join(desc, "meshes"), exist_ok=True)
    for side, nm in ((1, "left"), (-1, "right")):
        write_stl(os.path.join(desc, "meshes", f"finger_{nm}.stl"), finger_prism(side))
        write_stl(os.path.join(desc, "meshes", f"finger_{nm}_tip.stl"), tip_prism(side))

    # standalone gripper: root link "flange" = tool frame origin
    g = '<?xml version="1.0"?>\n<robot name="pgea_100_40_o">\n  <link name="flange"/>\n'
    g += gripper_links("../meshes/", "flange", "0 0 0") + "</robot>\n"
    open(os.path.join(desc, "urdf", "pgea_100_40_o.urdf"), "w", encoding="utf-8").write(g)

    # FR5 + gripper
    arm = open(os.path.join(HERE, "fr5", "fr5.urdf"), encoding="utf-8").read()
    arm = arm[: arm.rindex("</robot>")]
    arm += gripper_links("../pgea_description/meshes/", "wrist3_link", "0 0 0.100") + "</robot>\n"
    open(os.path.join(HERE, "fr5", "fr5_pgea_v2.urdf"), "w", encoding="utf-8").write(arm)
    print("written:", os.path.join(desc, "urdf", "pgea_100_40_o.urdf"), os.path.join(HERE, "fr5", "fr5_pgea_v2.urdf"))


if __name__ == "__main__":
    main()
