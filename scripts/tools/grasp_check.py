"""잡기 판정 검증: 기하 판정(손가락 영역 16x22mm 비었나)과 실제로 집어서 들어 본 결과를 비교.

배치마다 큐브 하나를 무작위로 골라 대상(0번)으로 바꾸고 -> 판정 -> 실제 잡기.
판정이 "열림"이면 열린 축으로, "막힘"이면 0° 축으로 잡아 본다 (막힘 판정이 맞는지도 확인).

사용:
  python scripts/tools/grasp_check.py [--noise]   (fr5_push 폴더에서)
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--noise", action="store_true", help="카메라 노이즈 켜기 (판정·잡는 위치에 적용)")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--rounds", type=int, default=4)
parser.add_argument("--finger_k", type=float, default=None, help="손가락 stiffness 바꾸기")
parser.add_argument("--finger_d", type=float, default=None)
parser.add_argument("--finger_f", type=float, default=None, help="손가락 최대 힘 (N)")
parser.add_argument("--open_each", type=float, default=None, help="잡기 전 손가락 하나 벌림 (m)")
parser.add_argument("--clear_depth", type=float, default=None, help="판정: 손가락 영역 깊이 (m)")
args = parser.parse_args()
app = AppLauncher(headless=True).app

import math  # noqa: E402

import torch  # noqa: E402

from fr5_push.tasks.direct.fr5_push.fr5_push_env import Fr5PushEnv, _wrap90  # noqa: E402
from fr5_push.tasks.direct.fr5_push.fr5_push_env_cfg import CUBE, Fr5PushEnvCfg  # noqa: E402
from isaaclab.utils.math import quat_from_euler_xyz  # noqa: E402

cfg = Fr5PushEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.camera_bottom_above_tip = 0.070  # 권장 장착 가정 (카메라가 벽보다 높음)
if not args.noise:
    cfg.obs_pos_noise_std = 0.0
    cfg.obs_yaw_noise_deg = 0.0
if args.open_each is not None:
    cfg.grasp_open_each = args.open_each
if args.clear_depth is not None:
    cfg.finger_clear_depth = args.clear_depth
print(f"[cfg] open_each={cfg.grasp_open_each} clear_depth={cfg.finger_clear_depth}")
act = cfg.robot_cfg.actuators["fingers"]
if args.finger_k is not None:
    act.stiffness = args.finger_k
if args.finger_d is not None:
    act.damping = args.finger_d
if args.finger_f is not None:
    act.effort_limit_sim = args.finger_f
print(f"[cfg] finger k={act.stiffness} d={act.damping} f={act.effort_limit_sim}")
env = Fr5PushEnv(cfg)
env.reset()
n, k = env.num_envs, env.cubes.num_objects
ar = torch.arange(n, device=env.device)
dt = env.physics_dt


def settle(seconds):
    for _ in range(round(seconds / dt)):
        env.robot.set_joint_position_target(env._hold_pos)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt)


# 판정(열림/막힘) x 결과(잡힘/안 잡힘)
tab = {(j, g): 0 for j in (True, False) for g in (True, False)}
held_only = 0
fpos_ok = []
for r in range(args.rounds):
    with torch.inference_mode():
        # 배치 만들고, 무작위 큐브를 0번(대상)으로 바꿔 넣기
        xy, yaw = env._sample_layout(n, k)
        t = torch.randint(0, k, (n,), device=env.device)
        perm = torch.arange(k, device=env.device).repeat(n, 1)
        perm[ar, 0], perm[ar, t] = t, 0
        xy = torch.gather(xy, 1, perm[..., None].expand(-1, -1, 2))
        yaw = torch.gather(yaw, 1, perm)
        q = quat_from_euler_xyz(torch.zeros_like(yaw).flatten(), torch.zeros_like(yaw).flatten(), yaw.flatten()).view(n, k, 4)
        pose = torch.cat([xy + env.scene.env_origins[:, None, :2], torch.full((n, k, 1), CUBE / 2 + 0.0005, device=env.device), q], -1)
        env.cubes.write_object_pose_to_sim(pose)
        env.cubes.write_object_velocity_to_sim(torch.zeros(n, k, 6, device=env.device))
        settle(0.3)

        # 판정은 "카메라로 본" 위치로
        oxy, oyaw = env.cubes_observed()
        clear = env.grasp_axes_clear(oxy, oyaw, torch.zeros(n, dtype=torch.long, device=env.device))  # (n, 2)
        judged = clear.any(-1)
        axis = torch.where(clear[:, 0] | ~clear[:, 1], 0, 1)
        ang = _wrap90(oyaw[:, 0]) + axis * math.pi / 2
        held, lifted = env.grasp_attempt(oxy[:, 0], ang)
        ok = held & lifted
    for jv in (True, False):
        for gv in (True, False):
            tab[(jv, gv)] += ((judged == jv) & (ok == gv)).sum().item()
    held_only += (held & ~lifted).sum().item()
    fpos_ok.append(env.last_finger_pos[ok].mean().item() if ok.any() else float("nan"))
    print(f"[round {r}] judged open {judged.float().mean()*100:.1f}%, grasped {ok.float().mean()*100:.1f}%")

tot = sum(tab.values())
o_tot = tab[(True, True)] + tab[(True, False)]
b_tot = tab[(False, True)] + tab[(False, False)]
print(f"\n===== 잡기 검증 ({'노이즈 있음' if args.noise else '노이즈 없음'}) : {tot}회 =====")
print(f"  판정 '열림' {o_tot}회 -> 실제로 잡힘 {tab[(True, True)] / max(o_tot, 1) * 100:5.1f} %   (낮으면 판정이 너무 느슨)")
print(f"  판정 '막힘' {b_tot}회 -> 실제로 잡힘 {tab[(False, True)] / max(b_tot, 1) * 100:5.1f} %   (높으면 판정이 너무 엄격)")
print(f"  위치값은 맞는데 들리지 않음 {held_only}회")
print(f"  잡혔을 때 손가락 위치 평균 {sum(fpos_ok) / len(fpos_ok) * 1000:.2f} mm (성공 범위 12.3~13.2)")
env.close()
app.close()
