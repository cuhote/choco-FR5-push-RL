"""처음 배치에서 1/2/3 번 밀기 조합으로 대상을 열 수 있다고 '예측'되는 비율 (기하 예측, 물리 없음).

규칙이 도달할 수 있는 대략의 상한. 2·3 번째는 '진전'(대상 주변 빈 깊이) 상위 K 개 후보만 펼침.
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--topk", type=int, default=8)
parser.add_argument("--clear_depth", type=float, default=None)
parser.add_argument("--wall", type=float, default=None, help="트레이 벽 높이 (m)")
parser.add_argument("--dists", type=str, default=None, help="예: 0.055,0.040,0.030")
args = parser.parse_args()
app = AppLauncher(headless=True).app

import torch  # noqa: E402

from fr5_push.tasks.direct.fr5_push.fr5_push_env import Fr5PushEnv  # noqa: E402
from fr5_push.tasks.direct.fr5_push.fr5_push_env_cfg import Fr5PushEnvCfg  # noqa: E402

cfg = Fr5PushEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.camera_bottom_above_tip = 0.070  # 권장 장착 가정
cfg.obs_pos_noise_std = 0.0
cfg.obs_yaw_noise_deg = 0.0
if args.wall is not None:
    cfg.tray_wall_height = args.wall
if args.clear_depth is not None:
    cfg.finger_clear_depth = args.clear_depth
if args.dists is not None:
    cfg.cand_dists = tuple(float(v) for v in args.dists.split(","))
print(f"[look] wall={cfg.tray_wall_height} clear_depth={cfg.finger_clear_depth} dists={cfg.cand_dists}")
env = Fr5PushEnv(cfg)
env.reset()
n = env.num_envs
zero = torch.zeros(n, dtype=torch.long, device=env.device)


def opens_any(xy, yaw):
    c = env.build_candidates(xy, yaw)
    return (c["valid"] & c["opens"]).any(-1), c


def progress(xy, yaw):
    """대상의 '덜 열린 정도': 두 축 중 나은 쪽의 (필요 깊이 - 실제 빈 깊이) 합. 작을수록 열림에 가까움."""
    need = cfg.finger_clear_depth
    depths = []
    for d in torch.linspace(0.001, need, 12, device=env.device):
        cfg.finger_clear_depth = float(d)
        depths.append(env.sides_clear(xy, yaw, zero))
    cfg.finger_clear_depth = need
    depths = torch.stack(depths, -1).float()  # (n, 4, 12): 그 깊이까지 비었나
    free = depths.cumprod(-1).sum(-1) / 12 * need  # 각 면 빈 깊이 근사
    deficit = (need - free).clamp(min=0)
    ax = torch.stack([deficit[:, 0] + deficit[:, 1], deficit[:, 2] + deficit[:, 3]], -1)
    return ax.min(-1).values


with torch.inference_mode():
    xy, yaw, _, _ = env.cubes_2d()
    o1, c1 = opens_any(xy, yaw)
    o2 = o1.clone()
    o3 = o1.clone()
    C = c1["valid"].shape[1]
    # 1번째 후보 중 진전 상위 K 개
    prog1 = torch.stack([progress(c1["xy_pred"][:, i], yaw) for i in range(C)], 1)
    prog1[~c1["valid"]] = 1e9
    top1 = prog1.topk(args.topk, dim=1, largest=False).indices
    ar = torch.arange(n, device=env.device)
    for a in range(args.topk):
        i = top1[:, a]
        ok_i = c1["valid"][ar, i]
        xy1 = c1["xy_pred"][ar, i]
        o2_i, c2 = opens_any(xy1, yaw)
        o2 |= o2_i & ok_i
        prog2 = torch.stack([progress(c2["xy_pred"][:, q], yaw) for q in range(C)], 1)
        prog2[~c2["valid"]] = 1e9
        top2 = prog2.topk(args.topk, dim=1, largest=False).indices
        for b in range(args.topk):
            q = top2[:, b]
            ok_q = c2["valid"][ar, q] & ok_i
            o3_q, _ = opens_any(c2["xy_pred"][ar, q], yaw)
            o3 |= o3_q & ok_q
        o3 |= o2
        if False: print(f"[look] 첫 수 {a + 1}/{args.topk} 탐색 중... 2수 {o2.float().mean()*100:.1f}% 3수 {o3.float().mean()*100:.1f}%")
# ---- 못 푼 경우 분류
from fr5_push.tasks.direct.fr5_push.fr5_push_env import _ray_to_tray, _wrap90  # noqa: E402
import math  # noqa: E402
fail = ~o3
hx, hy = cfg.tray_inner_half
c, th = xy[:, 0], _wrap90(yaw[:, 0])
wall_d = torch.stack([_ray_to_tray(c, th + q * math.pi / 2, hx, hy) for q in range(4)], -1)  # 네 면 방향 벽까지
near_wall = (wall_d.min(-1).values - 0.0125) < (cfg.finger_clear_depth + 0.025)  # 한 면이 벽 쪽 손가락 영역+여유 안
far = torch.full_like(xy, 10.0); far[:, 0] = xy[:, 0]
wall_axes = env.grasp_axes_clear(far, yaw, zero)  # 이웃이 없을 때 축별로 열리나 (벽만 고려)
wall_one_axis = wall_axes.sum(-1) == 1
wall_none = wall_axes.sum(-1) == 0
nb = (torch.linalg.norm(xy[:, 1:] - c[:, None], dim=-1) < 0.045).sum(-1)
print(f"\n[why] 못 푼 경우 {fail.sum().item()}개 / 전체 {n}")
print(f"[why]   벽 때문에 한 축은 원래 불가능 (남은 한 축만 가능): {(fail & wall_one_axis).float().sum()/fail.sum()*100:.0f}%  (전체 중 이런 배치 {wall_one_axis.float().mean()*100:.0f}%)")
print(f"[why]   벽 때문에 두 축 다 불가능:                  {(fail & wall_none).float().sum()/fail.sum()*100:.0f}%  (전체 중 {wall_none.float().mean()*100:.0f}%)")
print(f"[why]   벽과 무관:                                  {(fail & ~wall_one_axis & ~wall_none).float().sum()/fail.sum()*100:.0f}%")
for q in range(5):
    print(f"[why]   대상 주변 45mm 안 이웃 {q}개: 못 푼 비율 {((fail & (nb == q)).sum() / (nb == q).sum().clamp(min=1) * 100):.0f}%  (배치 {(nb == q).float().mean()*100:.0f}%)")
print(f"\n===== 처음 배치 {n}개: 기하 예측으로 열 수 있는 비율 =====")
print(f"  1번 밀기로      {o1.float().mean()*100:5.1f} %")
print(f"  2번 조합으로    {o2.float().mean()*100:5.1f} %")
print(f"  3번 조합으로    {o3.float().mean()*100:5.1f} %   (탐색: 매 단계 진전 상위 {args.topk}개)")
env.close()
app.close()
