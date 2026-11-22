"""후보 종류별로 실제 물리 결과 비교: 이웃 밀기 / 대상 면 밀기 / 벽 맞추기 / 벽 따라 밀기.

고른 후보의 '열림 예측'과 밀고 난 뒤 참값(물리)으로 열렸는지, 성공·넘어짐·쌓임을 종류별로 센다.
사용: python scripts/tools/cand_type_check.py [--pushes 9] [--policy look|demo]
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--pushes", type=int, default=9)
parser.add_argument("--policy", default="look", choices=["look", "demo"])
parser.add_argument("--tip", type=float, default=None, help="밀 때 손가락 끝 높이 (m)")
parser.add_argument("--square", action="store_true", help="벽 맞추기 후보 켜기")
args = parser.parse_args()
app = AppLauncher(headless=True).app

import torch  # noqa: E402

from fr5_push.tasks.direct.fr5_push.fr5_push_env import Fr5PushEnv  # noqa: E402
from fr5_push.tasks.direct.fr5_push.fr5_push_env_cfg import Fr5PushEnvCfg  # noqa: E402

cfg = Fr5PushEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = 0
if args.tip is not None:
    cfg.push_tip_z = args.tip
if args.square:
    cfg.cand_wall_square = True
print(f"[cfg] push_tip_z={cfg.push_tip_z} wall_square={cfg.cand_wall_square} mode={cfg.wall_square_mode}")
env = Fr5PushEnv(cfg)
env.reset()

n_nb = cfg.cand_neighbours * 2 * len(cfg.cand_angles) * len(cfg.cand_dists)
n_tg = 4 * len(cfg.cand_dists) if cfg.cand_target else 0
per_wall = 1 + 2 * len(cfg.cand_dists)


def kind_of(idx):
    k = torch.zeros_like(idx)  # 0 이웃
    k[idx >= n_nb] = 1  # 대상 면
    w = idx - n_nb - n_tg
    k[(w >= 0) & (w % per_wall == 0)] = 2  # 벽 맞추기
    k[(w >= 0) & (w % per_wall != 0)] = 3  # 벽 따라
    return k


names = ["이웃 밀기", "대상 면 밀기", "벽 맞추기", "벽 따라 밀기"]
keys = ["n", "pred", "open", "pred_open", "succ", "tilt", "stack"]
st = {i: dict.fromkeys(keys, 0) for i in range(4)}
for _ in range(args.pushes):
    with torch.inference_mode():
        act = env.demo_push_action(lookahead=args.policy == "look")
        env.step(act)
    kind = kind_of(env.chosen)
    real = ~env._skip
    for i in range(4):
        m = (kind == i) & real
        s = st[i]
        s["n"] += m.sum().item()
        s["pred"] += (m & env.chosen_opens).sum().item()
        s["open"] += (m & env.axis_open).sum().item()
        s["pred_open"] += (m & env.chosen_opens & env.axis_open).sum().item()
        s["succ"] += (m & env.success).sum().item()
        s["tilt"] += (m & env.fail_tilt).sum().item()
        s["stack"] += (m & env.fail_stack).sum().item()

tot = sum(s["n"] for s in st.values())
print(f"\n===== 후보 종류별 물리 결과 ({args.policy}, 밀기 {tot}회) =====")
print(f"{'종류':<10} {'횟수':>6} {'비율':>6} {'열림예측':>8} {'예측->실제열림':>14} {'실제열림':>8} {'성공':>6} {'넘어짐':>6} {'쌓임':>6}")
for i, s in st.items():
    nn = max(s["n"], 1)
    print(
        f"{names[i]:<10} {s['n']:>6} {s['n'] / max(tot, 1) * 100:5.1f}% {s['pred'] / nn * 100:7.1f}% "
        f"{s['pred_open'] / max(s['pred'], 1) * 100:13.1f}% {s['open'] / nn * 100:7.1f}% "
        f"{s['succ'] / nn * 100:5.1f}% {s['tilt'] / nn * 100:5.1f}% {s['stack'] / nn * 100:5.1f}%"
    )
env.close()
app.close()
