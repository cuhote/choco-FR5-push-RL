"""기하 계산(후보·안전 검사·관측·규칙 점수)이 코드 정리 전후로 똑같은지 확인용 스냅샷.

리셋 직후 배치(물리 진행 없음)에서 관측, 후보, 규칙 점수를 저장한다. 같은 시드면 정리 전후 값이 정확히 같아야 한다.
--fixture 를 주면 실물 코드(choco_ws push_geom) 시험용 JSON 도 만든다 (카메라로 본 큐브 위치 -> 기대 관측).

사용 (fr5_push 폴더에서):
  python scripts/tools/geom_snapshot.py --out logs/geom_before.pt
  python scripts/tools/geom_snapshot.py --out logs/geom_after.pt --compare logs/geom_before.pt
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", required=True)
parser.add_argument("--compare", default=None)
parser.add_argument("--fixture", default=None, help="실물 시험용 JSON 경로 (환경 4개분)")
args = parser.parse_args()
app = AppLauncher(headless=True).app

import json  # noqa: E402

import torch  # noqa: E402

from fr5_push.tasks.direct.fr5_push.fr5_push_env import Fr5PushEnv  # noqa: E402
from fr5_push.tasks.direct.fr5_push.fr5_push_env_cfg import Fr5PushEnvCfg  # noqa: E402

cfg = Fr5PushEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.residual_rule = False
env = Fr5PushEnv(cfg)
obs, _ = env.reset()
with torch.inference_mode():
    snap = {
        "obs": obs["policy"].cpu(),
        "obs_xy": env._obs_xy.cpu(),
        "obs_yaw": env._obs_yaw.cpu(),
        "valid": env._cand["valid"].cpu(),
        "opens": env._cand["opens"].cpu(),
        "look": env.demo_push_action(lookahead=True).cpu(),
        "demo": env.demo_push_action().cpu(),
    }
torch.save(snap, args.out)
print(f"[snap] 저장 {args.out}: obs {tuple(snap['obs'].shape)}, 안전한 후보 평균 {snap['valid'].float().sum(-1).mean():.1f}")

if args.compare:
    ref = torch.load(args.compare)
    ok = True
    for k, v in snap.items():
        same = torch.equal(v, ref[k])
        diff = (v.float() - ref[k].float()).abs().max().item()
        print(f"[snap] {k:8s} 같음={same} 최대 차이 {diff:.3e}")
        ok &= same
    print("[snap] 결과:", "완전히 같음" if ok else "다름")

if args.fixture:
    m = 4
    from fr5_push.push_geom import geom_cfg_dict  # noqa: E402

    fix = {
        "geom_cfg": geom_cfg_dict(cfg),
        "push_count": [0] * m,
        "obs_xy": snap["obs_xy"][:m].tolist(),
        "obs_yaw": snap["obs_yaw"][:m].tolist(),
        "expected_obs": snap["obs"][:m].tolist(),
        "expected_valid": snap["valid"][:m].tolist(),
        "expected_look": [[float(x) if x == x else None for x in row] for row in snap["look"][:m].tolist()],
    }
    with open(args.fixture, "w") as f:
        json.dump(fix, f)
    print(f"[snap] 실물 시험용 JSON: {args.fixture}")
env.close()
app.close()
