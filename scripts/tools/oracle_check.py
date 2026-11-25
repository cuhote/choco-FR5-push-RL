"""분석 2: 물리 상한 (밀기 1번). 같은 배치에서 후보를 하나씩 실제 물리로 밀어 보고 비교한다.

질문: 탐색 규칙이 고른 밀기로 안 열린 배치에서, 다른 후보 중 실제로 여는 것이 있었나?
  -> 있으면: 지금 후보 안에 규칙보다 나은 답이 있다 (학습 방식 개선으로 넘을 여지)
  -> 거의 없으면: 후보(행동) 자체를 넓혀야 한다

방법 (배치마다):
  1) 리셋 -> 0.5초 안정 -> 큐브 위치·자세 저장 -> 후보 220개 계산 (노이즈 없음, 참값)
  2) 후보 i = 0..219 마다: 저장한 배치로 되돌림 -> 후보 i 로 밀기 1회 (물리) -> 결과 기록
  3) 결과 = 참값으로 한 축 열림(18mm) + 실패 없음 (넘어짐·쌓임·큐브 위 착지)
     실제 잡기는 하지 않음 (판정이 보수적이라 실제 성공률보다 낮게 나옴. 규칙과 후보를 같은 기준으로 비교하는 용도)

사용 (fr5_push 폴더에서):
  python -u scripts/tools/oracle_check.py --num_envs 256 > logs/analysis_2_oracle.log 2>&1
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out", default="logs/analysis_2_oracle.pt")
args = parser.parse_args()
app = AppLauncher(headless=True).app

import time  # noqa: E402

import torch  # noqa: E402

from fr5_push.tasks.direct.fr5_push.fr5_push_env import Fr5PushEnv  # noqa: E402
from fr5_push.tasks.direct.fr5_push.fr5_push_env_cfg import (  # noqa: E402
    CAND_ANGLES, CAND_DISTS, CAND_NEIGHBOURS, Fr5PushEnvCfg,
)

cfg = Fr5PushEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.residual_rule = False  # 넘겨준 점수 그대로 실행
cfg.obs_pos_noise_std = 0.0  # 상한을 보려는 것이라 계획도 참값으로
cfg.obs_yaw_noise_deg = 0.0
env = Fr5PushEnv(cfg)
env.reset()
dev, n, dt = env.device, env.num_envs, env.physics_dt
zero = torch.zeros(n, dtype=torch.long, device=dev)


def physics_steps(k):
    for _ in range(k):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt)


with torch.inference_mode():
    # 1) 안정 -> 저장 -> 후보 계산
    physics_steps(round(0.5 / dt))
    pose0 = torch.cat([env.cubes.data.object_pos_w, env.cubes.data.object_quat_w], -1).clone()
    env._get_observations()  # 참값으로 후보 + 탐색 정보 계산 (env._cand)
    cand = env._cand
    valid = cand["valid"].clone()
    C = valid.shape[1]
    look = env.demo_push_action(lookahead=True).masked_fill(~valid, float("-inf")).argmax(-1)
    print(f"[oracle] 배치 {n}개, 후보 {C}개, 배치당 안전한 후보 평균 {valid.float().sum(-1).mean():.1f}개", flush=True)

    # 2) 후보마다 실제로 밀어 보기
    opened = torch.zeros(n, C, dtype=torch.bool, device=dev)
    failed = torch.zeros(n, C, dtype=torch.bool, device=dev)
    t0 = time.time()
    for i in range(C):
        env.cubes.write_object_pose_to_sim(pose0)
        env.cubes.write_object_velocity_to_sim(torch.zeros(n, pose0.shape[1], 6, device=dev))
        env._cand = cand
        act = torch.zeros(n, C, device=dev)
        act[:, i] = 1e6
        env._pre_physics_step(act)
        for _ in range(cfg.decimation):
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt)
        xy, yaw, upright, z = env.cubes_2d()
        bad = ~upright[:, 0] | (z > cfg.stacked_z).any(-1) | env.landed_on_cube
        opened[:, i] = env.grasp_axes_clear(xy, yaw, zero).any(-1) & ~bad & valid[:, i]
        failed[:, i] = bad & valid[:, i]
        if (i + 1) % 20 == 0 or i == C - 1:
            el = time.time() - t0
            print(f"[oracle] 후보 {i + 1}/{C} 완료 ({el / 60:.1f}분, 남은 약 {el / (i + 1) * (C - i - 1) / 60:.1f}분)", flush=True)

    # 3) 정리
    ar = torch.arange(n, device=dev)
    has_valid = valid.any(-1)
    look_open = opened[ar, look] & has_valid
    any_open = opened.any(-1)
    look_fail = ~look_open & has_valid
    rescue = look_fail & any_open  # 규칙은 못 열었는데 다른 후보는 연 배치
    pct = lambda m, d=None: (m.float().sum() / (has_valid.float().sum() if d is None else d.float().sum().clamp(min=1)) * 100).item()  # noqa: E731

    n_nb = CAND_NEIGHBOURS * 2 * len(CAND_ANGLES) * len(CAND_DISTS)
    n_tg = 4 * len(CAND_DISTS)
    kinds = {"이웃 밀기": slice(0, n_nb), "대상 밀기": slice(n_nb, n_nb + n_tg), "벽 관련": slice(n_nb + n_tg, C)}

    print(f"\n===== 분석 2: 밀기 1번, 물리 결과 (배치 {int(has_valid.sum())}개, 참값 한 축 18mm 열림 + 실패 없음) =====")
    print(f"  탐색 규칙이 고른 후보로 열림      {pct(look_open):5.1f} %")
    print(f"  220개 중 하나라도 열림 (물리 상한) {pct(any_open & has_valid):5.1f} %")
    print(f"  -> 차이 (규칙을 넘을 수 있는 여유)  {pct(any_open & has_valid) - pct(look_open):5.1f} %p")
    print(f"  규칙이 못 연 배치 중, 다른 후보로는 열린 비율  {pct(rescue, look_fail):5.1f} %  ({int(rescue.sum())}/{int(look_fail.sum())})")
    print(f"  규칙이 고른 후보의 실패(넘어짐·쌓임·착지)  {pct(failed[ar, look] & has_valid):5.1f} %")
    print(f"  배치당 여는 후보 수 (열리는 배치만) 평균  {opened.float().sum(-1)[any_open].mean():.1f}개")
    print("\n  규칙이 못 연 배치를 연 후보의 종류 (한 배치에 여러 종류면 각각 셈):")
    for name, sl in kinds.items():
        print(f"    {name:6s} {pct(rescue & opened[:, sl].any(-1), rescue):5.1f} %")
    torch.save(dict(opened=opened.cpu(), failed=failed.cpu(), valid=valid.cpu(), look=look.cpu(), pose0=pose0.cpu()), args.out)
    print(f"\n저장: {args.out}")

env.close()
app.close()
