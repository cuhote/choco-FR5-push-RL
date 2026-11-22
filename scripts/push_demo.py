"""학습 없이 규칙대로 밀어 보기 (화면 확인용).

규칙(demo): 안전한 후보 중 "밀면 대상이 열린다"고 예측되는 것을 가까운 이웃부터 고른다.
탐색(look): 바로 열리는 후보가 없으면 두 수 앞을 내다보고 고른다.
비교(random): 안전한 후보 중 무작위.

사용:
  python scripts/push_demo.py --num_envs 4                       # 화면으로 보기
  python scripts/push_demo.py --num_envs 128 --headless --pushes 12   # 성공률 측정
  python scripts/push_demo.py --num_envs 128 --headless --pushes 12 --policy random
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FR5 push demo (rule-based)")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--task", type=str, default="Template-Fr5-Push-Direct-v0")
parser.add_argument("--policy", type=str, default="demo", choices=["demo", "look", "random"])
parser.add_argument("--pushes", type=int, default=0, help="밀기 횟수 (0 = 창을 닫을 때까지)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import fr5_push.tasks  # noqa: F401, E402


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped  # Fr5PushEnv
    env.reset()

    episodes = successes = failures = first_total = first_success = landed = pushes = 0
    i = 0
    while simulation_app.is_running() and (args_cli.pushes == 0 or i < args_cli.pushes):
        with torch.inference_mode():
            if args_cli.policy == "demo":
                actions = u.demo_push_action()
            elif args_cli.policy == "look":
                actions = u.demo_push_action(lookahead=True)
            else:
                actions = torch.rand(u.num_envs, u.cfg.action_space, device=u.device)
            first = u.push_count == 0
            _, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
        # u.success / u.failed / u.landed_on_cube 는 리셋 직전 값이 보관돼 있음
        episodes += done.sum().item()
        successes += (u.success & done).sum().item()
        failures += (u.failed & done).sum().item()
        first_total += first.sum().item()
        first_success += (u.success & first).sum().item()
        landed += u.landed_on_cube.sum().item()
        pushes += u.num_envs
        i += 1
        print(
            f"[push {i:3d}] 이번: 성공 {(u.success).sum().item()}, 실패 {(u.failed).sum().item()}, "
            f"큐브 위 착지 {u.landed_on_cube.sum().item()} / {u.num_envs}"
        )

    if episodes > 0:
        print(f"\n===== {args_cli.policy} 결과 =====")
        print(f"에피소드 {episodes}개 (최대 {u.cfg.max_pushes}회 밀기)")
        print(f"  성공률            {successes / episodes * 100:5.1f} %")
        print(f"  실패(넘어짐/쌓임)  {failures / episodes * 100:5.1f} %")
        print(f"  첫 밀기로 성공     {first_success / max(first_total, 1) * 100:5.1f} %")
        print(f"  큐브 위에 착지     {landed / pushes * 100:5.1f} % (밀기 1회당)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
