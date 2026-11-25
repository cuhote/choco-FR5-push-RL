"""학습한 정책 평가 (headless). 규칙·무작위와 같은 기준으로 비교한다.

사용 (fr5_push 폴더에서):
  python scripts/rsl_rl/evaluate.py --task Template-Fr5-Push-Direct-v0 --num_envs 256 --headless
  python scripts/rsl_rl/evaluate.py ... --policy demo            # 규칙
  python scripts/rsl_rl/evaluate.py ... --policy random          # 무작위
  python scripts/rsl_rl/evaluate.py ... --checkpoint logs/rsl_rl/fr5_push/<날짜>/model_499.pt
체크포인트를 안 주면 logs/rsl_rl/fr5_push 의 가장 최근 실행, 가장 마지막 모델을 쓴다.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate FR5 push policy")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Template-Fr5-Push-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--pushes", type=int, default=12, help="환경마다 밀기 횟수")
parser.add_argument("--max_pushes", type=int, default=None, help="에피소드당 최대 밀기 횟수 (기본 = cfg)")
parser.add_argument("--policy", type=str, default="rl", choices=["rl", "demo", "look", "random"])
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import fr5_push.tasks  # noqa: F401, E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.residual_rule = args_cli.policy == "rl"  # rl 은 학습 때처럼 규칙+보정, 규칙 기준선은 점수 그대로
    if args_cli.max_pushes is not None:
        env_cfg.max_pushes = args_cli.max_pushes
        env_cfg.episode_length_s = (args_cli.max_pushes + 1) * env_cfg.decimation * env_cfg.sim.dt
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped

    policy = None
    if args_cli.policy == "rl":
        if args_cli.checkpoint:
            path = retrieve_file_path(args_cli.checkpoint)
        else:
            root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            path = get_checkpoint_path(root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] checkpoint: {path}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(path)
        policy = runner.get_inference_policy(device=u.device)  # 평균 행동 (탐색 노이즈 없음)

    obs = env.get_observations()
    n = u.num_envs
    ep = succ = fail = landed = pushes = skipped = pred_open = 0
    succ_at = {}
    po_n = po_s = f_tilt = f_stack = 0
    g_try = g_ok = g_axis_true = 0
    for _ in range(args_cli.pushes):
        with torch.inference_mode():
            if args_cli.policy == "rl":
                act = policy(obs)
            elif args_cli.policy == "demo":
                act = u.demo_push_action()
            elif args_cli.policy == "look":
                act = u.demo_push_action(lookahead=True)
            else:
                act = torch.rand(n, u.cfg.action_space, device=u.device)
            u._last_count = (u.push_count + 1).clone()  # 이번 밀기가 몇 번째인지 (리셋 전에 기록)
            obs, _, dones, _ = env.step(act)
        done = dones.bool()
        if done.any():
            for c_ in u._last_count[done & u.success].tolist():
                succ_at[c_] = succ_at.get(c_, 0) + 1
        ep += done.sum().item()
        succ += (u.success & done).sum().item()
        fail += (u.failed & done).sum().item()
        landed += u.landed_on_cube.sum().item()
        skipped += u._skip.sum().item()
        pred_open += u.chosen_opens.sum().item()
        po_n += u.chosen_opens.sum().item()
        po_s += (u.chosen_opens & u.success).sum().item()
        f_tilt += (u.fail_tilt & done).sum().item()
        g_try += u.grasp_tried.sum().item()
        g_ok += u.grasp_ok.sum().item()
        g_axis_true += (u.grasp_tried & u.axis_open).sum().item()
        f_stack += (u.fail_stack & done).sum().item()
        pushes += n

    print(f"\n===== {args_cli.policy} : 에피소드 {ep}개, 밀기 {pushes}회 =====")
    label = "성공률 (실제로 들어 올림)" if u.cfg.do_grasp else "성공률 (한 축 열림)"
    print(f"  {label:24s} {succ / max(ep, 1) * 100:5.1f} %")
    print(f"  실패(넘어짐/쌓임/착지)     {fail / max(ep, 1) * 100:5.1f} %")
    print(f"  큐브 위 착지 (밀기당)      {landed / pushes * 100:5.1f} %")
    print(f"  안전한 후보 없음 (밀기당)  {skipped / pushes * 100:5.1f} %  (실물: 실행 안 함)")
    print(f"  고른 후보가 열림 예측      {pred_open / pushes * 100:5.1f} %")
    if u.cfg.do_grasp:
        print(f"  잡기 시도 {g_try}회 -> 들림 {g_ok / max(g_try, 1) * 100:5.1f} %   (시도 중 참값으로도 열림 {g_axis_true / max(g_try, 1) * 100:.1f} %)")
    print(f"  '열림 예측' 밀기 -> 성공       {po_s / max(po_n, 1) * 100:5.1f} %  (예측 정확도)")
    print(f"  실패 내역: 대상 기울어짐 {f_tilt / max(ep, 1) * 100:.1f}%, 큐브 쌓임 {f_stack / max(ep, 1) * 100:.1f}%")
    acc = 0
    for c_ in sorted(succ_at):
        acc += succ_at[c_]
        print(f"    {c_}번 이내 성공 (누적)      {acc / max(ep, 1) * 100:5.1f} %")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
