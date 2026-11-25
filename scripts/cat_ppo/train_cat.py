"""후보 선택용 categorical PPO (11/25).

rsl_rl 학습(연속값 점수 220개 + 노이즈 -> 최고점 실행)은 학습 신호가 220개에 흩어져 규칙(87.8%)을 못 넘었다.
분석 2(scripts/tools/oracle_check.py)에서 밀기 1번 물리 상한이 규칙보다 13.7%p 높아 여유는 확인됨.
여기서는 두 가지를 바꾼다.

  1) categorical 정책: 후보 i 를 고를 확률 = softmax(점수). 실제로 고른 후보에 학습 신호가 정확히 간다.
     안전하지 않은 후보(valid=False)는 확률 0 (점수 -inf).
  2) 후보마다 같은 신경망: 점수_i = f(대상 주변 정보 51, 후보 i 특징 12). 220개 후보에 같은 f 를 쓴다.
     한 후보에서 배운 것("이런 이웃을 이 방향으로 밀면 열린다")이 비슷한 후보에 그대로 쓰인다.

환경·후보·보상은 그대로 (residual_rule 은 끔: 정책이 직접 고름).

사용 (fr5_push 폴더에서):
  학습: python -u scripts/cat_ppo/train_cat.py --iters 300 --run_name cat_v1 > logs/train_cat_v1.log 2>&1
  평가: python -u scripts/cat_ppo/train_cat.py --eval logs/cat_ppo/<run>/model_100.pt --num_envs 256 --pushes 12
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--iters", type=int, default=300)
parser.add_argument("--steps", type=int, default=8, help="iteration 당 환경마다 밀기 수")
parser.add_argument("--run_name", default="cat")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--entropy", type=float, default=0.01)
parser.add_argument("--save_every", type=int, default=10)
parser.add_argument("--resume", default=None, help="이어서 학습할 model_*.pt")
parser.add_argument("--eval", default=None, help="평가할 model_*.pt (탐색 없이 최고 확률 후보)")
parser.add_argument("--pushes", type=int, default=12, help="평가: 환경마다 밀기 수")
parser.add_argument("--eval_policy", default="cat", choices=["cat", "look", "demo"],
                    help="평가할 정책: cat = --eval 모델, look/demo = 규칙 (같은 시드 = 같은 배치로 비교)")
parser.add_argument("--gui", action="store_true", help="평가를 화면으로 보기 (환경 수는 16 정도로)")
args = parser.parse_args()
app = AppLauncher(headless=not args.gui).app

import os  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from fr5_push.tasks.direct.fr5_push.fr5_push_env import Fr5PushEnv  # noqa: E402
from fr5_push.tasks.direct.fr5_push.fr5_push_env_cfg import Fr5PushEnvCfg  # noqa: E402

CTX = 51  # 대상 주변 정보 (관측 앞부분)
FEAT = 12  # 후보 하나의 특징 (10 + 탐색 정보 2)


class CandidatePolicy(nn.Module):
    """점수_i = scorer([ctx, 후보 i 특징]) (후보 공통). 가치 = critic([ctx, 후보 임베딩의 평균·최대])."""

    def __init__(self, hidden=128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(CTX + FEAT, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1)
        )
        self.embed = nn.Sequential(nn.Linear(CTX + FEAT, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.critic = nn.Sequential(
            nn.Linear(CTX + 2 * hidden, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1)
        )

    @staticmethod
    def split(obs):
        n = obs.shape[0]
        ctx = obs[:, :CTX]
        cand = obs[:, CTX:].reshape(n, -1, FEAT)
        x = torch.cat([ctx[:, None].expand(-1, cand.shape[1], -1), cand], dim=-1)
        return ctx, x

    def forward(self, obs, mask):
        ctx, x = self.split(obs)
        logits = self.scorer(x).squeeze(-1).masked_fill(~mask, -1e9)
        e = self.embed(x)
        m = mask[..., None].float()
        mean = (e * m).sum(1) / m.sum(1).clamp(min=1.0)
        mx = e.masked_fill(~mask[..., None], -1e9).max(1).values
        mx = torch.where(mask.any(-1, keepdim=True), mx, torch.zeros_like(mx))
        value = self.critic(torch.cat([ctx, mean, mx], dim=-1)).squeeze(-1)
        return logits, value


def make_env(num_envs, seed):
    cfg = Fr5PushEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.residual_rule = False  # 정책이 직접 고름 (넘긴 점수 그대로 argmax)
    env = Fr5PushEnv(cfg)
    obs, _ = env.reset()
    return env, obs["policy"]


def to_action(idx, C, device):
    """고른 후보만 큰 점수 -> env 가 그 후보를 실행 (안전한 후보 중 argmax)."""
    a = torch.zeros(idx.shape[0], C, device=device)
    a[torch.arange(idx.shape[0], device=device), idx] = 1e6
    return a


def evaluate(path):
    env, obs = make_env(args.num_envs, args.seed + 1000)
    dev = env.device
    if args.gui:  # 트레이 격자 전체가 보이게 카메라 배치
        o = env.scene.env_origins
        c, span = o.mean(0).tolist(), (o.max(0).values - o.min(0).values).max().item() + 0.6
        env.sim.set_camera_view(eye=[c[0] + 0.1, c[1] - span * 0.9, span * 1.1], target=[c[0], c[1], 0.0])
    if args.eval_policy == "cat":
        net = CandidatePolicy().to(dev)
        net.load_state_dict(torch.load(path, map_location=dev)["model"])
        net.eval()
    C = env.cfg.action_space
    k = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
    by_k = [0] * (env.cfg.max_pushes + 1)
    ep = succ = fail = blocked = corner = 0
    # 실패 원인 분석 (11/25): 판마다 기록
    nenv = env.num_envs
    zero = torch.zeros(nenv, dtype=torch.long, device=dev)
    wall_cls = torch.zeros(nenv, dtype=torch.long, device=dev)  # 판 시작 때 벽만으로 열리는 축 수 (2 = 벽 무관, 1 = 한 축 벽에 막힘, 0 = 두 축 다)
    tried = torch.zeros(nenv, dtype=torch.bool, device=dev)  # 이 판에서 잡기를 시도했나
    tried_closed = torch.zeros(nenv, dtype=torch.bool, device=dev)  # 참값으로는 안 열렸는데 (인식 오차로) 잡기를 시도했나
    cnt = {key: [0, 0, 0] for key in ("ep", "succ", "A", "B", "B_closed", "F")}  # 벽 분류(0,1,2)별
    with torch.inference_mode():
        for i in range(args.pushes):
            new = k == 0  # 새 판: 시작 배치의 벽 상황 기록
            if new.any():
                xy0, yaw0, _, _ = env.cubes_2d()
                far = torch.full_like(xy0, 10.0)
                far[:, 0] = xy0[:, 0]
                wall_cls = torch.where(new, env.grasp_axes_clear(far, yaw0, zero).sum(-1), wall_cls)
            if args.eval_policy == "cat":
                logits, _ = net(obs, env._cand["valid"])
                act = to_action(logits.argmax(-1), C, dev)
            else:
                act = env.demo_push_action(lookahead=args.eval_policy == "look")
            o, r, term, trunc, _ = env.step(act)
            obs = o["policy"]
            done = term | trunc
            k += 1
            tried |= env.grasp_tried
            tried_closed |= env.grasp_tried & ~env.axis_open
            for w in range(3):
                d = done & (wall_cls == w)
                ok, bad = env.success & d, env.failed & d
                rest = d & ~env.success & ~env.failed  # 3번 다 쓰고 끝
                cnt["ep"][w] += d.sum().item()
                cnt["succ"][w] += ok.sum().item()
                cnt["F"][w] += bad.sum().item()
                cnt["A"][w] += (rest & ~tried).sum().item()
                cnt["B"][w] += (rest & tried).sum().item()
                cnt["B_closed"][w] += (rest & tried_closed).sum().item()
            tried[done] = False
            tried_closed[done] = False
            for j in range(1, env.cfg.max_pushes + 1):
                by_k[j] += (env.success & done & (k == j)).sum().item()
            k[done] = 0
            ep += done.sum().item()
            succ += (env.success & done).sum().item()
            fail += (env.failed & done).sum().item()
            blocked += (env.fail_blocked & done).sum().item()
            corner += (env.fail_corner & done).sum().item()
            print(f"[eval] 밀기 {i + 1}/{args.pushes}", flush=True)
    print(f"\n===== 평가 ({args.eval_policy}): {path if args.eval_policy == 'cat' else '규칙'} | 시드 {args.seed + 1000}, 환경 {env.num_envs}개 =====")
    print(f"에피소드 {ep}개 (최대 {env.cfg.max_pushes}회 밀기)")
    print(f"  성공률 (실제로 들어 올림)  {succ / max(ep, 1) * 100:5.1f} %")
    print(f"  실패(전체)                {fail / max(ep, 1) * 100:5.1f} %  (그중 시작점 막힘 {blocked / max(ep, 1) * 100:4.1f} %)")
    cum = 0
    for j in range(1, env.cfg.max_pushes + 1):
        cum += by_k[j]
        print(f"  {j}번 안에 성공 (누적)      {cum / max(ep, 1) * 100:5.1f} %")
    E = max(sum(cnt["ep"]), 1)
    nf = sum(cnt["A"]) + sum(cnt["B"]) + sum(cnt["F"])
    print(f"\n  --- 실패 원인 (전체 판 대비 %, 못 든 판 {nf}개) ---")
    print(f"  A. 3번 밀어도 '열림' 판정이 한 번도 안 남       {sum(cnt['A']) / E * 100:5.1f} %")
    print(f"  B. '열림'으로 잡기를 시도했지만 못 듦           {sum(cnt['B']) / E * 100:5.1f} %  (그중 참값으로는 안 열려 있던 판 {sum(cnt['B_closed']) / E * 100:4.1f} %)")
    print(f"  F. 실패로 끝남 (넘어짐·쌓임·착지·시작점 막힘·모서리)  {sum(cnt['F']) / E * 100:5.1f} %  (그중 모서리에 끼어 실패 {corner / E * 100:4.1f} %)")
    print("\n  --- 시작 배치의 벽 상황별 ---")
    names = {2: "벽 무관 (두 축 모두 벽은 OK)", 1: "한 축이 벽에 막힘", 0: "두 축 모두 벽에 막힘"}
    for w in (2, 1, 0):
        e = max(cnt["ep"][w], 1)
        print(f"  {names[w]:24s} 판 {cnt['ep'][w] / E * 100:4.1f}% | 성공 {cnt['succ'][w] / e * 100:5.1f}% | A {cnt['A'][w] / e * 100:4.1f}% B {cnt['B'][w] / e * 100:4.1f}% F {cnt['F'][w] / e * 100:4.1f}%")
    env.close()


def train():
    torch.manual_seed(args.seed)
    env, obs = make_env(args.num_envs, args.seed)
    dev, n, T = env.device, env.num_envs, args.steps
    C = env.cfg.action_space
    net = CandidatePolicy().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    start_it = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_it = ck["iter"] + 1
    run_dir = os.path.join("logs", "cat_ppo", datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)
    print(f"[cat] 저장 위치 {run_dir}, 환경 {n}개, 후보 {C}개, 파라미터 {sum(p.numel() for p in net.parameters())}개", flush=True)

    gamma, lam, clip, epochs, mbs = 0.9, 0.95, 0.2, 5, 4
    for it in range(start_it, start_it + args.iters):
        t0 = time.time()
        B_obs = torch.zeros(T, n, obs.shape[1], device=dev)
        B_mask = torch.zeros(T, n, C, dtype=torch.bool, device=dev)
        B_act = torch.zeros(T, n, dtype=torch.long, device=dev)
        B_logp = torch.zeros(T, n, device=dev)
        B_val = torch.zeros(T, n, device=dev)
        B_rew = torch.zeros(T, n, device=dev)
        B_done = torch.zeros(T, n, device=dev)
        ep = succ = fail = 0
        with torch.inference_mode():
            for t in range(T):
                mask = env._cand["valid"].clone()
                logits, value = net(obs, mask)
                dist = torch.distributions.Categorical(logits=logits)
                idx = dist.sample()
                B_obs[t], B_mask[t], B_act[t] = obs, mask, idx
                B_logp[t], B_val[t] = dist.log_prob(idx), value
                o, r, term, trunc, _ = env.step(to_action(idx, C, dev))
                done = term | trunc
                B_rew[t], B_done[t] = r, done.float()
                ep += done.sum().item()
                succ += (env.success & done).sum().item()
                fail += (env.failed & done).sum().item()
                obs = o["policy"]
            _, last_val = net(obs, env._cand["valid"])
        # GAE
        adv = torch.zeros(T, n, device=dev)
        gae = torch.zeros(n, device=dev)
        for t in reversed(range(T)):
            nxt = last_val if t == T - 1 else B_val[t + 1]
            nonterm = 1.0 - B_done[t]
            delta = B_rew[t] + gamma * nxt * nonterm - B_val[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + B_val
        # PPO 업데이트
        N = T * n
        f = lambda x: x.reshape(N, *x.shape[2:]).clone()  # noqa: E731
        o_, m_, a_, lp_, ad_, rt_ = f(B_obs), f(B_mask), f(B_act), f(B_logp), f(adv), f(ret)
        ad_ = (ad_ - ad_.mean()) / (ad_.std() + 1e-8)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}
        cnt = 0
        for _ in range(epochs):
            perm = torch.randperm(N, device=dev)
            for b in perm.chunk(mbs):
                logits, value = net(o_[b], m_[b])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(a_[b])
                ratio = (lp - lp_[b]).exp()
                s1, s2 = ratio * ad_[b], ratio.clamp(1 - clip, 1 + clip) * ad_[b]
                pl = -torch.min(s1, s2).mean()
                vl = (value - rt_[b]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pl + 0.5 * vl - args.entropy * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    stats["policy_loss"] += pl.item()
                    stats["value_loss"] += vl.item()
                    stats["entropy"] += ent.item()
                    stats["approx_kl"] += (lp_[b] - lp).mean().item()
                    stats["clipfrac"] += ((ratio - 1).abs() > clip).float().mean().item()
                    cnt += 1
        stats = {k: v / cnt for k, v in stats.items()}
        sr, fr = succ / max(ep, 1), fail / max(ep, 1)
        dt = time.time() - t0
        writer.add_scalar("Episode/success_rate", sr, it)
        writer.add_scalar("Episode/fail_rate", fr, it)
        writer.add_scalar("Train/mean_reward", B_rew.mean().item(), it)
        for k, v in stats.items():
            writer.add_scalar(f"Loss/{k}", v, it)
        print(f"[cat] iter {it:4d} | 성공 {sr * 100:5.1f}% 실패 {fr * 100:4.1f}% (에피소드 {ep}) | 보상 {B_rew.mean().item():5.2f} "
              f"| 엔트로피 {stats['entropy']:.2f} kl {stats['approx_kl']:.4f} 가치손실 {stats['value_loss']:.2f} | {dt:.0f}s", flush=True)
        if (it + 1) % args.save_every == 0 or it == start_it + args.iters - 1:
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "iter": it}, os.path.join(run_dir, f"model_{it}.pt"))
    writer.close()
    env.close()


if args.eval or args.eval_policy != "cat":
    evaluate(args.eval)
else:
    train()
app.close()
