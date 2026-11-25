"""큐브 쌓임을 화면으로 보기: (1) 쌓인 경우 찾아 저장 -> (2) 그 밀기를 하나씩 천천히 다시 재생.

(1) 찾기 (화면 없이, 환경 512개, 탐색 규칙):
    python scripts/tools/stack_view.py find --cases 20
    -> logs/stack_cases.pt 에 "밀기 직전 큐브 상태 + 실행한 밀기" 저장
(2) 다시 보기 (화면, 환경 1개):
    python scripts/tools/stack_view.py replay --slow 5            # 전부 차례로, 5배 느리게
    python scripts/tools/stack_view.py replay --case 3 --slow 10  # 3번만
    노란 공 = 쌓일 큐브 (찾기 때 쌓였던 큐브), 하늘색 공 = 그리퍼가 민 큐브 (시작 때 그리퍼 앞에 있던 큐브)
    --headless 로 돌리면 화면 없이 재현 여부·단계만 출력

GPU 물리는 환경 수가 다르면 결과가 조금 달라질 수 있어, 다시 재생할 때 쌓이는지 매번 출력한다.
"""
import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["find", "replay"])
parser.add_argument("--cases", type=int, default=20, help="find: 저장할 경우 수")
parser.add_argument("--case", type=int, default=None, help="replay: 이 번호만")
parser.add_argument("--slow", type=float, default=5.0, help="replay: 몇 배 느리게 (화면)")
parser.add_argument("--hold", type=float, default=2.0, help="replay: 밀기 끝난 뒤 멈춰 보여 줄 시간 (초, 실제 시간)")
parser.add_argument("--file", default="logs/stack_cases.pt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.mode == "find":
    args.headless = True
app = AppLauncher(args).app

import math  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402

from fr5_push.tasks.direct.fr5_push.fr5_push_env import Fr5PushEnv  # noqa: E402
from fr5_push.tasks.direct.fr5_push.fr5_push_env_cfg import CUBE, Fr5PushEnvCfg  # noqa: E402

cfg = Fr5PushEnvCfg()
cfg.scene.num_envs = 512 if args.mode == "find" else 1
cfg.seed = 0
cfg.residual_rule = False  # 규칙 점수를 그대로 실행
cfg.do_grasp = args.mode == "find"  # 찾기는 평가와 같은 흐름, 다시 보기는 밀기만
env = Fr5PushEnv(cfg)
env.reset()
dt = env.physics_dt
T1 = cfg.t_descend
T2 = T1 + cfg.t_push
TB = T2 + cfg.t_backoff
T3 = T2 + cfg.t_lift


def phase_of(t):
    if t < T1:
        return "내려가기"
    if t < T2:
        return "밀기"
    if t < TB:
        return "물러나기"
    if t < T3:
        return "들기"
    return "대기"


tip_body = env.robot.find_bodies("tcp")[0][0]


def snapshot():
    """큐브 위치·자세 (환경 원점 기준)."""
    p = env.cubes.data.object_pos_w.clone()
    p[..., :2] -= env.scene.env_origins[:, None, :2]
    return p, env.cubes.data.object_quat_w.clone()


def pushed_cube(pos, start, d):
    """시작점에서 미는 방향 앞쪽 3cm 안, 옆으로 2cm 안에 있는 가장 가까운 큐브 = 민 큐브 (없으면 -1)."""
    rel = pos[..., :2] - start[:, None]
    along = (rel * d[:, None]).sum(-1)
    side = (rel[..., 0] * d[:, None, 1] - rel[..., 1] * d[:, None, 0]).abs()
    score = torch.where((along > 0) & (along < 0.035) & (side < 0.02), along, torch.full_like(along, 9.0))
    v, i = score.min(-1)
    return torch.where(v < 9.0, i, torch.full_like(i, -1))


# ====================================================================== (1) 찾기
if args.mode == "find":
    orig = env._apply_action
    st = {}

    def patched():
        t = env._sub * dt
        orig()
        z = env.cubes.data.object_pos_w[..., 2]
        hi = z > cfg.stacked_z
        new = hi.any(-1) & (st["onset_t"] < 0)
        if new.any():
            st["onset_t"][new] = t
            st["onset_cube"][new] = z[new].argmax(-1)

    env._apply_action = patched
    cases = []
    phases = {}
    pushes = 0
    while len(cases) < args.cases:
        with torch.inference_mode():
            act = env.demo_push_action(lookahead=True)
            pos0, quat0 = snapshot()
            st["onset_t"] = torch.full((env.num_envs,), -1.0, device=env.device)
            st["onset_cube"] = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
            env.step(act)
        pushes += env.num_envs
        hit = (env.fail_stack & ~env._skip).nonzero().flatten().tolist()
        for e in hit:
            t = st["onset_t"][e].item()
            ph = phase_of(t) if t >= 0 else "?"
            phases[ph] = phases.get(ph, 0) + 1
            if len(cases) < args.cases:
                cases.append(
                    dict(
                        pos=pos0[e].cpu(), quat=quat0[e].cpu(), start=env._start[e].cpu(), dir=env._dir[e].cpu(),
                        dist=env._dist[e].item(), yaw=env._yaw[e].item(), onset_t=t,
                        onset_cube=st["onset_cube"][e].item(), slot=env.chosen[e].item(),
                    )
                )
        print(f"[find] 밀기 {pushes}회, 쌓임 {sum(phases.values())}회, 저장 {len(cases)}개")
    torch.save(cases, args.file)
    tot = sum(phases.values())
    print(f"\n===== 쌓임 {tot}회 / 밀기 {pushes}회 ({tot / pushes * 100:.2f}%) =====")
    for ph, c in sorted(phases.items(), key=lambda x: -x[1]):
        print(f"  처음 쌓인 단계: {ph:<6} {c:4d}회 ({c / tot * 100:4.1f}%)")
    print(f"저장: {args.file} ({len(cases)}개)")
    env.close()
    app.close()
    raise SystemExit

# ====================================================================== (2) 다시 보기
cases = torch.load(args.file)
idxs = [args.case] if args.case is not None else list(range(len(cases)))
mk = VisualizationMarkers(
    VisualizationMarkersCfg(
        prim_path="/Visuals/stack_view",
        markers={
            "stack": sim_utils.SphereCfg(radius=0.005, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.9, 0.0))),
            "pushed": sim_utils.SphereCfg(radius=0.005, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 1.0))),
        },
    )
)
origin = env.scene.env_origins[0]
dev = env.device
summary = []
for ci in idxs:
    c = cases[ci]
    with torch.inference_mode():
        pos = c["pos"].to(dev).clone()
        pos[..., :2] += origin[:2]
        env.cubes.write_object_pose_to_sim(torch.cat([pos, c["quat"].to(dev)], -1)[None])
        env.cubes.write_object_velocity_to_sim(torch.zeros(1, pos.shape[0], 6, device=dev))
        start, d = c["start"].to(dev)[None], c["dir"].to(dev)[None]
        env._start, env._dir = start, d
        env._dist = torch.tensor([c["dist"]], device=dev)
        env._yaw = torch.tensor([c["yaw"]], device=dev)
        env._skip = torch.zeros(1, dtype=torch.bool, device=dev)
        env.set_finger_mode("push")
        jp = env._joint_targets(start, cfg.safe_tip_z)
        env.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        env.robot.set_joint_position_target(jp)
        env._sub = 0
        env.scene.write_data_to_sim()
        env.scene.update(dt)
        pj = pushed_cube(c["pos"].to(dev)[None], start, d)[0].item()
        sc = c["onset_cube"]
        cx, cy = c["pos"][max(sc, 0), 0].item() + origin[0].item(), c["pos"][max(sc, 0), 1].item() + origin[1].item()
        if not args.headless:
            env.sim.set_camera_view(eye=[cx + 0.16, cy - 0.20, 0.20], target=[cx, cy, 0.01])
        ang = math.degrees(math.atan2(c["dir"][1].item(), c["dir"][0].item()))
        print(f"\n===== 경우 {ci}: 밀기 {c['dist'] * 1000:.0f}mm, 방향 {ang:.0f}°, 후보 {c['slot']}번 | "
              f"찾기 때 쌓인 큐브 {sc}번 ({phase_of(c['onset_t'])} {c['onset_t']:.2f}s), 민 큐브 {pj}번 =====")
        onset = None
        log = []
        n_steps = cfg.decimation + round(args.hold / dt) if not args.headless else cfg.decimation
        for s in range(n_steps):
            t0 = time.time()
            if s < cfg.decimation:
                env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=not args.headless)
            env.scene.update(dt)
            p = env.cubes.data.object_pos_w[0]
            markers, kinds = [], []
            if sc >= 0:
                markers.append(p[sc] + torch.tensor([0, 0, 0.022], device=dev)); kinds.append(0)
            if pj >= 0 and pj != sc:
                markers.append(p[pj] + torch.tensor([0, 0, 0.022], device=dev)); kinds.append(1)
            if markers and not args.headless:
                mk.visualize(translations=torch.stack(markers), marker_indices=torch.tensor(kinds))
            t = s * dt
            if s < cfg.decimation and s % 12 == 0:
                tip = env.robot.data.body_pos_w[0, tip_body]
                fing = env.robot.data.joint_pos[0, env._fingers] * 1000
                j = sc if sc >= 0 else int(p[:, 2].argmax())
                dxy = torch.linalg.norm(p[j, :2] - tip[:2]).item() * 1000
                moved = torch.linalg.norm(p[j, :2] - pos[j, :2]).item() * 1000
                oth = torch.linalg.norm(p[:, :2] - p[j, :2], dim=-1)
                oth[j] = 9
                hx, hy = cfg.tray_inner_half
                wall = min(hx - abs(p[j, 0].item() - origin[0].item()), hy - abs(p[j, 1].item() - origin[1].item())) * 1000 - 12.5
                log.append(f"  t={t:4.2f}s {phase_of(t):<5} 끝 높이 {tip[2] * 1000:5.1f}mm | 큐브{j:2d} 높이 {p[j, 2] * 1000:5.1f}mm "
                           f"그리퍼까지 {dxy:5.1f}mm 이동 {moved:4.1f}mm 옆큐브 {oth.min().item() * 1000:4.1f}mm 벽 {wall:5.1f}mm | 손가락 {fing[0]:4.1f}/{fing[1]:4.1f}mm")
            if onset is None and s < cfg.decimation and (p[:, 2] > cfg.stacked_z).any():
                j = int(p[:, 2].argmax())
                tip = env.robot.data.body_pos_w[0, tip_body]
                others = torch.linalg.norm(p[:, :2] - p[j, :2], dim=-1)
                others[j] = 9
                onset = (t, j)
                log.append(f"  >>> t={t:.2f}s [{phase_of(t)}] 큐브 {j}번이 {p[j, 2] * 1000:.1f}mm 로 올라감. "
                           f"그리퍼 끝 높이 {tip[2] * 1000:.1f}mm, 그리퍼까지 {torch.linalg.norm(p[j, :2] - tip[:2]).item() * 1000:.1f}mm, "
                           f"아래 큐브 {int(others.argmin())}번 ({others.min().item() * 1000:.1f}mm 옆)")
            if not args.headless:
                wait = dt * args.slow - (time.time() - t0) if s < cfg.decimation else dt - (time.time() - t0)
                if wait > 0:
                    time.sleep(wait)
        print("\n".join(log))
        z_end = env.cubes.data.object_pos_w[0, :, 2]
        stacked = bool((z_end > cfg.stacked_z).any())
        print(f"  결과: {'쌓임 재현됨' if stacked else '이번에는 안 쌓임'}"
              + (f" (처음 {onset[0]:.2f}s {phase_of(onset[0])}, 큐브 {onset[1]}번{' = 민 큐브' if onset[1] == pj else ''})" if onset else ""))
        summary.append((ci, stacked, phase_of(onset[0]) if onset else "-", onset[1] == pj if onset else None))
print("\n===== 요약 =====")
for ci, s, ph, is_p in summary:
    print(f"  경우 {ci:2d}: {'쌓임' if s else '안 쌓임'}  단계 {ph:<5} {'민 큐브가 올라감' if is_p else ('다른 큐브가 올라감' if is_p is False else '')}")
env.close()
app.close()
