"""큐브 쌓임이 밀기의 어느 단계에서 생기는지 기록 (내려가기/밀기/들기/대기)."""
import sys
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import torch  # noqa: E402
from fr5_push.tasks.direct.fr5_push.fr5_push_env import Fr5PushEnv  # noqa: E402
from fr5_push.tasks.direct.fr5_push.fr5_push_env_cfg import Fr5PushEnvCfg  # noqa: E402

cfg = Fr5PushEnvCfg(); cfg.scene.num_envs = 512; cfg.camera_bottom_above_tip = 0.07
env = Fr5PushEnv(cfg); env.reset()
t1 = cfg.t_descend; t2 = t1 + cfg.t_push; t3 = t2 + cfg.t_lift
phase_of = lambda t: "descend" if t < t1 else "push" if t < t2 else "lift" if t < t3 else "settle"
orig = env._apply_action
state = {}
counts = {}
held_by_gripper = 0


def patched():
    t = env._sub * env.physics_dt
    orig()
    z = env.cubes.data.object_pos_w[..., 2]
    hi = (z > cfg.stacked_z).any(-1)
    new = hi & ~state["was"]
    if new.any():
        ph = phase_of(t)
        counts[ph] = counts.get(ph, 0) + new.sum().item()
        if ph == "lift" and len(state["samples"]) < 12:
            xy, yaw, upright, zz = env.cubes_2d()
            tip = env.robot.find_bodies("tcp")[0][0]
            gp = env.robot.data.body_pos_w[:, tip] - env.scene.env_origins
            for e in new.nonzero().flatten().tolist()[:3]:
                j = int(zz[e].argmax())
                d = torch.linalg.norm(xy[e, j] - gp[e, :2]).item()
                others = torch.linalg.norm(xy[e] - xy[e, j], dim=-1)
                others[j] = 9
                state["samples"].append(
                    f"cube{j:2d} z={zz[e,j]*1000:5.1f}mm upright={bool(upright[e,j])} "
                    f"그리퍼까지 {d*1000:5.1f}mm 그리퍼끝높이 {gp[e,2]*1000:5.1f}mm 가장 가까운 큐브 {others.min()*1000:5.1f}mm "
                    f"t={t:.2f}s")
    state["was"] |= hi
    if env._sub == round((t3 - 0.05) / env.physics_dt):  # 들기 끝 무렵: 큐브가 그리퍼와 함께 떠 있나
        tipz = env.robot.data.joint_pos[:, env._j["pz"]] - cfg.tcp_offset
        state["lifted_with"] += ((z > 0.030).any(-1) & (tipz > 0.025)).sum().item()


env._apply_action = patched
state["lifted_with"] = 0
state["samples"] = []
pushes = 0
with torch.inference_mode():
    for _ in range(9):
        state["was"] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env.step(env.demo_push_action())
        pushes += env.num_envs
print(f"[stack] 밀기 {pushes}회 중 쌓임이 처음 생긴 단계: {counts}")
print(f"[stack] 들기 끝 무렵 큐브가 바닥에서 30mm 넘게 떠 있음 (그리퍼에 딸려 올라감): {state['lifted_with']}회")
for x in state["samples"]:
    print("[stack]   ", x)
env.close(); app.close()
