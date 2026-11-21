from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

from .fr5_push_env_cfg import CUBE, TRAY_PARTS, Fr5PushEnvCfg


class Fr5PushEnv(DirectRLEnv):
    cfg: Fr5PushEnvCfg

    def __init__(self, cfg: Fr5PushEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._hold_pos = self.robot.data.default_joint_pos.clone()  # 그리퍼 대기 자세

    # ------------------------------------------------------------------ 장면
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cubes = RigidObjectCollection(self.cfg.cubes_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -0.005))

        # 트레이 (움직이지 않는 충돌체 5개)
        for name, (size, pos) in TRAY_PARTS.items():
            part = sim_utils.CuboidCfg(
                size=size,
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.4),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.6)),
            )
            part.func(f"/World/envs/env_0/Tray/{name}", part, translation=pos)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_object_collections["cubes"] = self.cubes

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ 행동 (C단계: 그리퍼는 대기)
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._hold_pos)

    # ------------------------------------------------------------------ 관측/보상/종료 (C단계: 임시)
    def _get_observations(self) -> dict:
        target_pos = self.cubes.data.object_pos_w[:, 0] - self.scene.env_origins
        return {"policy": target_pos}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.debug_print and self.common_step_counter % 30 == 0:
            speed = torch.linalg.norm(self.cubes.data.object_lin_vel_w, dim=-1)  # (env, 18)
            z = self.cubes.data.object_pos_w[..., 2]
            print(
                f"[debug] step {self.common_step_counter:5d} | cube max speed {speed.max().item() * 1000:7.2f} mm/s"
                f" | z min {z.min().item() * 1000:6.2f} mm, max {z.max().item() * 1000:6.2f} mm (정상 12.5)"
            )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(time_out), time_out

    # ------------------------------------------------------------------ 리셋: 큐브 무작위 배치
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)
        k = self.cubes.num_objects  # 18

        # 그리퍼: 대기 자세로
        jp = self.robot.data.default_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
        self.robot.set_joint_position_target(jp, env_ids=env_ids)

        # 큐브 xy, yaw: 겹치지 않게 하나씩 배치 (트레이 중심 기준 좌표)
        xy, yaw = self._sample_layout(n, k)
        xy = xy + self.scene.env_origins[env_ids, None, :2]
        yaw = yaw.reshape(-1)
        zero = torch.zeros_like(yaw)
        quat = quat_from_euler_xyz(zero, zero, yaw).view(n, k, 4)

        z = torch.full((n, k, 1), CUBE / 2 + 0.0005, device=self.device)  # 바닥 위 0.5mm에서 내려놓기
        pose = torch.cat([xy, z, quat], dim=-1)
        self.cubes.write_object_pose_to_sim(pose, env_ids=env_ids)
        self.cubes.write_object_velocity_to_sim(torch.zeros(n, k, 6, device=self.device), env_ids=env_ids)

    # ------------------------------------------------------------------ 배치 도우미
    def _sample_layout(self, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """큐브 k개의 (xy, yaw)를 n개 환경에 대해 뽑는다 (트레이 중심 기준). 0번 = 대상(막힌 큐브)."""
        dev, cfg = self.device, self.cfg
        hx, hy = cfg.tray_inner_half
        h = CUBE / 2
        ar = torch.arange(n, device=dev)
        xy = torch.zeros(n, k, 2, device=dev)
        yaw = torch.zeros(n, k, device=dev)
        M = cfg.num_candidates

        # 1) 트레이 전체에 하나씩 흩뿌리기 (겹치지 않는 후보 중 무작위)
        for i in range(k):
            cand = torch.stack(
                [sample_uniform(-hx, hx, (n, M), dev), sample_uniform(-hy, hy, (n, M), dev)], dim=-1
            )
            cyaw = sample_uniform(-math.pi / 4, math.pi / 4, (n, M), dev)
            ok = _rect_inside_tray(cand, cyaw, (h, h), hx, hy)
            if i > 0:
                ok &= ~_overlap_matrix(cand, cyaw, (h + cfg.min_gap / 2,) * 2, xy[:, :i], yaw[:, :i]).any(-1)
            score = torch.rand(n, M, device=dev)
            score[~ok] = -1.0
            best = score.argmax(dim=1)
            xy[:, i], yaw[:, i] = cand[ar, best], cyaw[ar, best]

        # 2) 대상 고르기
        #    blocked   : 두 축 모두 막힘
        #    wall_only : 이웃이 없어도(벽만으로) 막힘 -> 밀어서 틈 만드는 문제가 아님
        far = torch.full_like(xy, 10.0)  # 이웃을 멀리 치운 상태
        blocked, wall_only = [], []
        for j in range(k):
            tj = torch.full((n,), j, device=dev)
            blocked.append(~self.grasp_axes_clear(xy, yaw, tj).any(-1))
            far_j = far.clone()
            far_j[:, j] = xy[:, j]
            wall_only.append(~self.grasp_axes_clear(far_j, yaw, tj).any(-1))
        blocked, wall_only = torch.stack(blocked, 1), torch.stack(wall_only, 1)  # (n, k)
        allow_wall = (torch.rand(n, device=dev) < cfg.wall_case_prob)[:, None]
        eligible = blocked & (~wall_only | allow_wall)  # 대상으로 쓸 수 있는 큐브
        fallback = ~wall_only  # 없으면: 벽에 안 막힌 큐브를 골라 이웃을 붙임
        score = torch.rand(n, k, device=dev) + 2.0 * eligible + 1.0 * fallback
        score += 1.0 * (eligible & wall_only)  # 벽 상황으로 뽑힌 환경은 벽 옆 큐브를 우선
        t = score.argmax(dim=1)

        # 3) 대상이 아직 잡히는 환경: 다른 큐브를 대상 옆으로 옮겨 붙이기
        for _ in range(cfg.max_cluster_moves):
            need = self.grasp_axes_clear(xy, yaw, t).any(-1)
            if not need.any():
                break
            j = (t + torch.randint(1, k, (n,), device=dev)) % k  # 대상이 아닌 큐브 하나
            r = cfg.near_radius * torch.sqrt(torch.rand(n, M, device=dev))
            th = torch.rand(n, M, device=dev) * 2 * math.pi
            cand = xy[ar, t][:, None] + torch.stack([r * torch.cos(th), r * torch.sin(th)], dim=-1)
            cyaw = sample_uniform(-math.pi / 4, math.pi / 4, (n, M), dev)
            hit = _overlap_matrix(cand, cyaw, (h + cfg.min_gap / 2,) * 2, xy, yaw)  # (n, M, k)
            hit[ar, :, j] = False  # 자기 자신은 제외
            ok = _rect_inside_tray(cand, cyaw, (h, h), hx, hy) & ~hit.any(-1)
            score = torch.linalg.norm(cand - xy[ar, t][:, None], dim=-1)
            score[~ok] = float("inf")
            best = score.argmin(dim=1)
            move = need & torch.isfinite(score[ar, best])
            xy[move, j[move]] = cand[ar, best][move]
            yaw[move, j[move]] = cyaw[ar, best][move]

        # 4) 대상을 0번 자리로 (0번 = 빨간 큐브)
        perm = torch.arange(k, device=dev).repeat(n, 1)
        perm[ar, 0], perm[ar, t] = t, 0
        xy = torch.gather(xy, 1, perm[..., None].expand(-1, -1, 2))
        yaw = torch.gather(yaw, 1, perm)
        return xy, yaw

    def grasp_axes_clear(self, xy: torch.Tensor, yaw: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """큐브 t 를 잡을 수 있는지 두 축(0°, 90°)에 대해 검사. 반환 (n, 2) bool.

        한 축이 열림 = 그 축 양쪽 손가락 영역이 비어 있고(다른 큐브·벽 없음) 하우징이 벽에 안 닿음.
        xy: (n, k, 2) 트레이 중심 기준, yaw: (n, k), t: (n,) 대상 인덱스.
        """
        cfg = self.cfg
        hx, hy = cfg.tray_inner_half
        h = CUBE / 2
        n, k = yaw.shape
        ar = torch.arange(n, device=xy.device)
        c, th = xy[ar, t], yaw[ar, t]
        not_self = torch.ones(n, k, dtype=torch.bool, device=xy.device)
        not_self[ar, t] = False
        region_half = (cfg.finger_clear_depth / 2, cfg.finger_clear_width / 2)

        result = []
        for a in (th, th + math.pi / 2):
            u = torch.stack([torch.cos(a), torch.sin(a)], dim=-1)
            clear = _rect_inside_tray(c, a, cfg.housing_half, hx, hy)
            for s in (1.0, -1.0):
                rc = c + s * u * (h + cfg.finger_clear_depth / 2)  # 손가락 영역 중심
                clear &= _rect_inside_tray(rc, a, region_half, hx, hy)
                hit = _overlap_matrix(rc[:, None], a[:, None], region_half, xy, yaw)[:, 0]  # (n, k)
                clear &= ~(hit & not_self).any(-1)
            result.append(clear)
        return torch.stack(result, dim=-1)


# ---------------------------------------------------------------------- 회전된 사각형 기하
def _extent(ang: torch.Tensor, half: tuple[float, float], axis: torch.Tensor) -> torch.Tensor:
    """각도 ang 인 사각형(반 크기 half)을 axis 방향에 투영한 반 길이."""
    d = ang - axis
    return half[0] * torch.abs(torch.cos(d)) + half[1] * torch.abs(torch.sin(d))


def _rect_inside_tray(c, ang, half, hx: float, hy: float) -> torch.Tensor:
    """사각형이 트레이 안쪽에 완전히 들어가면 True. c: (..., 2), ang: (...)."""
    zero = torch.zeros_like(ang)
    ex = _extent(ang, half, zero)
    ey = _extent(ang, half, zero + math.pi / 2)
    return (c[..., 0].abs() + ex <= hx - 0.0005) & (c[..., 1].abs() + ey <= hy - 0.0005)


def _overlap_matrix(c, ang, half, cubes_c, cubes_ang) -> torch.Tensor:
    """사각형 (n, M) 들과 큐브 (n, P) 들의 겹침 여부 (n, M, P). 분리축(SAT) 검사."""
    hc = (CUBE / 2, CUBE / 2)
    d = c[:, :, None, :] - cubes_c[:, None, :, :]
    a = ang[:, :, None].expand(-1, -1, cubes_c.shape[1])
    b = cubes_ang[:, None, :].expand_as(a)
    separated = torch.zeros_like(a, dtype=torch.bool)
    for ax in (a, a + math.pi / 2, b, b + math.pi / 2):
        proj = torch.abs(d[..., 0] * torch.cos(ax) + d[..., 1] * torch.sin(ax))
        separated |= proj > _extent(a, half, ax) + _extent(b, hc, ax)
    return ~separated
