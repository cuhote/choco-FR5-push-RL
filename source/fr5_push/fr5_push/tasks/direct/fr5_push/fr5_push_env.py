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
        self._j = {name: self.robot.find_joints(name)[0][0] for name in ("px", "py", "pz", "yaw")}
        self._fingers = self.robot.find_joints("finger_.*")[0]

        # 밀기 1회 계획 (환경마다)
        n = self.num_envs
        self._start = torch.zeros(n, 2, device=self.device)  # 시작점 (트레이 중심 기준)
        self._dir = torch.zeros(n, 2, device=self.device)  # 미는 방향 단위벡터
        self._dist = torch.zeros(n, device=self.device)  # 미는 거리
        self._yaw = torch.zeros(n, device=self.device)  # 그리퍼 yaw
        self._sub = 0  # 밀기 1회 안에서 몇 번째 물리 스텝인지
        # 결과 기록 (_get_dones 에서 채움. 리셋 전에 읽을 수 있게 따로 보관)
        self.push_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self.landed_on_cube = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.failed = torch.zeros(n, dtype=torch.bool, device=self.device)

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

    # ------------------------------------------------------------------ 행동: 밀기 1회
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """행동 [-1,1] x4 -> 밀기 계획. 그리퍼를 시작점 위로 순간 이동."""
        cfg = self.cfg
        a = actions.clamp(-1.0, 1.0)
        self.actions = a.clone()
        target_xy = self.cubes.data.object_pos_w[:, 0, :2] - self.scene.env_origins[:, :2]

        self._start = target_xy + a[:, 0:2] * cfg.start_range
        theta = a[:, 2] * math.pi
        self._dir = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        lo, hi = cfg.push_dist_range
        self._dist = lo + (a[:, 3] + 1.0) / 2.0 * (hi - lo)
        # 손가락 Y면(평평한 면)이 미는 방향을 보게: yaw = theta + 90°. 양쪽 Y면이 같으니 [-90°, 90°)로
        self._yaw = torch.remainder(theta + math.pi, math.pi) - math.pi / 2

        jp = self._joint_targets(self._start, self.cfg.safe_tip_z)
        self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        self.robot.set_joint_position_target(jp)
        self._sub = 0
        self.push_count += 1

    def _apply_action(self) -> None:
        """물리 스텝마다 호출: 시간표대로 관절 목표 위치를 준다."""
        cfg = self.cfg
        t = self._sub * self.physics_dt
        self._sub += 1
        t1 = cfg.t_descend
        t2 = t1 + cfg.t_push
        t3 = t2 + cfg.t_lift
        if t < t1:  # 내려가기
            tip = cfg.safe_tip_z + (cfg.push_tip_z - cfg.safe_tip_z) * t / t1
        elif t < t2:  # 밀기
            tip = cfg.push_tip_z
        elif t < t3:  # 들기
            tip = cfg.push_tip_z + (cfg.safe_tip_z - cfg.push_tip_z) * (t - t2) / cfg.t_lift
        else:  # 안정 대기
            tip = cfg.safe_tip_z
        s = ((t - t1) * cfg.push_speed) * torch.ones_like(self._dist)
        s = torch.minimum(s.clamp(min=0.0), self._dist)
        xy = self._start + self._dir * s[:, None]
        self.robot.set_joint_position_target(self._joint_targets(xy, tip))

        # 내려가기 끝 + 0.1초 (관절이 목표를 따라잡을 시간): 손가락 끝이 목표 높이에 못 갔으면 큐브 위에 내려앉은 것
        if self._sub == round((t1 + 0.1) / self.physics_dt):
            tip_now = self.robot.data.joint_pos[:, self._j["pz"]] - cfg.tcp_offset
            self.landed_on_cube = tip_now > cfg.push_tip_z + cfg.landed_tol

    def _joint_targets(self, xy: torch.Tensor, tip_z: float) -> torch.Tensor:
        jp = self._hold_pos.clone()
        jp[:, self._j["px"]] = xy[:, 0]
        jp[:, self._j["py"]] = xy[:, 1]
        jp[:, self._j["pz"]] = tip_z + self.cfg.tcp_offset
        jp[:, self._j["yaw"]] = self._yaw
        jp[:, self._fingers] = 0.0  # 손가락 닫기
        return jp

    # ------------------------------------------------------------------ 현재 큐브 상태 (2D)
    def cubes_2d(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """트레이 중심 기준 xy (n,k,2), yaw (n,k), 바로 놓였는지 (n,k), 높이 z (n,k)."""
        p = self.cubes.data.object_pos_w
        xy = p[..., :2] - self.scene.env_origins[:, None, :2]
        w, x, y, z = self.cubes.data.object_quat_w.unbind(-1)
        # 회전행렬의 열 = 큐브의 x, y, z 축 (월드 기준). 3번째 행 = 각 축의 z 성분
        ax = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], -1)
        ay = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], -1)
        az_z = 1 - 2 * (x * x + y * y)
        # 정육면체는 어느 면이 위로 가도 같은 모양 -> 세 축 중 하나라도 거의 수직이면 바로 놓인 것
        up = torch.stack([ax[..., 2].abs(), ay[..., 2].abs(), az_z.abs()], -1).max(-1).values
        upright = up > math.cos(math.radians(self.cfg.tilt_fail_deg))
        # yaw: 수평에 가까운 축 하나로 계산
        h = torch.where((ax[..., 2].abs() < 0.7)[..., None], ax, ay)
        yaw = torch.atan2(h[..., 1], h[..., 0])
        return xy, yaw, upright, p[..., 2]

    # ------------------------------------------------------------------ D단계: 규칙 기반 밀기
    def demo_push_action(self, push_dist: float = 0.035) -> torch.Tensor:
        """규칙: 대상에 가장 가까운 이웃을, 대상-이웃 선과 직각 방향으로 밀어 대상 옆에서 비켜나게 한다.

        그리퍼는 이웃의 옆면(틈이 아닌 쪽) 바깥 빈 바닥에 내린다. 양쪽 중 내릴 자리가 비어 있는 쪽을 고른다.
        반환: 행동 (n, 4) [-1, 1]
        """
        cfg = self.cfg
        h = CUBE / 2
        foot = (0.0105, 0.0105)  # 닫힌 손가락이 차지하는 바닥 면적 (반 크기)
        xy, yaw, _, _ = self.cubes_2d()
        n = xy.shape[0]
        ar = torch.arange(n, device=self.device)
        c = xy[:, 0]
        d = xy[:, 1:] - c[:, None]
        j = torch.linalg.norm(d, dim=-1).argmin(dim=1)  # 가장 가까운 이웃
        pj, yj = xy[:, 1:][ar, j], yaw[:, 1:][ar, j]
        v = d[ar, j] / torch.linalg.norm(d[ar, j], dim=-1, keepdim=True)  # 대상 -> 이웃

        lo, hi = cfg.push_dist_range
        action, ok = None, None
        for sgn in (1.0, -1.0):
            w = sgn * torch.stack([-v[:, 1], v[:, 0]], dim=-1)  # 직각 방향 = 미는 방향
            th = torch.atan2(w[:, 1], w[:, 0])
            start = pj - w * (_extent(yj, (h, h), th)[:, None] + foot[0] + 0.003)  # 이웃 옆면 바깥 3mm
            free = _rect_inside_tray(start, th, foot, *cfg.tray_inner_half)
            free &= ~_overlap_matrix(start[:, None], th[:, None], foot, xy, yaw)[:, 0].any(-1)
            a = torch.zeros(n, 4, device=self.device)
            a[:, 0:2] = (start - c) / cfg.start_range
            a[:, 2] = th / math.pi
            a[:, 3] = (push_dist - lo) / (hi - lo) * 2 - 1
            if action is None:
                action, ok = a, free
            else:
                action = torch.where((~ok & free)[:, None], a, action)
        return action.clamp(-1.0, 1.0)

    # ------------------------------------------------------------------ 관측/보상/종료 (E단계에서 완성)
    def _get_observations(self) -> dict:
        target_pos = self.cubes.data.object_pos_w[:, 0] - self.scene.env_origins
        return {"policy": target_pos}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        xy, yaw, upright, z = self.cubes_2d()
        t0 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success = self.grasp_axes_clear(xy, yaw, t0).any(-1) & upright[:, 0] & (z[:, 0] < self.cfg.stacked_z)
        self.failed = ~upright[:, 0] | (z > self.cfg.stacked_z).any(-1)
        self.success &= ~self.failed
        time_out = self.push_count >= self.cfg.max_pushes  # 최대 횟수만큼 밀었으면 끝
        return self.success | self.failed, time_out

    # ------------------------------------------------------------------ 리셋: 큐브 무작위 배치
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)
        k = self.cubes.num_objects  # 18

        self.push_count[env_ids] = 0

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
