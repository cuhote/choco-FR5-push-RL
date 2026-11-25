from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

from .fr5_push_env_cfg import CUBE, TRAY_CORNER, TRAY_PARTS, Fr5PushEnvCfg


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
        self._skip = torch.zeros(n, dtype=torch.bool, device=self.device)  # 안전한 후보가 없어 내려가지 않은 밀기
        self.chosen = torch.zeros(n, dtype=torch.long, device=self.device)
        self.chosen_opens = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.axis_open = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.grasp_tried = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.grasp_ok = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.fail_tilt = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.fail_stack = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.failed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.fail_blocked = torch.zeros(n, dtype=torch.bool, device=self.device)  # 시작점 막힘 (안전한 후보 없음)
        self.fail_corner = torch.zeros(n, dtype=torch.bool, device=self.device)  # 모서리에 낀 뒤 정해진 횟수 안에 못 듦
        self.in_corner = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.corner_count = torch.zeros(n, dtype=torch.long, device=self.device)  # 대상이 모서리에 낀 채로 끝난 밀기 수 (연속)
        self._deficit_before = torch.zeros(n, device=self.device)  # 밀기 전 모자란 깊이 (틈 보상용)
        self._deficit_after = torch.zeros(n, device=self.device)

    # ------------------------------------------------------------------ 장면
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cubes = RigidObjectCollection(self.cfg.cubes_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -0.005))

        # 트레이 (움직이지 않는 충돌체 5개)
        for name, (size, pos, yaw_deg) in TRAY_PARTS.items():
            part = sim_utils.CuboidCfg(
                size=size,
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.4),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.6)),
            )
            half = math.radians(yaw_deg) / 2
            part.func(f"/World/envs/env_0/Tray/{name}", part, translation=pos,
                      orientation=(math.cos(half), 0.0, 0.0, math.sin(half)))

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_object_collections["cubes"] = self.cubes

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ 행동: 후보 중 하나를 골라 밀기 1회
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """행동 = 후보 NUM_CAND 개 각각의 점수. 안전한 후보(valid) 중 점수가 가장 높은 것을 실행.

        안전한 후보가 하나도 없으면 내려가지 않는다 (실물 소프트웨어도 실행하지 않음). 밀기 1회는 소모.
        """
        self.actions = actions.clone()
        cand = self._cand
        if self.cfg.residual_rule:  # 규칙 + 정책 보정
            actions = actions + self.cfg.rule_weight * self.demo_push_action(lookahead=True)
        n = self.num_envs
        ar = torch.arange(n, device=self.device)
        scores = actions.masked_fill(~cand["valid"], float("-inf"))
        idx = scores.argmax(dim=1)
        self._skip = ~cand["valid"].any(dim=1)
        self.chosen = idx
        self.chosen_opens = cand["opens"][ar, idx] & ~self._skip  # 고른 후보가 "열림 예측"이었는지 (분석용)

        self._start = cand["start"][ar, idx]
        theta = cand["theta"][ar, idx]
        self._dist = torch.where(self._skip, torch.zeros_like(theta), cand["dist"][ar, idx])
        self._dir = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        # 손가락 Y면(평평한 면)이 미는 방향을 보게: yaw = theta + 90°. 양쪽 Y면이 같으니 [-90°, 90°)로
        self._yaw = torch.remainder(theta + math.pi, math.pi) - math.pi / 2

        xy_t, yaw_t, _, _ = self.cubes_2d()  # 틈 보상: 밀기 전 참값
        self._deficit_before = self.open_deficit(xy_t, yaw_t)
        self.set_finger_mode("push")
        jp = self._joint_targets(self._start, self.cfg.safe_tip_z)
        self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        self.robot.set_joint_position_target(jp)
        self._sub = 0
        self.push_count += 1

    def set_finger_mode(self, mode: str) -> None:
        """손가락 PD. push: 닫은 채 단단히 (밀 때 큐브가 손가락 사이로 끼어들지 않게)
        grasp: 약하게 5N (강하면 시뮬레이션에서 손가락이 15g 큐브를 뚫고 지나감)."""
        k, d, f = self.cfg.finger_gains[mode]
        ids = self._fingers
        n = self.num_envs
        self.robot.write_joint_stiffness_to_sim(torch.full((n, len(ids)), k, device=self.device), joint_ids=ids)
        self.robot.write_joint_damping_to_sim(torch.full((n, len(ids)), d, device=self.device), joint_ids=ids)
        self.robot.write_joint_effort_limit_to_sim(torch.full((n, len(ids)), f, device=self.device), joint_ids=ids)

    def _apply_action(self) -> None:
        """물리 스텝마다 호출: 시간표대로 관절 목표 위치를 준다."""
        cfg = self.cfg
        t = self._sub * self.physics_dt
        self._sub += 1
        t1 = cfg.t_descend
        t2 = t1 + cfg.t_push
        t3 = t2 + cfg.t_lift
        tb = t2 + cfg.t_backoff  # 물러나기 끝
        if t < t1:  # 내려가기
            tip = cfg.safe_tip_z + (cfg.push_tip_z - cfg.safe_tip_z) * t / t1
        elif t < tb:  # 밀기, 물러나기 (밀기 높이 유지)
            tip = cfg.push_tip_z
        elif t < t3:  # 들기
            tip = cfg.push_tip_z + (cfg.safe_tip_z - cfg.push_tip_z) * (t - tb) / (t3 - tb)
        else:  # 안정 대기
            tip = cfg.safe_tip_z
        s = ((t - t1) * cfg.push_speed) * torch.ones_like(self._dist)
        s = torch.minimum(s.clamp(min=0.0), self._dist)
        # 밀기 끝: 미는 반대 방향으로 backoff 만큼 물러난 뒤 들기 (손가락이 큐브에 붙은 채 올라가면 마찰로 큐브를 끌어올림)
        if t >= t2:
            back = min((t - t2) / cfg.t_backoff, 1.0) * cfg.push_backoff
            s = (s - back).clamp(min=0.0)
        xy = self._start + self._dir * s[:, None]
        tip_z = torch.where(self._skip, cfg.safe_tip_z, tip)  # 안전한 후보가 없던 환경은 내려가지 않음
        self.robot.set_joint_position_target(self._joint_targets(xy, tip_z))
        if cfg.lock_fingers_push:  # 손가락을 닫힘 위치에 고정 (실물 그리퍼는 닫힌 채 위치를 유지)
            zf = torch.zeros(self.num_envs, len(self._fingers), device=self.device)
            self.robot.write_joint_state_to_sim(zf, zf, joint_ids=self._fingers)

        # 내려가기 끝 + 0.1초 (관절이 목표를 따라잡을 시간): 손가락 끝이 목표 높이에 못 갔으면 큐브 위에 내려앉은 것
        if self._sub == round((t1 + 0.1) / self.physics_dt):
            tip_now = self.robot.data.joint_pos[:, self._j["pz"]] - cfg.tcp_offset
            self.landed_on_cube = (tip_now > cfg.push_tip_z + cfg.landed_tol) & ~self._skip

    # ------------------------------------------------------------------ 밀기 후보 (실물 소프트웨어에도 같은 로직)
    def build_candidates(self, xy: torch.Tensor, yaw: torch.Tensor) -> dict:
        """안전한 밀기 후보 만들기 (NUM_CAND 개 슬롯, 순서 고정).

        (1) 이웃 밀기 = 가까운 이웃(cand_neighbours) x 양쪽 옆면 x (옆으로, 45° 바깥으로) x 거리(cand_dists)
            이웃을 대상 옆에서 비켜나게 한다.
        (2) 대상 밀기 = 대상의 네 면 방향 x 거리. 대상을 무리 밖으로 옮긴다.
        그리퍼는 밀 큐브의 반대쪽, 옆면에서 start_clearance 떨어진 빈 바닥에 내린다.
          valid : 이웃이 obs_neighbour_radius 안 + 착지 자리·하우징이 안전 여유(safety_margin)까지 비어 있음
          opens : 민 큐브를 민 만큼 옮긴 배치에서 대상이 열리는지 (기하 예측. 실제 물리와 다를 수 있음)
          chain : 옮긴 큐브가 다른 큐브·벽과 겹침 = 줄줄이 밀림 (예측이 덜 정확)
          xy_pred, yaw_pred: 밀고 난 뒤 예상 배치 (n, k, 2), (n, k) - 앞을 내다보는 탐색용
          feat  : 후보마다 10개 (대상 좌표계) = 시작점 x',y' /0.1, 미는 방향 sin/cos, 거리 /0.04,
                  밀 큐브 위치 x',y' /0.1, opens, chain, valid.  valid=0 이면 전부 0
        """
        cfg = self.cfg
        h = CUBE / 2
        n, k = yaw.shape
        ar = torch.arange(n, device=self.device)
        zero = torch.zeros(n, dtype=torch.long, device=self.device)
        c, th_t = xy[:, 0], _wrap90(yaw[:, 0])
        dn = torch.linalg.norm(xy[:, 1:] - c[:, None], dim=-1)
        order = dn.argsort(dim=1)[:, : cfg.cand_neighbours]

        out = {key: [] for key in ("start", "theta", "dist", "valid", "opens", "feat", "xy_pred", "yaw_pred")}
        hx, hy = cfg.tray_inner_half

        def add_one(j, pj, yj, u, ok, dist, new_p, new_yaw, strict=False):
            """큐브 j 를 u 방향으로 dist (n,) 만큼 미는 후보 1개. 결과 예상: j 가 new_p, new_yaw 로 감.
            strict: 옮긴 큐브가 다른 큐브와 겹치면 아예 무효 (벽에 밀어 넣는 후보: 꽉 끼면 그리퍼가 누름)."""
            th = torch.atan2(u[:, 1], u[:, 0])
            gap = _extent(yj, (h, h), th) + cfg.finger_foot_half[0] + cfg.start_clearance
            start = pj - u * gap[:, None]  # 밀 큐브의 반대쪽 빈 바닥
            not_j = torch.ones(n, k, dtype=torch.bool, device=self.device)
            not_j[ar, j] = False
            ok = ok & self.landing_free(start, start + u * dist[:, None], th, xy, yaw, cfg.safety_margin)
            xy_pred, yaw_pred = xy.clone(), yaw.clone()
            xy_pred[ar, j], yaw_pred[ar, j] = new_p, new_yaw
            # 밀린 큐브가 다른 큐브·벽과 부딪힘(줄줄이 밀림) -> 예측이 덜 정확함. 정책이 보도록 특징으로 넣음
            hit = (_overlap_matrix(new_p[:, None], new_yaw[:, None], (h, h), xy_pred, yaw_pred)[:, 0] & not_j).any(-1)
            if strict:
                ok = ok & ~hit
            chain = hit | ~_rect_inside_tray(new_p, new_yaw, (h, h), hx, hy)
            opens = self.grasp_axes_clear(xy_pred, yaw_pred, zero).any(-1)
            rel = th - th_t
            feat = torch.cat(
                [
                    _rotate(start - c, -th_t) / 0.1,
                    torch.sin(rel)[:, None],
                    torch.cos(rel)[:, None],
                    (dist / 0.04)[:, None],
                    _rotate(pj - c, -th_t) / 0.1,
                    opens.float()[:, None],
                    chain.float()[:, None],
                    ok.float()[:, None],
                ],
                dim=-1,
            ) * ok.float()[:, None]
            vals = (("start", start), ("theta", th), ("dist", dist), ("valid", ok), ("opens", opens),
                    ("feat", feat), ("xy_pred", xy_pred), ("yaw_pred", yaw_pred))
            for key, val in vals:
                out[key].append(val)

        def wall_room(pj, yj, u):
            """큐브 j 가 u 방향으로 벽(어느 모서리든)에 닿기까지 중심이 갈 수 있는 거리 (n,)."""
            room = torch.full((n,), 1.0, device=self.device)
            for ang, W in ((0.0, hx), (math.pi, hx), (math.pi / 2, hy), (-math.pi / 2, hy)):
                nx, ny = math.cos(ang), math.sin(ang)
                un = u[:, 0] * nx + u[:, 1] * ny
                gap = W - (pj[:, 0] * nx + pj[:, 1] * ny) - _extent(yj, (h, h), torch.full_like(yj, ang))
                room = torch.where(un > 1e-6, torch.minimum(room, gap / un.clamp(min=1e-6)), room)
            return room.clamp(min=0.0)

        def add(j, pj, yj, u, near_ok):
            """큐브 j 를 u 방향으로 미는 후보 (거리별로 추가). 결과 예상 = 회전 없이 밀린 만큼 이동.
            벽 앞 멈춤: 민 큐브가 벽 wall_push_gap 앞에 오면 그리퍼를 멈춤 (벽에 눌러 끼우면 들 때 딸려 올라감,
            실물에서는 큐브를 벽에 세게 누름). 줄인 거리로 큐브를 3mm 도 못 밀면 후보에서 뺌."""
            max_d = wall_room(pj, yj, u) - cfg.wall_push_gap + cfg.start_clearance
            for dist in cfg.cand_dists:  # 큐브가 밀리는 거리 -> 손가락 이동 = + start_clearance
                d = torch.clamp(torch.full((n,), dist + cfg.start_clearance, device=self.device), max=max_d)
                ok = near_ok & (d >= cfg.start_clearance + 0.003)
                add_one(j, pj, yj, u, ok, d, pj + u * (d - cfg.start_clearance)[:, None], yj)

        # (1) 이웃 밀기: 가까운 이웃 순서대로, 양쪽 옆면 x 방향 (0° 옆으로 비키기 ~ 90° 대상 반대쪽으로)
        for rank in range(cfg.cand_neighbours):
            j = order[:, rank] + 1  # 큐브 인덱스 (0 = 대상)
            pj, yj = xy[ar, j], yaw[ar, j]
            near_ok = dn[ar, j - 1] < cfg.obs_neighbour_radius
            v = (pj - c) / dn[ar, j - 1, None].clamp(min=1e-6)  # 대상 -> 이웃
            for sgn in (1.0, -1.0):
                w = sgn * torch.stack([-v[:, 1], v[:, 0]], dim=-1)  # 직각
                for a_deg in cfg.cand_angles:  # 0° = 옆으로 비키기, 90° = 대상 반대쪽으로 곧게
                    a = math.radians(a_deg)
                    add(j, pj, yj, math.cos(a) * w + math.sin(a) * v, near_ok)
        # (2) 대상 밀기: 대상의 네 면 방향
        ones = torch.ones(n, dtype=torch.bool, device=self.device)
        if cfg.cand_target:
            for q in range(4):
                ang = th_t + q * math.pi / 2
                u = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
                add(zero, c, yaw[:, 0], u, ones)
        # (3) 벽 근처 대상: 벽마다 (가까울 때만 유효)
        #   벽 맞추기: 벽 쪽으로 밀어 벽에 붙이며 벽과 나란히 세움 (회전 예측). 벽에서 2mm 앞에서 멈춤
        #   벽 따라 밀기: 벽과 나란히 양쪽으로 cand_dists 만큼
        if cfg.cand_wall:
            yt = yaw[:, 0]
            for wall_ang, W in ((0.0, hx), (math.pi, hx), (math.pi / 2, hy), (-math.pi / 2, hy)):
                nw = torch.tensor([math.cos(wall_ang), math.sin(wall_ang)], device=self.device).expand(n, 2)
                cn = (c * nw).sum(-1)  # 벽 방향 좌표
                ext = _extent(yt, (h, h), torch.full_like(yt, wall_ang))
                gap_w = W - cn - ext  # 대상 앞 끝 ~ 벽
                near_w = gap_w < cfg.finger_clear_depth + cfg.safety_margin
                # 벽 맞추기: 대상을 벽 쪽으로 밀어 벽에 붙이며 벽과 나란히 세움 (회전 예측). 벽에서 stop 앞에서 멈춤
                #   wall_square_mode "normal": 벽과 직각으로 민다 (손가락 바닥이 벽과 나란 -> 붙은 큐브 뒤 면까지 감)
                #   "face": 벽에 가까운 면의 법선으로 민다 (기운 손가락 모서리가 먼저 멈춰 큐브가 벽까지 못 감)
                d = _wrap90(yt - wall_ang)  # 큐브가 벽과 어긋난 각 [-45°, 45°)
                stop = cfg.wall_stop_gap
                foot = cfg.finger_foot_half[0]
                tw = torch.stack([-nw[:, 1], nw[:, 0]], dim=-1)  # 벽 따라 방향
                travel_n = W - h - stop / 2 - cn
                if cfg.wall_square_mode == "normal":
                    u = nw
                    dist = W - cn - 2 * h - stop + ext + cfg.start_clearance
                    new_p = c + nw * travel_n[:, None]
                else:
                    cd, sd = torch.cos(d), torch.sin(d)
                    u = nw * cd[:, None] + tw * sd[:, None]
                    start_n = cn - (h + foot + cfg.start_clearance) * cd
                    dist = (W - 2 * h - stop - foot * (cd.abs() + sd.abs()) - start_n) / cd
                    new_p = c + nw * travel_n[:, None] + tw * (travel_n * torch.tan(d))[:, None]
                ok = near_w & (dist > cfg.start_clearance) & (dist <= cfg.t_push * cfg.push_speed)
                ok &= cfg.cand_wall_square
                add_one(zero, c, yt, u, ok, dist.clamp(min=0.0), new_p, yt - d, strict=True)
                # 벽 따라 밀기
                for sgn in (1.0, -1.0):
                    t = sgn * torch.stack([-nw[:, 1], nw[:, 0]], dim=-1)
                    max_d = wall_room(c, yt, t) - cfg.wall_push_gap + cfg.start_clearance
                    for dd in cfg.cand_dists:  # 큐브 기준 거리
                        d = torch.clamp(torch.full((n,), dd + cfg.start_clearance, device=self.device), max=max_d)
                        ok = near_w & (d >= cfg.start_clearance + 0.003)
                        add_one(zero, c, yt, t, ok, d, c + t * (d - cfg.start_clearance)[:, None], yt)
        return {key: torch.stack(val, dim=1) for key, val in out.items()}

    def landing_free(self, start, end, theta, xy, yaw, margin: float = 0.0) -> torch.Tensor:
        """그리퍼를 start 에 내려 end 까지 밀 수 있는지. 반환 (n,) bool. (계획용: 규칙·관측에서 사용)

        - 시작점: 닫힌 손가락 자리(바닥 면적)에 큐브가 없어야 함
        - 시작점·끝점: 손가락·하우징·카메라(브래킷 포함)가 트레이 벽 안에 있어야 함
          (밀 때 하우징 아래 55mm, 카메라 아래 약 72mm < 벽 80mm 라서 벽에 걸릴 수 있음)
        theta: 미는 방향. 그리퍼 X축(손가락 벌림)이 미는 방향과 직각이므로
               반 크기를 (미는 방향 = 그리퍼 Y, 직각 = 그리퍼 X) 순서로 넣는다.
        """
        cfg = self.cfg
        hx, hy = cfg.tray_inner_half
        foot = (cfg.finger_foot_half[0] + margin, cfg.finger_foot_half[1] + margin)
        housing = (cfg.housing_half[1] + margin, cfg.housing_half[0] + margin)
        camera = (cfg.camera_half[1] + margin, cfg.camera_half[0] + margin)
        cam_low = cfg.push_tip_z + cfg.camera_bottom_above_tip < cfg.tray_wall_height  # 카메라가 벽 높이보다 낮게 내려가나
        housing_low = cfg.push_tip_z + cfg.housing_bottom_above_tip < cfg.tray_wall_height
        ok = ~_overlap_matrix(start[:, None], theta[:, None], foot, xy, yaw)[:, 0].any(-1)
        for p in (start, end):
            ok &= _rect_inside_tray(p, theta, foot, hx, hy)
            if housing_low:
                ok &= _rect_inside_tray(p, theta, housing, hx, hy)
            if cam_low:
                ok &= _rect_inside_tray(p, theta, camera, hx, hy)
        return ok

    def _joint_targets(self, xy: torch.Tensor, tip_z) -> torch.Tensor:
        jp = self._hold_pos.clone()
        jp[:, self._j["px"]] = xy[:, 0]
        jp[:, self._j["py"]] = xy[:, 1]
        jp[:, self._j["pz"]] = tip_z + self.cfg.tcp_offset  # tip_z: 숫자 또는 환경별 텐서
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

    def cubes_observed(self) -> tuple[torch.Tensor, torch.Tensor]:
        """카메라 측정을 흉내: 참 위치·yaw 에 노이즈 (obs_pos_noise_std, obs_yaw_noise_deg). 매번 새로 뽑음.

        obs_noise_dist "uniform": 각 값에 [-크기, +크기] 균등 / "normal": 크기를 σ 로 하는 가우시안.
        """
        cfg = self.cfg
        xy, yaw, _, _ = self.cubes_2d()
        uniform = getattr(cfg, "obs_noise_dist", "normal") == "uniform"

        def noise(t):
            return (torch.rand_like(t) * 2 - 1) if uniform else torch.randn_like(t)

        if cfg.obs_pos_noise_std > 0:
            xy = xy + noise(xy) * cfg.obs_pos_noise_std
        if cfg.obs_yaw_noise_deg > 0:
            yaw = yaw + noise(yaw) * math.radians(cfg.obs_yaw_noise_deg)
        return xy, yaw

    # ------------------------------------------------------------------ 잡기 (검증용: 실제로 집어서 들어 보기)
    def grasp_attempt(
        self, center: torch.Tensor, axis_angle: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """center (n,2) 에서 axis_angle 방향으로 손가락을 벌려 잡고 들어 본다. env.step 밖에서 물리를 직접 돌림.

        순서: 손가락 벌림(grasp_open_each) -> 잡기 높이로 내려감 -> 닫음 -> 위치값 검사 -> 60mm 들기
        mask: 잡기를 실행할 환경 (None = 전부). 나머지 환경의 그리퍼는 지금 자세로 기다림.
        반환: (위치값이 성공 범위 56~60% 인지, 큐브가 30mm 이상 들렸는지) 각 (n,) bool. 대상 = 0번 큐브.
        """
        cfg = self.cfg
        dt = self.physics_dt
        n = self.num_envs
        m = torch.ones(n, dtype=torch.bool, device=self.device) if mask is None else mask
        hold = self.robot.data.joint_pos.clone()
        self.set_finger_mode("grasp")
        yaw_g = torch.remainder(axis_angle + math.pi / 2, math.pi) - math.pi / 2  # 그리퍼 X(벌림) = 잡기 축
        self._yaw = torch.where(m, yaw_g, self._yaw)
        open_w = cfg.grasp_open_each

        def targets(tip, finger):
            jp = self._joint_targets(center, tip)
            jp[:, self._fingers] = finger
            return torch.where(m[:, None], jp, hold)

        jp = targets(cfg.safe_tip_z, open_w)
        ids = m.nonzero().flatten()
        self.robot.write_joint_state_to_sim(jp[ids], torch.zeros_like(jp[ids]), env_ids=ids)

        def run(seconds, fn, label=""):
            steps = max(1, round(seconds / dt))
            for i in range(steps):
                tip, finger = fn((i + 1) / steps)
                self.robot.set_joint_position_target(targets(tip, finger))
                self.scene.write_data_to_sim()
                self.sim.step(render=self.sim.has_gui())  # 화면을 켰을 때만 잡기 동작도 그림 (학습은 headless라 영향 없음)
                self.scene.update(dt)
            if getattr(self, "debug_grasp", False):
                jp = self.robot.data.joint_pos[0]
                cz = self.cubes.data.object_pos_w[0, 0] - self.scene.env_origins[0]
                print(f"[grasp] {label:8s} tip {(jp[self._j['pz']] - cfg.tcp_offset) * 1000:6.1f}mm "
                      f"finger {jp[self._fingers[0]] * 1000:5.1f}/{jp[self._fingers[1]] * 1000:5.1f}mm "
                      f"cube ({cz[0] * 1000:6.1f},{cz[1] * 1000:6.1f},{cz[2] * 1000:5.1f})")

        top, g = cfg.safe_tip_z, cfg.grasp_tip_z
        run(0.2, lambda f: (top, open_w), "open")
        run(0.4, lambda f: (top + (g - top) * f, open_w), "descend")  # 내려가기
        run(0.2, lambda f: (g, open_w), "wait")
        run(0.5, lambda f: (g, open_w * (1 - f)), "close")  # 닫기
        run(0.2, lambda f: (g, 0.0), "hold")
        fpos = self.robot.data.joint_pos[:, self._fingers]
        lo, hi = cfg.grasp_ok_range
        held = ((fpos >= lo) & (fpos <= hi)).all(-1)
        run(0.5, lambda f: (g + 0.06 * f, 0.0), "lift")  # 들기
        run(0.3, lambda f: (g + 0.06, 0.0), "lifted")
        lifted = self.cubes.data.object_pos_w[:, 0, 2] > CUBE / 2 + 0.03
        self.last_finger_pos = fpos.clone()
        return held & m, lifted & m

    # ------------------------------------------------------------------ 규칙 (비교 기준선)
    def demo_push_action(self, lookahead: bool = False, topk: int = 6) -> torch.Tensor:
        """규칙 (비교 기준선). 반환: 후보 점수 (n, NUM_CAND). 안전하지 않은 후보는 env 가 알아서 뺌.

        기본  : 바로 "열림 예측"인 후보 (줄줄이 밀림 없는 것 우선, 가까운 이웃 먼저)
        lookahead=True : 바로 열리는 후보가 없을 때
            진전(open_deficit: 대상이 열리기까지 모자란 빈 깊이)을 가장 줄이는 후보 topk 개를 골라
            그다음 수에 "열림 예측" 후보가 생기는지 미리 본다 -> 생기는 것 우선, 그다음 진전 큰 것
        """
        cand = self._cand
        valid, opens = cand["valid"], cand["opens"]
        n, num = valid.shape
        order_bonus = -0.001 * torch.arange(num, device=self.device)  # 앞 슬롯 = 가까운 이웃, 긴 거리
        chain = cand["feat"][..., 8] > 0.5
        score = 2.0 * opens.float() + 0.5 * (opens & ~chain).float() + order_bonus
        if not lookahead:
            return score
        stuck = ~(valid & opens).any(-1)
        if not stuck.any():
            return score
        if "deficit" not in cand:  # 관측에서 이미 계산했으면 재사용
            cand["deficit"], cand["next_open"] = self.lookahead_info(cand, self._obs_yaw, topk)
        # 다음 수 열림(+4)이 우선, 그다음 모자란 깊이 1mm 당 -0.1 (최대 36mm -> -3.6 < 4 라 순서는 이전과 같음).
        # 이전 (1, -10/m) 은 10mm 차이가 0.1점이라 정책 노이즈(0.3~0.5)에 묻혔음 (11/25)
        score2 = 4.0 * cand["next_open"].float() - 100.0 * cand["deficit"] + order_bonus
        return torch.where(stuck[:, None], score2, score)

    def lookahead_info(self, cand: dict, yaw: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
        """후보마다 (밀고 난 뒤 열리기까지 모자란 깊이 (n,C) m, 다음 수에 "열림 예측" 후보가 생기나 (n,C) bool).

        다음 수 계산은 무거워서, 모자란 깊이가 가장 작은 안전한 후보 topk 개만 펼친다 (나머지는 False).
        """
        valid = cand["valid"]
        n, num = valid.shape
        ar = torch.arange(n, device=self.device)
        deficit = torch.stack(
            [self.open_deficit(cand["xy_pred"][:, i], cand["yaw_pred"][:, i]) for i in range(num)], dim=1
        )
        deficit = torch.where(valid, deficit, torch.full_like(deficit, self.cfg.finger_clear_depth * 2))
        top = deficit.topk(min(topk, num), dim=1, largest=False).indices
        next_open = torch.zeros(n, num, dtype=torch.bool, device=self.device)
        for r in range(top.shape[1]):
            i = top[:, r]
            c2 = self.build_candidates(cand["xy_pred"][ar, i], cand["yaw_pred"][ar, i])
            next_open[ar, i] = (c2["valid"] & c2["opens"]).any(-1) & valid[ar, i]
        return deficit, next_open

    def open_deficit(self, xy: torch.Tensor, yaw: torch.Tensor, steps: int = 8) -> torch.Tensor:
        """대상이 열리기까지 모자란 빈 깊이 (n,) m. 두 축 중 나은 쪽의 (양쪽 면 부족분 합). 0 = 열림."""
        cfg = self.cfg
        need = cfg.finger_clear_depth
        zero = torch.zeros(yaw.shape[0], dtype=torch.long, device=self.device)
        frees = []
        for d in torch.linspace(need / steps, need, steps).tolist():
            cfg.finger_clear_depth = d
            frees.append(self.sides_clear(xy, yaw, zero))
        cfg.finger_clear_depth = need
        free = torch.stack(frees, -1).float().cumprod(-1).sum(-1) / steps * need  # (n, 4) 면별 빈 깊이
        lack = (need - free).clamp(min=0)
        return torch.minimum(lack[:, 0] + lack[:, 1], lack[:, 2] + lack[:, 3])

    # ------------------------------------------------------------------ 관측/보상/종료
    def _get_observations(self) -> dict:
        """대상 큐브 좌표계로 표현한 관측 (51 + 후보 NUM_CAND x 12; 지금 148개 -> 1827).

        [0:2]   대상 yaw (트레이 기준) sin/cos(4θ)  - 정사각형이라 90° 주기
        [2:6]   대상 중심에서 +x', -x', +y', -y' 방향으로 벽까지 거리 / 0.1m
        [6:10]  손가락 영역 4곳(+x', -x', +y', -y')이 비었는지 (1 = 빔)
        [10:50] 가까운 이웃 8개 x (상대 x', y' / 0.1m, sin/cos(4Δθ), 있음 여부)
        [50]    민 횟수 / 최대 횟수
        [51:]   밀기 후보 x (특징 10 + 탐색 정보 2) (build_candidates, lookahead_info 참고). 행동 i = 후보 i 의 점수
        """
        cfg = self.cfg
        xy, yaw = self.cubes_observed()  # 카메라로 본 위치 (측정 노이즈 포함). 관측·후보·안전 검사 모두 이 값 사용
        n, k = yaw.shape
        c, th = xy[:, 0], _wrap90(yaw[:, 0])
        hx, hy = cfg.tray_inner_half

        target_yaw = torch.stack([torch.sin(4 * th), torch.cos(4 * th)], dim=-1)

        walls = []
        for ang in (th, th + math.pi, th + math.pi / 2, th - math.pi / 2):
            walls.append(_ray_to_tray(c, ang, hx, hy))
        walls = (torch.stack(walls, dim=-1) / 0.1).clamp(max=2.0)

        sides = self.sides_clear(xy, yaw, torch.zeros(n, dtype=torch.long, device=self.device)).float()

        rel = _rotate(xy[:, 1:] - c[:, None], -th[:, None])  # (n, 17, 2) 대상 좌표계
        dist = torch.linalg.norm(rel, dim=-1)
        near = dist.topk(cfg.obs_neighbours, dim=1, largest=False).indices  # (n, 8)
        rel_n = torch.gather(rel, 1, near[..., None].expand(-1, -1, 2))
        dyaw = torch.gather(yaw[:, 1:], 1, near) - th[:, None]
        valid = (torch.gather(dist, 1, near) < cfg.obs_neighbour_radius).float()
        nb = torch.cat(
            [rel_n / 0.1, torch.sin(4 * dyaw)[..., None], torch.cos(4 * dyaw)[..., None], valid[..., None]], dim=-1
        ) * valid[..., None]

        count = (self.push_count.float() / cfg.max_pushes)[:, None]
        self._obs_xy, self._obs_yaw = xy, yaw
        self._cand = self.build_candidates(xy, yaw)  # 다음 행동은 이 후보 중에서 고름
        feat = self._cand["feat"]
        if cfg.obs_lookahead:
            # 탐색 규칙이 쓰는 정보를 RL 에도: 후보마다 (모자란 깊이, 다음 수에 열리는 후보가 생기나)
            self._cand["deficit"], self._cand["next_open"] = self.lookahead_info(self._cand, yaw, cfg.lookahead_topk)
            v = self._cand["valid"].float()[..., None]
            extra = torch.stack(
                [self._cand["deficit"] / cfg.finger_clear_depth, self._cand["next_open"].float()], dim=-1
            )
            feat = torch.cat([feat, extra * v], dim=-1)
        obs = torch.cat([target_yaw, walls, sides, nb.reshape(n, -1), count, feat.reshape(n, -1)], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """성공 +10, 밀기 1회 -1, 1cm 당 -0.1, 실패(넘어짐·쌓임·큐브 위 착지) -5."""
        cfg = self.cfg
        r = cfg.rew_success * self.success.float()
        r += cfg.rew_push
        r += cfg.rew_per_cm * (self._dist * 100.0)
        r += cfg.rew_fail * self.failed.float()
        # 틈 보상: 모자란 깊이가 줄어든 mm 당 +0.17 (늘면 감점). 실패한 판은 주지 않음
        gain_mm = (self._deficit_before - self._deficit_after) * 1000.0
        r += cfg.rew_gap_per_mm * gain_mm * (~self.failed).float()
        return r

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """밀기 1회가 끝난 뒤: 실패 확인 -> 카메라로 본 위치로 "열림" 판정 -> 열렸으면 실제로 잡아서 들기.

        성공 = 실제로 들림 (do_grasp=True). 잡기에 실패하고 밀기 횟수가 남았으면 다시 보고 계속 민다.
        """
        cfg = self.cfg
        n = self.num_envs
        xy, yaw, upright, z = self.cubes_2d()
        t0 = torch.zeros(n, dtype=torch.long, device=self.device)
        # 실패: 대상이 기울어짐 / 큐브가 쌓임 / 그리퍼가 큐브 위에 내려앉음(실물에서는 충돌·정지)
        self.fail_tilt = ~upright[:, 0]
        self.fail_stack = (z > cfg.stacked_z).any(-1)
        self.fail_blocked = self._skip & cfg.fail_on_blocked
        self.failed = self.fail_tilt | self.fail_stack | self.landed_on_cube | self.fail_blocked
        self._deficit_after = self.open_deficit(xy, yaw)  # 틈 보상: 밀기 후 참값 (잡기 전에 계산)
        # 참값으로 본 "한 축 열림" (분석용)
        self.axis_open = self.grasp_axes_clear(xy, yaw, t0).any(-1) & ~self.failed

        self.grasp_tried = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.grasp_ok = torch.zeros(n, dtype=torch.bool, device=self.device)
        if cfg.do_grasp:
            oxy, oyaw = self.cubes_observed()  # 실물처럼 카메라로 본 위치로 판정
            clear = self.grasp_axes_clear(oxy, oyaw, t0)
            self.grasp_tried = clear.any(-1) & ~self.failed
            if self.grasp_tried.any():
                axis = torch.where(clear[:, 0], 0, 1)
                ang = _wrap90(oyaw[:, 0]) + axis * math.pi / 2
                held, lifted = self.grasp_attempt(oxy[:, 0], ang, mask=self.grasp_tried)
                self.grasp_ok = self.grasp_tried & held & lifted
            self.success = self.grasp_ok
        else:
            self.success = self.axis_open
        # 모서리 규칙: 대상이 모서리에 낀 뒤 corner_fail_after 번 더 밀어도 못 들면 실패
        self.in_corner = self.target_in_corner(xy, yaw)
        self.corner_count = torch.where(self.in_corner, self.corner_count + 1, torch.zeros_like(self.corner_count))
        self.fail_corner = self.in_corner & (self.corner_count > cfg.corner_fail_after) & ~self.success & (cfg.corner_fail_after >= 0)
        self.failed = self.failed | self.fail_corner
        time_out = self.push_count >= cfg.max_pushes  # 최대 횟수만큼 밀었으면 끝
        return self.success | self.failed, time_out

    # ------------------------------------------------------------------ 리셋: 큐브 무작위 배치
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)
        k = self.cubes.num_objects  # 18

        # 끝난 에피소드 결과를 학습 로그로 (TensorBoard: Episode/...)
        if self.common_step_counter > 0:
            self.extras["log"] = {
                "success_rate": self.success[env_ids].float().mean().item(),
                "fail_rate": self.failed[env_ids].float().mean().item(),
                "blocked_rate": self.fail_blocked[env_ids].float().mean().item(),
                "corner_fail_rate": self.fail_corner[env_ids].float().mean().item(),
                "corner_rate": self.in_corner[env_ids].float().mean().item(),
                "landed_rate": self.landed_on_cube[env_ids].float().mean().item(),
                "grasp_tried": self.grasp_tried[env_ids].float().mean().item(),
                "no_candidate_rate": self._skip[env_ids].float().mean().item(),
                "chosen_predicted_open": self.chosen_opens[env_ids].float().mean().item(),
                "pushes_used": self.push_count[env_ids].float().mean().item(),
            }
        self.corner_count[env_ids] = 0
        self.push_count[env_ids] = 0  # landed_on_cube 는 밀 때마다 새로 계산되므로 리셋 안 함 (결과 읽기용)

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
            ok = _rect_inside_tray(cand, cyaw, (h, h), hx, hy) & ~self._corner_zone(cand)
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
            ok = _rect_inside_tray(cand, cyaw, (h, h), hx, hy) & ~hit.any(-1) & ~self._corner_zone(cand)
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

    def _corner_zone(self, c: torch.Tensor) -> torch.Tensor:
        """배치용: 중심 c (..., 2) 가 모서리 금지 구역(두 벽 모두에서 corner_free 안)인가."""
        hx, hy = self.cfg.tray_inner_half
        f = self.cfg.corner_free
        return (hx - c[..., 0].abs() < f) & (hy - c[..., 1].abs() < f)

    def target_in_corner(self, xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """대상(0번)이 모서리에 끼었나 (n,): x 벽 쪽과 y 벽 쪽 틈이 모두 잡기 틈(finger_clear_depth) 미만."""
        hx, hy = self.cfg.tray_inner_half
        h = CUBE / 2
        c, th = xy[:, 0], yaw[:, 0]
        gx = hx - c[:, 0].abs() - _extent(th, (h, h), torch.zeros_like(th))
        gy = hy - c[:, 1].abs() - _extent(th, (h, h), torch.full_like(th, math.pi / 2))
        d = self.cfg.finger_clear_depth
        return (gx < d) & (gy < d)

    def sides_clear(self, xy: torch.Tensor, yaw: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """큐브 t 의 네 면(+x', -x', +y', -y') 바깥 손가락 영역이 비었는지. 반환 (n, 4) bool.

        손가락 영역 = 면에서 바깥으로 깊이 finger_clear_depth (벽은 + finger_wall_extra), 폭 finger_clear_width. 다른 큐브·벽이 없어야 빔.
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
        out = []
        for a in (th, th + math.pi / 2):
            u = torch.stack([torch.cos(a), torch.sin(a)], dim=-1)
            for sgn in (1.0, -1.0):
                rc = c + sgn * u * (h + cfg.finger_clear_depth / 2)  # 손가락 영역 중심
                # 벽: 손가락 윗부분(더 두꺼움)이 벽 높이에 걸리므로 finger_wall_extra 만큼 더 깊게 검사
                dw = cfg.finger_clear_depth + cfg.finger_wall_extra
                ok = _rect_inside_tray(c + sgn * u * (h + dw / 2), a, (dw / 2, region_half[1]), hx, hy)
                hit = _overlap_matrix(rc[:, None], a[:, None], region_half, xy, yaw)[:, 0]  # (n, k)
                out.append(ok & ~(hit & not_self).any(-1))
        return torch.stack(out, dim=-1)

    def grasp_axes_clear(self, xy: torch.Tensor, yaw: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """큐브 t 를 잡을 수 있는지 두 축(0°, 90°)에 대해 검사. 반환 (n, 2) bool.

        한 축이 열림 = 그 축 양쪽 손가락 영역이 비어 있고 하우징·카메라가 벽에 안 닿음.
        """
        cfg = self.cfg
        hx, hy = cfg.tray_inner_half
        ar = torch.arange(yaw.shape[0], device=xy.device)
        c, th = xy[ar, t], yaw[ar, t]
        sides = self.sides_clear(xy, yaw, t)
        ax0 = sides[:, 0] & sides[:, 1]
        ax1 = sides[:, 2] & sides[:, 3]
        # 잡을 때 하우징이 벽 높이보다 낮으면 벽에 닿으면 안 됨
        if cfg.grasp_tip_z + cfg.housing_bottom_above_tip < cfg.tray_wall_height:
            ax0 &= _rect_inside_tray(c, th, cfg.housing_half, hx, hy)
            ax1 &= _rect_inside_tray(c, th + math.pi / 2, cfg.housing_half, hx, hy)
        # 잡을 때 카메라(브래킷 포함)도 벽에 닿으면 안 됨 (카메라 아래 = 잡기 높이 + camera_bottom_above_tip)
        if cfg.grasp_tip_z + cfg.camera_bottom_above_tip < cfg.tray_wall_height:
            ax0 &= _rect_inside_tray(c, th, cfg.camera_half, hx, hy)
            ax1 &= _rect_inside_tray(c, th + math.pi / 2, cfg.camera_half, hx, hy)
        return torch.stack([ax0, ax1], dim=-1)


# ---------------------------------------------------------------------- 회전된 사각형 기하
def _extent(ang: torch.Tensor, half: tuple[float, float], axis: torch.Tensor) -> torch.Tensor:
    """각도 ang 인 사각형(반 크기 half)을 axis 방향에 투영한 반 길이."""
    d = ang - axis
    return half[0] * torch.abs(torch.cos(d)) + half[1] * torch.abs(torch.sin(d))


def _rect_inside_tray(c, ang, half, hx: float, hy: float, corner: float = TRAY_CORNER) -> torch.Tensor:
    """사각형이 트레이 안쪽(모서리 삼각형 블록 제외)에 완전히 들어가면 True. c: (..., 2), ang: (...)."""
    zero = torch.zeros_like(ang)
    ex = _extent(ang, half, zero)
    ey = _extent(ang, half, zero + math.pi / 2)
    ok = (c[..., 0].abs() + ex <= hx - 0.0005) & (c[..., 1].abs() + ey <= hy - 0.0005)
    if corner > 0:
        # 모서리 블록: 꼭짓점 4개가 모두 (hx-|x|) + (hy-|y|) >= corner 이어야 함 (안쪽 영역이 볼록 팔각형)
        cs, sn = torch.cos(ang), torch.sin(ang)
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            px = c[..., 0] + sx * half[0] * cs - sy * half[1] * sn
            py = c[..., 1] + sx * half[0] * sn + sy * half[1] * cs
            ok &= (hx - px.abs()) + (hy - py.abs()) >= corner + 0.0005
    return ok


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


def _wrap90(a: torch.Tensor) -> torch.Tensor:
    """각도를 [-45°, 45°) 로 (정사각형은 90° 마다 같은 모양)."""
    return torch.remainder(a + math.pi / 4, math.pi / 2) - math.pi / 4


def _wrap180(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def _rotate(v: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """벡터 v (..., 2) 를 ang (...) 만큼 회전."""
    cs, sn = torch.cos(ang), torch.sin(ang)
    return torch.stack([cs * v[..., 0] - sn * v[..., 1], sn * v[..., 0] + cs * v[..., 1]], dim=-1)


def _ray_to_tray(c: torch.Tensor, ang: torch.Tensor, hx: float, hy: float) -> torch.Tensor:
    """점 c (n,2) 에서 ang 방향으로 트레이 안쪽 벽까지 거리."""
    dx, dy = torch.cos(ang), torch.sin(ang)
    eps = 1e-6
    tx = torch.where(dx.abs() > eps, (torch.sign(dx) * hx - c[:, 0]) / torch.where(dx.abs() > eps, dx, eps), 1e3)
    ty = torch.where(dy.abs() > eps, (torch.sign(dy) * hy - c[:, 1]) / torch.where(dy.abs() > eps, dy, eps), 1e3)
    return torch.minimum(tx, ty).clamp(min=0.0)
