from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

from fr5_push.push_geom import CUBE as GEOM_CUBE
from fr5_push.push_geom import TRAY_CORNER as GEOM_TRAY_CORNER
from fr5_push.push_geom import (  # noqa: F401  (다른 스크립트가 여기서 가져다 씀)
    PushGeometry,
    _extent,
    _overlap_matrix,
    _ray_to_tray,
    _rect_inside_tray,
    _rotate,
    _wrap90,
    _wrap180,
)

from .fr5_push_env_cfg import CUBE, TRAY_CORNER, TRAY_PARTS, Fr5PushEnvCfg

assert CUBE == GEOM_CUBE and TRAY_CORNER == GEOM_TRAY_CORNER, "push_geom 의 상수와 설정이 다름"


class Fr5PushEnv(PushGeometry, DirectRLEnv):
    """기하 계산(후보·안전 검사·잡기 판정·관측·규칙)은 PushGeometry (fr5_push/push_geom.py) 에서 상속."""
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

    # ------------------------------------------------------------------ 관측/보상/종료
    def _get_observations(self) -> dict:
        """관측은 push_geom.PushGeometry.observe (실물과 같은 코드). 시뮬은 카메라 노이즈를 섞은 위치를 넘긴다."""
        xy, yaw = self.cubes_observed()
        return {"policy": self.observe(xy, yaw, self.push_count)}

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
