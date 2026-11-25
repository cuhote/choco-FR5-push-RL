"""밀기 정책의 기하 계산 (Isaac Sim 없이 돈다). 시뮬(fr5_push_env.py)과 실물(choco_ws fr5_task)이 같은 코드를 쓴다.

후보 생성(build_candidates), 안전 검사(landing_free), 잡기 판정(sides_clear, grasp_axes_clear),
모자란 틈(open_deficit), 앞을 내다본 정보(lookahead_info), 관측(observe), 탐색 규칙 점수(demo_push_action),
모서리 판정(target_in_corner).

좌표: 트레이(상자) 안쪽 중심이 원점, 단위 m, 각도 rad. 큐브 배열의 0번 = 대상.
필요한 것은 torch 뿐이다. 설정값은 self.cfg (속성 접근), 장치는 self.device.
  - 시뮬: Fr5PushEnv 가 이 클래스를 상속 (cfg = Fr5PushEnvCfg)
  - 실물: StandaloneGeometry(geom_cfg_dict) 로 만든다. 설정값은 학습 때 값(geom_cfg.json)을 그대로 쓴다

주의: 이 파일을 바꾸면 학습한 정책의 입력이 바뀐다. 시뮬에서 scripts/tools/geom_snapshot.py 로 전후가 같은지 확인하고,
실물 쪽 복사본도 같이 바꾼다.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

CUBE = 0.025  # 큐브 한 변 (m)
TRAY_CORNER = 0.0  # 트레이 모서리 삼각형 블록 변 길이 (0 = 없음)

# 기하 계산이 쓰는 설정값 이름 (geom_cfg_dict 로 내보내 실물에서 그대로 씀)
GEOM_CFG_KEYS = [
    "camera_bottom_above_tip",
    "camera_half",
    "cand_angles",
    "cand_dists",
    "cand_neighbours",
    "cand_target",
    "cand_wall",
    "cand_wall_square",
    "finger_clear_depth",
    "finger_clear_width",
    "finger_foot_half",
    "finger_wall_extra",
    "grasp_tip_z",
    "housing_bottom_above_tip",
    "housing_half",
    "lookahead_topk",
    "max_pushes",
    "obs_lookahead",
    "obs_neighbour_radius",
    "obs_neighbours",
    "push_speed",
    "push_tip_z",
    "safety_margin",
    "start_clearance",
    "t_push",
    "tray_inner_half",
    "tray_wall_height",
    "wall_push_gap",
    "wall_square_mode",
    "wall_stop_gap",
]


def geom_cfg_dict(cfg) -> dict:
    """설정 객체에서 기하 계산에 필요한 값만 dict 로 (JSON 저장용). 튜플은 리스트가 된다."""
    out = {}
    for k in GEOM_CFG_KEYS:
        v = getattr(cfg, k)
        out[k] = list(v) if isinstance(v, tuple) else v
    return out


class PushGeometry:
    """기하 계산 묶음. self.cfg, self.device 가 있어야 한다."""

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
        # 동점(모자란 깊이가 같은 후보, 특히 0)이 많아서 topk 는 CPU/GPU 에 따라 고르는 후보가 달라진다 (11/26 확인).
        # 실물(CPU)과 시뮬(GPU)이 같은 후보를 펼치도록 안정 정렬: 같은 값이면 앞 슬롯 우선
        top = torch.sort(deficit, dim=1, stable=True).indices[:, : min(topk, num)]
        next_open = torch.zeros(n, num, dtype=torch.bool, device=self.device)
        for r in range(top.shape[1]):
            i = top[:, r]
            c2 = self.build_candidates(cand["xy_pred"][ar, i], cand["yaw_pred"][ar, i])
            next_open[ar, i] = (c2["valid"] & c2["opens"]).any(-1) & valid[ar, i]
        return deficit, next_open

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

    def target_in_corner(self, xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """대상(0번)이 모서리에 끼었나 (n,): x 벽 쪽과 y 벽 쪽 틈이 모두 잡기 틈(finger_clear_depth) 미만."""
        hx, hy = self.cfg.tray_inner_half
        h = CUBE / 2
        c, th = xy[:, 0], yaw[:, 0]
        gx = hx - c[:, 0].abs() - _extent(th, (h, h), torch.zeros_like(th))
        gy = hy - c[:, 1].abs() - _extent(th, (h, h), torch.full_like(th, math.pi / 2))
        d = self.cfg.finger_clear_depth
        return (gx < d) & (gy < d)

    def observe(self, xy: torch.Tensor, yaw: torch.Tensor, push_count: torch.Tensor) -> torch.Tensor:
        """대상 큐브 좌표계로 표현한 관측 (n, 51 + 후보 수 x 12). 부르면 self._cand (후보) 도 채워진다.

        [0:2]   대상 yaw (트레이 기준) sin/cos(4θ)  - 정사각형이라 90° 주기
        [2:6]   대상 중심에서 +x', -x', +y', -y' 방향으로 벽까지 거리 / 0.1m
        [6:10]  손가락 영역 4곳(+x', -x', +y', -y')이 비었는지 (1 = 빔)
        [10:50] 가까운 이웃 8개 x (상대 x', y' / 0.1m, sin/cos(4Δθ), 있음 여부)
        [50]    민 횟수 / 최대 횟수
        [51:]   밀기 후보 x (특징 10 + 탐색 정보 2) (build_candidates, lookahead_info 참고). 행동 i = 후보 i 의 점수
        """
        cfg = self.cfg
        # xy, yaw: 카메라로 본 큐브 위치 (n, k, 2), (n, k). 트레이 중심 기준 m, 0번 = 대상. 관측·후보·안전 검사 모두 이 값 사용
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
        near = torch.sort(dist, dim=1, stable=True).indices[:, : cfg.obs_neighbours]  # (n, 8) 가까운 순. 장치와 무관하게 같은 결과
        rel_n = torch.gather(rel, 1, near[..., None].expand(-1, -1, 2))
        dyaw = torch.gather(yaw[:, 1:], 1, near) - th[:, None]
        valid = (torch.gather(dist, 1, near) < cfg.obs_neighbour_radius).float()
        nb = torch.cat(
            [rel_n / 0.1, torch.sin(4 * dyaw)[..., None], torch.cos(4 * dyaw)[..., None], valid[..., None]], dim=-1
        ) * valid[..., None]

        count = (push_count.float() / cfg.max_pushes)[:, None]
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
        return obs


class StandaloneGeometry(PushGeometry):
    """시뮬 없이 쓰는 기하 계산 (실물용). cfg_dict = geom_cfg_dict(학습 설정) 또는 그 JSON."""

    def __init__(self, cfg_dict: dict, device: str = "cpu"):
        missing = [k for k in GEOM_CFG_KEYS if k not in cfg_dict]
        if missing:
            raise KeyError(f"설정값이 빠짐: {missing}")
        vals = {k: tuple(v) if isinstance(v, list) else v for k, v in cfg_dict.items()}
        self.cfg = SimpleNamespace(**vals)
        self.device = torch.device(device)
        self._cand = None
        self._obs_xy = None
        self._obs_yaw = None



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
