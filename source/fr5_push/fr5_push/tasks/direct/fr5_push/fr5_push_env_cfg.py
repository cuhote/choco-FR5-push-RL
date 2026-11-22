import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# ---------------------------------------------------------------- 실측값 (m)
GRIPPER_USD = "/home/dync/cuhote_ws/assets/pgea/usd/pgea_floating.usd"
TCP_OFFSET = 0.160  # 플랜지 면 -> 손가락 끝
CUBE = 0.025  # 큐브 한 변
NUM_CUBES = 18  # Cube_00 = 대상(빨강), 나머지 = 이웃(노랑)

# 트레이 = 손잡이 쟁반 (상품 표기 40 x 27.5 cm, 높이 2 cm). 바닥 윗면 = z 0
# ★가정★ 40 x 27.5 를 안쪽 크기로 사용 (손잡이 포함 바깥 크기라면 실제 바닥은 더 좁음 -> 실측 후 수정)
#        테두리는 수직 벽으로 단순화 (실제는 약간 비스듬함), 모서리는 직각
# 이전 트레이(240 x 320, 벽 80mm)는 하우징이 벽에 걸려 못 푸는 경우가 많았음 -> 낮은 테두리는 하우징·카메라가 위로 지나감
TRAY_IN_X = 0.400  # 안쪽 가로 (x)
TRAY_IN_Y = 0.275  # 안쪽 세로 (y)
TRAY_WALL_H = 0.020  # 테두리 높이
TRAY_WALL_T = 0.005  # 테두리 두께 (가정)
TRAY_CORNER = 0.0  # 모서리 삼각형 블록 변 길이 (0 = 없음, 직각 모서리)
_H, _T = TRAY_WALL_H, TRAY_WALL_T
_OX, _OY = TRAY_IN_X + 2 * _T, TRAY_IN_Y + 2 * _T  # 바깥 크기
TRAY_PARTS = {
    #  이름        크기 (x, y, z)              중심 위치 (x, y, z)        yaw(도)
    "Floor": ((_OX, _OY, 0.004), (0.0, 0.0, -0.002), 0.0),
    "WallXPos": ((_T, _OY, _H), (TRAY_IN_X / 2 + _T / 2, 0.0, _H / 2), 0.0),
    "WallXNeg": ((_T, _OY, _H), (-(TRAY_IN_X / 2 + _T / 2), 0.0, _H / 2), 0.0),
    "WallYPos": ((_OX, _T, _H), (0.0, TRAY_IN_Y / 2 + _T / 2, _H / 2), 0.0),
    "WallYNeg": ((_OX, _T, _H), (0.0, -(TRAY_IN_Y / 2 + _T / 2), _H / 2), 0.0),
}
if TRAY_CORNER > 0:
    _S = TRAY_CORNER * 2 ** 0.5  # 45° 돌린 정사각형 -> 트레이 안쪽에 변 TRAY_CORNER 인 삼각형
    for _n, (_x, _y) in {"CornerPP": (1, 1), "CornerPN": (1, -1), "CornerNP": (-1, 1), "CornerNN": (-1, -1)}.items():
        TRAY_PARTS[_n] = ((_S, _S, _H), (_x * TRAY_IN_X / 2, _y * TRAY_IN_Y / 2, _H / 2), 45.0)


def _cube_cfg(i: int) -> RigidObjectCfg:
    color = (0.85, 0.1, 0.1) if i == 0 else (0.95, 0.75, 0.1)
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Cube_{i:02d}",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE, CUBE, CUBE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.015),  # 재질 확인 후 수정
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        # 처음 생성 위치만 겹치지 않게. 실제 배치는 _reset_idx 에서 함
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.04 * (i % 6) - 0.1, 0.04 * (i // 6) - 0.04, 0.0135)),
    )


GRIPPER_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=GRIPPER_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=0.5),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        # pz = 플랜지 높이. 손가락 끝 높이 = pz - 0.160 -> 0.35면 끝이 바닥에서 19cm
        joint_pos={"px": 0.0, "py": 0.0, "pz": 0.35, "yaw": 0.0, "finger_.*": 0.0},
    ),
    actuators={
        "gantry": ImplicitActuatorCfg(
            joint_names_expr=["px", "py", "pz"], effort_limit_sim=200.0, stiffness=5000.0, damping=300.0
        ),
        "yaw": ImplicitActuatorCfg(joint_names_expr=["yaw"], effort_limit_sim=50.0, stiffness=500.0, damping=20.0),
        # 손가락: 힘을 5N 으로 제한. 100N 이면 15g 큐브를 쥘 때 물리 엔진에서 손가락이 큐브를 뚫고 지나감
        # (잡기 검증: 100N -> 잡힘 0%, 5N -> 열림 판정의 99.3% 잡힘). 실물 쥐는 힘과는 별개
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["finger_.*"], effort_limit_sim=5.0, stiffness=200.0, damping=20.0
        ),
    },
)


# 밀기 1회 시간표 (초): 내려가기 -> 밀기 -> 들기 -> 안정 대기
SIM_DT = 1 / 120
T_DESCEND, T_PUSH, T_LIFT, T_SETTLE = 0.4, 1.6, 0.4, 0.4
PUSH_STEPS = round((T_DESCEND + T_PUSH + T_LIFT + T_SETTLE) / SIM_DT)  # 336 물리 스텝 = 행동 1번
MAX_PUSHES = 3  # 대상 하나당 최대 밀기 횟수 (2번 58.8% / 3번 64.7%, 규칙 기준 측정). 넘으면 사람 호출
CAND_NEIGHBOURS = 6  # 후보: 가까운 이웃 6개
CAND_DISTS = (0.040, 0.030)  # 후보: 미는 거리 2종 (앞쪽을 규칙이 우선)
CAND_TARGET = True  # 후보: 대상 자체를 네 면 방향으로 밀기
# 이웃 6 x 옆면 2 x 방향 2(옆으로 / 45° 바깥) x 거리 2 = 48, 대상 4면 x 거리 2 = 8 -> 56
NUM_CAND = CAND_NEIGHBOURS * 2 * 2 * len(CAND_DISTS) + (4 * len(CAND_DISTS) if CAND_TARGET else 0)


@configclass
class Fr5PushEnvCfg(DirectRLEnvCfg):
    # env: 행동 1번 = 밀기 1회 (물리 336스텝 = 2.8초)
    decimation = PUSH_STEPS
    episode_length_s = (MAX_PUSHES + 1) * PUSH_STEPS * SIM_DT  # 여유. 실제 종료는 max_pushes 로 판단
    max_pushes = MAX_PUSHES
    action_space = NUM_CAND  # 후보 56개 각각의 점수 -> 안전한 후보 중 최고점을 실행
    observation_space = 51 + NUM_CAND * 10  # 611: 대상 주변 51 + 후보 56 x 10
    state_space = 0

    # render_interval 4: 화면을 켰을 때 미는 동작이 보이도록 (학습은 headless라 영향 없음)
    sim: SimulationCfg = SimulationCfg(dt=SIM_DT, render_interval=4)

    # 밀기 동작
    t_descend, t_push, t_lift = T_DESCEND, T_PUSH, T_LIFT
    tcp_offset = TCP_OFFSET  # 플랜지 -> 손가락 끝 (pz = 손가락 끝 높이 + 0.160)
    safe_tip_z = 0.035  # 이동할 때 손가락 끝 높이 (큐브 윗면 25mm + 10mm)
    push_tip_z = 0.017  # 밀 때 손가락 끝 높이 (설계 16~18mm)
    push_speed = 0.025  # 2.5 cm/s
    push_backoff = 0.005  # 밀기 후 반대로 5mm 물러난 뒤 들기 (안 하면 들 때 큐브를 끌어올려 쌓임 8%)
    t_backoff = 0.15  # 물러나는 시간 (들기 시간 t_lift 안에 포함)
    cand_neighbours = CAND_NEIGHBOURS
    cand_dists = CAND_DISTS
    cand_target = CAND_TARGET
    safety_margin = 0.004  # 안전 필터 여유: 착지 자리·하우징 검사 면적을 4mm 넓혀 검사 (카메라 오차 대비)
    start_clearance = 0.008  # 후보: 밀 큐브 옆면에서 8mm 떨어져 내림 (안전 여유 4mm보다 크게)
    # 실패 판정
    landed_tol = 0.003  # 내려간 뒤 손가락 끝이 목표보다 3mm 넘게 높음 = 큐브 위에 내려앉음
    tilt_fail_deg = 15.0  # 큐브가 15° 넘게 기울어 있음 (모서리로 서거나 다른 큐브에 기댐)
    stacked_z = CUBE / 2 + 0.004  # 큐브 중심이 16.5mm보다 높음 = 다른 큐브 위에 올라감

    # 관측
    obs_neighbours = 8  # 가까운 이웃 몇 개를 볼지
    obs_neighbour_radius = 0.12  # 이보다 먼 이웃은 "없음"으로

    # 보상 (설계서 초안)
    rew_success = 10.0
    rew_push = -1.0  # 밀기 1회
    rew_per_cm = -0.1  # 민 거리 1cm 당
    rew_fail = -5.0  # 넘어짐 / 쌓임 / 큐브 위 착지
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=0.6, replicate_physics=True)

    robot_cfg: ArticulationCfg = GRIPPER_CFG
    cubes_cfg: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={f"cube_{i:02d}": _cube_cfg(i) for i in range(NUM_CUBES)}
    )

    # 큐브 배치: 트레이 전체에 무작위로 흩뿌린 뒤, 두 잡기 축이 모두 막힌 큐브를 대상으로 고름
    tray_inner_half = (TRAY_IN_X / 2, TRAY_IN_Y / 2)  # 트레이 안쪽 반 크기
    num_candidates = 128  # 큐브 1개 놓을 때 뽑는 후보 위치 수
    min_gap = 0.0005  # 큐브끼리 최소 간격 0.5mm (생성 시 겹침 방지)
    # 막힌 큐브가 없을 때: 무작위 대상 옆으로 다른 큐브를 옮겨 붙임
    near_radius = 0.045  # 옮겨 붙일 후보 반경 (대상 중심 기준)
    max_cluster_moves = 8  # 최대 옮기는 횟수
    wall_case_prob = 0.2  # 벽 때문에만 막힌 큐브를 대상으로 허용하는 비율 (나머지는 이웃 큐브 때문에 막힌 경우)

    # 잡기 판정 (E단계 성공 조건과 같은 함수 사용)
    # 큐브 면에서 바깥으로 필요한 빈 깊이 = (벌림 19.5 - 큐브 반 12.5) + 손가락 두께 13 + 여유 0.5
    # (16mm 였을 때 카메라 노이즈가 있으면 "열림" 판정의 77% 만 실제로 잡힘 -> 20.5mm 에서 97.6%)
    finger_clear_depth = 0.0205
    finger_clear_width = 0.022  # 손가락 폭 20 + 여유 2 (손가락 Y 폭 실측 후 수정)
    housing_half = (0.0275, 0.020)  # 하우징 반 크기 (잡기 축 방향, 직각 방향) - 벽 충돌 검사
    # D435 + 흰색 브래킷: 아래를 수직으로 봄. ★추정값 (사진 비율) - 실측 후 교체★
    #   위치를 몰라서 그리퍼 중심 기준으로 브래킷 판 전체를 덮는 넉넉한 상자로 잡음
    camera_half = (0.055, 0.045)  # 반 크기 (그리퍼 X = 손가락 벌림 방향, 그리퍼 Y)
    camera_bottom_above_tip = 0.055  # 카메라 아래 면이 손가락 끝보다 55mm 위 (플랜지에서 약 105mm)
    tray_wall_height = TRAY_WALL_H  # 트레이 벽 높이 (바닥 윗면 기준)
    housing_bottom_above_tip = 0.038  # 하우징 아래 면 = 손가락 끝 + 38mm (실측)
    grasp_tip_z = CUBE / 2  # 잡을 때 손가락 끝 높이 (큐브 중간 = 위쪽 절반을 잡음)
    # 손가락 PD (stiffness, damping, 최대 힘 N): 밀 때 / 잡을 때 다르게 (fr5_push_env.set_finger_mode)
    finger_gains = {"push": (2000.0, 100.0, 100.0), "grasp": (200.0, 20.0, 5.0)}
    grasp_open_each = 0.0195  # 잡기 전 벌림: 손가락마다 19.5mm (약 89%, 합 39mm). 70% 면 카메라 오차에 약함
    grasp_ok_range = (0.0123, 0.0132)  # 잡기 성공 위치값 56~60% (손가락마다, 실물 메모 기준)

    # 카메라 측정 노이즈 (D435 흉내). 계획(관측·후보·안전 검사)에만 적용, 물리·성공 판정은 참값
    obs_pos_noise_std = 0.002  # 위치 σ 2mm
    obs_yaw_noise_deg = 2.0  # 각도 σ 2°
    finger_foot_half = (0.0105, 0.0105)  # 닫힌 손가락이 밀 때 차지하는 바닥 면적 (반 크기, 약 21 x 21mm)

