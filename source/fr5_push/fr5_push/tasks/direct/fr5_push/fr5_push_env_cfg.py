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

# 트레이: 안쪽 240 x 320 x 80, 바깥쪽 255 x 336 x 84 -> 바닥 4, 벽 7.5 / 8
# 트레이 바닥 윗면 = z 0
TRAY_PARTS = {
    #  이름        크기 (x, y, z)              중심 위치 (x, y, z)
    "Floor": ((0.255, 0.336, 0.004), (0.0, 0.0, -0.002)),
    "WallXPos": ((0.0075, 0.336, 0.080), (0.12375, 0.0, 0.040)),
    "WallXNeg": ((0.0075, 0.336, 0.080), (-0.12375, 0.0, 0.040)),
    "WallYPos": ((0.255, 0.008, 0.080), (0.0, 0.164, 0.040)),
    "WallYNeg": ((0.255, 0.008, 0.080), (0.0, -0.164, 0.040)),
}


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
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["finger_.*"], effort_limit_sim=100.0, stiffness=2000.0, damping=100.0
        ),
    },
)


# 밀기 1회 시간표 (초): 내려가기 -> 밀기 -> 들기 -> 안정 대기
SIM_DT = 1 / 120
T_DESCEND, T_PUSH, T_LIFT, T_SETTLE = 0.4, 1.6, 0.4, 0.4
PUSH_STEPS = round((T_DESCEND + T_PUSH + T_LIFT + T_SETTLE) / SIM_DT)  # 336 물리 스텝 = 행동 1번
MAX_PUSHES = 2  # 에피소드당 최대 밀기 횟수


@configclass
class Fr5PushEnvCfg(DirectRLEnvCfg):
    # env: 행동 1번 = 밀기 1회 (물리 336스텝 = 2.8초)
    decimation = PUSH_STEPS
    episode_length_s = (MAX_PUSHES + 1) * PUSH_STEPS * SIM_DT  # 여유. 실제 종료는 max_pushes 로 판단
    max_pushes = MAX_PUSHES
    action_space = 4  # [-1,1]: 시작점 dx, dy (대상 기준), 방향, 거리
    observation_space = 3  # 임시: 대상 큐브 위치 (E단계에서 바꿈)
    state_space = 0

    # render_interval 4: 화면을 켰을 때 미는 동작이 보이도록 (학습은 headless라 영향 없음)
    sim: SimulationCfg = SimulationCfg(dt=SIM_DT, render_interval=4)

    # 밀기 동작
    t_descend, t_push, t_lift = T_DESCEND, T_PUSH, T_LIFT
    tcp_offset = TCP_OFFSET  # 플랜지 -> 손가락 끝 (pz = 손가락 끝 높이 + 0.160)
    safe_tip_z = 0.035  # 이동할 때 손가락 끝 높이 (큐브 윗면 25mm + 10mm)
    push_tip_z = 0.017  # 밀 때 손가락 끝 높이 (설계 16~18mm)
    push_speed = 0.025  # 2.5 cm/s
    push_dist_range = (0.01, 0.04)  # 1~4 cm
    start_range = 0.08  # 시작점: 대상 중심 기준 ±80mm
    # 실패 판정
    landed_tol = 0.003  # 내려간 뒤 손가락 끝이 목표보다 3mm 넘게 높음 = 큐브 위에 내려앉음
    tilt_fail_deg = 15.0  # 큐브가 15° 넘게 기울어 있음 (모서리로 서거나 다른 큐브에 기댐)
    stacked_z = CUBE / 2 + 0.004  # 큐브 중심이 16.5mm보다 높음 = 다른 큐브 위에 올라감
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=0.6, replicate_physics=True)

    robot_cfg: ArticulationCfg = GRIPPER_CFG
    cubes_cfg: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={f"cube_{i:02d}": _cube_cfg(i) for i in range(NUM_CUBES)}
    )

    # 큐브 배치: 트레이 전체에 무작위로 흩뿌린 뒤, 두 잡기 축이 모두 막힌 큐브를 대상으로 고름
    tray_inner_half = (0.120, 0.160)  # 트레이 안쪽 반 크기 (240 x 320)
    num_candidates = 128  # 큐브 1개 놓을 때 뽑는 후보 위치 수
    min_gap = 0.0005  # 큐브끼리 최소 간격 0.5mm (생성 시 겹침 방지)
    # 막힌 큐브가 없을 때: 무작위 대상 옆으로 다른 큐브를 옮겨 붙임
    near_radius = 0.045  # 옮겨 붙일 후보 반경 (대상 중심 기준)
    max_cluster_moves = 8  # 최대 옮기는 횟수
    wall_case_prob = 0.2  # 벽 때문에만 막힌 큐브를 대상으로 허용하는 비율 (나머지는 이웃 큐브 때문에 막힌 경우)

    # 잡기 판정 (E단계 성공 조건과 같은 함수 사용)
    finger_clear_depth = 0.016  # 큐브 면에서 바깥으로 필요한 빈 깊이 (벌림 여유 3 + 손가락 13)
    finger_clear_width = 0.022  # 손가락 폭 20 + 여유 2 (손가락 Y 폭 실측 후 수정)
    housing_half = (0.0275, 0.020)  # 하우징 반 크기 (잡기 축 방향, 직각 방향) - 벽 충돌 검사

