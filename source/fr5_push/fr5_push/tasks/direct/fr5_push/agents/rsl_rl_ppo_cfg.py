from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 에피소드가 1~2스텝으로 짧음 -> 환경당 8스텝이면 이터레이션마다 에피소드 4개 이상
    num_steps_per_env = 8
    max_iterations = 300
    save_interval = 20  # 중간에 멈춰도 평가할 모델이 남도록
    experiment_name = "fr5_push"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,  # 행동 범위가 [-1,1]이라 1.0이면 대부분 범위 밖으로 잘림
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,  # 후보 점수 노이즈가 이미 탐색 역할
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,  # 3e-4 는 40 이터레이션 이후 성능이 떨어짐
        schedule="adaptive",
        gamma=0.9,  # 2스텝 에피소드: 먼 미래를 볼 필요 없음
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
