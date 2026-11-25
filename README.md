# fr5_push: 막힌 큐브를 밀어서 잡기 축 열기 (Isaac Lab 강화학습)

FR5 + DH PGEA 그리퍼로 글자 큐브를 집을 때, 대상 큐브가 이웃·벽에 막혀 손가락이 들어갈 틈이 없으면
**밀어서 한 축의 틈을 만드는** 정책을 학습한다. 팀 저장소(choco_ws)의 `fr5_task/push_policy.py`에서 쓰는 모델을 만드는 별도 저장소.

- 강화학습이 하는 일: 후보 밀기 220개 중 **어느 것을 고를지**만 결정 (categorical PPO)
- 코드(규칙)가 하는 일: 후보 생성, 안전 검사, 밀기·잡기 동작, 열림 판정(틈 18mm), 보상
- 현재 성적 (같은 배치, 각 약 6,000판, 최대 3번 밀기, 실제로 집어 들어 올린 비율):

| 정책 | 성공률 |
|---|---|
| 무작위 | 52.2% |
| 탐색 규칙 | 85.9% |
| categorical PPO `model_139` | **91.1%** |

자세한 내용: [`docs/환경값_정리_20261125.md`](docs/환경값_정리_20261125.md) (설정값),
[`docs/변경내역_20261125.md`](docs/변경내역_20261125.md) (바꾼 이유),
[`docs/실험기록_20261125.md`](docs/실험기록_20261125.md) (테스트 T1~T14),
[`docs/인수인계_FR5.md`](docs/인수인계_FR5.md) (FR5 팔로 학습할 때 할 일).

## 개발 환경 (이 버전으로 학습함)

| 항목 | 버전 |
|---|---|
| OS | Ubuntu 22.04 |
| Isaac Sim | 5.1.0 (pip) |
| Isaac Lab | 2.3.0 (커밋 `3c6e67bb5`) |
| Python | 3.11 |
| PyTorch | 2.7.0+cu128 |
| GPU | RTX 5050 Laptop 8GB (환경 512개 기준 GPU 메모리 약 4GB) |

## 설치

```bash
# 1) Isaac Lab 2.3.0 설치 (공식 문서), 그 파이썬 환경을 활성화
# 2) 이 저장소를 Isaac Lab 폴더 바깥에 클론
git clone <이 저장소 주소> fr5_push
cd fr5_push
python -m pip install -e source/fr5_push
# 3) 확인
python scripts/list_envs.py      # Template-Fr5-Push-Direct-v0 이 보이면 됨
```

로봇 파일(USD·URDF)은 `assets/`에 들어 있어 추가로 받을 것이 없다. 다른 위치에 두려면 환경 변수 `FR5_PUSH_ASSETS`로 지정.

## 폴더

```
assets/
  usd/pgea_floating/     떠 있는 PGEA 그리퍼 (가상 관절 px, py, pz, yaw). 지금 학습에 쓰는 것
  usd/fr5_pgea_v2/       FR5 + PGEA + D435 브래킷 (FR5 학습용, 아직 환경에 연결 안 함)
  usd/fr5_pgea/          FR5 + PGEA (예전 버전)
  urdf/                  USD를 만든 원본 URDF와 메시 (fairino FR5, PGEA, D435 포함 버전)
  tools/                 PGEA URDF 생성 스크립트
source/fr5_push/fr5_push/push_geom.py   ★ 기하 계산 (후보·안전 검사·잡기 판정·관측·규칙). Isaac Sim 없이 돌고 실물(choco_ws)도 같은 코드를 씀
source/fr5_push/fr5_push/tasks/direct/fr5_push/
  fr5_push_env.py        환경: 배치, 밀기·잡기 동작, 실패 판정, 보상 (기하 계산은 push_geom 상속)
  fr5_push_env_cfg.py    설정값 (트레이, 큐브, 틈, 보상 등)
scripts/
  cat_ppo/train_cat.py   ★ categorical PPO 학습·평가·화면 보기
  push_demo.py           규칙 기준선 (demo / look / random)
  rsl_rl/                rsl_rl 학습 (이전 방식, 비교용)
  tools/                 분석 도구 (oracle_check: 물리 상한, lookahead_check: 기하 상한, stack_view: 쌓임 재생, make_figures,
                         geom_snapshot: push_geom 을 바꾼 전후로 관측이 같은지 확인 + 실물 시험용 JSON)
docs/                    설정값·변경 내역·실험 기록·그림
```

## 실행

모든 명령은 저장소 루트에서. Isaac Sim은 종료 시 버퍼 출력을 버리므로 파일로 보낼 때는 **`python -u`** 필수.

**학습** (환경 512개, iteration 1번 = 512 × 밀기 8번, 약 2분):
```bash
nohup python -u scripts/cat_ppo/train_cat.py --num_envs 512 --iters 300 --run_name cat_v1 > logs/train_cat_v1.log 2>&1 &
grep "^\[cat\] iter" logs/train_cat_v1.log | tail -5        # 진행 보기
```
모델은 `logs/cat_ppo/<시각>_<이름>/model_<iteration>.pt`에 10 iteration마다 저장.

**이어서 학습**: `--resume <model_*.pt> --iters <추가 횟수>`

**평가** (탐색 없이, 같은 배치 = 시드 1000, 실패 원인 분류 포함):
```bash
python -u scripts/cat_ppo/train_cat.py --eval <model_*.pt> --num_envs 384 --pushes 24 2>&1 | tee logs/eval.log | grep -v "^\[eval\]"
python -u scripts/cat_ppo/train_cat.py --eval_policy look --num_envs 384 --pushes 24      # 규칙 기준선 (같은 배치)
```

**화면으로 보기** (트레이 1개):
```bash
python -u scripts/cat_ppo/train_cat.py --gui --num_envs 1 --pushes 100 --eval <model_*.pt>
```

**규칙만으로 밀어 보기**: `python scripts/push_demo.py --num_envs 4 --policy look`

## 학습된 모델

모델 파일(`*.pt`)은 저장소에 커밋하지 않고 **GitHub Release**에 첨부한다. 받은 파일은 `--eval`, `--resume`에 경로로 넘기면 된다.

## 주의

- 학습 PC는 절전을 끈다 (절전 복귀 때 NVIDIA 드라이버 오류로 시뮬이 멈춘 적 있음):
  `sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target`
- 학습을 멈출 때 `kill <PID>`로 안 꺼지는 경우가 많음 → 30초 뒤 `kill -9 <PID>`. 저장 직후(iteration 번호가 9로 끝날 때) 멈추면 손실 없음
- `nohup ... &`의 `$!`가 셸 PID일 수 있음. 실제 PID는 `pgrep -af train_cat.py`로 확인
- 1~2%p 차이를 비교할 때는 같은 시드(`--eval`은 시드 1000 고정)와 약 6,000판 이상
