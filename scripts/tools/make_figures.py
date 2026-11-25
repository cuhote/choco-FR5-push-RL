"""포트폴리오용 그림 (docs/figures/*.png). 숫자는 docs/실험기록_20261125.md 의 T11~T13 (시드 1000, 같은 배치).

사용 (fr5_push 폴더에서): python scripts/tools/make_figures.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)

# 한글 글꼴
for name in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
# 기준선(규칙) = 중립 회색, 학습 모델 = 파랑 한 색의 밝기 단계 (학습 순서 = 진해짐)
COLORS = {
    "탐색 규칙": "#a3a29c",
    "model_39": "#86b6ef",
    "model_59": "#3987e5",
    "model_79": "#184f95",
}


def grouped_bars(cats, series, title, subtitle, fname, ylim=(0, 108)):
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    names = list(series)
    k = len(names)
    w = 0.8 / k
    for i, name in enumerate(names):
        xs = [c + (i - (k - 1) / 2) * w for c in range(len(cats))]
        bars = ax.bar(xs, series[name], width=w, color=COLORS[name], edgecolor=SURFACE, linewidth=2, label=name, zorder=3)
        for b, v in zip(bars, series[name]):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=10.5, color=INK)
    ax.set_ylim(*ylim)
    ax.set_ylabel("성공률 (%)", color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="y", colors=INK2)
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.text(0.06, 0.95, title, fontsize=14, fontweight="bold", color=INK)
    fig.text(0.06, 0.905, subtitle, fontsize=9.5, color=INK2)
    ax.legend(frameon=False, ncol=k, loc="lower left", bbox_to_anchor=(0, 1.04), fontsize=9.5, labelcolor=INK)
    fig.subplots_adjust(top=0.76, left=0.08, right=0.98, bottom=0.12)
    path = os.path.join(OUT, fname)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print("저장:", path)


grouped_bars(
    ["벽 무관\n(배치의 69%)", "한 축이 벽에 막힘\n(24%)", "두 축 모두 벽에 막힘\n(7%)"],
    {"탐색 규칙": [98.1, 69.7, 22.6], "model_59": [98.7, 74.2, 47.1], "model_79": [99.1, 74.0, 65.3]},
    "벽 상황별 성공률: 강화학습은 벽에 막힌 대상에서 규칙을 크게 앞섬",
    "실제로 집어 들어 올린 비율, 최대 3번 밀기 | 같은 배치(시드 1000), 정책마다 약 6,000판 | categorical PPO",
    "TS-RL-02_벽상황별_성공률.png",
)
grouped_bars(
    ["1번 안에", "2번 안에", "3번 안에"],
    {
        "탐색 규칙": [64.0, 81.4, 85.9],
        "model_39": [69.5, 84.3, 88.2],
        "model_59": [70.1, 85.2, 89.1],
        "model_79": [70.3, 86.4, 90.6],
    },
    "밀기 횟수별 누적 성공률: 학습할수록 규칙과의 차이가 커짐",
    "실제로 집어 들어 올린 비율 | 같은 배치(시드 1000), 정책마다 약 6,000판 | 3번 안에 85.9% → 90.6% (+4.7%p)",
    "TS-RL-02_밀기횟수별_누적성공률.png",
)
