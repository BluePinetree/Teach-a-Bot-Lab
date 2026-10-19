r"""Figures for the blog/slides from the real logs (re-run any time; numbers come from files).

    .venv\Scripts\python.exe scripts\make_learning_figures.py [--curve data\policies\bc_curve_*.csv]

fig13_bc_vs_demos.png      success rate and cubes per episode vs number of human demonstrations
fig14_rl_progress.png      RL baseline: held-out success / training cubes / return vs practice hours
fig16_takeover_effect.png  demos only vs demos + takeover corrections (scripts/compare_datasets.py CSV)
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "blog_materials" / "figures"
FONT = Path("C:/Windows/Fonts/malgun.ttf")       # used in place, never copied


def setup_font() -> None:
    if FONT.exists():
        font_manager.fontManager.addfont(str(FONT))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig_bc(curve: Path) -> Path:
    rows = read_csv(curve)
    k = [int(r["n_demos"]) for r in rows]
    sr = [100 * float(r["success_rate"]) for r in rows]
    cubes = [float(r["cubes_mean"]) for r in rows]
    eps = rows[0]["episodes"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar([str(x) for x in k], sr, color="#4c9be8")
    for i, v in enumerate(sr):
        ax[0].text(i, v + 1, f"{v:.0f}%", ha="center")
    ax[0].set(title="두 큐브 모두 성공한 비율", xlabel="사람 시범 수", ylabel="성공률 (%)", ylim=(0, max(40, max(sr) + 10)))
    ax[1].bar([str(x) for x in k], cubes, color="#f39c38")
    for i, v in enumerate(cubes):
        ax[1].text(i, v + 0.03, f"{v:.2f}", ha="center")
    ax[1].set(title="한 판에 상자에 넣은 큐브 (2개 중)", xlabel="사람 시범 수", ylabel="큐브 수", ylim=(0, 2))
    fig.suptitle(f"모방학습: 사람 시범이 늘수록 (처음 보는 배치 {eps}판, 실측)")
    fig.tight_layout()
    out = OUT / "fig13_bc_vs_demos.png"
    fig.savefig(out, dpi=150)
    return out


def fig_rl() -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for name, color, label in (("sparse", "#888888", "성공 점수만 (희소 보상)"), ("dense", "#2e8b57", "단계별 점수 (촘촘한 보상)")):
        runs = sorted(glob.glob(str(ROOT / "data" / "rl_runs" / f"{name}_2*")))
        if not runs:
            continue
        rows = read_csv(Path(runs[-1]) / "progress.csv")
        h = [float(r["practice_hours"]) for r in rows]
        ax[1].plot(h, [float(r["train_cubes_100"] or 0) for r in rows], color=color, label=label)
        ax[2].plot(h, [float(r["train_return_100"] or 0) for r in rows], color=color, label=label)
        ev = [(float(r["practice_hours"]), 100 * float(r["eval_success"])) for r in rows if r["eval_success"]]
        if ev:
            ax[0].plot(*zip(*ev), "o-", color=color, label=label)
    ax[0].set(title="처음 보는 배치 성공률", xlabel="로봇 연습 시간 (시간)", ylabel="%", ylim=(-2, 100))
    ax[1].set(title="연습 중 상자에 넣은 큐브 (최근 100판 평균)", xlabel="로봇 연습 시간 (시간)", ylim=(0, 2))
    ax[2].set(title="받은 점수 (최근 100판 평균)", xlabel="로봇 연습 시간 (시간)")
    for a in ax:
        a.legend(fontsize=8)
    fig.suptitle("강화학습 기준선: 시범 없이 스스로 (실측, 같은 관측·명령)")
    fig.tight_layout()
    out = OUT / "fig14_rl_progress.png"
    fig.savefig(out, dpi=150)
    return out


def fig_takeover(compare: Path) -> Path:
    rows = read_csv(compare)
    names = list(dict.fromkeys(r["dataset"] for r in rows))
    label = {"demos": "시범만", "demos+takeover": "시범 + 개입"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, (key, title, scale) in enumerate((("success_rate", "두 큐브 모두 성공한 비율 (%)", 100),
                                             ("cubes_mean", "한 판에 상자에 넣은 큐브 (2개 중)", 1))):
        for j, n in enumerate(names):
            v = [scale * float(r[key]) for r in rows if r["dataset"] == n]
            m = sum(v) / len(v)
            ax[i].bar(j, m, color=("#4c9be8", "#2e8b57")[j % 2], alpha=0.85)
            ax[i].scatter([j] * len(v), v, color="black", zorder=3, s=14)
            ax[i].text(j, m / 2, (f"평균 {m:.1f}%" if scale == 100 else f"평균 {m:.2f}"), ha="center",
                       color="white", fontsize=12, fontweight="bold")
        ticks = []
        for n in names:
            r = next(x for x in rows if x["dataset"] == n)
            data = f"시범 {r['n_demos']}" + (f" + 개입 {r['n_takeovers']}" if int(r["n_takeovers"]) else "")
            ticks.append(f"{label.get(n, n)}\n({data})")
        ax[i].set_xticks(range(len(names)), ticks)
        ax[i].set_title(title)
    eps, seeds = rows[0]["episodes"], len([r for r in rows if r["dataset"] == names[0]])
    fig.suptitle(f"넘겨받기(개입 수정)의 효과: 처음 보는 배치 {eps}판, 학습 seed {seeds}개 (점 = seed별)")
    fig.tight_layout()
    out = OUT / "fig16_takeover_effect.png"
    fig.savefig(out, dpi=150)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", type=Path, help="bc_curve_*.csv (default: newest with 40 episodes)")
    args = ap.parse_args()
    setup_font()
    OUT.mkdir(parents=True, exist_ok=True)
    curve = args.curve
    if curve is None:
        curves = [Path(p) for p in sorted(glob.glob(str(ROOT / "data" / "policies" / "bc_curve*.csv")))]
        curves = [c for c in curves if read_csv(c) and read_csv(c)[0].get("episodes") == "40"]
        curve = curves[-1]
    print(fig_bc(curve), "from", curve.name)
    print(fig_rl())
    compares = sorted(glob.glob(str(ROOT / "data" / "policies" / "compare_*.csv")))
    if compares:
        print(fig_takeover(Path(compares[-1])), "from", Path(compares[-1]).name)


if __name__ == "__main__":
    main()
