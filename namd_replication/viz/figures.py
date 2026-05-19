from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

if TYPE_CHECKING:
    from namd_replication.data.loader import CohortStep


def plot_consort(
    steps: list[CohortStep],
    out_path: Path,
    title: str = "Cohort selection flow",
) -> None:
    n = len(steps)
    if n == 0:
        raise ValueError("plot_consort requires at least one step")

    fig, ax = plt.subplots(figsize=(8, max(4, 1.6 * n)))

    box_w = 0.6
    box_h = 0.55
    x = 0.2

    for i, step in enumerate(steps):
        y = n - i - 0.5
        box = FancyBboxPatch(
            (x, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.02",
            linewidth=1.0,
            edgecolor="black",
            facecolor="#f0f4f9",
        )
        ax.add_patch(box)
        label = f"{step.label}\nn_eyes = {step.n_eyes}   n_patients = {step.n_patients}"
        ax.text(x + box_w / 2, y, label, ha="center", va="center", fontsize=9)

        if i > 0:
            prev = steps[i - 1]
            excluded = prev.n_eyes - step.n_eyes
            if excluded > 0:
                ax.annotate(
                    f"excluded: {excluded} eye{'s' if excluded != 1 else ''}",
                    xy=(x + box_w + 0.02, y + box_h / 2 + 0.05),
                    xytext=(x + box_w + 0.10, y + box_h / 2 + 0.05),
                    fontsize=8,
                    color="firebrick",
                    arrowprops={"arrowstyle": "-", "color": "firebrick", "lw": 0.6},
                )

        if i < n - 1:
            ax.annotate(
                "",
                xy=(x + box_w / 2, y - box_h / 2 - 0.20),
                xytext=(x + box_w / 2, y - box_h / 2),
                arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.0},
            )

    ax.set_xlim(0, 1.4)
    ax.set_ylim(-0.2, n + 0.4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
