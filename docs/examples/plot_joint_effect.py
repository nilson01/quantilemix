"""Plot the fixed Gaussian example; no simulation or resampling is performed."""

from pathlib import Path
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    here = Path(__file__).resolve().parent
    result = runpy.run_path(str(here / "gaussian.py"))["result"]
    levels = np.arange(4)
    mean = levels * result.psi
    lower, upper = levels * result.ci[0], levels * result.ci[1]
    output = here.parent / "site" / "assets"
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 11, "svg.fonttype": "none", "svg.hashsalt": "quantilemix"}):
        fig, ax = plt.subplots(figsize=(6.2, 3.5), layout="constrained")
        ax.fill_between(levels, lower, upper, color="#0072B2", alpha=0.14)
        ax.plot(levels, lower, ":", color="#0072B2", linewidth=1)
        ax.plot(levels, upper, ":", color="#0072B2", linewidth=1)
        ax.plot(levels, mean, "o-", color="#0072B2", linewidth=1.5)
        ax.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
        ax.set(xticks=levels, xlabel=r"Joint exposure score $a$",
               ylabel=r"Estimated mean change $a\widehat{\psi}$")
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(output / "joint-effect.svg", metadata={"Date": None})
        svg = output / "joint-effect.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        fig.savefig(output / "joint-effect.png", dpi=180)
        plt.close(fig)
    np.savetxt(output / "joint-effect.csv", np.column_stack([levels, mean, lower, upper]),
               delimiter=",", header="score,mean_difference,lower_95,upper_95", comments="")


if __name__ == "__main__":
    main()
