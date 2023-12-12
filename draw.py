import enum
import numpy as np
import seaborn as sns
import typer


class Method(enum.StrEnum):
    PDM = "PDM"
    LS = "LS"
    PF = "PF"

    def full_name(self) -> str:
        if self == self.PDM:
            return "Phase Dispersion Minimization"
        elif self == self.LS:
            return "Lomb--Scargle"
        elif self == self.PF:
            return "Duty Cycle"


class FRBName(enum.StrEnum):
    FRB20180916B = "FRB20180916B"  # (77)
    FRB20190915D = "FRB20190915D"  # 👍 (10)
    FRB20191106C = "FRB20191106C"  # 👎 (7)


def main(eventname: FRBName, method: Method):
    freq_grid = np.load("freq-grid.npy")
    power = np.load(f"{FRBName.value}-{method.value}-power.npy")
    if method != Method.PDM:
        period = 1 / freq_grid[np.nanargmax(power)]
    else:
        period = 1 / freq_grid[np.nanargmin(power)]
    g = sns.lineplot(x=1 / freq_grid, y=power)
    g.axvline(period.value, color="red", alpha=1)
    # g.axvspan(low_, high_, alpha=0.3)
    g.set_xscale("log")
    g.set_xlabel("period")
    g.set_ylabel("power")
    g.set_title(
        f"{method.full_name()} Periodogram of {eventname.value} ({period.value:.2f} d)"
    )

    g.figure.savefig(f"graph-{eventname.value}-periodogram-{method.value}.png")


if __name__ == "__main__":
    typer.run(main)
