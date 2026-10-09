"""SM figure sm_large: Fig. 3(a)-(c) for the large network (H=500), where both
protocols reach 100% training accuracy.

(a) switching-set size |S|/n, (b) persistence of the switching set with the
random-set chance level of each protocol (dotted), (c) switching distance d_ij.
Same data conventions and style as fig3_subpanels_v3.py (panels are drawn by
the same function).

Output: <outdir>/sm_large.png and .pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import fig_common as fc
from fig3_subpanels import draw_top_row

SGD_FOLDER = Path("/data/samuel_lozano/dynamical-sgd/with_bumps/spiral-1_150_bumps_large/")      # EDIT: H=500, A=1
DYNAMICAL_FOLDER = Path("/data/samuel_lozano/dynamical-sgd/with_bumps/spiral-150_bumps_large/")      # EDIT: H=500, A>1
MAX_CYCLES: int | None = None    # None = up to the last dynamical cycle of the DB runs
OUT_NAME = "fig_sm_large"
FIG_W, FIG_H = 11.6, 3.7


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-o", "--outdir", type=Path, default=Path("figures"))
    parser.add_argument("--sgd-folder", type=Path, default=SGD_FOLDER)
    parser.add_argument("--dyn-folder", type=Path, default=DYNAMICAL_FOLDER)
    parser.add_argument("--max-cycles", type=int, default=MAX_CYCLES)
    parser.add_argument("--band", choices=("sd", "sem"), default=fc.BAND)
    args = parser.parse_args()
    fc.BAND = args.band

    ub = fc.load_folder(args.sgd_folder, fc.UB_REGIME, "UB (H=500)")
    db = fc.load_folder(args.dyn_folder, fc.DB_REGIME, "DB (H=500)")
    if not ub and not db:
        raise SystemExit("No switching data found in either folder.")
    max_cycle = args.max_cycles or fc.last_cycle(db or ub, "dec/switch_frac")
    if max_cycle is None:
        raise SystemExit("No switch_decomposition.csv data: run switching_plots_full.py once.")

    with plt.rc_context(fc.RC):
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        gs = fig.add_gridspec(1, 3, left=0.085, right=0.99, bottom=0.17, top=0.80, wspace=0.55)
        draw_top_row([fig.add_subplot(gs[0, i]) for i in range(3)], ub, db, max_cycle)
        handles = fc.condition_handles(bool(ub), bool(db))
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), handlelength=2.4,
                   columnspacing=1.8, handletextpad=0.7, borderaxespad=0.0,
                   bbox_to_anchor=(0.52, 0.99), fontsize=18)
        fc.save(fig, args.outdir, OUT_NAME)


if __name__ == "__main__":
    main()
