"""Plot CAPGen training curves (attack loss + temperature) from TensorBoard logs.

Reads the scalars written by capgen_trainer.py (`Loss/attack`, `Train/temperature`)
and renders a static PNG so you can eyeball the loss trajectory at any time
(during or after training) without launching the TensorBoard server.

Usage:
    python plot_loss.py                                  # logdir=runs/capgen, out=loss_curve.png
    python plot_loss.py --logdir runs/capgen --out loss_curve.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")  # headless backend, just save to file
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="runs/capgen",
                    help="TensorBoard log dir written by the trainer")
    ap.add_argument("--out", default="loss_curve.png", help="output PNG path")
    args = ap.parse_args()

    # size_guidance scalars=0 -> keep ALL points (no downsampling)
    ea = EventAccumulator(args.logdir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if "Loss/attack" not in tags:
        raise SystemExit(f"No 'Loss/attack' in {args.logdir}. Found scalars: {tags}")

    loss = ea.Scalars("Loss/attack")
    steps = [s.step for s in loss]
    vals = [s.value for s in loss]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(steps, vals, color="tab:red", marker="o", ms=3, lw=1.6, label="attack loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("attack loss  (max person conf, lower = better)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(True, alpha=0.3)

    # mark the minimum (this is the patch that gets saved as best_*)
    imin = min(range(len(vals)), key=lambda i: vals[i])
    ax1.scatter([steps[imin]], [vals[imin]], color="black", zorder=5)
    ax1.annotate(f"min {vals[imin]:.3f} @ ep{steps[imin]}",
                 (steps[imin], vals[imin]), textcoords="offset points",
                 xytext=(6, 8), fontsize=9)

    # overlay temperature on a secondary axis if present
    if "Train/temperature" in tags:
        tau = ea.Scalars("Train/temperature")
        ax2 = ax1.twinx()
        ax2.plot([s.step for s in tau], [s.value for s in tau],
                 color="tab:blue", ls="--", lw=1.2, label="tau")
        ax2.set_ylabel("temperature  tau", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("CAPGen training: attack loss vs epoch")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}  ({len(steps)} pts | "
          f"first={vals[0]:.4f}  last={vals[-1]:.4f}  "
          f"min={vals[imin]:.4f}@ep{steps[imin]})")


if __name__ == "__main__":
    main()
