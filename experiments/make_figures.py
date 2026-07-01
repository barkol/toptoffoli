#!/usr/bin/env python3
"""Generate the paper's figures from the measured benchmark data.
All numbers are taken from experiments/{scale_results,sensitivity_results}.md
and the Q1/Q2 evaluation; no synthetic data. Outputs vector PDFs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
OUT = os.path.dirname(os.path.abspath(__file__))
COL = 3.35    # single-column width (inches)
GRAY, STEEL, OURS, OURS2 = "#8a8f98", "#4878a8", "#2e7d32", "#7cbf7c"

# ---------------------------------------------------------------- Fig 1: safety
fig, ax = plt.subplots(figsize=(COL, 1.9))
labels = ["count-greedy\n(QContext/Maslov-style)", "ours,\nno gate", "ours,\ngated"]
errs = [6, 6, 0]
bars = ax.bar(labels, errs, color=[GRAY, STEEL, OURS], width=0.62, zorder=3)
ax.set_ylabel("silently corrupted\ncircuits (of 12)")
ax.set_ylim(0, 7); ax.set_yticks(range(0, 7, 2))
ax.grid(axis="y", lw=0.4, color="0.85", zorder=0)
for b, v in zip(bars, errs):
    ax.text(b.get_x()+b.get_width()/2, v+0.12, f"{v}/12",
            ha="center", va="bottom", fontsize=7,
            fontweight="bold" if v == 0 else "normal")
ax.text(2, 0.35, "certified\nerror-free", ha="center", va="bottom",
        fontsize=6.5, color=OURS)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.savefig(f"{OUT}/fig_safety.pdf"); plt.close(fig)

# ------------------------------------------------------------ Fig 2: budget cmp
methods = ["Qiskit\nopt-3", "tket", "exact-\nonly", "ours\n(pair)", "ours\n(phase)"]
infid   = [2.690, 2.660, 2.744, 2.192, 1.736]
count   = [281, 279, 281, 221, 170]
colors  = [GRAY, GRAY, STEEL, OURS2, OURS]
fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.3))
for ax, data, ylab, ttl in [
        (axs[0], infid, "summed 2q infidelity", "(a) error budget"),
        (axs[1], count, "summed 2q-gate count", "(b) two-qubit gates")]:
    b = ax.bar(range(len(methods)), data, color=colors, width=0.7, zorder=3)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=7)
    ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=7)
    ax.grid(axis="y", lw=0.4, color="0.85", zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.margins(y=0.16)
    ax.text(len(methods)-1, data[-1]*1.02, f"{data[-1]:g}", ha="center",
            va="bottom", fontsize=6, color=OURS, fontweight="bold")
fig.tight_layout(w_pad=1.2)
fig.savefig(f"{OUT}/fig_budget.pdf"); plt.close(fig)

# ------------------------------------------------------------ Fig 3: device pts
dev   = ["Quantinuum\nH2", "IBM\nHeron r2", "Google\nWillow", "IonQ\nForte", "IonQ\nAria"]
r     = [43, 11, 9, 20, 8]
ex    = [0.241, 0.482, 0.541, 0.585, 0.619]
ours  = [0.136, 0.290, 0.333, 0.370, 0.394]
red   = [43.7, 39.7, 38.4, 36.7, 36.4]
x = np.arange(len(dev)); w = 0.38
fig, ax = plt.subplots(figsize=(COL, 2.0))
ax.bar(x-w/2, ex,   w, label="exact-only", color=STEEL, zorder=3)
ax.bar(x+w/2, ours, w, label="ours",       color=OURS,  zorder=3)
ax.set_xticks(x); ax.set_xticklabels(dev, fontsize=6)
ax.set_ylabel("estimated circuit infidelity")
ax.set_ylim(0, 0.78)
ax.grid(axis="y", lw=0.4, color="0.85", zorder=0)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for xi, o, rd in zip(x, ours, red):
    ax.text(xi+w/2, o+0.012, f"$-${rd:.0f}%", ha="center", va="bottom",
            fontsize=5.8, color=OURS)
ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.1)
fig.savefig(f"{OUT}/fig_device.pdf"); plt.close(fig)

# --------------------------------------------- Fig 4: workload + scaling (wide)
fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.25))
# (a) per-family 2q reduction (large suite, 20 circuits)
fam  = ["Grover\noracle", "CLA\nadder", "ripple\nadder", "modular\nincr.", "array\nmult."]
fred = [49.4, 27.5, 5.5, 0.0, 0.0]
fcol = [OURS, OURS, OURS2, GRAY, GRAY]
b = axs[0].bar(range(len(fam)), fred, color=fcol, width=0.7, zorder=3)
axs[0].set_xticks(range(len(fam))); axs[0].set_xticklabels(fam, fontsize=6)
axs[0].set_ylabel("mean 2q-gate reduction (%)")
axs[0].set_title("(a) workload dependence", fontsize=7.5)
axs[0].grid(axis="y", lw=0.4, color="0.85", zorder=0); axs[0].margins(y=0.16)
for bi, v in zip(b, fred):
    axs[0].text(bi.get_x()+bi.get_width()/2, v+0.8,
                f"{v:.0f}" if v > 0 else "0",
                ha="center", va="bottom", fontsize=6.3)
axs[0].text(3.5, 6, "kept exact\n(all ANDs live)", ha="center", fontsize=5.8,
            color=GRAY)
for s in ("top", "right"): axs[0].spines[s].set_visible(False)
# (b) verification time vs n: measured QCEC points + the exhaustive anchor
n_qcec = [12,12,12,12,13,13,13,16,16,16,16,16,16,17,17,20,22,22,22,22,24]
t_qcec = [0.0181,0.0173,0.0148,0.0135,0.0250,0.0150,0.0174,0.0189,0.0248,
          0.0391,0.0211,0.0161,0.0224,0.0162,0.0224,0.0173,0.0238,0.0551,
          0.2005,0.0203,0.0201]
axs[1].scatter(n_qcec, t_qcec, s=14, color=OURS, zorder=3,
               label="QCEC (decision diagram)")
axs[1].scatter([12], [55.7], s=42, color="#c0392b", marker="X", zorder=4,
               label="exhaustive (measured)")
axs[1].annotate("55.7 s\n(infeasible $n\\geq14$)", (12, 55.7),
                xytext=(13.4, 18), fontsize=6, color="#c0392b",
                arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.6))
axs[1].axhspan(8, 200, color="#c0392b", alpha=0.06, zorder=0)
axs[1].set_yscale("log"); axs[1].set_xlabel("circuit width $n$ (qubits)")
axs[1].set_ylabel("verification time (s)")
axs[1].set_title("(b) verification cost", fontsize=7.5)
axs[1].set_xlim(11, 25); axs[1].set_ylim(8e-3, 2e2)
axs[1].grid(True, which="both", lw=0.35, color="0.88", zorder=0)
axs[1].legend(frameon=False, loc="center right", fontsize=6, handletextpad=0.3)
for s in ("top", "right"): axs[1].spines[s].set_visible(False)
fig.tight_layout(w_pad=1.6)
fig.savefig(f"{OUT}/fig_scale.pdf"); plt.close(fig)

print("wrote: fig_safety.pdf fig_budget.pdf fig_device.pdf fig_scale.pdf")
