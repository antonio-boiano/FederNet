#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot federated-learning accuracies from log files.

Features (toggle at the CONFIG section):
  • SEPARATE_PLOTS – one PNG per run vs. combined plot.
  • ENABLE_ZOOM    – optional inset zoom (start/end).
  • USE_TIME       – X-axis in seconds instead of training rounds.
  • Recursive scan of `log_folder`, only *.log whose name contains "ps".
"""

import os, re, ast, csv
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

###############################################################################
# -------------------------------  CONFIG  ---------------------------------- #
###############################################################################
log_folder      = 'src/output/long_run_3_async'      # root with log sub-dirs
results_root    = './results'                 # where CSVs are stored
output_plot     = './results/accuracy_plot.png'  # used if SEPARATE_PLOTS=False
zoom_mode       = "end"      # 'start' • 'end' • ''   (ignored if ENABLE_ZOOM=False)

SEPARATE_PLOTS  = True      # True → one PNG per run
ENABLE_ZOOM     = False       # build the inset zoom?
USE_TIME        = True      # X-axis = seconds elapsed

###############################################################################
# ---------------------------  HELPER FUNCTIONS  ---------------------------  #
###############################################################################
def extract_config(txt: str):
    m = re.search(r"INFO \| run\.py:\d+ \| ({.*})", txt, re.DOTALL)
    if m:
        try:
            return ast.literal_eval(m.group(1))
        except Exception:
            pass
    return None
_TS_START = (r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| INFO \| "
          r"run\.py:\d+ \|")

def extract_t0(txt: str):
    m = re.search(_TS_START, txt)
    if m:
        return pd.to_datetime(m.group(1), format='%Y-%m-%d %H:%M:%S,%f')
    return None

_TS_RE = (r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| INFO \| "
          r"abstract\.py:\d+ \| ACCURACY: (\d+\.\d+)")
def extract_accuracies(txt: str):
    return re.findall(_TS_RE, txt)


def generate_label(cfg: dict | None, fname: str, mmap: dict):
    if fname in mmap:
        return mmap[fname]

    if cfg:
        lab, prm = cfg.get('fl_method', 'Unknown'), []
        cl_cfg = cfg.get('client_config', {})
        if lr := cl_cfg.get('local_step_size'): prm.append(f"LR={lr}")
        if a  := cl_cfg.get('alpha'):          prm.append(f"α={a}")
        if cfg.get('model_pruning'):           prm.append("Prune")
        if cfg.get('client_clustering'):       prm.append("CS")
        return f"{lab} ({', '.join(prm)})" if prm else lab

    return fname


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def process_log(log_path: str, use_time: bool, mmap: dict):
    """Return (label, csv_path, parent_dir)."""
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.read()

    cfg          = extract_config(txt)
    pairs        = extract_accuracies(txt)
    rows, t0     = [[0,0]], extract_t0(txt)
        
    for idx, (ts_s, acc_s) in enumerate(pairs):
        x = (pd.to_datetime(ts_s, format='%Y-%m-%d %H:%M:%S,%f') - (t0 := t0 or
             pd.to_datetime(ts_s, format='%Y-%m-%d %H:%M:%S,%f'))).total_seconds() \
            if use_time else idx
        rows.append((x, float(acc_s)))

    parent_dir   = os.path.basename(os.path.dirname(log_path))
    out_dir      = ensure_dir(os.path.join(results_root, parent_dir))
    base         = os.path.splitext(os.path.basename(log_path))[0]
    csv_path     = os.path.join(out_dir, f"{base}.csv")
    with open(csv_path, 'w', newline='') as c:
        csv.writer(c).writerows([[('time_sec' if use_time else 'round'), 'accuracy']] + rows)

    label = generate_label(cfg, base, mmap)
    return label, csv_path, parent_dir


###############################################################################
# -----------------------------  MAIN ROUTINE  ----------------------------- #
###############################################################################
manual_label_map = {}
plot_data, dataframes, parents = [], [], {}

for root, _, files in os.walk(log_folder):
    for f in files:
        if f.endswith('.log') and 'ps' in f:
            lbl, csv_p, pdir = process_log(os.path.join(root, f), USE_TIME,
                                           manual_label_map)
            plot_data.append((csv_p, lbl))
            parents[lbl] = pdir

if not plot_data:
    raise SystemExit("No matching log files found.")

###############################################################################
# -------------------------------- PLOTS ----------------------------------- #
###############################################################################
x_col = 'time_sec' if USE_TIME else 'round'
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

def new_axes():
    fig = plt.figure(figsize=(15, 12))
    return fig.add_subplot(111), fig


def add_zoom(ax, dfs):
    zx = zoomed_inset_axes(ax, 2.3 if zoom_mode == "end" else 1.3,
                           bbox_to_anchor=(.45, .34, .6, .5) if zoom_mode=="end"
                           else (.55, .34, .6, .5),
                           bbox_transform=ax.transAxes,
                           loc=4 if zoom_mode=="end" else 3)
    for i, df in enumerate(dfs):
        zx.plot(df[x_col], df['accuracy'], linewidth=2,
                color=colors[i % len(colors)])
    zx.tick_params(axis='both', labelsize=22)
    zx.grid(True, linestyle='--', linewidth=1)
    mark_inset(ax, zx, loc1=4 if zoom_mode=="end" else 3,
               loc2=2 if zoom_mode=="end" else 1, fc="none", ec="0.5")


if SEPARATE_PLOTS:
    for i, (csv_p, lbl) in enumerate(plot_data):
        df, pdir = pd.read_csv(csv_p), parents[lbl]
        ax, fig  = new_axes()
        ax.plot(df[x_col], df['accuracy'], linewidth=2, color=colors[0], label=lbl)
        ax.set_xlabel('Time [s]' if USE_TIME else 'Round', fontsize=36)
        ax.set_ylabel('Accuracy', fontsize=36)
        ax.tick_params(axis='both', labelsize=36)
        ax.grid(True, linestyle='-', linewidth=2)
        ax.legend(loc='lower right', fontsize=28)
        if ENABLE_ZOOM and zoom_mode:
            add_zoom(ax, [df])

        png_path = os.path.join(results_root, pdir, f"{lbl.replace(' ', '_')}.png")
        plt.tight_layout(); fig.savefig(png_path, bbox_inches='tight'); plt.close(fig)
        print(f"Saved → {png_path}")

else:  # combined
    ax, fig = new_axes()
    dfs     = []
    for i, (csv_p, lbl) in enumerate(plot_data):
        df = pd.read_csv(csv_p)
        dfs.append(df)
        ax.plot(df[x_col], df['accuracy'],
                linewidth=2, color=colors[i % len(colors)], label=lbl)

    ax.set_xlabel('Time [s]' if USE_TIME else 'Round', fontsize=36)
    ax.set_ylabel('Accuracy', fontsize=36)
    ax.tick_params(axis='both', labelsize=36)
    ax.grid(True, linestyle='-', linewidth=2)
    ax.legend(loc='lower right', fontsize=28)

    if ENABLE_ZOOM and zoom_mode:
        add_zoom(ax, dfs)

    comb_dir  = ensure_dir(os.path.join(results_root,
                       os.path.basename(os.path.normpath(log_folder))))
    comb_png  = os.path.join(comb_dir, 'accuracy_plot.png')
    plt.tight_layout(); fig.savefig(comb_png, bbox_inches='tight'); plt.show()
    print(f"Saved → {comb_png}")