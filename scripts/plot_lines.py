#!/usr/bin/env python3
"""Plot the track walls, centerline, and any generated cut line into log/.

Usage (ROS python for numpy/matplotlib):
    /usr/bin/python3 plot_lines.py [--cut src/csv_data/test_worldv5_cut.csv]
Outputs: src/lpv_mpc_gazebo/log/line_compare.png
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, '..', 'log')


def load(path):
    r = [[float(v) for v in l.strip().split(';')]
         for l in open(path) if not l.startswith('#') and l.strip()]
    a = np.array(r)
    return a[:, 1], a[:, 2], a[:, 7], a[:, 8], a[:, 4]   # x, y, w_r, w_l, kappa


def lnorm(x, y):
    N = len(x)
    dx = np.array([x[(i + 1) % N] - x[(i - 1) % N] for i in range(N)])
    dy = np.array([y[(i + 1) % N] - y[(i - 1) % N] for i in range(N)])
    t = np.stack([dx, dy], 1)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.stack([-t[:, 1], t[:, 0]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--center', default='src/csv_data/test_worldv5.csv')
    ap.add_argument('--cut', default='src/csv_data/test_worldv5_cut.csv')
    ap.add_argument('--out', default=os.path.join(LOGDIR, 'line_compare.png'))
    args = ap.parse_args()

    cx, cy, wr, wl, ck = load(args.center)
    n = lnorm(cx, cy)
    rwx, rwy = cx - wr * n[:, 0], cy - wr * n[:, 1]     # right wall
    lwx, lwy = cx + wl * n[:, 0], cy + wl * n[:, 1]     # left wall
    kmax = np.tan(0.4189) / 0.33

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.plot(np.append(rwx, rwx[0]), np.append(rwy, rwy[0]), 'k-', lw=1.5)
    ax.plot(np.append(lwx, lwx[0]), np.append(lwy, lwy[0]), 'k-', lw=1.5, label='track walls')
    ax.plot(cx, cy, '--', color='tab:blue', lw=1.2, label='centerline')
    bad = np.abs(ck) > kmax
    ax.scatter(cx[bad], cy[bad], c='red', s=60, zorder=5,
               label=f'undrivable centerline pts ({int(bad.sum())})')

    if os.path.exists(args.cut):
        ux, uy, _, _, uk = load(args.cut)
        ax.plot(ux, uy, '-', color='tab:orange', lw=1.6, label='cut line')

    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_title('test_worldv5: walls / centerline / cut line')
    ax.grid(alpha=0.3)
    os.makedirs(LOGDIR, exist_ok=True)
    plt.savefig(args.out, dpi=90, bbox_inches='tight')
    print('saved', os.path.normpath(args.out))


if __name__ == '__main__':
    main()
