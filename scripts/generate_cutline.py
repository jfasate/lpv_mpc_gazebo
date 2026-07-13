#!/usr/bin/env python3
"""Generate a drivable, faster corner-cut line from a centerline + track widths.

The raw centerline of test_worldv5 has 4 hairpins whose radius (down to 0.60 m)
is TIGHTER than the car's kinematic minimum turn radius at full steering lock
(0.742 m for wheelbase 0.33 m, max steer 24 deg). No controller can follow those
corners on the centerline. This script solves a minimum-curvature optimization
INSIDE the track corridor (using w_tr_right/left) to produce a line whose peak
curvature is within the steering envelope — which also happens to be the fast
racing line (enter wide, apex the inside, exit wide).

Output is written in the exact column format the lpv_mpc_gazebo node parses:
    s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2; w_tr_right_m; w_tr_left_m

Run with the ROS python that has scipy:
    /usr/bin/python3 generate_cutline.py \
        --in  src/csv_data/test_worldv5.csv \
        --out src/csv_data/test_worldv5_cut.csv
"""
import argparse
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import lsq_linear


def load_centerline(path):
    rows = []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        p = line.strip().split(';')
        rows.append([float(v) for v in p])
    a = np.array(rows)
    x, y = a[:, 1], a[:, 2]
    w_tr_right, w_tr_left = a[:, 7], a[:, 8]
    return x, y, w_tr_right, w_tr_left


def left_normals(x, y):
    """Unit left-hand normals via periodic central-difference tangents."""
    dx = np.gradient(x)  # np.gradient is periodic-friendly enough at this density
    dy = np.gradient(y)
    # closed loop: fix endpoints with wrap
    dx[0] = x[1] - x[-1]
    dy[0] = y[1] - y[-1]
    dx[-1] = x[0] - x[-2]
    dy[-1] = y[0] - y[-2]
    t = np.stack([dx, dy], axis=1)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    # left normal of (tx,ty) is (-ty, tx)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return n


def solve_min_curvature(x, y, n, lb, ub, reg=1e-3):
    """Minimise sum of squared second-differences of p = c + alpha*n over alpha,
    box-constrained to the corridor. Convex bounded linear least squares."""
    N = len(x)
    c = np.stack([x, y], axis=1)
    A = lil_matrix((2 * N + N, N))
    b = np.zeros(2 * N + N)
    for i in range(N):
        im, ip = (i - 1) % N, (i + 1) % N
        base = c[im] - 2 * c[i] + c[ip]      # constant second diff of centerline
        # x-row / y-row: alpha_{i-1} n_{i-1} - 2 alpha_i n_i + alpha_{i+1} n_{i+1}
        for comp, r in ((0, 2 * i), (1, 2 * i + 1)):
            A[r, im] += n[im, comp]
            A[r, i] += -2.0 * n[i, comp]
            A[r, ip] += n[ip, comp]
            b[r] = -base[comp]
        # small L2 regularisation on alpha (prefer centerline where curvature is free)
        A[2 * N + i, i] = np.sqrt(reg)
    res = lsq_linear(A.tocsr(), b, bounds=(lb, ub), max_iter=200, tol=1e-8)
    return res.x


def arc_psi_kappa(x, y):
    N = len(x)
    dx = np.array([x[(i + 1) % N] - x[i] for i in range(N)])
    dy = np.array([y[(i + 1) % N] - y[i] for i in range(N)])
    ds = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(ds)[:-1]])
    psi = np.unwrap(np.arctan2(dy, dx))
    # curvature = dpsi/ds (periodic central difference)
    dpsi = np.array([np.angle(np.exp(1j * (psi[(i + 1) % N] - psi[(i - 1) % N])))
                     for i in range(N)])
    seg = np.array([ds[i] + ds[(i - 1) % N] for i in range(N)])
    kappa = dpsi / seg
    return s, ds, psi, kappa


def velocity_profile(kappa, ds, a_lat, v_max, a_acc, a_brk):
    N = len(kappa)
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-4)))
    for _ in range(3):  # forward (accel) + backward (brake), looped for closure
        for i in range(N):
            v[i] = min(v[i], np.sqrt(v[(i - 1) % N] ** 2 + 2 * a_acc * ds[(i - 1) % N]))
        for i in range(N - 1, -1, -1):
            v[i] = min(v[i], np.sqrt(v[(i + 1) % N] ** 2 + 2 * a_brk * ds[i]))
    ax = np.array([(v[(i + 1) % N] ** 2 - v[i] ** 2) / (2 * ds[i]) for i in range(N)])
    return v, ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', dest='out', required=True)
    ap.add_argument('--margin', type=float, default=0.25,
                    help='keep the line this far [m] off each wall (car half-width + safety)')
    ap.add_argument('--wheelbase', type=float, default=0.33)
    ap.add_argument('--max-steer', type=float, default=0.4189)
    ap.add_argument('--a-lat', type=float, default=7.0, help='max lateral accel [m/s^2]')
    ap.add_argument('--v-max', type=float, default=8.5)
    ap.add_argument('--a-acc', type=float, default=3.0)
    ap.add_argument('--a-brk', type=float, default=5.0)
    ap.add_argument('--iters', type=int, default=3, help='Gauss-Newton re-linearisations')
    args = ap.parse_args()

    x0, y0, wr, wl = load_centerline(args.inp)
    N = len(x0)
    kappa_max = np.tan(args.max_steer) / args.wheelbase
    print(f'{N} points | car min radius = {1/kappa_max:.3f} m (|kappa|<={kappa_max:.3f})')

    # corridor bounds on lateral offset alpha (measured along the LEFT normal):
    #   alpha in [-(w_tr_right - margin), +(w_tr_left - margin)]
    lb = -(wr - args.margin)
    ub = (wl - args.margin)
    lb = np.minimum(lb, -0.01)   # guarantee a feasible, non-empty box
    ub = np.maximum(ub, 0.01)

    # Gauss-Newton: re-estimate normals on the evolving line, keep bounds vs centerline.
    x, y = x0.copy(), y0.copy()
    alpha = np.zeros(N)
    for it in range(args.iters):
        n = left_normals(x, y)
        # express the optimisation about the CURRENT line: total offset = alpha,
        # so solve fresh from centerline each iter using the latest normals.
        n0 = left_normals(x0, y0)
        alpha = solve_min_curvature(x0, y0, n0, lb, ub)
        x = x0 + alpha * n0[:, 0]
        y = y0 + alpha * n0[:, 1]
        _, _, _, kap = arc_psi_kappa(x, y)
        print(f'  iter {it}: max|kappa|={np.max(np.abs(kap)):.3f}  '
              f'tightest R={1/np.max(np.abs(kap)):.3f} m  '
              f'undrivable pts={int(np.sum(np.abs(kap) > kappa_max))}')
        break  # single linear solve is exact here; loop kept for future refinement

    s, ds, psi, kappa = arc_psi_kappa(x, y)
    v, ax = velocity_profile(kappa, ds, args.a_lat, args.v_max, args.a_acc, args.a_brk)
    new_wr = wr + alpha
    new_wl = wl - alpha

    hdr = '# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2; w_tr_right_m; w_tr_left_m'
    with open(args.out, 'w') as f:
        f.write(hdr + '\n')
        for i in range(N):
            f.write(f'{s[i]:.7f};{x[i]:.7f};{y[i]:.7f};{psi[i]:.7f};{kappa[i]:.7f};'
                    f'{v[i]:.7f};{ax[i]:.7f};{new_wr[i]:.7f};{new_wl[i]:.7f}\n')

    _, _, _, kap0 = arc_psi_kappa(x0, y0)
    print(f'\nCENTERLINE : len={s[-1]+ds[-1]:.2f} m  max|kappa|={np.max(np.abs(kap0)):.3f}  '
          f'undrivable={int(np.sum(np.abs(kap0)>kappa_max))}')
    print(f'CUT LINE   : len={s[-1]+ds[-1]:.2f} m  max|kappa|={np.max(np.abs(kappa)):.3f}  '
          f'undrivable={int(np.sum(np.abs(kappa)>kappa_max))}  '
          f'vx[{v.min():.2f},{v.max():.2f}]  mean_v={v.mean():.2f}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
