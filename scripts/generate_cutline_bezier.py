#!/usr/bin/env python3
"""Control-point Bezier/spline corner-cut line generator.

Instead of offsetting all ~500 waypoints, we place a FEW control points and let a
smooth periodic cubic spline (piecewise-cubic Bezier, C2 continuous) generate the
curve through them:

  * straights  -> sparse control points on the centerline (offset 0)
  * each corner -> 3 control points: OUTER on entry, INNER at the apex, OUTER on
                   exit (the out-in-out racing line -> larger effective radius)

A SAFETY LOOP deepens a corner's apex/entry cut (toward the corridor limit) if its
peak |kappa| still exceeds the car's steering-limit curvature, until drivable.

Emitted in the exact format the lpv_mpc_gazebo node parses:
    s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2; w_tr_right_m; w_tr_left_m

Run with the ROS python (numpy+scipy):
    /usr/bin/python3 generate_cutline_bezier.py \
        --in src/csv_data/test_worldv5.csv --out src/csv_data/test_worldv5_cut.csv
"""
import argparse
import numpy as np
from scipy.interpolate import CubicSpline


def load_centerline(path):
    rows = [ [float(v) for v in l.strip().split(';')]
             for l in open(path) if not l.startswith('#') and l.strip() ]
    a = np.array(rows)
    return a[:, 1], a[:, 2], a[:, 7], a[:, 8]   # x, y, w_tr_right, w_tr_left


def left_normals(x, y):
    N = len(x)
    dx = np.array([x[(i + 1) % N] - x[(i - 1) % N] for i in range(N)])
    dy = np.array([y[(i + 1) % N] - y[(i - 1) % N] for i in range(N)])
    t = np.stack([dx, dy], 1)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.stack([-t[:, 1], t[:, 0]], 1)      # left normal (-ty, tx)


def arc_psi_kappa(x, y):
    N = len(x)
    dx = np.array([x[(i + 1) % N] - x[i] for i in range(N)])
    dy = np.array([y[(i + 1) % N] - y[i] for i in range(N)])
    ds = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(ds)[:-1]])
    psi = np.unwrap(np.arctan2(dy, dx))
    dpsi = np.array([np.angle(np.exp(1j * (psi[(i + 1) % N] - psi[(i - 1) % N])))
                     for i in range(N)])
    seg = np.array([ds[i] + ds[(i - 1) % N] for i in range(N)])
    return s, ds, psi, dpsi / seg


def detect_corners(kappa, kt, ds_mean):
    """Return [(apex_idx, half_len_idx, sign)] for each high-curvature run."""
    N = len(kappa)
    mask = np.abs(kappa) > kt
    if not mask.any():
        return []
    corners = []
    starts = [s for s in range(N) if mask[s] and not mask[(s - 1) % N]]
    for s in starts:
        run, e = [], s
        for _ in range(N):
            if not mask[e % N]:
                break
            run.append(e % N)
            e += 1
        apex = run[int(np.argmax([abs(kappa[k]) for k in run]))]
        corners.append((apex, max(len(run) // 2, 1), int(np.sign(kappa[apex]))))
    return corners


def build_control_points(x0, y0, wr, wl, n0, corners, margin,
                         fa, fo, straight_step_idx, pad_idx, ds_mean):
    """Return ordered (index, offset) control anchors around the loop."""
    N = len(x0)
    # mark indices influenced by a corner so straights skip them
    corner_mask = np.zeros(N, bool)
    anchors = {}   # idx -> offset (keep the larger-magnitude if clash)

    def put(idx, off):
        idx %= N
        off = float(np.clip(off, -(wr[idx] - margin), (wl[idx] - margin)))
        if idx not in anchors or abs(off) > abs(anchors[idx]):
            anchors[idx] = off

    for (m, h, k) in corners:
        inner = max((wl[m] if k > 0 else wr[m]) - margin, 0.0)
        outer = max((wr[m] if k > 0 else wl[m]) - margin, 0.0)
        e_idx, x_idx = m - h - pad_idx, m + h + pad_idx
        put(m, k * inner * fa)            # apex -> inner wall
        put(e_idx, -k * outer * fo)       # entry -> outer wall
        put(x_idx, -k * outer * fo)       # exit  -> outer wall
        for d in range(-h - pad_idx, h + pad_idx + 1):
            corner_mask[(m + d) % N] = True

    # sparse straight control points on the centerline
    i = 0
    while i < N:
        if not corner_mask[i]:
            put(i, 0.0)
            i += straight_step_idx
        else:
            i += 1

    idxs = sorted(anchors)
    return idxs, np.array([anchors[i] for i in idxs])


def spline_line(x0, y0, n0, idxs, offs, N):
    """Fit a periodic C2 cubic spline through the control points and resample N."""
    cx = x0[idxs] + offs * n0[idxs, 0]
    cy = y0[idxs] + offs * n0[idxs, 1]
    # close the loop for periodic bc
    cxp = np.append(cx, cx[0])
    cyp = np.append(cy, cy[0])
    chord = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(cxp), np.diff(cyp)))])
    u = chord / chord[-1]
    sx = CubicSpline(u, cxp, bc_type='periodic')
    sy = CubicSpline(u, cyp, bc_type='periodic')
    uu = np.linspace(0.0, 1.0, N, endpoint=False)
    return sx(uu), sy(uu)


def velocity_profile(kappa, ds, a_lat, v_max, a_acc, a_brk):
    N = len(kappa)
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-4)))
    for _ in range(3):
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
    ap.add_argument('--margin', type=float, default=0.25)
    ap.add_argument('--wheelbase', type=float, default=0.33)
    ap.add_argument('--max-steer', type=float, default=0.4189)
    ap.add_argument('--kappa-thresh', type=float, default=0.25)
    ap.add_argument('--straight-step', type=float, default=2.5, help='straight ctrl-pt spacing [m]')
    ap.add_argument('--corner-pad', type=float, default=0.7, help='entry/exit ctrl-pt offset from corner [m]')
    ap.add_argument('--a-lat', type=float, default=7.0)
    ap.add_argument('--v-max', type=float, default=8.5)
    ap.add_argument('--a-acc', type=float, default=3.0)
    ap.add_argument('--a-brk', type=float, default=5.0)
    args = ap.parse_args()

    x0, y0, wr, wl = load_centerline(args.inp)
    N = len(x0)
    n0 = left_normals(x0, y0)
    s0, ds0, _, kap0 = arc_psi_kappa(x0, y0)
    ds_mean = float(np.mean(ds0))
    kmax = np.tan(args.max_steer) / args.wheelbase
    straight_step_idx = max(int(round(args.straight_step / ds_mean)), 2)
    pad_idx = max(int(round(args.corner_pad / ds_mean)), 2)

    corners = detect_corners(kap0, args.kappa_thresh, ds_mean)
    print(f'{N} pts | car min radius {1/kmax:.3f} m (|kappa|<={kmax:.3f})')
    print(f'detected {len(corners)} corners at wp {[m for m, _, _ in corners]}')

    fa = {i: 0.9 for i in range(len(corners))}   # apex inner fraction
    fo = {i: 0.6 for i in range(len(corners))}   # entry/exit outer fraction
    x = y = kappa = None
    n_ctrl = 0
    for it in range(14):
        # rebuild with per-corner fractions
        idxs, offs = [], []
        anchors = {}

        def put(idx, off):
            idx %= N
            off = float(np.clip(off, -(wr[idx] - args.margin), (wl[idx] - args.margin)))
            if idx not in anchors or abs(off) > abs(anchors[idx]):
                anchors[idx] = off
        corner_mask = np.zeros(N, bool)
        for ci, (m, h, k) in enumerate(corners):
            inner = max((wl[m] if k > 0 else wr[m]) - args.margin, 0.0)
            outer = max((wr[m] if k > 0 else wl[m]) - args.margin, 0.0)
            put(m, k * inner * fa[ci])
            put(m - h - pad_idx, -k * outer * fo[ci])
            put(m + h + pad_idx, -k * outer * fo[ci])
            for d in range(-h - pad_idx, h + pad_idx + 1):
                corner_mask[(m + d) % N] = True
        i = 0
        while i < N:
            if not corner_mask[i]:
                put(i, 0.0); i += straight_step_idx
            else:
                i += 1
        idxs = sorted(anchors); offs = np.array([anchors[j] for j in idxs])
        n_ctrl = len(idxs)

        x, y = spline_line(x0, y0, n0, np.array(idxs), offs, N)
        _, _, _, kappa = arc_psi_kappa(x, y)
        over = []
        for ci, (m, h, k) in enumerate(corners):
            win = [(m + d) % N for d in range(-h - pad_idx - 3, h + pad_idx + 4)]
            if max(abs(kappa[j]) for j in win) > kmax * 0.95:
                over.append(ci); fa[ci] = min(fa[ci] + 0.08, 1.0); fo[ci] = min(fo[ci] + 0.08, 1.0)
        if not over:
            print(f'  iter {it}: {n_ctrl} control pts, all drivable, max|kappa|={np.max(np.abs(kappa)):.3f}')
            break
        print(f'  iter {it}: {n_ctrl} ctrl pts, {len(over)} corner(s) over -> deepen, '
              f'max|kappa|={np.max(np.abs(kappa)):.3f}')

    s, ds, psi, kappa = arc_psi_kappa(x, y)
    v, ax = velocity_profile(kappa, ds, args.a_lat, args.v_max, args.a_acc, args.a_brk)
    off = (x - x0) * n0[:, 0] + (y - y0) * n0[:, 1]
    new_wr, new_wl = wr + off, wl - off

    hdr = '# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2; w_tr_right_m; w_tr_left_m'
    with open(args.out, 'w') as f:
        f.write(hdr + '\n')
        for i in range(N):
            f.write(f'{s[i]:.7f};{x[i]:.7f};{y[i]:.7f};{psi[i]:.7f};{kappa[i]:.7f};'
                    f'{v[i]:.7f};{ax[i]:.7f};{new_wr[i]:.7f};{new_wl[i]:.7f}\n')

    length = s[-1] + ds[-1]
    print(f'\nCENTERLINE : max|kappa|={np.max(np.abs(kap0)):.3f}  '
          f'undrivable={int(np.sum(np.abs(kap0) > kmax))}  len={s0[-1]+ds0[-1]:.2f} m')
    print(f'CUT LINE   : {n_ctrl} control pts  max|kappa|={np.max(np.abs(kappa)):.3f}  '
          f'undrivable={int(np.sum(np.abs(kappa) > kmax))}  '
          f'vx[{v.min():.2f},{v.max():.2f}] mean_v={v.mean():.2f}  '
          f'min wall clr={min(new_wr.min(), new_wl.min()):.2f} m  len={length:.2f} m')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
