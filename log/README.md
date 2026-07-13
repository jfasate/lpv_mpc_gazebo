# lpv_mpc_gazebo run logs

Each run overwrites a single `run.csv` here (created by `lpv_mpc_node` when
`enable_csv_log` is true; the launch file points `log_dir` at this folder). One
row per control tick (~50 Hz). It's always the same path, so no timestamp hunting.

## Columns

| column | meaning |
|--------|---------|
| `wall_t`, `sim_t` | seconds since node start (wall clock) and ROS sim clock |
| `iter`, `lap`, `wp_idx` | control-tick counter, lap number, nearest waypoint index |
| `status` | `ok` normal, `startup` (v<1.5 open-loop), `qp_infeasible`, `qp_exception` |
| `x_dot`,`y_dot`,`psi`,`psi_dot`,`X`,`Y` | measured state (body-frame vels, heading, yaw-rate, position) |
| `slip_deg` | body slip angle atan2(y_dot,x_dot) — spikes = losing grip |
| `ref_vx`,`ref_psi`,`ref_X`,`ref_Y` | reference (first horizon step) fed to the MPC |
| `err_v`,`err_psi_deg`,`err_X`,`err_Y` | measured − reference |
| `pos_err`,`lat_err`,`lon_err` | position error magnitude; cross-track and along-track in ref frame |
| `sq_err_v`,`sq_pos_err` | per-tick squared errors |
| `mse_v`,`mse_pos` | running mean-square error over `ok` rows |
| `steer_deg`,`accel` | applied control (U1, U2) |
| `du_steer`,`du_accel` | first MPC step deltas |
| `speed_cmd`,`ref_speed_ff`,`mpc_speed` | published velocity setpoint and its two components |
| `t_build_ms`,`t_solve_ms` | per-tick linearize/build and QP solve times |

On shutdown the node prints a summary line with final `RMSE_v` and `RMSE_pos`.

## Quick look
```bash
column -s, -t < run_*.csv | less -S      # tabular view
```
Only `README.md` is kept in git; the `run_*.csv` files are gitignored.
