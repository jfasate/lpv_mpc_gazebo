#!/usr/bin/env python3
"""
LPV-MPC (Linear Parameter Varying Model Predictive Control) node for F1Tenth.

Subscribes to odometry, runs a dynamic bicycle-model MPC (adapted from
autonomous360), and publishes AckermannDriveStamped commands to follow
waypoints loaded from a CSV file.
"""

import csv
import math
import os
import time

# Cap BLAS/OpenMP threads BEFORE numpy is imported. The MPC's matrices are
# small (~60x60), so multi-threaded OpenBLAS gives no speedup but spin-waits
# across every core (~1200% CPU observed), starving the sim's plant loop and
# making the car stutter in Gazebo. Must be set before numpy loads its BLAS.
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
from qpsolvers import solve_qp

# OSQP + scipy.sparse are used for the persistent, warm-started solver path.
# Fall back to qpsolvers.solve_qp if either is unavailable.
try:
    import osqp
    from scipy import sparse
    _OSQP_AVAILABLE = True
except ImportError:
    _OSQP_AVAILABLE = False

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from lpv_mpc_gazebo.support_files import SupportFilesF1Tenth
from lpv_mpc_gazebo.utils import (
    nearest_point, nearest_point_windowed, precompute_segments)


class LPVMPCNode(Node):

    def __init__(self):
        super().__init__('lpv_mpc_node')

        # ── ROS parameters ────────────���─────────────────────────────
        # Single knob to choose the reference line (see lpv_mpc_params.yaml).
        # Bare name -> src/csv_data/<name>.csv; a value with '/' is used as a path.
        self.declare_parameter('reference_csv', 'test_worldv5_optimize')
        self.declare_parameter('speed_scale', 1.0)
        self.declare_parameter('speed_ff_blend', 0.7)
        self.declare_parameter('adaptive_speed_ff_enabled', True)
        self.declare_parameter('adaptive_speed_ff_blend_max', 0.40)
        self.declare_parameter('adaptive_speed_ff_risk_start', 0.60)
        self.declare_parameter('adaptive_speed_ff_slip_deg', 5.0)
        self.declare_parameter('adaptive_speed_ff_clearance_soft', 1.30)
        self.declare_parameter('adaptive_speed_ff_clearance_hard', 0.90)
        self.declare_parameter('cmd_accel_horizon', 0.15)
        # Brake-lookahead: command the max speed that can still decelerate to
        # every upcoming reference speed within lookahead_time, using
        # lookahead_decel as the assumed braking capability. Prevents arriving
        # at a corner too hot (the trigger of the lap-1 spin-out).
        self.declare_parameter('speed_lookahead_time', 1.2)
        self.declare_parameter('speed_lookahead_decel', 4.0)
        self.declare_parameter('recovery_lat_err', 0.8)
        self.declare_parameter('recovery_heading_err_deg', 30.0)
        self.declare_parameter('recovery_hard_lat_err', 2.0)
        self.declare_parameter('recovery_hard_heading_err_deg', 60.0)
        self.declare_parameter('recovery_soft_speed_cap', 4.8)
        self.declare_parameter('recovery_speed_cap', 3.4)
        self.declare_parameter('recovery_soft_accel_cap', 1.0)
        self.declare_parameter('recovery_accel_cap', 0.0)
        self.declare_parameter('predictive_recovery_lat_growth', 0.85)
        self.declare_parameter('predictive_recovery_lat_end', 1.00)
        self.declare_parameter('predictive_recovery_heading_err_deg', 45.0)
        self.declare_parameter('predictive_recovery_hard_lat_growth', 2.50)
        self.declare_parameter('predictive_recovery_hard_lat_end', 3.00)
        self.declare_parameter('predictive_recovery_hard_heading_err_deg', 85.0)
        # Per-run CSV debug log. log_dir defaults to the source-tree log/ folder
        # so the recorded runs are easy to inspect after the fact.
        self.declare_parameter('enable_csv_log', True)
        self.declare_parameter(
            'log_dir',
            os.path.expanduser('~/sim_gazebo/src/lpv_mpc_gazebo/log'))
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('scan_front_angle_deg', 0.0)
        self.declare_parameter('scan_left_angle_deg', 90.0)
        self.declare_parameter('scan_right_angle_deg', -90.0)
        self.declare_parameter('scan_sector_width_deg', 10.0)
        self.declare_parameter('wall_avoidance_enabled', True)
        self.declare_parameter('wall_soft_clearance', 0.75)
        self.declare_parameter('wall_hard_clearance', 0.40)
        self.declare_parameter('wall_front_soft_clearance', 1.50)
        self.declare_parameter('wall_front_hard_clearance', 0.60)
        self.declare_parameter('wall_soft_speed_cap', 4.8)
        self.declare_parameter('wall_hard_speed_cap', 2.0)
        self.declare_parameter('wall_accel_cap', 0.0)
        self.declare_parameter('wall_brake_gain', 1.0)
        self.declare_parameter('wall_scan_max_age', 0.20)
        self.declare_parameter('obstacle_avoidance_enabled', True)
        self.declare_parameter('obstacle_path_aware_enabled', True)
        self.declare_parameter('obstacle_forward_angle_deg', 55.0)
        self.declare_parameter('obstacle_corridor_width', 0.75)
        self.declare_parameter('obstacle_soft_distance', 3.00)
        self.declare_parameter('obstacle_hard_distance', 0.90)
        self.declare_parameter('obstacle_soft_speed_cap', 6.8)
        self.declare_parameter('obstacle_hard_speed_cap', 1.5)
        self.declare_parameter('obstacle_ttc_soft', 1.35)
        self.declare_parameter('obstacle_ttc_hard', 0.75)
        self.declare_parameter('obstacle_brake_gain', 0.6)
        self.declare_parameter('config_file', '')
        self.declare_parameter('Ts', 0.02)
        self.declare_parameter('hz', 10)
        self.declare_parameter('m', 3.47)
        self.declare_parameter('Iz', 0.04712)
        self.declare_parameter('Cf', 90.0)
        self.declare_parameter('Cr', 110.0)
        self.declare_parameter('lf', 0.15875)
        self.declare_parameter('lr', 0.17145)
        self.declare_parameter('mju', 0.015)
        self.declare_parameter('steer_rate_limit', 3.2)
        self.declare_parameter('d_a_max', 0.5)
        self.declare_parameter('Q_diag', [10.0, 500.0, 100.0, 100.0])
        self.declare_parameter('S_diag', [10.0, 500.0, 100.0, 100.0])
        self.declare_parameter('R_diag', [50.0, 5.0])
        self.declare_parameter('qp_solver', 'cvxopt')
        # Soft state constraints: add slack variables to the state-limit rows so
        # the QP is ALWAYS feasible (never returns None -> never freezes the
        # steering mid-slide). Slack is heavily penalised so limits are only
        # violated when the hard problem would otherwise be infeasible.
        self.declare_parameter('soft_constraints', True)
        self.declare_parameter('slack_penalty_lin', 1.0e4)   # L1 weight on slack
        self.declare_parameter('slack_penalty_quad', 1.0e2)  # L2 weight on slack

        reference_csv = self.get_parameter('reference_csv').value
        self.speed_scale = self.get_parameter('speed_scale').value
        self.speed_ff_blend = self.get_parameter('speed_ff_blend').value
        self.adaptive_speed_ff_enabled = self.get_parameter(
            'adaptive_speed_ff_enabled').value
        self.adaptive_speed_ff_blend_max = self.get_parameter(
            'adaptive_speed_ff_blend_max').value
        self.adaptive_speed_ff_risk_start = self.get_parameter(
            'adaptive_speed_ff_risk_start').value
        self.adaptive_speed_ff_slip = math.radians(
            self.get_parameter('adaptive_speed_ff_slip_deg').value)
        self.adaptive_speed_ff_clearance_soft = self.get_parameter(
            'adaptive_speed_ff_clearance_soft').value
        self.adaptive_speed_ff_clearance_hard = self.get_parameter(
            'adaptive_speed_ff_clearance_hard').value
        self.cmd_accel_horizon = self.get_parameter('cmd_accel_horizon').value
        self.speed_lookahead_time = self.get_parameter('speed_lookahead_time').value
        self.speed_lookahead_decel = self.get_parameter('speed_lookahead_decel').value
        self.recovery_lat_err = self.get_parameter('recovery_lat_err').value
        self.recovery_heading_err = math.radians(
            self.get_parameter('recovery_heading_err_deg').value)
        self.recovery_hard_lat_err = self.get_parameter('recovery_hard_lat_err').value
        self.recovery_hard_heading_err = math.radians(
            self.get_parameter('recovery_hard_heading_err_deg').value)
        self.recovery_soft_speed_cap = self.get_parameter('recovery_soft_speed_cap').value
        self.recovery_speed_cap = self.get_parameter('recovery_speed_cap').value
        self.recovery_soft_accel_cap = self.get_parameter('recovery_soft_accel_cap').value
        self.recovery_accel_cap = self.get_parameter('recovery_accel_cap').value
        self.predictive_recovery_lat_growth = self.get_parameter(
            'predictive_recovery_lat_growth').value
        self.predictive_recovery_lat_end = self.get_parameter(
            'predictive_recovery_lat_end').value
        self.predictive_recovery_heading_err = math.radians(
            self.get_parameter('predictive_recovery_heading_err_deg').value)
        self.predictive_recovery_hard_lat_growth = self.get_parameter(
            'predictive_recovery_hard_lat_growth').value
        self.predictive_recovery_hard_lat_end = self.get_parameter(
            'predictive_recovery_hard_lat_end').value
        self.predictive_recovery_hard_heading_err = math.radians(
            self.get_parameter('predictive_recovery_hard_heading_err_deg').value)
        self.enable_csv_log = self.get_parameter('enable_csv_log').value
        self.log_dir = self.get_parameter('log_dir').value
        odom_topic = self.get_parameter('odom_topic').value
        scan_topic = self.get_parameter('scan_topic').value
        drive_topic = self.get_parameter('drive_topic').value
        self.scan_front_angle = math.radians(
            self.get_parameter('scan_front_angle_deg').value)
        self.scan_left_angle = math.radians(
            self.get_parameter('scan_left_angle_deg').value)
        self.scan_right_angle = math.radians(
            self.get_parameter('scan_right_angle_deg').value)
        self.scan_sector_width = math.radians(
            self.get_parameter('scan_sector_width_deg').value)
        self.wall_avoidance_enabled = self.get_parameter('wall_avoidance_enabled').value
        self.wall_soft_clearance = self.get_parameter('wall_soft_clearance').value
        self.wall_hard_clearance = self.get_parameter('wall_hard_clearance').value
        self.wall_front_soft_clearance = self.get_parameter(
            'wall_front_soft_clearance').value
        self.wall_front_hard_clearance = self.get_parameter(
            'wall_front_hard_clearance').value
        self.wall_soft_speed_cap = self.get_parameter('wall_soft_speed_cap').value
        self.wall_hard_speed_cap = self.get_parameter('wall_hard_speed_cap').value
        self.wall_accel_cap = self.get_parameter('wall_accel_cap').value
        self.wall_brake_gain = self.get_parameter('wall_brake_gain').value
        self.wall_scan_max_age = self.get_parameter('wall_scan_max_age').value
        self.obstacle_avoidance_enabled = self.get_parameter(
            'obstacle_avoidance_enabled').value
        self.obstacle_path_aware_enabled = self.get_parameter(
            'obstacle_path_aware_enabled').value
        self.obstacle_forward_angle = math.radians(
            self.get_parameter('obstacle_forward_angle_deg').value)
        self.obstacle_corridor_width = self.get_parameter(
            'obstacle_corridor_width').value
        self.obstacle_soft_distance = self.get_parameter(
            'obstacle_soft_distance').value
        self.obstacle_hard_distance = self.get_parameter(
            'obstacle_hard_distance').value
        self.obstacle_soft_speed_cap = self.get_parameter(
            'obstacle_soft_speed_cap').value
        self.obstacle_hard_speed_cap = self.get_parameter(
            'obstacle_hard_speed_cap').value
        self.obstacle_ttc_soft = self.get_parameter('obstacle_ttc_soft').value
        self.obstacle_ttc_hard = self.get_parameter('obstacle_ttc_hard').value
        self.obstacle_brake_gain = self.get_parameter('obstacle_brake_gain').value
        config_file = self.get_parameter('config_file').value
        self.qp_solver = self.get_parameter('qp_solver').value
        self.soft_constraints = self.get_parameter('soft_constraints').value
        self.slack_rho = self.get_parameter('slack_penalty_lin').value
        self.slack_mu = self.get_parameter('slack_penalty_quad').value

        Ts = self.get_parameter('Ts').value
        hz = self.get_parameter('hz').value

        # Build support-class params from ROS parameters
        support_params = {
            'Ts': Ts, 'hz': hz,
            'm': self.get_parameter('m').value,
            'Iz': self.get_parameter('Iz').value,
            'Cf': self.get_parameter('Cf').value,
            'Cr': self.get_parameter('Cr').value,
            'lf': self.get_parameter('lf').value,
            'lr': self.get_parameter('lr').value,
            'mju': self.get_parameter('mju').value,
            'steer_rate_limit': self.get_parameter('steer_rate_limit').value,
            'd_a_max': self.get_parameter('d_a_max').value,
            'Q_diag': list(self.get_parameter('Q_diag').value),
            'S_diag': list(self.get_parameter('S_diag').value),
            'R_diag': list(self.get_parameter('R_diag').value),
        }

        # ── Support class (vehicle model + MPC matrices) ───────────
        self.support = SupportFilesF1Tenth(support_params)
        self.constants = self.support.constants
        self.Ts = self.constants['Ts']
        self.hz = self.constants['hz']
        self.inputs = self.constants['inputs']
        self.outputs = self.constants['outputs']

        # ── Load waypoints ──────────────────────────────────────────
        # reference_csv is the SINGLE reference-line selector. A bare name
        # resolves to src/csv_data/<name>.csv; a value containing '/' (or an
        # absolute/~ path) is used directly.
        ref = str(reference_csv)
        if '/' in ref:
            csv_file = os.path.abspath(os.path.expanduser(ref))
        else:
            if not ref.endswith('.csv'):
                ref += '.csv'
            csv_file = os.path.join(
                os.path.abspath(os.path.join('src', 'csv_data')), ref)
        self.get_logger().info(f'>>> REFERENCE LINE (reference_csv): {csv_file}')

        # CSV columns: s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2; ...
        self.waypoints = np.loadtxt(csv_file, delimiter=';', skiprows=0)

        # Drop duplicate points that would spike the geometry-derived heading:
        #   - interior consecutive duplicates
        #   - the closed-loop closure duplicate (last point repeats the first)
        _d = np.linalg.norm(np.diff(self.waypoints[:, 1:3], axis=0), axis=1)
        self.waypoints = self.waypoints[np.concatenate([[True], _d > 1e-6])]
        if np.linalg.norm(self.waypoints[0, 1:3] - self.waypoints[-1, 1:3]) < 1e-6:
            self.waypoints = self.waypoints[:-1]

        # Auto-orient the line to the car's driving direction (counter-clockwise,
        # matching the centerline). Some raceline exporters wind the opposite way;
        # following such a line would make the car drive it in reverse (heading
        # reference points backwards -> instant spin). Detect via the signed
        # (shoelace) area and reverse point order + flip heading by pi if needed.
        _x, _y = self.waypoints[:, 1], self.waypoints[:, 2]
        signed_area = 0.5 * np.sum(_x * np.roll(_y, -1) - np.roll(_x, -1) * _y)
        self._reference_reversed = signed_area < 0.0
        if self._reference_reversed:
            self.waypoints = self.waypoints[::-1].copy()
            self.get_logger().warn(
                'Reference line is clockwise; reversed it to match the CCW '
                'driving direction (order + heading flipped).')

        self.n_waypoints = self.waypoints.shape[0]
        self.wp_xy = self.waypoints[:, 1:3]            # (N, 2) for nearest-point
        self.wp_vx = self.waypoints[:, 5].copy()        # speed [m/s]

        # Heading reference is recomputed from the point geometry (heading toward
        # the next waypoint) rather than the file's psi column, which some
        # exporters populate inconsistently — an unreliable psi makes the car
        # fight a wrong heading reference. Computing it here guarantees psi always
        # matches the travel direction of the (already CCW-oriented) points.
        _dx = np.roll(self.wp_xy[:, 0], -1) - self.wp_xy[:, 0]
        _dy = np.roll(self.wp_xy[:, 1], -1) - self.wp_xy[:, 1]
        self.wp_psi = np.unwrap(np.arctan2(_dy, _dx))

        # Arc-length + spacing recomputed from geometry (robust to reversal and
        # to any 's' column convention in the file).
        _seg = np.linalg.norm(np.diff(self.wp_xy, axis=0), axis=1)
        self.wp_s = np.concatenate([[0.0], np.cumsum(_seg)])
        self.ds = float(np.mean(_seg))

        # ── Nearest-point: precomputed segments + local windowed search ──
        # Build the closed-loop segment vectors/lengths once (instead of
        # recomputing them every tick), and search only a local window around
        # the previous index. At 50 Hz the car advances < 1 waypoint per tick,
        # so a window of a few tens of segments always contains the true
        # nearest point while scanning ~15x fewer candidates than the full loop.
        self.wp_diffs, self.wp_l2s = precompute_segments(self.wp_xy)
        self.nn_back = 20            # segments to look behind the seed
        self.nn_fwd = 60            # segments to look ahead of the seed
        self.nn_idx = 0             # rolling seed index (last nearest segment)
        self.nn_seeded = False      # first tick does one global scan to seed

        # Minimum reference speed: ensure scaled speeds stay above the
        # dynamic model's stability threshold (1.5 m/s) with some margin.
        # Derived from CSV data so it adapts to any track / speed_scale.
        self.min_ref_speed = max(self.wp_vx.min() * self.speed_scale, 2.0)

        self.get_logger().info(
            f'Loaded {self.n_waypoints} waypoints, avg spacing={self.ds:.4f} m, '
            f'speed range [{self.wp_vx.min():.1f}, {self.wp_vx.max():.1f}] m/s, '
            f'min_ref_speed={self.min_ref_speed:.2f} m/s')

        # ── MPC state ────────────────────────────────────────────��─
        self.states = np.zeros(6)   # [x_dot, y_dot, psi, psi_dot, X, Y]
        self.U1 = 0.0               # current steering angle (delta)
        self.U2 = 0.0               # current acceleration (a)
        self.du = np.zeros((self.inputs * self.hz, 1))
        self.state_received = False
        self.scan_received = False
        self.last_scan_time = None
        self.scan_metrics = {
            'front': float('nan'),
            'left': float('nan'),
            'right': float('nan'),
            'front_min': float('nan'),
            'left_min': float('nan'),
            'right_min': float('nan'),
            'min': float('nan'),
            'age': float('nan'),
            'obstacle_front': float('nan'),
            'obstacle_x': float('nan'),
            'obstacle_y': float('nan'),
            'obstacle_bearing': float('nan'),
        }
        self._last_wall_clearance = float('nan')
        self._last_wall_speed_cap = float('nan')
        self._last_wall_limit_active = False
        self._last_wall_limit_hard = False
        self._last_wall_source = ''
        self._last_speed_ff_blend = self.speed_ff_blend
        self.iteration = 0

        # ── Warm-started OSQP solver state ──────────────────────────
        # A single OSQP object is created once and then updated in place each
        # tick (same sparsity pattern → reuse symbolic factorization) and
        # warm-started from the previous du solution, instead of building a
        # fresh problem every call via qpsolvers.
        self._use_osqp = _OSQP_AVAILABLE and self.qp_solver == 'osqp'
        self._osqp_prob = None
        self._osqp_pattern = None   # (P.nnz, A.nnz, m) — re-setup if it changes
        self._z_warm = None         # warm-start vector for the augmented [du; slack]
        self._last_slack = 0.0      # max state-constraint violation last solve
        # Input-rate box rows (hard); the remaining constraint rows are the state
        # limits that get softened. G = vstack(I_mega_global, state_constraints),
        # I_mega_global has 2*inputs*hz rows.
        self._n_hard_rows = 2 * self.inputs * self.hz

        # ── Profiling accumulators (per-stage timing, averaged over a window) ──
        self._prof_build = 0.0    # state_space + mpc_simplification
        self._prof_reg = 0.0      # Hessian symmetrize + eigvalsh regularization
        self._prof_solve = 0.0    # QP solve
        self._prof_total = 0.0    # whole control_loop
        self._prof_count = 0

        # Lap timing
        self.prev_wp_idx = 0
        self.nr_laps = 0
        self.lap_start_time = None
        self.lap_crossed_half = False

        # ── ROS pub/sub ───────────────���──────────��──────────────────
        # The Gazebo diff_drive plugin publishes odom as BEST_EFFORT (matching
        # pose_reset.py / sysid_excite.py). The subscription must use the same
        # reliability or QoS is incompatible and no odom is ever delivered.
        odom_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, odom_qos)
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, odom_qos)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 1)

        self.vis_waypoints_pub = self.create_publisher(Marker, '/lpv_mpc_gazebo/waypoints', 1)
        self.vis_ref_pub = self.create_publisher(Marker, '/lpv_mpc_gazebo/ref_traj', 1)
        self.vis_pred_pub = self.create_publisher(Marker, '/lpv_mpc_gazebo/pred_path', 1)

        # Publish waypoints once
        self._publish_waypoints_marker()

        # Control timer at 1/Ts Hz
        self.control_timer = self.create_timer(self.Ts, self.control_loop)

        self.get_logger().info(
            f'LPV-MPC node started  |  Ts={self.Ts}s  hz={self.hz}  '
            f'solver={self.qp_solver}  disc=rk4  '
            f'speed_scale={self.speed_scale}  '
            f'ff_blend={self.speed_ff_blend}  cmd_horizon={self.cmd_accel_horizon}s')
        self.get_logger().info(f'LPV-MPC code file: {__file__}')
        self.get_logger().info(f'LPV-MPC config file: {config_file}')
        self.get_logger().info(
            'RViz topics: /lpv_mpc_gazebo/waypoints, '
            '/lpv_mpc_gazebo/ref_traj, /lpv_mpc_gazebo/pred_path')

        # ── Per-run CSV debug log ───────────────────────────────────
        self._init_csv_log()

    # ───────────��──────────────────────────────��─────────────────────
    #  Odometry callback — extract state from sim
    # ────────────────────────────────────────────────────────────────
    def odom_callback(self, msg):
        """Store latest vehicle state from the simulator odometry."""
        pose = msg.pose.pose
        twist = msg.twist.twist

        X = pose.position.x
        Y = pose.position.y

        # Quaternion → yaw
        q = pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        # Body-frame velocities (the sim already provides these in body frame)
        x_dot = twist.linear.x
        y_dot = twist.linear.y
        psi_dot = twist.angular.z

        self.states = np.array([x_dot, y_dot, yaw, psi_dot, X, Y])
        self.state_received = True

    def scan_callback(self, msg):
        """Store wall-clearance metrics from the latest LaserScan."""
        ranges = np.asarray(msg.ranges, dtype=float)
        valid = (
            np.isfinite(ranges)
            & (ranges >= msg.range_min)
            & (ranges <= msg.range_max))

        def beam_at(angle):
            if angle < msg.angle_min or angle > msg.angle_max:
                return float('nan')
            idx = int(round((angle - msg.angle_min) / msg.angle_increment))
            if idx < 0 or idx >= ranges.size or not valid[idx]:
                return float('nan')
            return float(ranges[idx])

        def sector_min(angle):
            half_width = 0.5 * self.scan_sector_width
            lo = max(angle - half_width, msg.angle_min)
            hi = min(angle + half_width, msg.angle_max)
            if lo > hi:
                return float('nan')
            i0 = max(0, int(math.floor((lo - msg.angle_min) / msg.angle_increment)))
            i1 = min(ranges.size - 1,
                     int(math.ceil((hi - msg.angle_min) / msg.angle_increment)))
            sector = ranges[i0:i1 + 1]
            sector_valid = valid[i0:i1 + 1]
            if not np.any(sector_valid):
                return float('nan')
            return float(np.min(sector[sector_valid]))

        if np.any(valid):
            scan_min = float(np.min(ranges[valid]))
        else:
            scan_min = float('nan')

        angles = msg.angle_min + np.arange(ranges.size) * msg.angle_increment
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        corridor_half_width = 0.5 * self.obstacle_corridor_width
        obstacle_mask = (
            valid
            & (xs > 0.0)
            & (np.abs(ys) <= corridor_half_width)
            & (np.abs(angles) <= self.obstacle_forward_angle))
        if self.obstacle_avoidance_enabled and np.any(obstacle_mask):
            obstacle_idxs = np.flatnonzero(obstacle_mask)
            nearest_local_idx = int(np.argmin(xs[obstacle_idxs]))
            obstacle_idx = int(obstacle_idxs[nearest_local_idx])
            obstacle_front = float(ranges[obstacle_idx])
            obstacle_x = float(xs[obstacle_idx])
            obstacle_y = float(ys[obstacle_idx])
            obstacle_bearing = float(angles[obstacle_idx])
        else:
            obstacle_front = obstacle_x = obstacle_y = obstacle_bearing = float('nan')

        self.scan_metrics = {
            'front': beam_at(self.scan_front_angle),
            'left': beam_at(self.scan_left_angle),
            'right': beam_at(self.scan_right_angle),
            'front_min': sector_min(self.scan_front_angle),
            'left_min': sector_min(self.scan_left_angle),
            'right_min': sector_min(self.scan_right_angle),
            'min': scan_min,
            'age': 0.0,
            'obstacle_front': obstacle_front,
            'obstacle_x': obstacle_x,
            'obstacle_y': obstacle_y,
            'obstacle_bearing': obstacle_bearing,
        }
        self.last_scan_time = self.get_clock().now().nanoseconds * 1e-9
        self.scan_received = True

    def _scan_metrics_for_log(self):
        scan = dict(self.scan_metrics)
        if self.scan_received and self.last_scan_time is not None:
            now = self.get_clock().now().nanoseconds * 1e-9
            scan['age'] = max(0.0, now - self.last_scan_time)
        return scan

    def _inactive_wall_guard(self):
        return {
            'active': False,
            'hard': False,
            'clearance': float('nan'),
            'speed_cap': float('nan'),
            'source': '',
        }

    def _record_wall_guard(self, guard):
        self._last_wall_clearance = guard['clearance']
        self._last_wall_speed_cap = guard['speed_cap']
        self._last_wall_limit_active = guard['active']
        self._last_wall_limit_hard = guard['hard']
        self._last_wall_source = guard['source']

    def _compute_wall_guard(self, current_speed=0.0, path_heading=0.0):
        """LiDAR clearance speed guard, inspired by MPPI wall soft-hinge costs."""
        guard = self._inactive_wall_guard()
        if not self.wall_avoidance_enabled or not self.scan_received:
            return guard

        scan = self._scan_metrics_for_log()
        age = scan.get('age', float('nan'))
        if math.isfinite(age) and age > self.wall_scan_max_age:
            return guard

        candidates = []
        side_vals = [
            v for v in (scan.get('left_min'), scan.get('right_min'))
            if v is not None and math.isfinite(v)
        ]
        if side_vals:
            candidates.append((
                'side', min(side_vals),
                self.wall_soft_clearance, self.wall_hard_clearance,
                self.wall_soft_speed_cap, self.wall_hard_speed_cap))

        front = scan.get('front_min')
        if front is not None and math.isfinite(front):
            candidates.append((
                'front', front,
                self.wall_front_soft_clearance, self.wall_front_hard_clearance,
                self.wall_soft_speed_cap, self.wall_hard_speed_cap))

        obstacle_x = scan.get('obstacle_x')
        if (self.obstacle_avoidance_enabled
                and obstacle_x is not None and math.isfinite(obstacle_x)):
            obstacle_y = scan.get('obstacle_y')
            use_obstacle = True
            if (self.obstacle_path_aware_enabled
                    and obstacle_y is not None and math.isfinite(obstacle_y)
                    and math.isfinite(path_heading)):
                heading = float(np.clip(
                    path_heading, -self.obstacle_forward_angle,
                    self.obstacle_forward_angle))
                path_y = obstacle_x * math.tan(heading)
                path_lateral_error = obstacle_y - path_y
                corridor_half_width = 0.5 * self.obstacle_corridor_width
                use_obstacle = abs(path_lateral_error) <= corridor_half_width
            if use_obstacle:
                candidates.append((
                    'obstacle', obstacle_x,
                    self.obstacle_soft_distance, self.obstacle_hard_distance,
                    self.obstacle_soft_speed_cap, self.obstacle_hard_speed_cap))

        if not candidates:
            return guard

        scored = []
        for source, clearance, soft, hard, soft_cap, hard_cap in candidates:
            denom = max(soft - hard, 1e-3)
            distance_risk = float(np.clip((soft - clearance) / denom, 0.0, 1.0))
            risk = distance_risk
            if source == 'obstacle':
                ttc = clearance / max(float(current_speed), 0.1)
                ttc_denom = max(self.obstacle_ttc_soft - self.obstacle_ttc_hard, 1e-3)
                ttc_risk = float(np.clip(
                    (self.obstacle_ttc_soft - ttc) / ttc_denom, 0.0, 1.0))
                risk = max(distance_risk, ttc_risk)
            scored.append((risk, clearance, source, soft, hard, soft_cap, hard_cap))

        if any(risk > 0.0 for risk, *_ in scored):
            risk, clearance, source, soft, hard, soft_cap, hard_cap = max(
                scored, key=lambda item: (item[0], -item[1]))
        else:
            risk, clearance, source, soft, hard, soft_cap, hard_cap = min(
                scored, key=lambda item: item[1])
        guard['clearance'] = clearance
        guard['source'] = source
        if risk <= 0.0:
            return guard

        guard['speed_cap'] = (
            (1.0 - risk) * soft_cap
            + risk * hard_cap)
        guard['active'] = True
        guard['hard'] = clearance <= hard
        return guard

    def _apply_wall_guard_to_speed(self, speed_cmd, current_speed, guard):
        if not guard['active'] or not math.isfinite(guard['speed_cap']):
            return speed_cmd

        cap = guard['speed_cap']
        if guard['hard'] and current_speed > cap:
            brake_gain = (self.obstacle_brake_gain
                          if guard.get('source') == 'obstacle'
                          else self.wall_brake_gain)
            return min(speed_cmd, cap - brake_gain * (current_speed - cap))
        return min(speed_cmd, cap)

    def _adaptive_speed_ff_blend(self, lat_err, heading_err, pred_lat_growth,
                                 pred_lat_end, pred_heading_peak, slip,
                                 wall_guard, recovery_active,
                                 hard_recovery_active):
        if not self.adaptive_speed_ff_enabled:
            return float(self.speed_ff_blend)
        if hard_recovery_active:
            return float(self.speed_ff_blend)

        base = float(self.speed_ff_blend)
        max_blend = max(base, float(self.adaptive_speed_ff_blend_max))
        risk_start = float(np.clip(self.adaptive_speed_ff_risk_start, 0.0, 0.99))

        pred_risk = max(
            max(pred_lat_growth, 0.0) / max(self.predictive_recovery_lat_growth, 1e-3),
            pred_lat_end / max(self.predictive_recovery_lat_end, 1e-3),
            pred_heading_peak / max(self.predictive_recovery_heading_err, 1e-3))
        current_risk = max(
            abs(lat_err) / max(self.recovery_lat_err, 1e-3),
            abs(heading_err) / max(self.recovery_heading_err, 1e-3))
        slip_risk = abs(slip) / max(self.adaptive_speed_ff_slip, 1e-3)

        clearance_risk = 0.0
        clearances = []
        source = wall_guard.get('source', '')
        if source in ('side', 'front'):
            clearance = wall_guard.get('clearance', float('nan'))
            if math.isfinite(clearance):
                clearances.append(clearance)
        scan = self._scan_metrics_for_log()
        for key in ('min', 'left_min', 'right_min', 'front_min'):
            value = scan.get(key, float('nan'))
            if math.isfinite(value):
                clearances.append(value)
        if clearances:
            clearance = min(clearances)
            denom = max(self.adaptive_speed_ff_clearance_soft
                        - self.adaptive_speed_ff_clearance_hard, 1e-3)
            clearance_risk = float(np.clip(
                (self.adaptive_speed_ff_clearance_soft - clearance) / denom,
                0.0, 1.0))

        risk = max(pred_risk, current_risk, slip_risk, clearance_risk)
        if recovery_active:
            risk = max(risk, 1.0)
        if risk <= risk_start:
            scale = 1.0
        elif risk >= 1.0:
            scale = 0.0
        else:
            scale = (1.0 - risk) / max(1.0 - risk_start, 1e-3)
        return float(base + (max_blend - base) * scale)

    # ────────────────────────────────────────────────────────────────
    #  Main control loop (timer callback)
    # ───────────────────────────────────────────���────────────────────
    def control_loop(self):
        if not self.state_received:
            return

        t_loop_start = time.perf_counter()
        states = self.states.copy()
        self._record_wall_guard(self._inactive_wall_guard())

        # Ensure minimum forward velocity for the dynamic model.
        # The dynamic bicycle model is numerically unstable (forward-Euler)
        # below ~1.2 m/s for F1Tenth params.  Wait until the car is fast
        # enough, sending an open-loop speed command in the meantime.
        if states[0] < 1.5:
            self._publish_drive(self.U1, 2.0)
            self._log_row('startup', states, speed_cmd=2.0)
            return

        hz = self.hz  # local copy (stays constant for closed track)

        # ── 1. Find nearest waypoint (local windowed search) ────────
        point = np.array([states[4], states[5]])
        if not self.nn_seeded:
            # Seed the rolling index once with a full scan, then go local.
            _, _, _, seed = nearest_point(point, self.wp_xy)
            self.nn_idx = int(seed)
            self.nn_seeded = True
        _, _, _, wp_idx = nearest_point_windowed(
            point, self.wp_xy, self.wp_diffs, self.wp_l2s,
            self.nn_idx, self.nn_back, self.nn_fwd)
        wp_idx = int(wp_idx)
        self.nn_idx = wp_idx  # advance the rolling seed

        # ── Lap timing ──────────────────────────────────────────────
        self._update_lap_timing(wp_idx)

        # ── 2. Build reference vector for the horizon ──────────────
        r = self._build_reference(states, wp_idx, hz)

        # Brake-aware feedforward speed target (also used by the QP-failure
        # fallbacks below so they BRAKE toward the reference instead of
        # coasting at the current speed, which used to cancel braking exactly
        # at hard corner entries where the QP is most likely infeasible).
        ff_ref_speed = self._lookahead_ref_speed(wp_idx, states[0])
        path_heading = math.atan2(
            math.sin(float(r[1]) - states[2]),
            math.cos(float(r[1]) - states[2]))

        # ── 3. Linearize & build QP ───────────────────────────���────
        t_build_start = time.perf_counter()
        Ad, Bd, Cd, Dd = self.support.state_space(states, self.U1, self.U2)

        x_aug_t = np.array([[states[0]], [states[1]], [states[2]],
                            [states[3]], [states[4]], [states[5]],
                            [self.U1], [self.U2]])

        Hdb, Fdbt, Cdb, Adc, G, ht = self.support.mpc_simplification(
            Ad, Bd, Cd, Dd, hz, x_aug_t, self.du)

        ft = np.concatenate((x_aug_t.flatten(), r)) @ Fdbt
        t_build = time.perf_counter() - t_build_start

        # ── 4. Solve QP ────────────────────────────────────────────
        # Symmetrize and add a small fixed ridge for numerical robustness.
        # Hdb = Cdb.T Qdb Cdb + Rdb is PD by construction (Rdb is strictly
        # positive-definite), so the previous per-tick eigendecomposition was
        # pure overhead — a fixed ridge guarantees PD far more cheaply and
        # also keeps the sparsity pattern stable for the warm-started solver.
        t_reg_start = time.perf_counter()
        Hdb = 0.5 * (Hdb + Hdb.T)
        Hdb[np.diag_indices_from(Hdb)] += 1e-6
        t_reg = time.perf_counter() - t_reg_start

        t_solve_start = time.perf_counter()
        try:
            du_sol = self._solve_qp(Hdb, ft, G, ht)
            if du_sol is None:
                if self.iteration % 50 == 0:
                    # Check which constraints are infeasible at du=0
                    slack = ht - G @ np.zeros(G.shape[1])
                    n_violated = np.sum(slack < 0)
                    self.get_logger().warn(
                        f'QP infeasible: {n_violated} constraints violated at du=0  '
                        f'states={np.round(states, 3)}  U1={self.U1:.4f} U2={self.U2:.4f}  '
                        f'min_slack={slack.min():.4f}')
                # Brake toward the reference (do NOT coast at current speed).
                brake_cmd = min(states[0], ff_ref_speed)
                wall_guard = self._compute_wall_guard(states[0], path_heading)
                self._record_wall_guard(wall_guard)
                brake_cmd = self._apply_wall_guard_to_speed(
                    brake_cmd, states[0], wall_guard)
                self._publish_drive(self.U1, brake_cmd)
                self._log_row('qp_infeasible', states, wp_idx, r,
                              speed_cmd=brake_cmd, ref_speed=ff_ref_speed, t_build=t_build)
                return
            self.du = du_sol.reshape(-1, 1)
        except Exception as e:
            if self.iteration % 50 == 0:
                self.get_logger().warn(f'QP exception: {e}  states={np.round(states, 3)}')
            brake_cmd = min(states[0], ff_ref_speed)
            wall_guard = self._compute_wall_guard(states[0], path_heading)
            self._record_wall_guard(wall_guard)
            brake_cmd = self._apply_wall_guard_to_speed(
                brake_cmd, states[0], wall_guard)
            self._publish_drive(self.U1, brake_cmd)
            self._log_row('qp_exception', states, wp_idx, r,
                          speed_cmd=brake_cmd, ref_speed=ff_ref_speed, t_build=t_build)
            return
        t_solve = time.perf_counter() - t_solve_start

        # ── 5. Update inputs ──────────────────────────��─────────────
        self.U1 += self.du[0][0]   # steering angle
        self.U2 += self.du[1][0]   # acceleration

        # Clamp steering and acceleration to stay within constraint bounds.
        # Forward accel is additionally traction-limited at low speed (matching
        # the soft QP cap in support_files) so the car can't floor the throttle
        # from a standstill and spin the tires. Braking is not limited.
        max_steer = 0.4189
        self.U1 = np.clip(self.U1, -max_steer, max_steer)
        accel_cap = min(3.0, 0.5 + 0.6 * max(states[0], 1.5))
        self.U2 = np.clip(self.U2, -3.0, accel_cap)

        pred_aug = (Adc @ x_aug_t + Cdb @ self.du).flatten()

        ref_psi = float(r[1])
        err_X = states[4] - float(r[2])
        err_Y = states[5] - float(r[3])
        lat_err = -math.sin(ref_psi) * err_X + math.cos(ref_psi) * err_Y
        heading_err = math.atan2(
            math.sin(states[2] - ref_psi),
            math.cos(states[2] - ref_psi))
        recovery_active = (
            abs(lat_err) > self.recovery_lat_err
            or abs(heading_err) > self.recovery_heading_err)
        hard_recovery_active = (
            abs(lat_err) > self.recovery_hard_lat_err
            or abs(heading_err) > self.recovery_hard_heading_err)

        n_aug = 6 + self.inputs
        pred_lat = []
        pred_heading_err = []
        for k in range(hz):
            p0 = n_aug * k
            r0 = self.outputs * k
            pred_X = float(pred_aug[p0 + 4])
            pred_Y = float(pred_aug[p0 + 5])
            pred_psi = float(pred_aug[p0 + 2])
            ref_psi_k = float(r[r0 + 1])
            err_X_k = pred_X - float(r[r0 + 2])
            err_Y_k = pred_Y - float(r[r0 + 3])
            pred_lat.append(
                -math.sin(ref_psi_k) * err_X_k
                + math.cos(ref_psi_k) * err_Y_k)
            pred_heading_err.append(math.atan2(
                math.sin(pred_psi - ref_psi_k),
                math.cos(pred_psi - ref_psi_k)))
        pred_lat_start = abs(pred_lat[0]) if pred_lat else abs(lat_err)
        pred_lat_end = abs(pred_lat[-1]) if pred_lat else abs(lat_err)
        pred_lat_growth = pred_lat_end - pred_lat_start
        pred_heading_peak = max((abs(e) for e in pred_heading_err), default=abs(heading_err))
        predictive_recovery_active = (
            pred_lat_growth > self.predictive_recovery_lat_growth
            or pred_lat_end > self.predictive_recovery_lat_end
            or pred_heading_peak > self.predictive_recovery_heading_err)
        predictive_hard_recovery_active = (
            pred_lat_growth > self.predictive_recovery_hard_lat_growth
            or pred_lat_end > self.predictive_recovery_hard_lat_end
            or pred_heading_peak > self.predictive_recovery_hard_heading_err)
        recovery_active = recovery_active or predictive_recovery_active
        hard_recovery_active = hard_recovery_active or predictive_hard_recovery_active
        if recovery_active:
            accel_limit = (self.recovery_accel_cap if hard_recovery_active
                           else self.recovery_soft_accel_cap)
            self.U2 = min(self.U2, accel_limit)

        wall_guard = self._compute_wall_guard(states[0], path_heading)
        if wall_guard['active']:
            self.U2 = min(self.U2, self.wall_accel_cap)
        self._record_wall_guard(wall_guard)

        # Compute desired speed (a velocity setpoint for the sim servo).
        # Feedforward the profiled raceline speed instead of anchoring the
        # command to measured velocity + one tiny Ts accel step (which starved
        # the servo and capped the car near its current speed). r[0] is the
        # first-horizon-step x_dot reference — already scaled by speed_scale
        # and floored by min_ref_speed. The MPC accel term is integrated over
        # cmd_accel_horizon (>> Ts) and blended in via speed_ff_blend.
        # When decelerating, use reverse (negative speed) to brake harder: the
        # simulator treats negative speed as reverse thrust for strong braking.
        ref_speed = ff_ref_speed
        mpc_speed = states[0] + self.U2 * self.cmd_accel_horizon
        if self.U2 < -0.5 and states[0] > ref_speed:
            # Car is faster than reference → brake hard using reverse
            speed_cmd = ref_speed - (states[0] - ref_speed)
            speed_ff_blend = 0.0
        else:
            slip = math.atan2(states[1], states[0]) if abs(states[0]) > 1e-3 else 0.0
            speed_ff_blend = self._adaptive_speed_ff_blend(
                lat_err, heading_err, pred_lat_growth, pred_lat_end,
                pred_heading_peak, slip, wall_guard, recovery_active,
                hard_recovery_active)
            speed_cmd = (speed_ff_blend * ref_speed
                         + (1.0 - speed_ff_blend) * mpc_speed)
        self._last_speed_ff_blend = speed_ff_blend
        if recovery_active:
            cap = (self.recovery_speed_cap if hard_recovery_active
                   else self.recovery_soft_speed_cap)
            if hard_recovery_active and states[0] > cap:
                speed_cmd = min(
                    speed_cmd,
                    cap - (states[0] - cap))
            else:
                speed_cmd = min(speed_cmd, cap)
        speed_cmd = self._apply_wall_guard_to_speed(
            speed_cmd, states[0], wall_guard)

        # ── 6. Publish drive ─────────────────────���──────────────────
        self._publish_drive(self.U1, speed_cmd)

        # ── 7. Visualization ────────────────────────────────────────
        self._publish_ref_marker(r, hz)
        self._publish_pred_marker(pred_aug, hz)

        # ── 8. Logging ─────────────────────────────────────────────
        self._log_row('ok', states, wp_idx, r, pred_aug=pred_aug, speed_cmd=speed_cmd,
                      ref_speed=ref_speed, mpc_speed=mpc_speed,
                      t_build=t_build, t_solve=t_solve)
        self.iteration += 1

        # Profiling accumulation (averaged + logged every 50 iters)
        t_total = time.perf_counter() - t_loop_start
        self._prof_build += t_build
        self._prof_reg += t_reg
        self._prof_solve += t_solve
        self._prof_total += t_total
        self._prof_count += 1

        if self.iteration % 50 == 0:
            self.get_logger().info(
                f'[iter={self.iteration}] wp={wp_idx}  '
                f'v={states[0]:.2f} m/s  steer={math.degrees(self.U1):.1f}deg  '
                f'accel={self.U2:.2f}  speed_cmd={speed_cmd:.2f}  '
                f'wall={self._last_wall_clearance:.2f}/{self._last_wall_speed_cap:.2f}  '
                f'slack={self._last_slack:.3f}')
            n = max(self._prof_count, 1)
            self.get_logger().info(
                f'[PROFILE avg over {n}]  '
                f'build={1e3 * self._prof_build / n:.2f}ms  '
                f'reg={1e3 * self._prof_reg / n:.2f}ms  '
                f'solve={1e3 * self._prof_solve / n:.2f}ms  '
                f'total={1e3 * self._prof_total / n:.2f}ms  '
                f'(budget={1e3 * self.Ts:.0f}ms)')
            self._prof_build = self._prof_reg = self._prof_solve = self._prof_total = 0.0
            self._prof_count = 0

    # ────────────────────────────────────────────────────────────────
    #  QP solve — persistent, warm-started OSQP
    # ────────────────────────────────────────────────────────────────
    def _solve_qp(self, Hdb, ft, G, ht):
        """Solve the MPC QP; returns the du vector (len inputs*hz) or None.

        With soft_constraints=True the STATE-limit rows get a slack variable
        s>=0 so the problem is ALWAYS feasible (the input-rate box stays hard):
            min 0.5[du;s]'diag(Hdb, mu I)[du;s] + [ft; rho 1]'[du;s]
            s.t.  G_hard du <= ht_hard            (input-rate box, hard)
                  G_soft du - s <= ht_soft        (state limits, soft)
                  -s <= 0                          (slack non-negative)
        Slack is heavily penalised (L1 rho + L2 mu) so limits are only violated
        when the hard problem would otherwise be infeasible — which used to
        return None and freeze the steering mid-slide.
        """
        if not self.soft_constraints:
            return self._solve_qp_hard(Hdb, ft, G, ht)

        n_du = G.shape[1]
        n_hard = self._n_hard_rows
        n_soft = G.shape[0] - n_hard

        if not self._use_osqp:
            # Dense augmentation for the qpsolvers fallback path.
            Isoft = np.eye(n_soft)
            P = np.zeros((n_du + n_soft, n_du + n_soft))
            P[:n_du, :n_du] = Hdb
            P[n_du:, n_du:] = self.slack_mu * Isoft
            q = np.concatenate([ft, self.slack_rho * np.ones(n_soft)])
            A = np.zeros((n_hard + 2 * n_soft, n_du + n_soft))
            A[:n_hard, :n_du] = G[:n_hard]
            A[n_hard:n_hard + n_soft, :n_du] = G[n_hard:]
            A[n_hard:n_hard + n_soft, n_du:] = -Isoft
            A[n_hard + n_soft:, n_du:] = -Isoft
            b = np.concatenate([ht[:n_hard], ht[n_hard:], np.zeros(n_soft)])
            z = solve_qp(P, q, A, b, solver=self.qp_solver)
            return None if z is None else z[:n_du]

        # OSQP sparse augmentation (form: min 0.5 z'P z + q'z s.t. l <= A z <= u).
        Isoft = sparse.identity(n_soft, format='csc')
        P = sparse.triu(
            sparse.block_diag([sparse.csc_matrix(Hdb), self.slack_mu * Isoft],
                              format='csc'), format='csc')
        q = np.concatenate([ft, self.slack_rho * np.ones(n_soft)])
        Gsp = sparse.csc_matrix(G)
        A = sparse.vstack([
            sparse.hstack([Gsp[:n_hard], sparse.csc_matrix((n_hard, n_soft))]),
            sparse.hstack([Gsp[n_hard:], -Isoft]),
            sparse.hstack([sparse.csc_matrix((n_soft, n_du)), -Isoft]),
        ], format='csc')
        m = A.shape[0]
        u = np.concatenate([ht[:n_hard], ht[n_hard:], np.zeros(n_soft)])
        pattern = (P.nnz, A.nnz, m)

        if self._z_warm is None or self._z_warm.shape[0] != n_du + n_soft:
            self._z_warm = np.zeros(n_du + n_soft)

        if self._osqp_prob is None or pattern != self._osqp_pattern:
            self._osqp_prob = osqp.OSQP()
            self._osqp_prob.setup(
                P=P, q=q, A=A, l=np.full(m, -np.inf), u=u,
                verbose=False, warm_start=True, polish=False,
                eps_abs=1e-3, eps_rel=1e-3, max_iter=4000)
            self._osqp_pattern = pattern
        else:
            self._osqp_prob.update(Px=P.data, Ax=A.data, q=q, u=u)
            self._osqp_prob.warm_start(x=self._z_warm)

        res = self._osqp_prob.solve()
        if (res.info.status_val not in (1, 2)
                or res.x is None or not np.all(np.isfinite(res.x))):
            return None
        self._z_warm = res.x
        self._last_slack = float(np.max(res.x[n_du:])) if n_soft else 0.0
        return res.x[:n_du]

    def _solve_qp_hard(self, Hdb, ft, G, ht):
        """Original hard-constrained QP (min 0.5 du' Hdb du + ft' du s.t. G du <= ht).

        Used when soft_constraints=False. Can return None (infeasible), which the
        caller treats as a fault and coasts — this is the behaviour soft
        constraints replace.
        """
        if not self._use_osqp:
            return solve_qp(Hdb, ft, G, ht, solver=self.qp_solver)

        m = G.shape[0]
        P = sparse.triu(sparse.csc_matrix(Hdb), format='csc')
        A = sparse.csc_matrix(G)
        pattern = (P.nnz, A.nnz, m)

        if self._osqp_prob is None or pattern != self._osqp_pattern:
            self._osqp_prob = osqp.OSQP()
            self._osqp_prob.setup(
                P=P, q=ft, A=A, l=np.full(m, -np.inf), u=ht,
                verbose=False, warm_start=True, polish=False,
                eps_abs=1e-3, eps_rel=1e-3, max_iter=4000)
            self._osqp_pattern = pattern
        else:
            self._osqp_prob.update(Px=P.data, Ax=A.data, q=ft, u=ht)
            self._osqp_prob.warm_start(x=self.du.flatten())

        res = self._osqp_prob.solve()
        if (res.info.status_val not in (1, 2)
                or res.x is None or not np.all(np.isfinite(res.x))):
            return None
        return res.x

    # ────────────────────────────────────��───────────────────────���───
    #  Reference trajectory builder
    # ─────────────────────────────────────────────────���──────────────
    def _build_reference(self, states, wp_idx, hz):
        """Build the reference signal vector r for the MPC horizon.

        r = [x_dot_ref_1, psi_ref_1, X_ref_1, Y_ref_1, ...,
             x_dot_ref_hz, psi_ref_hz, X_ref_hz, Y_ref_hz]

        Advances the reference based on predicted travel distance at each
        step (speed * Ts), so the reference matches where the car will
        actually be — not a fixed 1-waypoint-per-step which overshoots
        on turns.
        """
        speed = max(states[0], 1.5)
        current_psi = states[2]

        # Small lookahead offset so MPC sees just ahead of nearest point
        lookahead_dist = speed * self.Ts * 2  # ~2 timesteps ahead
        lookahead_indices = max(1, int(round(lookahead_dist / self.ds)))

        # Distance the car travels per MPC step
        dist_per_step = speed * self.Ts
        # How many waypoint indices that corresponds to
        indices_per_step = dist_per_step / self.ds

        r = np.zeros(self.outputs * hz)
        for k in range(hz):
            # Advance proportional to predicted travel distance
            advance = lookahead_indices + k * indices_per_step
            idx = (wp_idx + int(round(advance))) % self.n_waypoints
            ref_vx = max(self.wp_vx[idx] * self.speed_scale, self.min_ref_speed)
            ref_psi = self.wp_psi[idx]

            # Adjust psi reference to be close to current psi (avoid 2*pi jumps)
            while ref_psi - current_psi > np.pi:
                ref_psi -= 2.0 * np.pi
            while ref_psi - current_psi < -np.pi:
                ref_psi += 2.0 * np.pi

            r[self.outputs * k + 0] = ref_vx      # x_dot ref
            r[self.outputs * k + 1] = ref_psi      # psi ref
            r[self.outputs * k + 2] = self.wp_xy[idx, 0]  # X ref
            r[self.outputs * k + 3] = self.wp_xy[idx, 1]  # Y ref

        return r

    def _lookahead_ref_speed(self, wp_idx, v):
        """Brake-aware feedforward speed target.

        Scans the profiled reference speed over a braking-distance window ahead
        of the car and returns the highest speed that can still be decelerated
        (at speed_lookahead_decel) down to every upcoming reference point in
        time. This keeps the car fast on straights but starts braking early
        enough for corners, instead of tracking the instantaneous point speed
        (which dropped only once the car was already in the corner → spin-out).
        """
        # Window length: distance covered in lookahead_time at the greater of
        # current and profiled speed, converted to waypoint indices.
        horizon_m = max(v, 1.0) * self.speed_lookahead_time
        span = int(max(1, round(horizon_m / self.ds)))
        idxs = (wp_idx + np.arange(span)) % self.n_waypoints
        v_prof = self.wp_vx[idxs] * self.speed_scale        # upcoming target speeds
        dist = np.arange(span) * self.ds                    # distance to each point
        # Max speed now that still allows braking to v_prof[i] over dist[i]:
        #   v_now^2 <= v_prof[i]^2 + 2 * a_brake * dist[i]
        v_allow = np.sqrt(v_prof ** 2 + 2.0 * self.speed_lookahead_decel * dist)
        return max(float(v_allow.min()), self.min_ref_speed)

    # ────────────────────��──────────────────────────��────────────────
    #  Lap timing
    # ─────────────────────────────────────────────��──────────────────
    def _update_lap_timing(self, wp_idx):
        if self.lap_start_time is None:
            self.lap_start_time = time.perf_counter()

        half = self.n_waypoints // 2
        if half * 0.4 < wp_idx < half * 1.6:
            self.lap_crossed_half = True

        if (self.lap_crossed_half
                and wp_idx < self.n_waypoints * 0.05
                and self.prev_wp_idx > self.n_waypoints * 0.9):
            lap_time = time.perf_counter() - self.lap_start_time
            self.get_logger().info(
                f'========== LAP {self.nr_laps} FINISHED  |  time: {lap_time:.2f}s ==========')
            self.nr_laps += 1
            self.lap_start_time = time.perf_counter()
            self.lap_crossed_half = False

        self.prev_wp_idx = wp_idx

    # ────────────────────────────────────────────────────────────────
    #  Per-run CSV debug logging
    # ────────────────────────────────────────────────────────────────
    LOG_COLUMNS = [
        'wall_t', 'sim_t', 'iter', 'lap', 'wp_idx', 'status',
        # measured state
        'x_dot', 'y_dot', 'psi', 'psi_dot', 'X', 'Y', 'slip_deg',
        # lidar wall clearance from /scan
        'scan_front_m', 'scan_left_m', 'scan_right_m',
        'scan_front_min_m', 'scan_left_min_m', 'scan_right_min_m',
        'scan_min_m', 'scan_age_s',
        'obstacle_front_m', 'obstacle_x_m', 'obstacle_y_m',
        'obstacle_bearing_deg',
        'wall_clearance_m', 'wall_speed_cap', 'wall_limit_active',
        'wall_limit_hard', 'wall_source',
        # reference (first horizon step)
        'ref_vx', 'ref_psi', 'ref_X', 'ref_Y',
        # tracking errors
        'err_v', 'err_psi_deg', 'err_X', 'err_Y',
        'pos_err', 'lat_err', 'lon_err',
        # squared errors + running mean-square errors
        'sq_err_v', 'sq_pos_err', 'mse_v', 'mse_pos',
        # control
        'steer_deg', 'accel', 'du_steer', 'du_accel',
        'speed_cmd', 'ref_speed_ff', 'mpc_speed', 'speed_ff_blend_used',
        # timing (ms)
        't_build_ms', 't_solve_ms',
        # full horizons as space-separated scalar series
        'ref_horizon_vx', 'ref_horizon_psi', 'ref_horizon_X', 'ref_horizon_Y',
        'pred_horizon_vx', 'pred_horizon_psi', 'pred_horizon_X', 'pred_horizon_Y',
    ]

    def _init_csv_log(self):
        """Open a fresh timestamped CSV for this run and write the header."""
        self._csv_file = None
        self._csv_writer = None
        self._log_path = None
        self._sse_v = 0.0        # accumulated squared speed error
        self._sse_pos = 0.0      # accumulated squared position error
        self._n_log = 0          # number of valid tracking rows
        self._log_flush_ctr = 0
        self._t0 = time.perf_counter()

        if not self.enable_csv_log:
            self.get_logger().info('CSV logging disabled (enable_csv_log=false)')
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            # Single fixed file, overwritten each run, so it's always the same
            # path to read back (no timestamp hunting).
            self._log_path = os.path.join(self.log_dir, 'run.csv')
            self._csv_file = open(self._log_path, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(self.LOG_COLUMNS)
            self._csv_file.flush()
            self.get_logger().info(f'Logging run data to {self._log_path}')
        except Exception as e:
            self.get_logger().warn(f'Failed to open CSV log: {e}')
            self._csv_file = None
            self._csv_writer = None

    def _log_row(self, status, states, wp_idx=-1, r=None, pred_aug=None,
                 speed_cmd=float('nan'), ref_speed=float('nan'),
                 mpc_speed=float('nan'), t_build=float('nan'),
                 t_solve=float('nan')):
        """Append one control-tick row. Missing fields are logged as nan.

        `status` is 'ok' on the normal path, or 'startup' / 'qp_infeasible' /
        'qp_exception' on the early-return paths so failures are visible.
        """
        if self._csv_writer is None:
            return
        nan = float('nan')
        x_dot, y_dot, psi, psi_dot, X, Y = (float(v) for v in states)
        slip = math.degrees(math.atan2(y_dot, x_dot)) if abs(x_dot) > 1e-3 else 0.0
        scan = self._scan_metrics_for_log()

        if r is not None:
            ref_vx, ref_psi, ref_X, ref_Y = (float(r[i]) for i in range(4))
            err_v = x_dot - ref_vx
            err_psi = math.atan2(math.sin(psi - ref_psi), math.cos(psi - ref_psi))
            err_X = X - ref_X
            err_Y = Y - ref_Y
            pos_err = math.hypot(err_X, err_Y)
            # cross-track (lateral) and along-track error in the reference frame
            lat_err = -math.sin(ref_psi) * err_X + math.cos(ref_psi) * err_Y
            lon_err = math.cos(ref_psi) * err_X + math.sin(ref_psi) * err_Y
            sq_err_v = err_v * err_v
            sq_pos = pos_err * pos_err
            # only accumulate MSE on rows where the car is actually tracking
            if status == 'ok':
                self._sse_v += sq_err_v
                self._sse_pos += sq_pos
                self._n_log += 1
        else:
            ref_vx = ref_psi = ref_X = ref_Y = nan
            err_v = err_psi = err_X = err_Y = nan
            pos_err = lat_err = lon_err = sq_err_v = sq_pos = nan

        ref_horizon_vx = ref_horizon_psi = ref_horizon_X = ref_horizon_Y = ''
        pred_horizon_vx = pred_horizon_psi = pred_horizon_X = pred_horizon_Y = ''
        if r is not None:
            ref_horizon_vx = self._format_horizon_series(r, 0, self.outputs)
            ref_horizon_psi = self._format_horizon_series(r, 1, self.outputs)
            ref_horizon_X = self._format_horizon_series(r, 2, self.outputs)
            ref_horizon_Y = self._format_horizon_series(r, 3, self.outputs)
        if pred_aug is not None:
            n_aug = 6 + self.inputs
            pred_horizon_vx = self._format_horizon_series(pred_aug, 0, n_aug)
            pred_horizon_psi = self._format_horizon_series(pred_aug, 2, n_aug)
            pred_horizon_X = self._format_horizon_series(pred_aug, 4, n_aug)
            pred_horizon_Y = self._format_horizon_series(pred_aug, 5, n_aug)

        n = max(self._n_log, 1)
        mse_v = self._sse_v / n
        mse_pos = self._sse_pos / n

        def R(x, p=4):
            return round(x, p)

        row = [
            R(time.perf_counter() - self._t0), R(self.get_clock().now().nanoseconds * 1e-9),
            self.iteration, self.nr_laps, wp_idx, status,
            R(x_dot), R(y_dot), R(psi), R(psi_dot), R(X), R(Y), R(slip, 3),
            R(scan['front']), R(scan['left']), R(scan['right']),
            R(scan['front_min']), R(scan['left_min']), R(scan['right_min']),
            R(scan['min']), R(scan['age']),
            R(scan['obstacle_front']), R(scan['obstacle_x']), R(scan['obstacle_y']),
            R(math.degrees(scan['obstacle_bearing'])
              if math.isfinite(scan['obstacle_bearing']) else nan, 3),
            R(self._last_wall_clearance), R(self._last_wall_speed_cap),
            int(self._last_wall_limit_active), int(self._last_wall_limit_hard),
            self._last_wall_source,
            R(ref_vx), R(ref_psi), R(ref_X), R(ref_Y),
            R(err_v), R(math.degrees(err_psi) if r is not None else nan, 3),
            R(err_X), R(err_Y), R(pos_err), R(lat_err), R(lon_err),
            R(sq_err_v), R(sq_pos), R(mse_v), R(mse_pos),
            R(math.degrees(self.U1), 3), R(self.U2),
            R(float(self.du[0][0])), R(float(self.du[1][0])),
            R(speed_cmd), R(ref_speed), R(mpc_speed),
            R(self._last_speed_ff_blend, 3),
            R(t_build * 1e3, 3), R(t_solve * 1e3, 3),
            ref_horizon_vx, ref_horizon_psi, ref_horizon_X, ref_horizon_Y,
            pred_horizon_vx, pred_horizon_psi, pred_horizon_X, pred_horizon_Y,
        ]
        self._csv_writer.writerow(row)
        # flush every ~0.4 s so a Ctrl-C / crash keeps almost all the data
        self._log_flush_ctr += 1
        if self._log_flush_ctr >= 20:
            self._csv_file.flush()
            self._log_flush_ctr = 0

    @staticmethod
    def _format_horizon_series(values, offset, stride, precision=4):
        arr = np.asarray(values).reshape(-1)
        return ' '.join(
            f'{float(arr[i]):.{precision}f}'
            for i in range(offset, arr.size, stride))

    def _close_log(self):
        """Flush + close the CSV and print a run summary."""
        if getattr(self, '_csv_file', None) is None:
            return
        try:
            self._csv_file.flush()
            self._csv_file.close()
            n = max(self._n_log, 1)
            self.get_logger().info(
                f'Closed log {self._log_path}  |  rows={self.iteration}  '
                f'RMSE_v={(self._sse_v / n) ** 0.5:.3f} m/s  '
                f'RMSE_pos={(self._sse_pos / n) ** 0.5:.3f} m')
        except Exception:
            pass
        self._csv_file = None
        self._csv_writer = None

    # ────────────────────────────────────────────────────────────────
    #  Publishing helpers
    # ────────────────────────────────────────────────────────────────
    def _publish_drive(self, steering, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

    def _publish_waypoints_marker(self):
        m = Marker()
        m.header.frame_id = 'map'
        m.ns = 'lpv_mpc_waypoints'
        m.action = Marker.ADD
        m.type = Marker.POINTS
        m.pose.orientation.w = 1.0
        m.color.g = 0.75
        m.color.a = 1.0
        m.scale.x = 0.05
        m.scale.y = 0.05
        m.id = 0
        for i in range(self.n_waypoints):
            m.points.append(Point(
                x=float(self.wp_xy[i, 0]),
                y=float(self.wp_xy[i, 1]),
                z=0.1))
        self.vis_waypoints_pub.publish(m)

    def _publish_ref_marker(self, r, hz):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'lpv_mpc_ref_horizon'
        m.action = Marker.ADD
        m.type = Marker.LINE_STRIP
        m.pose.orientation.w = 1.0
        m.color.b = 0.9
        m.color.a = 1.0
        m.scale.x = 0.06
        m.id = 1
        for k in range(hz):
            m.points.append(Point(
                x=float(r[self.outputs * k + 2]),
                y=float(r[self.outputs * k + 3]),
                z=0.2))
        self.vis_ref_pub.publish(m)

    def _publish_pred_marker(self, pred_aug, hz):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'lpv_mpc_predicted_horizon'
        m.action = Marker.ADD
        m.type = Marker.LINE_STRIP
        m.pose.orientation.w = 1.0
        m.color.r = 1.0
        m.color.g = 0.35
        m.color.a = 1.0
        m.scale.x = 0.07
        m.id = 2

        n_aug = 6 + self.inputs
        for k in range(hz):
            base = n_aug * k
            m.points.append(Point(
                x=float(pred_aug[base + 4]),
                y=float(pred_aug[base + 5]),
                z=0.28))
        self.vis_pred_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = LPVMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Flush + close the CSV log so Ctrl-C runs are still fully saved.
        node._close_log()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
