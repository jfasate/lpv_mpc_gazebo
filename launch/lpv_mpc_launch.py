import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _resolve_log_dir():
    """Return the source-tree log/ folder (created if missing).

    The node runs from the install space, but we want the per-run CSV logs in
    the source package (src/lpv_mpc_gazebo/log) so they are easy to inspect and are
    not wiped by `colcon build`. Prefer the LPV_MPC_LOG_DIR env override, then
    the conventional ~/sim_gazebo/src/lpv_mpc_gazebo/log path.
    """
    log_dir = os.environ.get(
        'LPV_MPC_LOG_DIR',
        os.path.expanduser('~/sim_gazebo/src/lpv_mpc_gazebo/log'))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('lpv_mpc_gazebo'),
        'config',
        'lpv_mpc_params.yaml')

    log_dir = _resolve_log_dir()

    lpv_mpc_node = Node(
        package='lpv_mpc_gazebo',
        executable='lpv_mpc_node',
        name='lpv_mpc_node',
        output='screen',
        # config first, then the resolved log_dir override so every run lands
        # in the source-tree log/ folder (enable_csv_log stays YAML-controlled).
        parameters=[config, {'log_dir': log_dir, 'config_file': config}],
    )

    return LaunchDescription([lpv_mpc_node])
