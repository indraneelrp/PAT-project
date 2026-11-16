"""One-command launch of closed-loop ROS 2 fine pointing demo."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def typed(name, typ):
    return ParameterValue(LaunchConfiguration(name), value_type=typ)


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('imu_hz', default_value='200.0'),
        DeclareLaunchArgument('camera_hz', default_value='30.0'),
        DeclareLaunchArgument('latency_ms', default_value='20.0'),
        DeclareLaunchArgument('seed', default_value='42'),
        DeclareLaunchArgument('enabled', default_value='false'),
        DeclareLaunchArgument('lock_threshold', default_value='0.0002'),
        DeclareLaunchArgument('loss_threshold', default_value='0.0004'),
        DeclareLaunchArgument('csv_path', default_value='/tmp/pat_tracking.csv'),
        Node(package='pat_core', executable='simulator_node', output='screen',
             parameters=[{'imu_hz': typed('imu_hz', float),
                          'camera_hz': typed('camera_hz', float),
                          'latency_ms': typed('latency_ms', float),
                          'seed': typed('seed', int)}]),
        Node(package='pat_core', executable='estimator_node', output='screen'),
        Node(package='pat_core', executable='controller_node', output='screen',
             parameters=[{'enabled': typed('enabled', bool),
                          'imu_hz': typed('imu_hz', float),
                          'lock_threshold': typed('lock_threshold', float),
                          'loss_threshold': typed('loss_threshold', float)}]),
        Node(package='pat_core', executable='monitor_node', output='screen',
             parameters=[{'csv_path': LaunchConfiguration('csv_path')}]),
    ])
