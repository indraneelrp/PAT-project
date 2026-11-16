"""ROS wrapper for the deterministic fine pointing plant."""
import rclpy
from std_srvs.srv import SetBool
from rclpy.node import Node
from pat_interfaces.msg import (PatImu, PatCamera, PatFsmCommand,
                                PatFsmState, PatTruth)
from .simulator_core import PATSimulator


class SimulatorNode(Node):
    def __init__(self):
        super().__init__('simulator_node')
        for name, value in [
            ('imu_hz', 200.0), ('camera_hz', 30.0), ('latency_ms', 20.0),
            ('seed', 42), ('initial_x', 0.0012), ('initial_y', -0.0006),
            ('fsm_tau', 0.010), ('fsm_limit', 0.002),
            ('camera_std', 0.00005), ('gyro_std', 0.0002),
            ('step_offset_x', 0.0015), ('step_offset_y', 0.0),
        ]:
            self.declare_parameter(name, value)
        param = lambda n: self.get_parameter(n).value
        self.plant = PATSimulator(**{k: param(k) for k in [
            'imu_hz', 'camera_hz', 'latency_ms', 'seed', 'initial_x',
            'initial_y', 'fsm_tau', 'fsm_limit', 'camera_std', 'gyro_std',
            'step_offset_x', 'step_offset_y'
        ]})
        # deep queues to avoid silently losing simulated IMU ticks.
        self.imu_pub = self.create_publisher(PatImu, '/pat/imu', 1000)
        self.cam_pub = self.create_publisher(PatCamera, '/pat/camera', 100)
        self.state_pub = self.create_publisher(PatFsmState, '/pat/fsm_state', 1000)
        self.truth_pub = self.create_publisher(PatTruth, '/pat/truth', 1000)
        self.create_subscription(PatFsmCommand, '/pat/fsm_command', self.on_command, 30)
        self.create_service(SetBool, '/pat/inject_offset', self.on_inject_offset)
        self.cmd_x = self.cmd_y = 0.0
        self.create_timer(self.plant.dt, self.on_step)
        self.get_logger().info('Plant %.0f Hz, camera ~%.1f Hz, delay %.0f ms' % (
            1 / self.plant.dt, 1 / (self.plant.dt * self.plant.camera_ticks),
            self.plant.dt * self.plant.latency_ticks * 1000))

    def on_inject_offset(self, request, response):
        self.plant.offset_enabled = bool(request.data)
        response.success = True
        response.message = 'Simulated target angle offset %s' % (
            'enabled' if request.data else 'disabled')
        self.get_logger().info(response.message)
        return response

    def on_command(self, msg):
        self.cmd_x, self.cmd_y = float(msg.command_x), float(msg.command_y)

    def on_step(self):
        s = self.plant.step(self.cmd_x, self.cmd_y)
        imu = PatImu()
        imu.tick, imu.dt = s.tick, s.dt
        imu.gyro_x, imu.gyro_y = s.gyro_x, s.gyro_y
        state = PatFsmState()
        state.tick, state.position_x, state.position_y = s.tick, s.fsm_x, s.fsm_y
        truth = PatTruth()
        truth.tick, truth.theta_x, truth.theta_y = s.tick, s.theta_x, s.theta_y
        truth.residual_x, truth.residual_y = s.residual_x, s.residual_y
        self.imu_pub.publish(imu)
        self.state_pub.publish(state)
        self.truth_pub.publish(truth)
        for c in s.camera:
            msg = PatCamera()
            msg.capture_tick = c.capture_tick
            msg.error_x, msg.error_y = c.error_x, c.error_y
            msg.fsm_x_at_capture = c.fsm_x_at_capture
            msg.fsm_y_at_capture = c.fsm_y_at_capture
            msg.detected = c.detected
            self.cam_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
