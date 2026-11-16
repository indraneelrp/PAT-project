"""ROS 2 controller: supervisory fine-lock mode + two-axis FSM commands.

Receives target LOS from the estimator and actual mirror position from the
simulator. All errors/thresholds are radians. Ground truth is NEVER used.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool
from pat_interfaces.msg import PatCamera, PatEstimate, PatFsmState, PatFsmCommand

from .controller import FSMController
from .state_machine import FineTrackingStateMachine


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')
        for name, value in [
            ('gain', 0.7), ('fsm_limit', 0.002), ('enabled', False),
            ('imu_hz', 200.0), ('max_camera_age_s', 0.15),
            ('lock_threshold', 0.0002), ('loss_threshold', 0.0004),
            ('lock_samples', 15), ('loss_samples', 5),
        ]:
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value

        self.cx = FSMController(gain=float(p('gain')), limit=float(p('fsm_limit')))
        self.cy = FSMController(gain=float(p('gain')), limit=float(p('fsm_limit')))
        self.machine = FineTrackingStateMachine(
            lock_threshold=float(p('lock_threshold')),
            loss_threshold=float(p('loss_threshold')),
            lock_samples=int(p('lock_samples')),
            loss_samples=int(p('loss_samples')),
        )
        self.machine.set_enabled(bool(p('enabled')))
        self.imu_hz = float(p('imu_hz'))
        self.max_camera_age_s = float(p('max_camera_age_s'))
        self.fsm_x = self.fsm_y = 0.0
        self.last_camera_tick = None
        self.last_camera_error = None
        self.latest_estimate = None

        self.pub = self.create_publisher(PatFsmCommand, '/pat/fsm_command', 30)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.mode_pub = self.create_publisher(String, '/pat/mode', latched)
        self.enabled_pub = self.create_publisher(Bool, '/pat/tracking_enabled', latched)
        self.create_subscription(PatEstimate, '/pat/estimate', self.on_estimate, 1000)
        self.create_subscription(PatFsmState, '/pat/fsm_state', self.on_fsm, 1000)
        # Only for validity / freshness; estimator handles camera corrections.
        self.create_subscription(PatCamera, '/pat/camera', self.on_camera, 100)
        self.create_service(SetBool, '/pat/enable_tracking', self.on_enable)
        self.publish_mode()
        self.get_logger().info('Controller started in %s' % self.machine.state)

    def on_fsm(self, msg):
        self.fsm_x, self.fsm_y = float(msg.position_x), float(msg.position_y)

    def on_camera(self, msg):
        if msg.detected:
            if self.last_camera_tick is None or msg.capture_tick > self.last_camera_tick:
                self.last_camera_tick = int(msg.capture_tick)
                self.last_camera_error = math.hypot(msg.error_x, msg.error_y)
        else:
            # Explicitly invalidate camera observations on loss of beacon.
            self.last_camera_tick = None
            self.last_camera_error = None

    def camera_fresh(self, estimate_tick):
        if self.last_camera_tick is None:
            return False
        age_ticks = max(0, int(estimate_tick) - self.last_camera_tick)
        return age_ticks / self.imu_hz <= self.max_camera_age_s

    def on_estimate(self, msg):
        self.latest_estimate = msg
        if self.machine.state != FineTrackingStateMachine.DISABLED:
            ex = float(msg.theta_x) - self.fsm_x
            ey = float(msg.theta_y) - self.fsm_y
            # A recent camera residual is a conservative lock check.
            # This prevents an overconfident estimate from declaring lock
            # immediately after an unmodelled target step.
            estimated_error = math.hypot(ex, ey)
            mode_error = max(estimated_error, self.last_camera_error or 0.0)
            if self.machine.update(mode_error, 0.0,
                                   camera_fresh=self.camera_fresh(msg.tick)):
                self.publish_mode()
                self.get_logger().info('PAT mode -> %s (lock-check error %.3f mrad)' % (
                    self.machine.state, 1000 * mode_error))
        self.publish_command()

    def publish_command(self):
        cmd = PatFsmCommand()
        if (self.machine.state != FineTrackingStateMachine.DISABLED
                and self.latest_estimate is not None):
            ex = float(self.latest_estimate.theta_x) - self.fsm_x
            ey = float(self.latest_estimate.theta_y) - self.fsm_y
            # Both FINE_LOCK and TRACK use the same fine controller here.
            cmd.command_x = self.cx.compute_command(self.fsm_x, ex)
            cmd.command_y = self.cy.compute_command(self.fsm_y, ey)
        else:
            cmd.command_x = 0.0
            cmd.command_y = 0.0
        self.pub.publish(cmd)

    def publish_mode(self):
        enabled = self.machine.state != FineTrackingStateMachine.DISABLED
        self.mode_pub.publish(String(data=self.machine.state))
        self.enabled_pub.publish(Bool(data=enabled))

    def on_enable(self, request, response):
        self.machine.set_enabled(request.data)
        self.publish_mode()
        self.publish_command()
        response.success = True
        response.message = 'PAT fine control: %s' % self.machine.state
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
