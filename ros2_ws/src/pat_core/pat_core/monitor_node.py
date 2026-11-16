"""Match estimator and plant truth by tick; log lock and accuracy metrics.

Ground truth is used ONLY in this monitor. It does not reach the estimator or
controller. Tracking toggles segment the metrics and produce CSV mode labels.
"""
import csv
import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, String
from pat_interfaces.msg import PatEstimate, PatTruth, PatFsmState, PatFsmCommand


class MonitorNode(Node):
    def __init__(self):
        super().__init__('monitor_node')
        self.declare_parameter('csv_path', '/tmp/pat_tracking.csv')
        self.declare_parameter('lock_tolerance', 0.0002)
        self.csv_path = str(self.get_parameter('csv_path').value)
        self.lock_tolerance = float(self.get_parameter('lock_tolerance').value)
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = open(self.csv_path, 'w', newline='', buffering=1)
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['tick', 'mode', 'theta_x', 'theta_y', 'estimate_x',
                              'estimate_y', 'residual_x', 'residual_y', 'fsm_x',
                              'fsm_y', 'command_x', 'command_y',
                              'tracking_error_rad', 'estimation_error_rad'])
        self.truth = {}
        self.estimate = {}
        self.latest_fsm = (0.0, 0.0)
        self.latest_command = (0.0, 0.0)
        self.enabled = False
        self.mode = 'DISABLED'
        self.reset_stats()
        self.create_subscription(PatTruth, '/pat/truth', self.on_truth, 1000)
        self.create_subscription(PatEstimate, '/pat/estimate', self.on_estimate, 1000)
        self.create_subscription(PatFsmState, '/pat/fsm_state', self.on_fsm, 1000)
        self.create_subscription(PatFsmCommand, '/pat/fsm_command', self.on_command, 30)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/pat/tracking_enabled', self.on_enabled, qos)
        self.create_subscription(String, '/pat/mode', self.on_mode, qos)
        self.create_timer(2.0, self.report)
        self.get_logger().info('Logging per-tick metrics to %s' % self.csv_path)

    def reset_stats(self):
        self.n = 0
        self.sum_tracking_err_sq = 0.0
        self.sum_est_err_sq = 0.0
        self.n_locked = 0

    def on_enabled(self, msg):
        if self.enabled != msg.data:
            self.report()
            self.enabled = bool(msg.data)
            self.reset_stats()
            self.get_logger().info('New mode: %s' % ('TRACKING_ON' if self.enabled else 'TRACKING_OFF'))

    def on_mode(self, msg):
        if self.mode != msg.data:
            self.mode = msg.data
            self.get_logger().info('Supervisory PAT state: %s' % self.mode)

    def on_fsm(self, msg):
        self.latest_fsm = (msg.position_x, msg.position_y)

    def on_command(self, msg):
        self.latest_command = (msg.command_x, msg.command_y)

    def on_truth(self, msg):
        self.truth[int(msg.tick)] = msg
        self.consume(msg.tick)

    def on_estimate(self, msg):
        self.estimate[int(msg.tick)] = msg
        self.consume(msg.tick)

    def consume(self, tick):
        if tick not in self.truth or tick not in self.estimate:
            # Bound stale records, e.g. if a ROS subscription starts late.
            for cache in (self.truth, self.estimate):
                while len(cache) > 1500:
                    cache.pop(next(iter(cache)))
            return
        truth = self.truth.pop(tick)
        est = self.estimate.pop(tick)
        re = math.hypot(truth.residual_x, truth.residual_y)
        ee = math.hypot(est.theta_x - truth.theta_x, est.theta_y - truth.theta_y)
        self.n += 1
        self.sum_tracking_err_sq += re ** 2
        self.sum_est_err_sq += ee ** 2
        self.n_locked += int(re < self.lock_tolerance)
        self.writer.writerow([
            tick, self.mode,
            truth.theta_x, truth.theta_y, est.theta_x, est.theta_y,
            truth.residual_x, truth.residual_y,
            self.latest_fsm[0], self.latest_fsm[1],
            self.latest_command[0], self.latest_command[1], re, ee
        ])

    def report(self):
        if self.n:
            tracking_rmse = math.sqrt(self.sum_tracking_err_sq / self.n)
            estimate_rmse = math.sqrt(self.sum_est_err_sq / self.n)
            self.get_logger().info(
                '%s | N=%d | tracking RMSE=%.3f mrad | estimate RMSE=%.3f mrad '
                '| lock=%.1f%% (<%.3f mrad)' % (
                    self.mode, self.n,
                    1000 * tracking_rmse, 1000 * estimate_rmse,
                    100 * self.n_locked / self.n, 1000 * self.lock_tolerance))

    def destroy_node(self):
        self.csv_file.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
