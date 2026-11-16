import math
import numpy as np
import rclpy
from rclpy.node import Node
from pat_interfaces.msg import PatImu, PatCamera, PatEstimate
from .estimator import DelayAwareKalmanEstimator2D


class EstimatorNode(Node):
    def __init__(self):
        super().__init__('estimator_node')
        self.declare_parameter('history_capacity', 500)
        self.declare_parameter('camera_std', 0.00005)
        self.estimator = DelayAwareKalmanEstimator2D(
            history_capacity=int(self.get_parameter('history_capacity').value))
        camera_std = float(self.get_parameter('camera_std').value)
        for axis in (self.estimator.kf.x_axis, self.estimator.kf.y_axis):
            axis.R = camera_std ** 2
            axis.P = np.diag([1e-6, 1e-6])
            axis.Q = np.diag([1e-11, 1e-9])
        self.last_gyro_x = self.last_gyro_y = 0.0
        self.pending_cameras = []
        self.pub = self.create_publisher(PatEstimate, '/pat/estimate', 1000)
        self.create_subscription(PatImu, '/pat/imu', self.on_imu, 1000)
        self.create_subscription(PatCamera, '/pat/camera', self.on_camera, 100)
        self.get_logger().info('Delay-aware Kalman filter ready')

    def on_imu(self, msg):
        if msg.tick <= self.estimator.tick:
            return
        gap = int(msg.tick) - self.estimator.tick - 1
        if gap:
            self.get_logger().warn('Skipped %d IMU ticks; extrapolating with last gyro' % gap)
            # Keeps camera capture ticks aligned with Kalman history.
            for _ in range(gap):
                self.estimator.predict(msg.dt, self.last_gyro_x, self.last_gyro_y)
        self.estimator.predict(msg.dt, msg.gyro_x, msg.gyro_y)
        self.last_gyro_x, self.last_gyro_y = msg.gyro_x, msg.gyro_y
        self.apply_pending_cameras()
        self.publish_estimate()

    def on_camera(self, msg):
        if not msg.detected:
            return   # no update; IMU prediction continues
        if msg.capture_tick <= self.estimator.last_camera_tick:
            self.get_logger().warn('Ignoring old/out-of-order camera sample')
            return
        self.pending_cameras.append(msg)
        self.pending_cameras.sort(key=lambda m: m.capture_tick)
        if self.apply_pending_cameras():
            self.publish_estimate()

    def apply_pending_cameras(self):
        updated = False
        while self.pending_cameras and self.pending_cameras[0].capture_tick <= self.estimator.tick:
            msg = self.pending_cameras.pop(0)
            if msg.capture_tick <= self.estimator.last_camera_tick:
                continue
            # Transform from residual camera error to absolute LOS direction.
            theta_x_meas = msg.error_x + msg.fsm_x_at_capture
            theta_y_meas = msg.error_y + msg.fsm_y_at_capture
            try:
                self.estimator.update_camera_at(
                    int(msg.capture_tick), theta_x_meas, theta_y_meas)
                updated = True
            except ValueError as exc:
                self.get_logger().warn('Dropping camera measurement: %s' % str(exc))
        return updated

    def publish_estimate(self):
        m = PatEstimate()
        m.tick = self.estimator.tick
        m.theta_x, m.theta_y = [float(a) for a in self.estimator.angle]
        m.sigma_x = math.sqrt(max(0.0, self.estimator.kf.x_axis.P[0, 0]))
        m.sigma_y = math.sqrt(max(0.0, self.estimator.kf.y_axis.P[0, 0]))
        self.pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = EstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
