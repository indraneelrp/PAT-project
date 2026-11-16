"""Deterministic 2D PAT plant + synthetic camera/IMU. No ROS imports.

Angles are radians. `theta` is beacon LOS relative to the moving platform,
`fsm.position` is *effective pointing correction* (1:1 abstracted mirror), and
camera residual = theta - fsm.position.  This is a *fine lock* model, not coarse
search/acquisition. Simulation ticks are distinct from wall-clock time.
"""
from dataclasses import dataclass
import math
import numpy as np
from .mocked_hardware import FirstOrderFSM


@dataclass(frozen=True)
class CameraSample:
    capture_tick: int
    error_x: float
    error_y: float
    fsm_x_at_capture: float
    fsm_y_at_capture: float
    detected: bool = True


@dataclass(frozen=True)
class PlantSample:
    tick: int
    dt: float
    gyro_x: float
    gyro_y: float
    fsm_x: float
    fsm_y: float
    theta_x: float
    theta_y: float
    residual_x: float
    residual_y: float
    camera: tuple  # zero or more CameraSample objects delivered at this tick


class PATSimulator:
    def __init__(self, imu_hz=200.0, camera_hz=30.0, latency_ms=20.0,
                 seed=42, initial_x=0.0012, initial_y=-0.0006,
                 fsm_tau=0.010, fsm_limit=0.002,
                 camera_std=0.00005, gyro_std=0.0002,
                 step_offset_x=0.0015, step_offset_y=0.0):
        if imu_hz <= 0 or camera_hz <= 0 or latency_ms < 0:
            raise ValueError('Invalid simulation rate/latency')
        if fsm_tau <= 0 or fsm_limit <= 0:
            raise ValueError('FSM tau and limit must be positive')
        self.dt = 1.0 / float(imu_hz)
        # Approximate the camera rate with an integer number of IMU ticks.
        self.camera_ticks = max(1, round(imu_hz / camera_hz))
        self.latency_ticks = round(latency_ms / 1000.0 * imu_hz)
        self.tick = 0
        self.theta_x = float(initial_x)
        self.theta_y = float(initial_y)
        self.fsm_x = FirstOrderFSM(tau=fsm_tau)
        self.fsm_y = FirstOrderFSM(tau=fsm_tau)
        self.fsm_limit = float(fsm_limit)
        self.camera_std = float(camera_std)
        self.gyro_std = float(gyro_std)
        self.rng = np.random.default_rng(seed)
        self.pending_camera = []
        # Reversible target LOS step for a demonstrable TRACK -> FINE_LOCK.
        # This models sudden counterpart movement, not IMU-measured rotation.
        self.offset_enabled = False
        self.step_offset_x = float(step_offset_x)
        self.step_offset_y = float(step_offset_y)

    def step(self, command_x=0.0, command_y=0.0):
        self.tick += 1
        t = self.tick * self.dt
        # Simulate two independently vibrating platform axes.
        true_gyro_x = (0.014 * math.sin(2 * math.pi * 2 * t)
                       + 0.006 * math.sin(2 * math.pi * 5 * t))
        true_gyro_y = (0.012 * math.cos(2 * math.pi * 3 * t)
                       + 0.004 * math.sin(2 * math.pi * 4 * t))
        self.theta_x -= true_gyro_x * self.dt
        self.theta_y -= true_gyro_y * self.dt
        command_x = float(np.clip(command_x, -self.fsm_limit, self.fsm_limit))
        command_y = float(np.clip(command_y, -self.fsm_limit, self.fsm_limit))
        fx = self.fsm_x.step(command_x, self.dt)
        fy = self.fsm_y.step(command_y, self.dt)
        true_x = self.theta_x + (self.step_offset_x if self.offset_enabled else 0.0)
        true_y = self.theta_y + (self.step_offset_y if self.offset_enabled else 0.0)
        ex = true_x - fx
        ey = true_y - fy

        if self.tick % self.camera_ticks == 0:
            captured = CameraSample(
                capture_tick=self.tick,
                error_x=float(ex + self.rng.normal(0, self.camera_std)),
                error_y=float(ey + self.rng.normal(0, self.camera_std)),
                fsm_x_at_capture=float(fx),
                fsm_y_at_capture=float(fy),
            )
            self.pending_camera.append((self.tick + self.latency_ticks, captured))

        due = tuple(c for deadline, c in self.pending_camera if deadline <= self.tick)
        self.pending_camera = [(deadline, c) for deadline, c in self.pending_camera
                               if deadline > self.tick]
        return PlantSample(
            tick=self.tick, dt=self.dt,
            gyro_x=float(true_gyro_x + self.rng.normal(0, self.gyro_std)),
            gyro_y=float(true_gyro_y + self.rng.normal(0, self.gyro_std)),
            fsm_x=float(fx), fsm_y=float(fy),
            theta_x=float(true_x), theta_y=float(true_y),
            residual_x=float(ex), residual_y=float(ey), camera=due,
        )
