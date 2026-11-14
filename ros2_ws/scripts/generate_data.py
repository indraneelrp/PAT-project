import csv
import math
import random
from pathlib import Path

DT = 0.001
DURATION = 3.0
CAMERA_EVERY = 33
CAMERA_DELAY = 0.060
rng = random.Random(7)

OUT = Path(__file__).resolve().parents[1] / "sample_data"
OUT.mkdir(exist_ok=True)

def body(t):
    """Platform attitude on two local angular axes."""
    return (
        0.0002 * math.sin(2 * math.pi * 2 * t)
        + 0.0001 * math.sin(2 * math.pi * 17 * t),
        0.0003 * math.cos(2 * math.pi * 1.5 * t),
    )

def body_rate(t):
    """True angular rate: derivative of body(t)."""
    return (
        0.0002 * 2 * math.pi * 2
        * math.cos(2 * math.pi * 2 * t)
        + 0.0001 * 2 * math.pi * 17
        * math.cos(2 * math.pi * 17 * t),
        -0.0003 * 2 * math.pi * 1.5
        * math.sin(2 * math.pi * 1.5 * t),
    )

def target(t):
    """Remote beacon's slowly moving direction."""
    return (
        0.0012 + 0.00015 * math.sin(2 * math.pi * 0.2 * t),
        -0.0007,
    )

with (
    open(OUT / "imu.csv", "w", newline="") as fi,
    open(OUT / "camera.csv", "w", newline="") as fc,
    open(OUT / "truth.csv", "w", newline="") as ft,
):
    imu = csv.writer(fi)
    camera = csv.writer(fc)
    truth = csv.writer(ft)

    imu.writerow(["t", "gyro_x", "gyro_y"])
    camera.writerow([
        "capture_t", "delivery_t", "error_x", "error_y",
        "gimbal_x", "gimbal_y", "fsm_x", "fsm_y"
    ])
    truth.writerow(["t", "q_x", "q_y"])

    for k in range(round(DURATION / DT)):
        t = k * DT

        b = body(t)
        w = body_rate(t)
        a = target(t)

        qx = a[0] - b[0]
        qy = a[1] - b[1]

        # Biased, noisy IMU.
        imu.writerow([
            t,
            w[0] + 0.0005 + rng.gauss(0, 0.0001),
            w[1] - 0.0003 + rng.gauss(0, 0.0001),
        ])

        # Only the simulator/evaluator sees true q.
        truth.writerow([t, qx, qy])

        # First fixture: actuators held at zero.
        if k % CAMERA_EVERY == 0:
            camera.writerow([
                t, t + CAMERA_DELAY,
                qx + rng.gauss(0, 0.00001),
                qy + rng.gauss(0, 0.00001),
                0.0, 0.0, 0.0, 0.0,
            ])

print("Created imu.csv, camera.csv, truth.csv")
