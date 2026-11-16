# ROS 2 Pointing, Acquisition and Tracking simulation — Fine Pointing and Tracking

A simulation of a **Pointing, Acquisition and Tracking (PAT)** subsystem implemented with ROS 2 Jazzy and Python. It demonstrates **fine pointing lock** and **tracking** using an IMU, a slower delayed camera, a delay-aware Kalman filter, and a simulated fast steering mirror (FSM).

> **Scope:** `SEARCH`, coarse gimbal `ACQUIRE`, coarse-to-fine handoff is not implemented. The demo assumes the beacon is already visible and initially within the FSM's correction range.

## Features

- Simulated platform vibration, noisy IMU and camera, and finite-speed FSM response.
- High-rate IMU prediction and low-rate camera reading capture-- accounted for in the estimator.
- Kalman correction based on the capture timestamp (as opposed to correction at the time of publishing) 
- Closed-loop FSM command and position feedback.
- Supervisory states: `DISABLED`, `FINE_LOCK`, `TRACK` -- a basic finite state machine that would be expanded on in a full implementation.
- Live ROS 2 demonstration, target-offset injection service, tests, and sample simulations.

## Repository layout

```text
ros2_ws/
└── src/
    ├── pat_interfaces/                 # ROS 2 custom messages and any existing services
    │   ├── msg/                        # PatImu, PatCamera, PatEstimate,
    │   │                               # PatFsmCommand, PatFsmState, PatTruth
    │   ├── CMakeLists.txt
    │   └── package.xml
    └── pat_core/                       # Python ROS 2 package (ament_python)
        ├── pat_core/
        │   ├── estimator.py            # 1D/2D Kalman + delayed replay
        │   ├── controller.py           # Saturated proportional FSM commands
        │   ├── mocked_hardware.py      # First-order FSM actuator model
        │   ├── state_machine.py        # DISABLED/FINE_LOCK/TRACK
        │   ├── simulator_core.py       # Physics, sensors and camera delay queue
        │   ├── simulator_node.py       # ROS 2 simulated sensors/actuator
        │   ├── estimator_node.py       # ROS 2 Kalman wrapper
        │   ├── controller_node.py      # ROS 2 controller and mode supervisor
        │   ├── monitor_node.py         # RMSE/CSV output
        │   ├── state_demo.py           # Offline state-transition scenario
        │   └── offline_validation.py   # Offline baseline comparison
        ├── launch/pat_demo.launch.py
        ├── test/
        ├── setup.py
        ├── setup.cfg
        └── package.xml
```

Optional plotting script: `plot_state_demo.py` in `ros2_ws/` if it has been added to the repository. It is **not required** to run ROS 2.

## Prerequisites

- Git and Docker with a running Linux container or a native **Ubuntu 24.04 + ROS 2 Jazzy** installation.
- Within the ROS environment: Python 3, `rclpy`, `colcon`, `numpy`, ROS 2 interface-generation packages, `std_msgs` and `std_srvs`.
- For tests: `pytest`. For plots only: `matplotlib`; apt-get update and then install them respectively.

### 1. Clone

Run on your **host computer**:

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <YOUR_REPO_NAME>
```

Replace the placeholders with the actual repository URL and directory name.

### 2. Choose your ROS execution environment

**Option A — existing Docker container (recommended for the macOS setup used in development).** On macOS, do not run `/opt/ros/...` `source` commands in the Mac shell.

```bash
docker ps -a
# If the container is stopped:
docker start pat_ros2
docker exec -it pat_ros2 bash
```

These commands assume the Docker container is named `pat_ros2` **and** the repository's `ros2_ws/` directory is already available inside it at `/workspace/ros2_ws` (typically by a bind mount). Check with:

```bash
ls /workspace/ros2_ws/src
```

You should see `pat_core` and `pat_interfaces`. If the directory is missing, the host repo has **not** been mounted into that container. You can instead create a new container with a bind mount (replace the image and host path with your existing ROS Jazzy Docker image and cloned repository):

```bash
# Run on the host, from the cloned repository root:
docker run -it --name pat_ros2_dev \
  -v "$(pwd)/ros2_ws:/workspace/ros2_ws" \
  -w /workspace/ros2_ws \
  <YOUR_ROS_JAZZY_IMAGE> bash
```

A base ROS image may not include `colcon`, interface generators, `numpy`, or pytest; install them using the appropriate Ubuntu/ROS apt repositories as needed.

**Option B — native Ubuntu 24.04.** Install and source ROS 2 Jazzy using the official ROS instructions; replace `/workspace/ros2_ws` in the commands below with the absolute path to your cloned `ros2_ws`.

### 3. Build the ROS workspace

All commands in this section run **inside Docker (or native Ubuntu)**:

```bash
cd /workspace/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-up-to pat_core
source install/setup.bash
```

After every *new terminal*, source both setup files again:

```bash
source /opt/ros/jazzy/setup.bash
source /workspace/ros2_ws/install/setup.bash
```

Check package registration and generated messages:

```bash
ros2 pkg list | grep '^pat_'
ros2 pkg executables pat_core
ros2 interface show pat_interfaces/msg/PatCamera
```

> If your custom interface package is actually named `pat_interface` (singular), the message imports/dependencies must match that package name. This README uses `pat_interfaces` (plural).

## Run the live ROS 2 demo (two terminals)

### Terminal 1 — start the whole system

On the host, enter the container if needed, then run:

```bash
cd /workspace/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch pat_core pat_demo.launch.py enabled:=true
```

This starts `simulator_node`, `estimator_node`, `controller_node`, and `monitor_node`. The launch runs **continuously** until Ctrl+C; it does not restart after 1,600 ticks and does not inject any offset automatically.

The default launch configuration requests **200 Hz IMU**, approximately **28.6 Hz camera** (rounded to 7 simulator ticks), and **20 ms camera latency**. These are simulated timing targets, not real-time scheduling guarantees.

### Terminal 2 — inspect and control the same running system

Open a **second host terminal**:

```bash
docker exec -it pat_ros2 bash
cd /workspace/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

If using `pat_ros2_dev` or another container name, substitute it for `pat_ros2`.

Inspect nodes, topics, and update rates:

```bash
ros2 node list
ros2 topic list -t
ros2 topic hz /pat/imu
# Ctrl+C to stop the rate viewer
ros2 topic hz /pat/camera
# Ctrl+C
ros2 topic echo /pat/estimate
# Ctrl+C
ros2 topic echo /pat/mode --qos-durability transient_local
# Ctrl+C
```

**Demonstrate loss and recovery of fine lock:**

```bash
# Inject a sudden simulated target-direction offset
ros2 service call /pat/inject_offset std_srvs/srv/SetBool '{data: true}'

# Remove the offset later
ros2 service call /pat/inject_offset std_srvs/srv/SetBool '{data: false}'
```

A sufficiently large offset should take the mode from `TRACK` back to `FINE_LOCK`. Once alignment is recovered and the lock criteria hold, the system transitions back to `TRACK`. Reacquisition is not guaranteed if the target remains outside the FSM's range; there is no gimbal recovery.

Enable/disable control:

```bash
ros2 service call /pat/enable_tracking std_srvs/srv/SetBool '{data: false}'
ros2 service call /pat/enable_tracking std_srvs/srv/SetBool '{data: true}'
```

To start with control disabled, run `ros2 launch pat_core pat_demo.launch.py` without `enabled:=true` (default is `false`).

### Change demo parameters

For example:

```bash
ros2 launch pat_core pat_demo.launch.py \
  enabled:=true imu_hz:=200.0 camera_hz:=30.0 latency_ms:=20.0 \
  lock_threshold:=0.0002 loss_threshold:=0.0004 \
  csv_path:=/tmp/pat_tracking.csv
```

Angles and thresholds use **radians**: `0.0002` rad = `0.2` mrad. Ensure `lock_threshold` is smaller than `loss_threshold`.

## Run Python simulations and tests without ROS

These use the same algorithm and plant classes directly (no topics/executor). Inside the container:

```bash
cd /workspace/ros2_ws
export PYTHONPATH="$PWD/src/pat_core:$PYTHONPATH"

# Unit and integration tests
python3 -m pytest -q src/pat_core/test

# 8-second scenario with mode transitions
python3 -m pat_core.state_demo

# Compare uncontrolled, camera-only, naive Kalman and delayed replay
python3 -m pat_core.offline_validation
```

`state_demo.py` takes 1,600 steps at 200 Hz, **automatically enables the target offset at tick 600 (~3.0 s)** and removes it at tick 950 (~4.75 s), then exits at tick 1,600 (8.0 s). It does not loop back to zero. This is independent of the continuously running ROS 2 launch.

## Plot tracking error

### Option A — standalone deterministic plot

If you added `plot_state_demo.py` to `ros2_ws/` and installed Matplotlib:

```bash
cd /workspace/ros2_ws
export PYTHONPATH="$PWD/src/pat_core:$PYTHONPATH"
MPLBACKEND=Agg python3 plot_state_demo.py
```

This produces `pat_state_demo_error.png` in the working directory. `MPLBACKEND=Agg` renders a file without opening a GUI inside Docker.

### Option B — CSV from the live ROS 2 demo

`monitor_node` writes per-tick metrics to `/tmp/pat_tracking.csv` inside the container. After running the demo, copy it to the host:

```bash
# Run on your host computer, not inside the container
docker cp pat_ros2:/tmp/pat_tracking.csv ./pat_tracking.csv
```

Plot on your host using Python with Matplotlib installed:

```python
import csv
import matplotlib.pyplot as plt

ticks, errors = [], []
with open('pat_tracking.csv', newline='') as f:
    for row in csv.DictReader(f):
        ticks.append(int(row['tick']))
        errors.append(1000 * float(row['tracking_error_rad']))

plt.plot([tick / 200 for tick in ticks], errors, label='Actual pointing error')
plt.axhline(0.2, linestyle='--', label='0.2 mrad tolerance')
plt.xlabel('Simulation time (s)')
plt.ylabel('2D residual pointing error (mrad)')
plt.title('ROS 2 PAT fine-tracking error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('ros_tracking_error.png', dpi=160)
```

The divisor `200` matches the default IMU rate; change it if you ran with another `imu_hz`. Running on your Mac avoids needing Matplotlib inside Docker.

If `python3 -m pip` is unavailable in Docker, avoid assuming pip is installed. On a properly configured Ubuntu 24.04 image, `apt-get update && apt-get install python3-matplotlib` is an option; if apt cannot locate the package, diagnose Ubuntu repository configuration rather than repeatedly attempting installation.

## How it works

```text
                         ┌────────────────────┐
           FSM command ──►│ simulator_node     │
                         │ simulated plant,   │
                         │ sensors and FSM    │
                         └──────┬─────────────┘
                                │ /pat/imu + delayed /pat/camera
                                ▼
                         ┌────────────────────┐
                         │ estimator_node     │
                         │ Kalman + replay    │
                         └──────┬─────────────┘
                                │ /pat/estimate
                                ▼
                         ┌────────────────────┐
                         │ controller_node    │◄── /pat/fsm_state
                         │ FSM + state machine│
                         └──────┬─────────────┘
                                │ /pat/fsm_command
                                └─────────────► simulator_node

               monitor_node receives /pat/truth and estimates
               for evaluation only; truth is not used in control.
```

- `controller.py`: computes **desired** FSM position and clips it to the configured limit.
- `mocked_hardware.py`: simulates **actual** mirror position approaching the commanded position with actuator lag.
- `simulator_core.py`: updates platform/beacon motion, actual FSM response, camera residual, noise, and delayed delivery.
- `estimator.py`: predicts target direction from IMU; corrects past state at camera capture time and replays IMU history.
- `state_machine.py`: decides whether fine lock is established (`TRACK`) or needs reacquisition within fine pointing (`FINE_LOCK`). It does not implement `SEARCH`/`ACQUIRE`.

The simplified plant defines camera residual as `error = theta - fsm_position`, while the Kalman filter estimates target direction `theta` in the platform frame. Thus the estimator uses **camera residual + FSM position at capture time**, not the FSM position at message arrival time.

### Main ROS 2 topics

| Topic | Publisher | Main subscriber | Data |
|---|---|---|---|
| `/pat/imu` | Simulator | Estimator | Gyro, tick, time step |
| `/pat/camera` | Simulator | Estimator, controller | Noisy residual, capture tick, FSM position at capture |
| `/pat/estimate` | Estimator | Controller, monitor | Predicted target direction |
| `/pat/fsm_command` | Controller | Simulator | Desired FSM position |
| `/pat/fsm_state` | Simulator | Controller, monitor | Actual FSM position |
| `/pat/mode` | Controller | Monitor, CLI | `DISABLED`, `FINE_LOCK`, `TRACK` |
| `/pat/truth` | Simulator | Monitor | Simulation ground truth, never used for control |

## Troubleshooting

**`source: no such file or directory: /opt/ros//setup.bash`** — You are likely on macOS instead of inside Docker, or `ROS_DISTRO` is unset. Enter the container and use `/opt/ros/jazzy/setup.bash`.

**Package or message type not found** — Build the workspace, source `install/setup.bash` in *each terminal*, check `pat_interfaces` spelling, and confirm the six `.msg` entries are registered in `CMakeLists.txt`.

**`ros2 launch` cannot find launch file** — Ensure `pat_core/setup.py` installs `launch/*.launch.py` to `share/pat_core/launch` and rebuild.

**`python3: No module named pip`** — Python pip is not installed in that environment. Use the host for plots or install dependencies through the container's configured package manager.

**`E: Unable to locate package python3-matplotlib`** — Check `cat /etc/os-release`, `apt-get update`, `apt-cache policy python3-matplotlib`, and `/etc/apt/sources.list.d/`; your container might have incomplete Ubuntu repositories.

**No `TRACK` state after startup** — Check tracking is enabled, camera messages are arriving, and errors fall under the lock threshold long enough. Inspect `/pat/mode`, `/pat/camera`, `/pat/estimate`, `/pat/fsm_state`.

**Poor or irregular sensor rates** — Python timer callbacks in Docker are not hard real-time; check CPU load and `ros2 topic hz`. Start with the default 200 Hz simulation rate.