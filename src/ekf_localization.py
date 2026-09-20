"""Simulate noisy sensors and EKF localization for a differential-drive robot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DT = 0.01
WHEEL_RADIUS = 0.02
WHEEL_DISTANCE = 0.1
LINEAR_RESPONSE_TIME = 0.25
ANGULAR_RESPONSE_TIME = 0.15
TARGET = np.array([0.5, 0.5])


def dynamics(state: np.ndarray, control: tuple[float, float]) -> np.ndarray:
    commanded_velocity, commanded_angular_velocity = control
    velocity_dot = (commanded_velocity - state[3]) / LINEAR_RESPONSE_TIME
    angular_velocity_dot = (
        commanded_angular_velocity - state[4]
    ) / ANGULAR_RESPONSE_TIME
    derivative = np.array(
        [
            state[3] * np.cos(state[2]),
            state[3] * np.sin(state[2]),
            state[4],
            velocity_dot,
            angular_velocity_dot,
        ]
    )
    return state + derivative * DT


def measurement_model(state: np.ndarray) -> np.ndarray:
    velocity, angular_velocity = state[3], state[4]
    return np.array(
        [
            velocity / WHEEL_RADIUS - angular_velocity * WHEEL_DISTANCE / (2 * WHEEL_RADIUS),
            velocity / WHEEL_RADIUS + angular_velocity * WHEEL_DISTANCE / (2 * WHEEL_RADIUS),
            angular_velocity,
        ]
    )


def controller(state: np.ndarray) -> tuple[float, float]:
    difference = TARGET - state[:2]
    distance = np.linalg.norm(difference)
    desired_heading = np.arctan2(difference[1], difference[0])
    heading_error = (desired_heading - state[2] + np.pi) % (2 * np.pi) - np.pi
    forward = min(0.25, 0.8 * distance) * max(0.0, np.cos(heading_error))
    turn = np.clip(3.0 * heading_error, -2.5, 2.5)
    return forward, float(turn)


def run_simulation(steps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    process_noise = np.diag([0.001] * 5)
    measurement_noise = np.diag([0.05**2, 0.05**2, 0.02**2])
    measurement_jacobian = np.zeros((3, 5))
    measurement_jacobian[0, 3:] = [1 / WHEEL_RADIUS, -WHEEL_DISTANCE / (2 * WHEEL_RADIUS)]
    measurement_jacobian[1, 3:] = [1 / WHEEL_RADIUS, WHEEL_DISTANCE / (2 * WHEEL_RADIUS)]
    measurement_jacobian[2, 4] = 1

    true_state = np.zeros(5)
    estimated_state = np.zeros(5)
    covariance = np.eye(5) * 0.01
    true_history: list[np.ndarray] = []
    estimate_history: list[np.ndarray] = []

    for _ in range(steps):
        control = controller(estimated_state)
        true_state = dynamics(true_state, control)
        measurement = measurement_model(true_state) + rng.multivariate_normal(np.zeros(3), measurement_noise)

        predicted_state = dynamics(estimated_state, control)
        theta, velocity = estimated_state[2], estimated_state[3]
        transition_jacobian = np.eye(5)
        transition_jacobian[0, 2] = -velocity * np.sin(theta) * DT
        transition_jacobian[0, 3] = np.cos(theta) * DT
        transition_jacobian[1, 2] = velocity * np.cos(theta) * DT
        transition_jacobian[1, 3] = np.sin(theta) * DT
        transition_jacobian[2, 4] = DT
        transition_jacobian[3, 3] = 1 - DT / LINEAR_RESPONSE_TIME
        transition_jacobian[4, 4] = 1 - DT / ANGULAR_RESPONSE_TIME
        predicted_covariance = transition_jacobian @ covariance @ transition_jacobian.T + process_noise

        innovation = measurement - measurement_model(predicted_state)
        innovation_covariance = (
            measurement_jacobian @ predicted_covariance @ measurement_jacobian.T + measurement_noise
        )
        kalman_gain = (
            predicted_covariance
            @ measurement_jacobian.T
            @ np.linalg.inv(innovation_covariance)
        )
        estimated_state = predicted_state + kalman_gain @ innovation
        covariance = (np.eye(5) - kalman_gain @ measurement_jacobian) @ predicted_covariance

        true_history.append(true_state.copy())
        estimate_history.append(estimated_state.copy())
        if np.linalg.norm(true_state[:2] - TARGET) < 0.05:
            break

    return np.asarray(true_history), np.asarray(estimate_history)


def save_plot(output: Path, steps: int, seed: int) -> None:
    true_states, estimated_states = run_simulation(steps, seed)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(true_states[:, 0], true_states[:, 1], label="True trajectory", lw=2)
    ax.plot(estimated_states[:, 0], estimated_states[:, 1], "--", label="EKF estimate", lw=2)
    ax.scatter(*TARGET, color="green", s=80, label="Target")
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Micromouse EKF Localization")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved localization plot to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/ekf-localization.png"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_plot(args.output, args.steps, args.seed)
