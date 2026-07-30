#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import rclpy

from instinct_nav_onboard.bridges import SensorSource
from instinct_nav_onboard.config import OnboardConfig
from instinct_nav_onboard.onnx_high_level_runtime import OnnxDebugDumpConfig, OnnxHighLevelRuntime
from instinct_nav_onboard.packets import SensorPacket
from instinct_nav_onboard.safety import FallSafetyMonitor, PersistentSafetyLogger, projected_gravity_tilt_deg

from instinct_onboard.agents.base import AgentStatus, ColdStartAgent
from instinct_onboard.agents.parkour_agent import ParkourAgent, ParkourStandAgent
from instinct_onboard.joystick import UnitreeJoyStick
from instinct_onboard.ros_nodes.realsense import RealsenseMPCamera
from instinct_onboard.ros_nodes.unitree import UnitreeNode

try:
    from instinct_onboard.scripts.g1_parkour import MAIN_LOOP_FREQUENCY_CHECK_INTERVAL
except Exception:
    MAIN_LOOP_FREQUENCY_CHECK_INTERVAL = 500


class G1NodeSensorSource(SensorSource):
    def __init__(self, node: "G1NavOnboardOnnxNode"):
        self.node = node

    def read(self) -> SensorPacket:
        self.node.refresh_camera_data()
        depth = self.node.get_depth_image()
        if depth is None:
            depth = np.full((36, 64), 2.5, dtype=np.float32)
        depth = self._resize_depth_to_training_shape(depth)
        return SensorPacket(
            timestamp_s=time.monotonic(),
            base_lin_vel_b=self.node.base_lin_vel_b.astype(np.float32),
            base_ang_vel_b=self.node._get_base_ang_vel_obs().astype(np.float32),
            projected_gravity_b=self.node._get_projected_gravity_obs().astype(np.float32),
            joint_pos=self.node._get_joint_pos_obs().astype(np.float32),
            joint_vel=self.node._get_joint_vel_obs().astype(np.float32),
            depth_m=depth,
            goal_b=self.node.nav_goal_b.astype(np.float32),
            reset=bool(self.node.nav_reset_requested),
        )

    @staticmethod
    def _resize_depth_to_training_shape(depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)
        if depth.shape == (36, 64):
            return depth
        import cv2
        return cv2.resize(depth, (64, 36), interpolation=cv2.INTER_NEAREST).astype(np.float32)


class G1NavOnboardOnnxNode(RealsenseMPCamera, UnitreeNode):
    def __init__(
        self,
        *args,
        sru_policy_onnx_path: str,
        sru_depth_encoder_onnx_path: str,
        deploy_manifest_path: str | None = None,
        debug_dump_dir: str | None = None,
        debug_dump_every_steps: int = 1,
        debug_dump_max_steps: int = 1000,
        debug_dump_stats_only: bool = False,
        sru_high_level_hz: float = 5.0,
        autonomous_command_timeout_s: float = 0.5,
        camera_frame_timeout_s: float = 0.25,
        auto_start_parkour: bool = False,
        joystick_topic: str = "/wirelesscontroller",
        goal_pose_topic: str = "/instinct_nav/goal_pose",
        goal_reached_threshold: float = 0.5,
        odom_topic: str = "/dog_odom",
        odom_twist_frame: str = "auto",
        odom_timeout_s: float = 0.5,
        use_odom_velocity: bool = True,
        command_min: tuple[float, float, float] = (-1.0, -0.6, -1.0),
        command_max: tuple[float, float, float] = (1.0, 0.6, 1.0),
        command_deadband: tuple[float, float, float] = (0.02, 0.02, 0.02),
        command_slew_rate: tuple[float, float, float] = (0.4, 0.3, 0.6),
        command_lowpass_alpha: float = 0.5,
        fall_tilt_threshold_deg: float = 55.0,
        fall_tilt_duration_s: float = 0.15,
        safety_log_dir: str | None = None,
        safety_telemetry_hz: float = 10.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.available_agents = {}
        self.current_agent_name: str | None = None
        self._joystick = UnitreeJoyStick(self, joy_stick_topic=joystick_topic)
        self.base_velocity_cmd = np.zeros(3, dtype=np.float32)
        self.nav_goal_b = np.asarray([1.0, 0.0, 0.0, np.log1p(2.0)], dtype=np.float32)
        self.nav_reset_requested = False
        self._sru_high_level_period_s = 1.0 / float(sru_high_level_hz)
        self._next_sru_update_time_s = 0.0
        self._last_sru_command_time_s = 0.0
        self._autonomous_command_timeout_s = float(autonomous_command_timeout_s)
        self._camera_frame_timeout_s = float(camera_frame_timeout_s)
        self._last_depth_frame_counter_seen = 0
        self._last_depth_frame_timestamp_s = 0.0
        self._auto_start_parkour = bool(auto_start_parkour)
        self._goal_pose_topic = str(goal_pose_topic)
        self._goal_reached_threshold = float(goal_reached_threshold)
        self._odom_topic = str(odom_topic)
        self._odom_timeout_s = float(odom_timeout_s)
        self._last_odom_timestamp_s = 0.0
        self._odom_twist_frame = str(odom_twist_frame)
        if self._odom_twist_frame not in {"auto", "body", "world"}:
            raise ValueError(f"odom_twist_frame must be auto/body/world, got {self._odom_twist_frame!r}")
        self._use_odom_velocity = bool(use_odom_velocity)
        self._goal_pose_subscription = None
        self._odom_subscription = None
        self._goal_start_requested = False
        self._autonomy_fault_latched = False
        self._last_autonomy_fault_reason: str | None = None
        self._last_autonomy_fault_log_time_s = 0.0
        self.base_lin_vel_b = np.zeros(3, dtype=np.float32)
        self._last_shaped_command = np.zeros(3, dtype=np.float32)
        self._last_command_shape_time_s = 0.0
        self._command_min = tuple(float(v) for v in command_min)
        self._command_max = tuple(float(v) for v in command_max)
        self._command_deadband = np.asarray(command_deadband, dtype=np.float32)
        self._command_slew_rate = np.asarray(command_slew_rate, dtype=np.float32)
        self._command_lowpass_alpha = float(np.clip(command_lowpass_alpha, 0.0, 1.0))
        self._fall_monitor = FallSafetyMonitor(
            threshold_deg=fall_tilt_threshold_deg,
            duration_s=fall_tilt_duration_s,
        )
        self._fall_shutdown_triggered = False
        self._safety_telemetry_period_s = 1.0 / max(float(safety_telemetry_hz), 1.0)
        self._next_safety_telemetry_time_s = 0.0
        self._safety_logger = PersistentSafetyLogger(safety_log_dir) if safety_log_dir is not None else None
        if self._safety_logger is not None:
            self.get_logger().info(f"Persistent safety log enabled: {self._safety_logger.path}")
            self._write_safety_event(
                "node_initialized",
                critical=True,
                dryrun=bool(self.dryrun),
                fall_tilt_threshold_deg=self._fall_monitor.threshold_deg,
                fall_tilt_duration_s=self._fall_monitor.duration_s,
            )

        debug_cfg = None
        if debug_dump_dir is not None:
            debug_cfg = OnnxDebugDumpConfig(
                dump_dir=Path(debug_dump_dir).expanduser(),
                every_steps=int(debug_dump_every_steps),
                max_steps=int(debug_dump_max_steps),
                save_arrays=not bool(debug_dump_stats_only),
            )
            self.get_logger().info(
                f"ONNX SRU debug dump enabled: dir={debug_cfg.dump_dir}, every_steps={debug_cfg.every_steps}, "
                f"max_steps={debug_cfg.max_steps}, save_arrays={debug_cfg.save_arrays}."
            )
        self.sru_runtime = OnnxHighLevelRuntime(
            cfg=OnboardConfig(high_level_hz=float(sru_high_level_hz), command_min=self._command_min, command_max=self._command_max),
            policy_onnx_path=sru_policy_onnx_path,
            depth_encoder_onnx_path=sru_depth_encoder_onnx_path,
            command_sink=None,
            debug_dump_config=debug_cfg,
            manifest_path=deploy_manifest_path,
        )
        self.get_logger().info(f"Loaded high-level SRU ONNX policy from {sru_policy_onnx_path}")
        self.get_logger().info(f"Loaded high-level SRU depth encoder ONNX from {sru_depth_encoder_onnx_path}")

    def register_agent(self, name: str, agent) -> None:
        self.available_agents[name] = agent

    def check_buffers_ready(self) -> bool:
        if not super().check_buffers_ready():
            return False
        return self._joystick.data.ly is not None

    def start_ros_handlers(self):
        super().start_ros_handlers()
        projected_gravity = self._get_projected_gravity_obs()
        preflight_ok, tilt_deg = self._fall_monitor.preflight_ok(projected_gravity)
        if not preflight_ok:
            self._trigger_fall_shutdown(
                reason="startup torso attitude is unsafe or IMU gravity is invalid",
                tilt_deg=tilt_deg,
                projected_gravity=projected_gravity,
            )
        self._write_safety_event(
            "attitude_preflight_passed",
            critical=True,
            tilt_deg=tilt_deg,
            projected_gravity_b=projected_gravity,
        )
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import JointState
        from tf2_ros import TransformBroadcaster

        self.joint_state_publisher = self.create_publisher(JointState, "joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self._goal_pose_subscription = self.create_subscription(PoseStamped, self._goal_pose_topic, self._on_goal_pose, 10)
        if self._use_odom_velocity:
            self._odom_subscription = self.create_subscription(Odometry, self._odom_topic, self._on_odom, 20)
            self.get_logger().info(f"Using odometry linear velocity from {self._odom_topic} for high-level actor obs.")
        else:
            self.get_logger().warn("Odom velocity disabled; high-level actor base_lin_vel_b will stay zero.")
        self.get_logger().info(
            f"Listening for SRU navigation goals on {self._goal_pose_topic}. "
            "For offline dry-run, publish PoseStamped in frame_id='base_link'."
        )
        main_loop_duration = 0.02
        self.get_logger().info(f"Starting nav ONNX onboard main loop with duration: {main_loop_duration} seconds.")
        self.main_loop_timer = self.create_timer(main_loop_duration, self.main_loop_callback)
        if MAIN_LOOP_FREQUENCY_CHECK_INTERVAL > 1:
            import queue
            self.main_loop_timer_counter = 0
            self.main_loop_timer_counter_time = time.time()
            self.main_loop_callback_time_consumptions = queue.Queue(maxsize=MAIN_LOOP_FREQUENCY_CHECK_INTERVAL)

    def main_loop_callback(self):
        start_time = time.time()
        now_monotonic = time.monotonic()
        self._check_fall_safety(now_monotonic)
        self._maybe_write_safety_telemetry(now_monotonic)
        if self.current_agent_name is None:
            if self.dryrun and self._auto_start_parkour:
                self.get_logger().info("Dry-run auto-start enabled; starting parkour without cold start.")
                self._auto_start_parkour = False
                self.current_agent_name = "parkour"
                self.available_agents[self.current_agent_name].reset()
                self.nav_reset_requested = True
                return
            self.get_logger().info("Starting cold start agent automatically.")
            self.current_agent_name = "cold_start"
            self.available_agents[self.current_agent_name].reset()
            self._write_safety_event("agent_transition", from_agent=None, to_agent="cold_start")
            return
        if self.current_agent_name == "cold_start":
            tjs, status = self.available_agents[self.current_agent_name].step()
            self.send_target_joint_state(tjs)
            if status != AgentStatus.Working:
                self.get_logger().info("ColdStartAgent done; switching to stand agent.", throttle_duration_sec=2.0)
                previous_agent = self.current_agent_name
                self.current_agent_name = "stand" if "stand" in self.available_agents else "parkour"
                self.available_agents[self.current_agent_name].reset()
                self._write_safety_event(
                    "agent_transition", from_agent=previous_agent, to_agent=self.current_agent_name
                )
        elif self.current_agent_name == "stand":
            tjs, _ = self.available_agents[self.current_agent_name].step()
            self.refresh_camera_data()
            self.send_target_joint_state(tjs)
            if self._autonomy_fault_latched:
                self._zero_autonomous_command()
            elif self._auto_start_parkour or self._goal_start_requested:
                reason = "auto_start_parkour enabled" if self._auto_start_parkour else "navigation goal received"
                self.get_logger().info(f"{reason}; switching to parkour agent.", once=True)
                self._auto_start_parkour = False
                previous_agent = self.current_agent_name
                self.current_agent_name = "parkour"
                self.available_agents[self.current_agent_name].reset()
                self._write_safety_event(
                    "agent_transition", from_agent=previous_agent, to_agent="parkour", reason=reason
                )
        elif self.current_agent_name == "parkour":
            if self._joystick.data.R1:
                self.get_logger().warn("R1 pressed; stopping autonomous navigation and switching to stand.")
                self._zero_autonomous_command()
                self.sru_runtime.reset()
                self._goal_start_requested = False
                previous_agent = self.current_agent_name
                self.current_agent_name = "stand"
                self.available_agents[self.current_agent_name].reset()
                self._write_safety_event(
                    "agent_transition", critical=True, from_agent=previous_agent, to_agent="stand", reason="R1"
                )
                return
            self._update_sru_velocity_command_if_due()
            tjs, _ = self.available_agents[self.current_agent_name].step()
            self.send_target_joint_state(tjs)
        if MAIN_LOOP_FREQUENCY_CHECK_INTERVAL > 1:
            self.main_loop_callback_time_consumptions.put(time.time() - start_time)
            self.main_loop_timer_counter += 1
            if self.main_loop_timer_counter % MAIN_LOOP_FREQUENCY_CHECK_INTERVAL == 0:
                values = [self.main_loop_callback_time_consumptions.get() for _ in range(MAIN_LOOP_FREQUENCY_CHECK_INTERVAL)]
                self.get_logger().info(
                    f"Actual main loop frequency: {(MAIN_LOOP_FREQUENCY_CHECK_INTERVAL / (time.time() - self.main_loop_timer_counter_time)):.2f} Hz. "
                    f"Mean callback time: {np.mean(values):.4f} s."
                )
                self.main_loop_timer_counter = 0
                self.main_loop_timer_counter_time = time.time()

    def _update_sru_velocity_command_if_due(self) -> None:
        now = time.monotonic()
        if not self._camera_depth_is_fresh(now):
            self._enter_autonomy_fault("Depth camera frame is stale or unavailable; stopping autonomous navigation.")
            self._next_sru_update_time_s = now + self._sru_high_level_period_s
            return
        if not self._odom_velocity_is_fresh(now):
            self._enter_autonomy_fault("Odometry velocity is stale or unavailable; stopping autonomous navigation.")
            self._next_sru_update_time_s = now + self._sru_high_level_period_s
            return
        if now < self._next_sru_update_time_s:
            if now - self._last_sru_command_time_s > self._autonomous_command_timeout_s:
                self._zero_autonomous_command()
            return
        try:
            packet = G1NodeSensorSource(self).read()
            command = self.sru_runtime.step(packet)
        except Exception as exc:
            self._enter_autonomy_fault(f"SRU high-level update failed: {exc}")
            self._next_sru_update_time_s = now + self._sru_high_level_period_s
            return
        self.nav_reset_requested = False
        self._last_sru_command_time_s = now
        self._next_sru_update_time_s = now + self._sru_high_level_period_s
        self.base_velocity_cmd = self._shape_command(command.as_array(), now)


    def _on_odom(self, msg) -> None:
        self._last_odom_timestamp_s = time.monotonic()
        linear = msg.twist.twist.linear
        vel = np.asarray([linear.x, linear.y, linear.z], dtype=np.float32)
        mode = self._odom_twist_frame
        if mode == "auto":
            child_frame = (msg.child_frame_id or "").strip("/")
            mode = "body" if child_frame in {"base_link", "base", "body", "torso_link", "robot_center"} else "world"
        if mode == "body":
            self.base_lin_vel_b = vel
            return
        q = msg.pose.pose.orientation
        self.base_lin_vel_b = _rotate_world_to_body(vel, np.asarray([q.x, q.y, q.z, q.w], dtype=np.float32))

    def _camera_depth_is_fresh(self, now_monotonic_s: float) -> bool:
        header = getattr(self, "_depth_header", None)
        if header is None:
            depth = self.get_depth_image()
            return depth is not None and np.isfinite(depth).any()
        frame_counter = int(getattr(header, "frame_counter", 0))
        timestamp_wall_s = float(getattr(header, "timestamp", 0.0))
        if frame_counter <= 0 or timestamp_wall_s <= 0.0:
            return False
        if frame_counter != self._last_depth_frame_counter_seen:
            self._last_depth_frame_counter_seen = frame_counter
            self._last_depth_frame_timestamp_s = now_monotonic_s
            return True
        return (now_monotonic_s - self._last_depth_frame_timestamp_s) <= self._camera_frame_timeout_s

    def _odom_velocity_is_fresh(self, now_monotonic_s: float) -> bool:
        if not self._use_odom_velocity:
            return True
        if self._last_odom_timestamp_s <= 0.0:
            return False
        return (now_monotonic_s - self._last_odom_timestamp_s) <= self._odom_timeout_s

    def _shape_command(self, command: np.ndarray, now: float) -> np.ndarray:
        command = np.asarray(command, dtype=np.float32).reshape(3)
        command = np.where(np.abs(command) < self._command_deadband, 0.0, command)
        if self._last_command_shape_time_s <= 0.0:
            dt = self._sru_high_level_period_s
        else:
            dt = max(now - self._last_command_shape_time_s, 1.0e-3)
        max_delta = self._command_slew_rate * dt
        delta = np.clip(command - self._last_shaped_command, -max_delta, max_delta)
        limited = self._last_shaped_command + delta
        shaped = self._command_lowpass_alpha * limited + (1.0 - self._command_lowpass_alpha) * self._last_shaped_command
        self._last_shaped_command = shaped.astype(np.float32)
        self._last_command_shape_time_s = now
        return self._last_shaped_command.copy()

    def _zero_autonomous_command(self) -> None:
        self.base_velocity_cmd = np.zeros(3, dtype=np.float32)
        self._last_shaped_command = np.zeros(3, dtype=np.float32)
        self._last_command_shape_time_s = 0.0

    def _enter_autonomy_fault(self, reason: str) -> None:
        now = time.monotonic()
        repeated = reason == self._last_autonomy_fault_reason
        should_log = (not repeated) or (now - self._last_autonomy_fault_log_time_s > 2.0)
        self._autonomy_fault_latched = True
        self._last_autonomy_fault_reason = reason
        if should_log:
            self.get_logger().error(f"Autonomy fault latched: {reason}")
            self._last_autonomy_fault_log_time_s = now
        self._zero_autonomous_command()
        self.sru_runtime.reset()
        self._goal_start_requested = False
        self.nav_reset_requested = True
        self._write_safety_event("autonomy_fault_latched", critical=True, reason=reason)
        if "stand" in self.available_agents and self.current_agent_name != "stand":
            self.current_agent_name = "stand"
            self.available_agents[self.current_agent_name].reset()

    def set_nav_goal_body(self, goal_b: np.ndarray) -> None:
        self.nav_goal_b = np.asarray(goal_b, dtype=np.float32).reshape(4)
        self.nav_reset_requested = True

    def _on_goal_pose(self, msg) -> None:
        frame_id = msg.header.frame_id.strip("/") if msg.header.frame_id else "base_link"
        if frame_id not in {"base_link", "base", "body", "torso_link"}:
            self.get_logger().warn(
                f"Received goal in frame '{msg.header.frame_id}', but only base_link/body-relative goals are supported. Ignoring.",
                throttle_duration_sec=2.0,
            )
            return
        goal_xyz = np.asarray([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32)
        distance = float(np.linalg.norm(goal_xyz))
        if self._autonomy_fault_latched:
            self.get_logger().warn(
                f"Ignoring navigation goal because autonomy fault is latched: {self._last_autonomy_fault_reason}. Restart the node to re-arm.",
                throttle_duration_sec=2.0,
            )
            self._zero_autonomous_command()
            return
        if distance < self._goal_reached_threshold:
            goal_b = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self._goal_start_requested = False
            self._zero_autonomous_command()
            self.sru_runtime.reset()
            if self.current_agent_name == "parkour" and "stand" in self.available_agents:
                self.get_logger().info("Navigation goal reached; switching back to stand agent.")
                previous_agent = self.current_agent_name
                self.current_agent_name = "stand"
                self.available_agents[self.current_agent_name].reset()
                self._write_safety_event(
                    "agent_transition", from_agent=previous_agent, to_agent="stand", reason="goal_reached"
                )
        else:
            direction = goal_xyz / max(distance, 1.0e-6)
            goal_b = np.asarray([direction[0], direction[1], direction[2], np.log1p(distance)], dtype=np.float32)
            self._goal_start_requested = True
        self.set_nav_goal_body(goal_b)
        self._write_safety_event(
            "body_goal_updated",
            critical=False,
            goal_xyz_b=goal_xyz,
            goal_b=goal_b,
            goal_start_requested=bool(self._goal_start_requested),
        )
        self.get_logger().info(
            f"Updated SRU goal from {self._goal_pose_topic}: xyz_b=({goal_xyz[0]:+.2f}, {goal_xyz[1]:+.2f}, {goal_xyz[2]:+.2f}) m, "
            f"goal_b=({goal_b[0]:+.3f}, {goal_b[1]:+.3f}, {goal_b[2]:+.3f}, {goal_b[3]:+.3f})."
        )

    def _check_fall_safety(self, now_monotonic_s: float) -> None:
        projected_gravity = self._get_projected_gravity_obs()
        triggered, tilt_deg = self._fall_monitor.update(projected_gravity, now_monotonic_s)
        if triggered:
            self._trigger_fall_shutdown(
                reason="sustained torso tilt exceeded fall threshold or IMU gravity became invalid",
                tilt_deg=tilt_deg,
                projected_gravity=projected_gravity,
            )

    def _trigger_fall_shutdown(self, *, reason: str, tilt_deg: float, projected_gravity: np.ndarray) -> None:
        if self._fall_shutdown_triggered:
            raise SystemExit("Fall safety shutdown already triggered.")
        self._fall_shutdown_triggered = True
        self._zero_autonomous_command()
        try:
            self._write_safety_event(
                "fall_safety_shutdown",
                critical=True,
                reason=reason,
                tilt_deg=float(tilt_deg),
                projected_gravity_b=np.asarray(projected_gravity, dtype=np.float32),
                agent=self.current_agent_name,
                joint_pos=self.joint_pos_.copy(),
                joint_vel=self.joint_vel_.copy(),
            )
            self.get_logger().fatal(
                f"Fall safety shutdown: {reason}; torso tilt={tilt_deg:.1f} deg. Turning motors off."
            )
        finally:
            # Motor-off must not be skipped even if persistent logging or ROS
            # logging itself fails while handling the incident.
            self._turn_off_motors()
        raise SystemExit("Fall safety shutdown triggered.")

    def _maybe_write_safety_telemetry(self, now_monotonic_s: float) -> None:
        if self._safety_logger is None or now_monotonic_s < self._next_safety_telemetry_time_s:
            return
        self._next_safety_telemetry_time_s = now_monotonic_s + self._safety_telemetry_period_s
        projected_gravity = self._get_projected_gravity_obs()
        odom_age_s = None
        if self._last_odom_timestamp_s > 0.0:
            odom_age_s = max(0.0, now_monotonic_s - self._last_odom_timestamp_s)
        depth_header = getattr(self, "_depth_header", None)
        depth_frame_counter = int(getattr(depth_header, "frame_counter", 0)) if depth_header is not None else 0
        last_target_position = None
        if self.last_sent_target_joint_state is not None:
            last_target_position = self.last_sent_target_joint_state.position.copy()
        self._write_safety_event(
            "telemetry",
            critical=False,
            agent=self.current_agent_name,
            tilt_deg=float(projected_gravity_tilt_deg(projected_gravity)),
            projected_gravity_b=projected_gravity,
            base_ang_vel_b=self._get_base_ang_vel_obs(),
            base_lin_vel_b=self.base_lin_vel_b,
            base_velocity_cmd=self.base_velocity_cmd,
            raw_high_level_action=self.sru_runtime.last_action.copy(),
            nav_goal_b=self.nav_goal_b,
            joint_pos=self.joint_pos_.copy(),
            joint_vel=self.joint_vel_.copy(),
            last_target_joint_pos=last_target_position,
            max_abs_joint_vel=float(np.max(np.abs(self.joint_vel_))),
            torque_clip_event_count=int(self.torque_clip_event_count),
            last_torque_clip_joint_count=int(self.last_torque_clip_joint_count),
            last_torque_clip_max_position_delta=float(self.last_torque_clip_max_position_delta),
            odom_age_s=odom_age_s,
            depth_frame_counter=depth_frame_counter,
            autonomy_fault_latched=bool(self._autonomy_fault_latched),
        )

    def _write_safety_event(self, event: str, *, critical: bool = False, **fields) -> None:
        if self._safety_logger is None:
            return
        try:
            self._safety_logger.write(event, critical=critical, **fields)
        except Exception as exc:
            self.get_logger().error(f"Failed to write persistent safety event {event!r}: {exc}")

    def destroy_node(self):
        if self._safety_logger is not None:
            self._write_safety_event("node_destroy", critical=True, agent=self.current_agent_name)
        try:
            super().destroy_node()
        finally:
            if self._safety_logger is not None:
                try:
                    self._safety_logger.close()
                except Exception as exc:
                    self.get_logger().error(f"Failed to close persistent safety log: {exc}")


def _rotate_world_to_body(vec_w: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat_xyzw]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-6:
        return np.asarray(vec_w, dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    # Rotation matrix world_from_body, then transpose for body_from_world.
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    vx, vy, vz = [float(v) for v in vec_w]
    return np.asarray(
        [
            r00 * vx + r10 * vy + r20 * vz,
            r01 * vx + r11 * vy + r21 * vz,
            r02 * vx + r12 * vy + r22 * vz,
        ],
        dtype=np.float32,
    )



def main(args: argparse.Namespace) -> None:
    if args.nodryrun and (args.safety_log_dir is None or not args.safety_log_dir.strip()):
        raise ValueError("--safety_log_dir is required with --nodryrun so incident telemetry survives process exit")
    rclpy.init()
    node = G1NavOnboardOnnxNode(
        depth_resolution=(480, 270),
        depth_fps=60,
        joint_pos_protect_ratio=2.0,
        robot_class_name="G1_29Dof_TorsoBase",
        dryrun=not args.nodryrun,
        sru_policy_onnx_path=args.sru_policy_onnx_path,
        sru_depth_encoder_onnx_path=args.sru_depth_encoder_onnx_path,
        deploy_manifest_path=args.deploy_manifest_path,
        debug_dump_dir=args.debug_dump_dir,
        debug_dump_every_steps=args.debug_dump_every_steps,
        debug_dump_max_steps=args.debug_dump_max_steps,
        debug_dump_stats_only=args.debug_dump_stats_only,
        sru_high_level_hz=args.sru_high_level_hz,
        camera_frame_timeout_s=args.camera_frame_timeout_s,
        auto_start_parkour=args.auto_start_parkour,
        joystick_topic=args.joystick_topic,
        goal_pose_topic=args.goal_pose_topic,
        goal_reached_threshold=args.goal_reached_threshold,
        odom_topic=args.odom_topic,
        odom_twist_frame=args.odom_twist_frame,
        odom_timeout_s=args.odom_timeout_s,
        use_odom_velocity=not args.disable_odom_velocity,
        command_min=tuple(args.command_min),
        command_max=tuple(args.command_max),
        command_deadband=tuple(args.command_deadband),
        command_slew_rate=tuple(args.command_slew_rate),
        command_lowpass_alpha=args.command_lowpass_alpha,
        fall_tilt_threshold_deg=args.fall_tilt_threshold_deg,
        fall_tilt_duration_s=args.fall_tilt_duration_s,
        safety_log_dir=args.safety_log_dir,
        safety_telemetry_hz=args.safety_telemetry_hz,
    )
    parkour_agent = ParkourAgent(logdir=args.logdir, ros_node=node, depth_vis=args.depth_vis, pointcloud_vis=args.pointcloud_vis)
    node.register_agent("parkour", parkour_agent)
    stand_agent = ParkourStandAgent(logdir=args.standdir, ros_node=node)
    node.register_agent("stand", stand_agent)
    cold_start_agent = ColdStartAgent(
        startup_step_size=args.startup_step_size,
        ros_node=node,
        joint_target_pos=parkour_agent.default_joint_pos,
        action_terms=parkour_agent.action_terms,
        p_gains=parkour_agent.p_gains * args.kpkd_factor,
        d_gains=parkour_agent.d_gains * args.kpkd_factor,
    )
    node.register_agent("cold_start", cold_start_agent)
    if args.depth_vis or args.pointcloud_vis:
        node.publish_auxiliary_static_transforms("camera_depth_link_transform")
    try:
        # start_ros_handlers() spins while waiting for the initial low-state,
        # IMU and joystick samples.  A joystick emergency stop can therefore
        # raise SystemExit here, before the regular rclpy.spin() begins.  Keep
        # startup inside this try/finally so the camera subprocess and shared
        # memory are released on that path as well.
        node.start_ros_handlers()
        node.get_logger().info("G1NavOnboardOnnxNode is ready to run.")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Node shutdown complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 SRU navigation onboard node using ONNX high-level policy")
    parser.add_argument("--standdir", type=str, required=True)
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--sru_policy_onnx_path", type=str, required=True)
    parser.add_argument("--sru_depth_encoder_onnx_path", type=str, required=True)
    parser.add_argument("--deploy_manifest_path", type=str, default=None)
    parser.add_argument("--debug_dump_dir", type=str, default=None)
    parser.add_argument("--debug_dump_every_steps", type=int, default=1)
    parser.add_argument("--debug_dump_max_steps", type=int, default=1000)
    parser.add_argument("--debug_dump_stats_only", action="store_true", default=False)
    parser.add_argument("--sru_high_level_hz", type=float, default=5.0)
    parser.add_argument("--camera_frame_timeout_s", type=float, default=0.25)
    parser.add_argument("--startup_step_size", type=float, default=0.2)
    parser.add_argument("--kpkd_factor", type=float, default=2.0)
    parser.add_argument("--depth_vis", action="store_true", default=False)
    parser.add_argument("--pointcloud_vis", action="store_true", default=False)
    parser.add_argument("--nodryrun", action="store_true", default=False)
    parser.add_argument("--auto_start_parkour", action="store_true", default=False)
    parser.add_argument("--joystick_topic", type=str, default="/wirelesscontroller")
    parser.add_argument("--goal_pose_topic", type=str, default="/instinct_nav/goal_pose")
    parser.add_argument("--goal_reached_threshold", type=float, default=0.5)
    parser.add_argument("--odom_topic", type=str, default="/dog_odom")
    parser.add_argument("--odom_twist_frame", type=str, choices=("auto", "body", "world"), default="auto")
    parser.add_argument("--odom_timeout_s", type=float, default=0.5)
    parser.add_argument("--disable_odom_velocity", action="store_true", default=False)
    parser.add_argument("--command_min", type=float, nargs=3, default=(-1.0, -0.6, -1.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--command_max", type=float, nargs=3, default=(1.0, 0.6, 1.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--command_deadband", type=float, nargs=3, default=(0.02, 0.02, 0.02), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--command_slew_rate", type=float, nargs=3, default=(0.4, 0.3, 0.6), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--command_lowpass_alpha", type=float, default=0.5)
    parser.add_argument("--fall_tilt_threshold_deg", type=float, default=55.0)
    parser.add_argument("--fall_tilt_duration_s", type=float, default=0.15)
    parser.add_argument("--safety_log_dir", type=str, default=None)
    parser.add_argument("--safety_telemetry_hz", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
