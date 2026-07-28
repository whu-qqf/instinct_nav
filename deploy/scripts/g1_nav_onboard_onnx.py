#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rclpy

from instinct_nav_onboard.bridges import SensorSource
from instinct_nav_onboard.config import OnboardConfig
from instinct_nav_onboard.instinct_onboard_bridge import InstinctOnboardVelocityCommandSink
from instinct_nav_onboard.onnx_high_level_runtime import OnnxDebugDumpConfig, OnnxHighLevelRuntime
from instinct_nav_onboard.packets import SensorPacket

from instinct_onboard.agents.base import AgentStatus, ColdStartAgent
from instinct_onboard.agents.parkour_agent import ParkourAgent, ParkourStandAgent
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
            base_lin_vel_b=np.zeros(3, dtype=np.float32),
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
        debug_dump_dir: str | None = None,
        debug_dump_every_steps: int = 1,
        debug_dump_max_steps: int = 1000,
        debug_dump_stats_only: bool = False,
        sru_high_level_hz: float = 5.0,
        autonomous_command_timeout_s: float = 0.5,
        auto_start_parkour: bool = False,
        goal_pose_topic: str = "/instinct_nav/goal_pose",
        goal_reached_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.available_agents = {}
        self.current_agent_name: str | None = None
        self.base_velocity_cmd = np.zeros(3, dtype=np.float32)
        self.nav_goal_b = np.asarray([1.0, 0.0, 0.0, np.log1p(2.0)], dtype=np.float32)
        self.nav_reset_requested = False
        self._sru_high_level_period_s = 1.0 / float(sru_high_level_hz)
        self._next_sru_update_time_s = 0.0
        self._last_sru_command_time_s = 0.0
        self._autonomous_command_timeout_s = float(autonomous_command_timeout_s)
        self._auto_start_parkour = bool(auto_start_parkour)
        self._goal_pose_topic = str(goal_pose_topic)
        self._goal_reached_threshold = float(goal_reached_threshold)
        self._goal_pose_subscription = None
        self._goal_start_requested = False

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
            cfg=OnboardConfig(high_level_hz=float(sru_high_level_hz)),
            policy_onnx_path=sru_policy_onnx_path,
            depth_encoder_onnx_path=sru_depth_encoder_onnx_path,
            command_sink=InstinctOnboardVelocityCommandSink(self),
            debug_dump_config=debug_cfg,
        )
        self.get_logger().info(f"Loaded high-level SRU ONNX policy from {sru_policy_onnx_path}")
        self.get_logger().info(f"Loaded high-level SRU depth encoder ONNX from {sru_depth_encoder_onnx_path}")

    def register_agent(self, name: str, agent) -> None:
        self.available_agents[name] = agent

    def start_ros_handlers(self):
        super().start_ros_handlers()
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState
        from tf2_ros import TransformBroadcaster

        self.joint_state_publisher = self.create_publisher(JointState, "joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self._goal_pose_subscription = self.create_subscription(PoseStamped, self._goal_pose_topic, self._on_goal_pose, 10)
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
        if self.current_agent_name is None:
            self.get_logger().info("Starting cold start agent automatically.")
            self.current_agent_name = "cold_start"
            self.available_agents[self.current_agent_name].reset()
            return
        if self.current_agent_name == "cold_start":
            tjs, status = self.available_agents[self.current_agent_name].step()
            self.send_target_joint_state(tjs)
            if status != AgentStatus.Working:
                self.get_logger().info("ColdStartAgent done; switching to stand agent.", throttle_duration_sec=2.0)
                self.current_agent_name = "stand" if "stand" in self.available_agents else "parkour"
                self.available_agents[self.current_agent_name].reset()
        elif self.current_agent_name == "stand":
            tjs, _ = self.available_agents[self.current_agent_name].step()
            self.refresh_camera_data()
            self.send_target_joint_state(tjs)
            if self._auto_start_parkour or self._goal_start_requested:
                reason = "auto_start_parkour enabled" if self._auto_start_parkour else "navigation goal received"
                self.get_logger().info(f"{reason}; switching to parkour agent.", once=True)
                self.current_agent_name = "parkour"
                self.available_agents[self.current_agent_name].reset()
        elif self.current_agent_name == "parkour":
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
        if now < self._next_sru_update_time_s:
            if now - self._last_sru_command_time_s > self._autonomous_command_timeout_s:
                self.base_velocity_cmd = np.zeros(3, dtype=np.float32)
            return
        packet = G1NodeSensorSource(self).read()
        command = self.sru_runtime.step(packet)
        self.nav_reset_requested = False
        self._last_sru_command_time_s = now
        self._next_sru_update_time_s = now + self._sru_high_level_period_s
        self.base_velocity_cmd = command.as_array()

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
        if distance < self._goal_reached_threshold:
            goal_b = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self._goal_start_requested = False
            self.base_velocity_cmd = np.zeros(3, dtype=np.float32)
            self.sru_runtime.reset()
            if self.current_agent_name == "parkour" and "stand" in self.available_agents:
                self.get_logger().info("Navigation goal reached; switching back to stand agent.")
                self.current_agent_name = "stand"
                self.available_agents[self.current_agent_name].reset()
        else:
            direction = goal_xyz / max(distance, 1.0e-6)
            goal_b = np.asarray([direction[0], direction[1], direction[2], np.log1p(distance)], dtype=np.float32)
            self._goal_start_requested = True
        self.set_nav_goal_body(goal_b)
        self.get_logger().info(
            f"Updated SRU goal from {self._goal_pose_topic}: xyz_b=({goal_xyz[0]:+.2f}, {goal_xyz[1]:+.2f}, {goal_xyz[2]:+.2f}) m, "
            f"goal_b=({goal_b[0]:+.3f}, {goal_b[1]:+.3f}, {goal_b[2]:+.3f}, {goal_b[3]:+.3f})."
        )


def main(args: argparse.Namespace) -> None:
    rclpy.init()
    node = G1NavOnboardOnnxNode(
        depth_resolution=(480, 270),
        depth_fps=60,
        joint_pos_protect_ratio=2.0,
        robot_class_name="G1_29Dof_TorsoBase",
        dryrun=not args.nodryrun,
        sru_policy_onnx_path=args.sru_policy_onnx_path,
        sru_depth_encoder_onnx_path=args.sru_depth_encoder_onnx_path,
        debug_dump_dir=args.debug_dump_dir,
        debug_dump_every_steps=args.debug_dump_every_steps,
        debug_dump_max_steps=args.debug_dump_max_steps,
        debug_dump_stats_only=args.debug_dump_stats_only,
        sru_high_level_hz=args.sru_high_level_hz,
        auto_start_parkour=args.auto_start_parkour,
        goal_pose_topic=args.goal_pose_topic,
        goal_reached_threshold=args.goal_reached_threshold,
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
    node.start_ros_handlers()
    node.get_logger().info("G1NavOnboardOnnxNode is ready to run.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("Node shutdown complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 SRU navigation onboard node using ONNX high-level policy")
    parser.add_argument("--standdir", type=str, required=True)
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--sru_policy_onnx_path", type=str, required=True)
    parser.add_argument("--sru_depth_encoder_onnx_path", type=str, required=True)
    parser.add_argument("--debug_dump_dir", type=str, default=None)
    parser.add_argument("--debug_dump_every_steps", type=int, default=1)
    parser.add_argument("--debug_dump_max_steps", type=int, default=1000)
    parser.add_argument("--debug_dump_stats_only", action="store_true", default=False)
    parser.add_argument("--sru_high_level_hz", type=float, default=5.0)
    parser.add_argument("--startup_step_size", type=float, default=0.2)
    parser.add_argument("--kpkd_factor", type=float, default=2.0)
    parser.add_argument("--depth_vis", action="store_true", default=False)
    parser.add_argument("--pointcloud_vis", action="store_true", default=False)
    parser.add_argument("--nodryrun", action="store_true", default=False)
    parser.add_argument("--auto_start_parkour", action="store_true", default=False)
    parser.add_argument("--goal_pose_topic", type=str, default="/instinct_nav/goal_pose")
    parser.add_argument("--goal_reached_threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
