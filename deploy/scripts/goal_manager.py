#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


@dataclass
class PlanarPose:
    x: float
    y: float
    yaw: float


class BodyGoalManager(Node):
    """Convert a persistent world-frame goal into continuously updated base-frame goals."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("instinct_nav_goal_manager")
        self.odom_topic = args.odom_topic
        self.world_goal_topic = args.world_goal_topic
        self.body_goal_topic = args.body_goal_topic
        self.goal_reached_threshold = float(args.goal_reached_threshold)
        self.publish_hz = float(args.publish_hz)
        self.goal: tuple[float, float, float] | None = None
        self.robot_pose: PlanarPose | None = None
        self.goal_active = False
        self.goal_done_latched = False

        self.body_goal_pub = self.create_publisher(PoseStamped, self.body_goal_topic, 10)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 20)
        self.create_subscription(PointStamped, self.world_goal_topic, self._on_world_goal, 10)
        self.timer = self.create_timer(1.0 / max(self.publish_hz, 1.0), self._publish_body_goal)
        self.get_logger().info(
            f"Goal manager ready: odom={self.odom_topic}, world_goal={self.world_goal_topic}, "
            f"body_goal={self.body_goal_topic}, threshold={self.goal_reached_threshold:.2f} m."
        )

    def _on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        yaw = _yaw_from_quat(q.x, q.y, q.z, q.w)
        p = msg.pose.pose.position
        self.robot_pose = PlanarPose(float(p.x), float(p.y), yaw)

    def _on_world_goal(self, msg: PointStamped) -> None:
        p = msg.point
        self.goal = (float(p.x), float(p.y), float(p.z))
        self.goal_active = True
        self.goal_done_latched = False
        self.get_logger().info(
            f"Received world goal: frame={msg.header.frame_id or 'odom'}, "
            f"xyz=({p.x:+.2f}, {p.y:+.2f}, {p.z:+.2f}) m."
        )

    def _publish_body_goal(self) -> None:
        if self.robot_pose is None or self.goal is None or not self.goal_active:
            return
        gx, gy, gz = self.goal
        dx_w = gx - self.robot_pose.x
        dy_w = gy - self.robot_pose.y
        distance = math.hypot(dx_w, dy_w)
        if distance < self.goal_reached_threshold:
            if not self.goal_done_latched:
                self.get_logger().info(f"Goal reached: distance={distance:.3f} m. Publishing stop goal.")
                self.goal_done_latched = True
            self._publish_goal_b(0.0, 0.0, 0.0)
            self.goal_active = False
            return

        cos_yaw = math.cos(self.robot_pose.yaw)
        sin_yaw = math.sin(self.robot_pose.yaw)
        x_b = cos_yaw * dx_w + sin_yaw * dy_w
        y_b = -sin_yaw * dx_w + cos_yaw * dy_w
        self._publish_goal_b(x_b, y_b, gz)

    def _publish_goal_b(self, x_b: float, y_b: float, z_b: float) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = float(x_b)
        msg.pose.position.y = float(y_b)
        msg.pose.position.z = float(z_b)
        msg.pose.orientation.w = 1.0
        self.body_goal_pub.publish(msg)


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PointStamped world goals into continuously updated base-frame goals for InstinctNav onboard control.")
    parser.add_argument("--odom_topic", type=str, default="/dog_odom")
    parser.add_argument("--world_goal_topic", type=str, default="/instinct_nav/world_goal_point")
    parser.add_argument("--body_goal_topic", type=str, default="/instinct_nav/goal_pose")
    parser.add_argument("--goal_reached_threshold", type=float, default=0.5)
    parser.add_argument("--publish_hz", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    rclpy.init()
    node = BodyGoalManager(parse_args())
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
