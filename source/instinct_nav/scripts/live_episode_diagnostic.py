"""Opt-in live observations and PhysX/height-scan comparison, no pose replay."""
from pathlib import Path
import json

import numpy as np


def terrain_query_height(query, origin):
    hits = []
    errors = []

    def collect(hit):
        try:
            path = str(hit.collision)
            body = str(hit.rigid_body)
            if any(p == "/World/ground" or p.startswith("/World/ground/") for p in (path, body)):
                hits.append({"position": list(hit.position), "collision": path, "body": body})
        except Exception as exc:
            # PhysX can swallow callback exceptions; surface them after the query.
            errors.append(exc)
            return False
        return True

    query.raycast_all(tuple(float(v) for v in origin), (0.0, 0.0, -1.0), 100.0, collect)
    if errors:
        raise RuntimeError("Failed to decode PhysX terrain query") from errors[0]
    return max(hits, key=lambda h: h["position"][2]) if hits else None


class LiveEpisodeDiagnostic:
    def __init__(self, directory, env_id, camera):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=False)
        self.env_id = env_id
        self.camera = camera
        self.step = 0

    def capture(self, env, actions):
        import torch
        from PIL import Image
        from omni.physx import get_physx_scene_query_interface
        from isaaclab.utils import math as math_utils

        env = env.unwrapped
        i = self.env_id
        robot = env.scene["robot"]
        if not 0 <= i < env.num_envs:
            raise ValueError("Diagnostic env_id outside environment batch")
        array = lambda v: v.detach().cpu().numpy().copy()
        root = array(robot.data.root_pos_w[i])
        scanner = env.scene["height_scanner_critic"]
        scan = array(scanner.data.ray_hits_w[i])
        valid = np.isfinite(scan).all(axis=1)
        indices = np.flatnonzero(valid)
        nearest = indices[np.argsort(np.linalg.norm(scan[indices, :2] - root[:2], axis=1))[:9]]
        query = get_physx_scene_query_interface()
        probes = []
        for n in nearest:
            origin = scan[n].copy()
            origin[2] = max(float(root[2]), float(origin[2])) + 20.0
            hit = terrain_query_height(query, origin)
            probes.append({"scan_index": int(n), "scan_xyz": scan[n].tolist(),
                           "physx": hit, "height_difference_m":
                           float(hit["position"][2] - scan[n, 2]) if hit else None})
        contact = env.scene["contact_forces"]
        manager = getattr(env, "_dynamic_obstacle_manager", None)
        record = {"step": self.step, "env_id": i,
                  "episode_step": int(env.episode_length_buf[i].item()),
                  "root_pos_w": root.tolist(),
                  "root_lin_vel_w": array(robot.data.root_lin_vel_w[i]).tolist(),
                  "root_quat_w": array(robot.data.root_quat_w[i]).tolist(),
                  "action": array(actions[i]).tolist(),
                  "contact_body_names": list(contact.body_names),
                  "obstacles_enabled": bool(manager.enabled) if manager else None,
                  "spawned_count": int(manager.spawned_obstacles[i].sum().item()) if manager else 0,
                  "nearest_scan_xy_distance_m":
                  float(np.linalg.norm(scan[nearest[0], :2] - root[:2])) if len(nearest) else None,
                  "probes": probes}
        arrays = {"height_scan_w": scan, "joint_pos": array(robot.data.joint_pos[i]),
                  "joint_vel": array(robot.data.joint_vel[i]),
                  "body_pos_w": array(robot.data.body_pos_w[i]),
                  "contact_forces_w": array(contact.data.net_forces_w[i]),
                  "contact_force_history_w": array(contact.data.net_forces_w_history[i])}
        for name in ("camera", "raycast_camera"):
            sensor = env.scene.sensors.get(name)
            if sensor is not None and "distance_to_image_plane" in sensor.data.output:
                arrays[name + "_depth"] = array(sensor.data.output["distance_to_image_plane"][i])
                arrays[name + "_pos_w"] = array(sensor.data.pos_w[i])
        tag = f"frame_{self.step:05d}"
        np.savez_compressed(self.path / (tag + ".npz"), **arrays)
        with (self.path / "states.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        target = torch.tensor(root[None], device=env.device)
        eye = target + torch.tensor([[-4.0, -4.0, 4.0]], device=env.device)
        rot = math_utils.create_rotation_matrix_from_view(eye, target, up_axis="Z", device=env.device)
        self.camera.set_world_poses(eye, math_utils.quat_from_matrix(rot), convention="opengl")
        env.sim.render()
        self.camera.update(env.physics_dt, force_recompute=True)
        rgb = array(self.camera.data.output["rgb"][0])[..., :3].astype(np.uint8)
        Image.fromarray(rgb).save(self.path / (tag + ".png"))
        self.step += 1
