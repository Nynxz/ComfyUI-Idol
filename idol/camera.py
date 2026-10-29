"""The camera Idol predicts next to the object: a 5-vector [direction (3), 1 / distance, half-width].

Idol's object frame is z-up. ComfyUI keeps gaussian splats in the 3DGS/PLY frame, (x, y, z) -> (x, -z, y), and
its Load 3D / Render Splat cameras live in three.js world space (right-handed, Y-up), which is that frame with
(x, y, z) -> (x, -y, -z). Composed: object (x, y, z) -> world (x, z, -y).
"""

import math

import torch


def decode_camera(camera: torch.Tensor) -> dict:
    """camera [5] -> direction (object frame, unit), radius, vertical fov (degrees), orthographic flag."""
    direction, inverse_distance, half_width = camera[:3].float(), float(camera[3]), float(camera[4])
    norm = float(torch.linalg.norm(direction))
    dx, dy, dz = (direction / norm).tolist() if norm > 1e-6 else (0.0, 0.0, 1.0)
    orthographic = inverse_distance < 1e-3
    if orthographic:
        radius = 1000.0
        fov = math.degrees(2 * math.atan(half_width / radius))
    else:
        radius = 1.0 / inverse_distance
        fov = math.degrees(2 * math.atan(half_width * inverse_distance))
    return {"direction": (dx, dy, dz), "radius": radius, "fov": fov, "orthographic": orthographic}


def world_position(cam: dict):
    """Camera position in three.js world space (the object sits at the origin)."""
    dx, dy, dz = cam["direction"]
    r = cam["radius"]
    return (r * dx, r * dz, -r * dy)


def orbit_angles(cam: dict):
    """(yaw, pitch) in degrees in the convention of ComfyUI's Create Camera Info 'orbit' mode:
    position = distance * (cos(pitch) sin(yaw), sin(pitch), cos(pitch) cos(yaw))."""
    x, y, z = world_position(cam)
    return math.degrees(math.atan2(x, z)), math.degrees(math.asin(max(-1.0, min(1.0, y / cam["radius"]))))


def _lookat_quaternion(position, target):
    """three.js lookAt: camera local +Z = eye - target, up = world +Y. Returns (w, x, y, z)."""
    z = torch.tensor(position, dtype=torch.float64) - torch.tensor(target, dtype=torch.float64)
    z = z / z.norm().clamp_min(1e-8)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    if abs(float(z.dot(up))) > 0.999:  # looking straight up or down
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    x = torch.linalg.cross(up, z)
    x = x / x.norm().clamp_min(1e-8)
    y = torch.linalg.cross(z, x)
    m = torch.stack([x, y, z], dim=1)  # columns are the camera's world axes
    # rotation matrix -> quaternion, picking the largest component for stability
    t = [1 + m[0, 0] + m[1, 1] + m[2, 2], 1 + m[0, 0] - m[1, 1] - m[2, 2],
         1 - m[0, 0] + m[1, 1] - m[2, 2], 1 - m[0, 0] - m[1, 1] + m[2, 2]]
    i = max(range(4), key=lambda k: float(t[k]))
    s = math.sqrt(max(float(t[i]), 1e-12)) * 2
    if i == 0:
        q = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif i == 1:
        q = [(m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif i == 2:
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s]
    else:
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4]
    q = torch.tensor([float(v) for v in q], dtype=torch.float64)
    q = q / q.norm()
    return q.tolist()


def camera_info(cam: dict, target=(0.0, 0.0, 0.0)) -> dict:
    """A LOAD3D_CAMERA dict (what Render Splat, Create Camera Info and the 3D previews use) looking at `target`."""
    position = world_position(cam)
    w, x, y, z = _lookat_quaternion(position, target)
    xyz = lambda v: {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
    return {"position": xyz(position), "target": xyz(target), "quaternion": {"x": x, "y": y, "z": z, "w": w},
            "fov": float(cam["fov"]), "cameraType": "perspective", "zoom": 1.0}
