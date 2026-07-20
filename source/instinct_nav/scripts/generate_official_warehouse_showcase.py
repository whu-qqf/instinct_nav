#!/usr/bin/env python3
"""Generate an Isaac official-asset warehouse showcase wrapper USD.

The wrapper references NVIDIA's official Simple Warehouse visual asset and adds
one local mesh used consistently for ray-cast depth, height scanning, and terrain
collision. IsaacLab's RayCaster currently supports only one static mesh prim, so
all navigable showcase proxy geometry is intentionally merged into RaycastProxy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

OFFICIAL_WAREHOUSE_USD_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "assets" / "showcase" / "official_warehouse" / "official_warehouse_showcase.usd"


def _add_quad(points: list[tuple[float, float, float]], faces: list[tuple[int, int, int]], verts):
    start = len(points)
    points.extend(verts)
    faces.append((start, start + 1, start + 2))
    faces.append((start, start + 2, start + 3))


def _add_box(points: list[tuple[float, float, float]], faces: list[tuple[int, int, int]], center, size):
    cx, cy, cz = center
    sx, sy, sz = size
    x0, x1 = cx - sx / 2.0, cx + sx / 2.0
    y0, y1 = cy - sy / 2.0, cy + sy / 2.0
    z0, z1 = cz - sz / 2.0, cz + sz / 2.0
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    start = len(points)
    points.extend(corners)
    quad_indices = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]
    for a, b, c, d in quad_indices:
        faces.append((start + a, start + b, start + c))
        faces.append((start + a, start + c, start + d))


def build_proxy_mesh() -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    _add_quad(points, faces, [(-20.0, -22.0, 0.0), (20.0, -22.0, 0.0), (20.0, 22.0, 0.0), (-20.0, 22.0, 0.0)])

    _add_box(points, faces, (0.0, -21.5, 1.6), (40.0, 0.6, 3.2))
    _add_box(points, faces, (0.0, 21.5, 1.6), (40.0, 0.6, 3.2))
    _add_box(points, faces, (-19.5, 0.0, 1.6), (0.6, 44.0, 3.2))
    _add_box(points, faces, (19.5, 0.0, 1.6), (0.6, 44.0, 3.2))

    for x in (-8.0, 0.0, 8.0):
        _add_box(points, faces, (x, -8.5, 1.15), (1.1, 11.0, 2.3))
        _add_box(points, faces, (x, 8.5, 1.15), (1.1, 11.0, 2.3))

    for center, size in [
        ((-15.0, -11.0, 0.45), (2.2, 2.0, 0.9)),
        ((-15.0, 9.0, 0.65), (2.6, 2.0, 1.3)),
        ((15.0, -9.0, 0.55), (2.4, 2.4, 1.1)),
        ((15.0, 12.0, 0.75), (2.8, 1.8, 1.5)),
        ((-2.0, 18.0, 0.45), (3.0, 1.5, 0.9)),
        ((10.5, -18.0, 0.45), (2.0, 1.8, 0.9)),
    ]:
        _add_box(points, faces, center, size)

    return points, faces


def _format_vec3(values: list[tuple[float, float, float]]) -> str:
    return ",\n            ".join(f"({x:.3f}, {y:.3f}, {z:.3f})" for x, y, z in values)


def _format_faces(faces: list[tuple[int, int, int]]) -> str:
    return ", ".join(str(i) for tri in faces for i in tri)


def write_official_warehouse_showcase(output_path: str | Path = DEFAULT_OUTPUT) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    points, faces = build_proxy_mesh()
    face_counts = ", ".join("3" for _ in faces)
    text = f'''#usda 1.0
(
    defaultPrim = "OfficialWarehouseShowcase"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "OfficialWarehouseShowcase"
{{
    def Xform "Visual" (
        prepend references = @{OFFICIAL_WAREHOUSE_USD_URL}@
    )
    {{
    }}

    def Xform "Geometry"
    {{
        def Mesh "RaycastProxy" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            uniform bool doubleSided = true
            int[] faceVertexCounts = [{face_counts}]
            int[] faceVertexIndices = [{_format_faces(faces)}]
            point3f[] points = [
            {_format_vec3(points)}
            ]
            color3f[] primvars:displayColor = [(0.37, 0.40, 0.37)] (
                interpolation = "constant"
            )
            float[] primvars:displayOpacity = [0.32] (
                interpolation = "constant"
            )
        }}
    }}
}}
'''
    output.write_text(text, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = write_official_warehouse_showcase(args.output)
    print(output)


if __name__ == "__main__":
    main()
