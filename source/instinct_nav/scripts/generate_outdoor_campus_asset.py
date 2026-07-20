#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def mat(n, c, r=0.85):
    return """    def Material \"%s\"
    {
        token outputs:surface.connect = </Campus/Looks/%s/Shader.outputs:surface>
        def Shader \"Shader\"
        {
            uniform token info:id = \"UsdPreviewSurface\"
            color3f inputs:diffuseColor = (%.3f, %.3f, %.3f)
            float inputs:roughness = %.3f
            token outputs:surface
        }
    }
""" % (n, n, c[0], c[1], c[2], r)


def cube(n, m, t, s, coll=False):
    api = " (\n        prepend apiSchemas = [\"PhysicsCollisionAPI\"]\n    )" if coll else ""
    return """    def Cube \"%s\"%s
    {
        double size = 1
        rel material:binding = </Campus/Looks/%s>
        double3 xformOp:translate = (%.3f, %.3f, %.3f)
        double3 xformOp:scale = (%.3f, %.3f, %.3f)
        uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:scale\"]
    }
""" % (n, api, m, t[0], t[1], t[2], s[0], s[1], s[2])


def ground_mesh():
    return """    def Mesh \"Ground\" (
        prepend apiSchemas = [\"PhysicsCollisionAPI\"]
    )
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-34, -26, 0), (34, -26, 0), (34, 26, 0), (-34, 26, 0)]
        uniform token subdivisionScheme = \"none\"
        rel material:binding = </Campus/Looks/Grass>
    }
"""


def cyl(n, m, t, rad, h):
    return """    def Cylinder \"%s\"
    {
        double radius = %.3f
        double height = %.3f
        rel material:binding = </Campus/Looks/%s>
        double3 xformOp:translate = (%.3f, %.3f, %.3f)
        uniform token[] xformOpOrder = [\"xformOp:translate\"]
    }
""" % (n, rad, h, m, t[0], t[1], t[2])


def sph(n, m, t, rad):
    return """    def Sphere \"%s\"
    {
        double radius = %.3f
        rel material:binding = </Campus/Looks/%s>
        double3 xformOp:translate = (%.3f, %.3f, %.3f)
        uniform token[] xformOpOrder = [\"xformOp:translate\"]
    }
""" % (n, rad, m, t[0], t[1], t[2])


def build_usda():
    looks = "".join([
        mat("Grass", (0.18, 0.38, 0.20)), mat("Asphalt", (0.08, 0.09, 0.09)),
        mat("Sidewalk", (0.58, 0.56, 0.51)), mat("Brick", (0.46, 0.20, 0.15)),
        mat("Concrete", (0.45, 0.47, 0.46)), mat("TreeBark", (0.33, 0.19, 0.10)),
        mat("TreeLeaf", (0.08, 0.31, 0.13)), mat("LampMetal", (0.08, 0.09, 0.10)),
        mat("LampGlow", (1.0, 0.78, 0.32), 0.25), mat("BlueMarker", (0.12, 0.32, 0.80)),
    ])
    geom = [
        ground_mesh(),
        cube("MainRoad", "Asphalt", (0, -8, 0.005), (24, 2, 0.025)),
        cube("CrossPath", "Sidewalk", (-4, 0, 0.02), (1.5, 22, 0.03)),
        cube("NorthWalk", "Sidewalk", (0, 7, 0.02), (23, 1.3, 0.03)),
        cube("SouthWalk", "Sidewalk", (0, -12, 0.02), (23, 1, 0.03)),
        cube("GentleRamp", "Concrete", (8, 0, 0.18), (5, 2, 0.18)),
        cube("CampusHall", "Brick", (15, 9.5, 1.6), (4.5, 2.7, 1.6)),
        cube("LabBlock", "Concrete", (-18, 10, 1.25), (3.5, 3, 1.25)),
        cube("LowWallNorth", "Concrete", (0, 19.5, 0.45), (18, 0.25, 0.45)),
        cube("LowWallWest", "Concrete", (-22.5, 0, 0.35), (0.25, 12, 0.35)),
        cube("BlueLandmark", "BlueMarker", (3, 13, 1), (0.8, 0.8, 1)),
    ]
    for i, (x, y) in enumerate([(-15, -2), (-11, 4), (-8, 13), (0, 13), (8, 13), (13, 3), (18, -3), (-17, -14)]):
        geom.extend([cyl(f"TreeTrunk_{i:02d}", "TreeBark", (x, y, 0.75), 0.16, 1.5), sph(f"TreeCrown_{i:02d}", "TreeLeaf", (x, y, 1.85), 0.95)])
    for i, (x, y) in enumerate([(-13, -10.5), (-5, -10.5), (3, -10.5), (11, -10.5), (-11, 8.5), (6, 8.5)]):
        geom.extend([cyl(f"LampPost_{i:02d}", "LampMetal", (x, y, 1.2), 0.06, 2.4), sph(f"LampGlow_{i:02d}", "LampGlow", (x, y, 2.45), 0.22)])
    return (
        "#usda 1.0\n"
        "(\n"
        "    defaultPrim = \"Campus\"\n"
        "    metersPerUnit = 1\n"
        "    upAxis = \"Z\"\n"
        ")\n\n"
        "def Xform \"Campus\" {\n"
        "  def Scope \"Looks\" {\n"
        + looks
        + "  }\n"
        + "  def Xform \"Geometry\" {\n"
        + "".join(geom)
        + "  }\n}\n"
    )

def main():
    ap = argparse.ArgumentParser(description="Generate a self-contained outdoor campus USD showcase scene.")
    ap.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parents[3] / "assets" / "showcase" / "outdoor_campus")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "campus.usd"
    out.write_text(build_usda())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
