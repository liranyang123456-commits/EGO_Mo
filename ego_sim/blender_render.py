"""Executed inside Blender: render stereo images described by a JSON job."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix
from bpy_extras.object_utils import world_to_camera_view


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=Path, required=True)
    return ap.parse_args(argv)


def _material(name, color, roughness=0.55, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def _checkerboard(config):
    square = config["square_mm"] / 1000.0
    cols, rows = config["board_cols"], config["board_rows"]
    verts, faces, materials = [], [], []
    for row in range(rows):
        for col in range(cols):
            x0, x1 = (col - 1) * square, col * square
            y0, y1 = (row - 1) * square, row * square
            base = len(verts)
            verts += [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]
            faces.append((base, base + 1, base + 2, base + 3))
            materials.append((row + col) % 2)
    mesh = bpy.data.meshes.new("GP050_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.materials.append(_material("white", (0.86, 0.86, 0.82), 0.7))
    mesh.materials.append(_material("black", (0.008, 0.008, 0.008), 0.65))
    for polygon, material in zip(mesh.polygons, materials):
        polygon.material_index = material
    obj = bpy.data.objects.new("GP050", mesh)
    bpy.context.collection.objects.link(obj)
    solid = obj.modifiers.new("board thickness", "SOLIDIFY")
    solid.thickness = 0.001
    return obj


def _tissue_plane():
    bpy.ops.mesh.primitive_plane_add(size=1.2, location=(0.0, 0.0, -0.006))
    plane = bpy.context.object
    plane.name = "procedural tissue background"
    mat = bpy.data.materials.new("tissue")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 7.0
    noise.inputs["Detail"].default_value = 4.0
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.035, 0.008, 0.006, 1.0)
    ramp.color_ramp.elements[1].color = (0.24, 0.055, 0.045, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.48
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    plane.data.materials.append(mat)


def _camera(name):
    data = bpy.data.cameras.new(name)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    data.clip_start = 0.01
    data.clip_end = 10.0
    return obj


def _set_intrinsics(scene, camera, K, width, height):
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    sensor_width = 36.0
    pixel_aspect_x = 1.0
    pixel_aspect_y = 1.0
    if fx > fy:
        pixel_aspect_y = fx / fy
    else:
        pixel_aspect_x = fy / fx
    pixel_aspect_ratio = pixel_aspect_y / pixel_aspect_x
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = sensor_width
    camera.data.lens = fx * sensor_width / width
    camera.data.shift_x = (cx - (width - 1) * 0.5) / -width
    camera.data.shift_y = (cy - (height - 1) * 0.5) / width * pixel_aspect_ratio
    scene.render.pixel_aspect_x = pixel_aspect_x
    scene.render.pixel_aspect_y = pixel_aspect_y


def _matrix(rows):
    return Matrix(rows)


def main():
    args = _args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    config = job["config"]
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    # Blender 5.2 exposes Eevee as BLENDER_EEVEE.
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = config["width"]
    scene.render.resolution_y = config["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 92
    scene.render.film_transparent = False
    scene.world.color = (0.003, 0.003, 0.005)
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -1.0
    scene.render.filepath = ""

    board = _checkerboard(config)
    _tissue_plane()
    camera = _camera("StereoCamera")
    scene.camera = camera

    bpy.ops.object.light_add(type="AREA", location=(0.05, -0.04, 0.16))
    key = bpy.context.object
    key.data.energy = 1.5 * config["light_level"]
    key.data.shape = "DISK"
    key.data.size = 0.18
    key.rotation_euler = (0.0, 0.0, 0.0)
    bpy.ops.object.light_add(type="POINT", location=(-0.12, 0.08, 0.20))
    fill = bpy.context.object
    fill.data.energy = 0.25 * config["light_level"]
    fill.data.color = (1.0, 0.55, 0.45)

    width, height = config["width"], config["height"]
    left_dir = Path(job["left_dir"])
    right_dir = Path(job["right_dir"])
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(job["frames"]):
        board.matrix_world = _matrix(frame["board_matrix"])
        camera.matrix_world = _matrix(frame["left_matrix"])
        _set_intrinsics(scene, camera, job["K0"], width, height)
        if i == 0:
            sample = []
            step = max(1, len(board.data.vertices) // 8)
            for vertex_index in range(0, len(board.data.vertices), step):
                vertex = board.data.vertices[vertex_index]
                ndc = world_to_camera_view(scene, camera, board.matrix_world @ vertex.co)
                sample.append((round(ndc.x, 3), round(ndc.y, 3), round(ndc.z, 3)))
            print(f"BLENDER_BOARD_NDC {sample}", flush=True)
        scene.render.filepath = str(left_dir / f"{i:06d}.jpg")
        bpy.ops.render.render(write_still=True)
        camera.matrix_world = _matrix(frame["right_matrix"])
        _set_intrinsics(scene, camera, job["K1"], width, height)
        scene.render.filepath = str(right_dir / f"{i:06d}.jpg")
        bpy.ops.render.render(write_still=True)
        print(f"BLENDER_FRAME {i + 1}/{len(job['frames'])}", flush=True)


if __name__ == "__main__":
    main()
