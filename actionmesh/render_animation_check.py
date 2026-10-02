"""Blender 3.5.1: validate and render the animation imported from an actual GLB.

blender -b -P render_animation_check.py -- --mesh animated_mesh.glb --output NEWDIR
Only imported GLB geometry, animation, materials and UVs are used; no deformation
arrays or replacement materials participate in this check.
"""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=16,
                        help="At most this many actual animation key times, including endpoints")
    parser.add_argument("--view-index", type=int, choices=(0, 1, 2, 3), default=0,
                        help="Camera view index matching render_application_mesh.py")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    if not 3 <= args.samples <= 16:
        parser.error("--samples must be between 3 and 16")
    if not args.mesh.is_file():
        parser.error("--mesh must name an existing GLB")
    return args


def evaluated_vertices(scene, objects, frame):
    import bpy

    whole_frame = math.floor(frame)
    scene.frame_set(whole_frame, subframe=frame - whole_frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    result = {}
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        try:
            vertices = [tuple(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices]
            if not vertices or not all(math.isfinite(v) for point in vertices for v in point):
                raise ValueError(f"Empty or non-finite evaluated vertices: {obj.name}, frame {frame}")
            result[obj.name] = vertices
        finally:
            evaluated.to_mesh_clear()
    return result


def inspect_materials(objects):
    material_records, image_records = {}, {}
    meshes = []
    for obj in objects:
        meshes.append({
            "object": obj.name, "mesh": obj.data.name,
            "vertices": len(obj.data.vertices), "polygons": len(obj.data.polygons),
            "uv_layers": [{"name": uv.name, "loops": len(uv.data)} for uv in obj.data.uv_layers],
            "material_slots": [slot.material.name if slot.material else None for slot in obj.material_slots],
            "shape_keys": ([key.name for key in obj.data.shape_keys.key_blocks]
                           if obj.data.shape_keys else []),
        })
        for slot in obj.material_slots:
            material = slot.material
            if material is None or material.name in material_records:
                continue
            record = {"name": material.name, "use_nodes": material.use_nodes,
                      "blend_method": material.blend_method, "image_nodes": []}
            if material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type != "TEX_IMAGE" or node.image is None:
                        continue
                    image = node.image
                    record["image_nodes"].append({"node": node.name, "image": image.name})
                    if image.name not in image_records:
                        # Reading pixels also verifies that an embedded/imported texture
                        # is available to the actual renderer, not just named in JSON.
                        pixel_count = len(image.pixels)
                        if pixel_count == 0:
                            raise ValueError(f"Material image has no loaded pixels: {image.name}")
                        pixels = array("f", [0.0]) * pixel_count
                        image.pixels.foreach_get(pixels)
                        if not all(math.isfinite(value) for value in pixels):
                            raise ValueError(f"Non-finite texture pixels: {image.name}")
                        image_records[image.name] = {
                            "name": image.name, "size": list(image.size),
                            "channels": image.channels, "colorspace": image.colorspace_settings.name,
                            "packed": bool(image.packed_file), "source": image.source,
                            "float_pixel_count": pixel_count,
                            "decoded_float32_pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                        }
            material_records[material.name] = record
    return {"meshes": meshes, "materials": list(material_records.values()),
            "images": list(image_records.values()), "texture_image_count": len(image_records),
            "material_policy": "Imported GLB materials and UVs; no replacement or recoloring"}


def run(args, report):
    import bpy
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.mesh.resolve()))
    scene = bpy.context.scene
    objects = sorted((obj for obj in scene.objects if obj.type == "MESH"), key=lambda obj: obj.name)
    if not objects:
        raise ValueError("Imported GLB contains no mesh objects")
    actions, keys = [], set()
    for action in bpy.data.actions:
        times = sorted({float(point.co.x) for curve in action.fcurves for point in curve.keyframe_points})
        if not times:
            continue
        if not all(math.isfinite(value) for value in times):
            raise ValueError(f"Non-finite animation key times: {action.name}")
        keys.update(times)
        actions.append({"name": action.name, "frame_range": list(action.frame_range),
                        "fcurve_count": len(action.fcurves), "key_times": times})
    key_times = sorted(keys)
    if len(key_times) < 2 or key_times[-1] <= key_times[0]:
        raise ValueError("GLB has no animation spanning at least two distinct key times")
    count = min(args.samples, len(key_times))
    selected = [key_times[round(i * (len(key_times) - 1) / (count - 1))] for i in range(count)]
    fps = scene.render.fps / scene.render.fps_base
    report.update(
        blender_version=bpy.app.version_string, renderer="Cycles CPU", samples=16,
        resolution=[512, 512], actions=actions, animation_frame_range=[key_times[0], key_times[-1]],
        imported_scene_fps=fps, selected_key_times=selected,
        selected_relative_seconds=[(frame - key_times[0]) / fps for frame in selected],
        selection="Evenly spaced actual imported key indices, including first and last",
        animation_policy="Evaluate the GLB importer's active animation and default NLA state",
        material_inspection=inspect_materials(objects),
    )
    lower, upper = [math.inf] * 3, [-math.inf] * 3
    reference, motion_records = None, []
    for frame in selected:
        current = evaluated_vertices(scene, objects, frame)
        if reference is None:
            reference = current
        displacement = 0.0
        for name, vertices in current.items():
            if len(vertices) != len(reference[name]):
                raise ValueError(f"Evaluated topology changes vertex count for {name}")
            for point, first in zip(vertices, reference[name]):
                displacement = max(displacement, math.dist(point, first))
                for axis in range(3):
                    lower[axis] = min(lower[axis], point[axis])
                    upper[axis] = max(upper[axis], point[axis])
        motion_records.append({"key_time": frame, "max_world_vertex_displacement_from_first": displacement})
    center = (Vector(lower) + Vector(upper)) / 2
    scale = max(upper[axis] - lower[axis] for axis in range(3))
    threshold = max(1e-8, scale * 1e-7)
    maximum_motion = max(row["max_world_vertex_displacement_from_first"] for row in motion_records)
    report.update(world_bounds={"min": lower, "max": upper}, frame_geometry=motion_records,
                  world_vertex_counts={name: len(vertices) for name, vertices in reference.items()},
                  evaluated_world_vertices_finite=True,
                  max_world_vertex_displacement=maximum_motion, motion_threshold=threshold,
                  evaluated_animation_moves=maximum_motion > threshold)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Degenerate mesh bounds")
    if maximum_motion <= threshold:
        raise ValueError("Imported GLB animation does not change evaluated world-space vertices")
    # Imported materials and meshes remain untouched. Replace only cameras/lights
    # so every frame uses the selected view from render_application_mesh.py.
    for obj in list(scene.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for position, power, size in [((2, -3, 4), 180, 3), ((-3, -1, 2), 120, 3), ((0, 3, 3), 180, 3)]:
        bpy.ops.object.light_add(type="AREA", location=center + Vector(position) * scale)
        light = bpy.context.object
        light.data.energy, light.data.size = power * scale * scale, size * scale
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
    camera_offsets = [(0, -4, 0.7), (4, 0, 0.7), (0, 4, 0.7), (-4, 0, 0.7)]
    camera_offset = camera_offsets[args.view_index]
    bpy.ops.object.camera_add(location=center + Vector(camera_offset) * scale)
    camera = bpy.context.object
    camera.data.type, camera.data.ortho_scale = "ORTHO", 1.35 * scale
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera
    scene.world = bpy.data.worlds.new("AnimationCheckWhiteWorld")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.7
    scene.render.engine, scene.cycles.device, scene.cycles.samples = "CYCLES", "CPU", 16
    scene.cycles.seed = 42
    scene.cycles.use_animated_seed = False
    scene.render.threads_mode, scene.render.threads = "FIXED", 8
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform, scene.view_settings.look = "Standard", "Medium High Contrast"
    report.update(view_index=args.view_index, camera_offset=list(camera_offset), camera_center=list(center),
                  camera_ortho_scale=camera.data.ortho_scale, fixed_camera_and_lighting=True,
                  bounds_policy="Union of evaluated world-space vertices across all selected times",
                  rendered_frames=[])
    for index, frame in enumerate(selected):
        scene.frame_set(math.floor(frame), subframe=frame - math.floor(frame))
        bpy.context.view_layer.update()
        target = args.output / f"frame_{index:03d}.png"
        scene.render.filepath = str(target)
        bpy.ops.render.render(write_still=True)
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"Renderer did not produce {target}")
        report["rendered_frames"].append({"file": target.name, "key_time": frame,
                                          "sha256": sha256(target), "bytes": target.stat().st_size})
        write_json(args.output / "render-check.json", report)
    report["status"] = "verified"


def main():
    args = parse_args()
    args.mesh, args.output = args.mesh.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "source_glb": str(args.mesh),
              "source_sha256": sha256(args.mesh), "uses_deformation_arrays": False}
    try:
        run(args, report)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "render-check.json", report)


if __name__ == "__main__":
    main()
