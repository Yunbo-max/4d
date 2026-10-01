"""Render generated geometry with Blender CPU Cycles; does not change the model output.

blender -b -t 8 -P render_preview.py -- /path/to/inference/output
"""
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector

folder = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
vertices = np.load(folder / 'deformations_vertices.npy')
faces = np.load(folder / 'deformations_faces.npy')
assert np.isfinite(vertices).all()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
mesh = bpy.data.meshes.new('Generated geometry')
mesh.from_pydata(vertices[0].tolist(), [], faces.tolist())
mesh.update()
obj = bpy.data.objects.new('ActionMesh output', mesh)
bpy.context.collection.objects.link(obj)
for poly in mesh.polygons:
    poly.use_smooth = True
material = bpy.data.materials.new('Preview blue')
material.use_nodes = True
bsdf = material.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Base Color'].default_value = (0.16, 0.36, 0.72, 1)
bsdf.inputs['Roughness'].default_value = 0.45
obj.data.materials.append(material)
lo = vertices.min(axis=(0, 1))
hi = vertices.max(axis=(0, 1))
center = Vector(((lo + hi) / 2).tolist())
scale = float(max(hi - lo))
bpy.ops.object.camera_add(location=center + Vector((2.4, -4.0, 1.7)) * scale)
camera = bpy.context.object
camera.rotation_euler = (center - camera.location).to_track_quat('-Z', 'Y').to_euler()
camera.data.type = 'ORTHO'
camera.data.ortho_scale = 1.5 * scale
bpy.context.scene.camera = camera
for position, power, size in [((2, -3, 4), 650, 3), ((-3, -1, 2), 350, 3), ((0, 3, 3), 800, 2)]:
    bpy.ops.object.light_add(type='AREA', location=center + Vector(position) * scale)
    light = bpy.context.object
    light.data.energy = power * scale * scale
    light.data.shape = 'DISK'
    light.data.size = size * scale
    light.rotation_euler = (center - light.location).to_track_quat('-Z', 'Y').to_euler()
scene = bpy.context.scene
scene.world.color = (0.3, 0.3, 0.3)
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 16
scene.cycles.use_denoising = True
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGBA'
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'Medium High Contrast'
dest = folder / 'preview_frames'
dest.mkdir(exist_ok=True)
for index, frame in enumerate(vertices):
    mesh.vertices.foreach_set('co', np.asarray(frame, dtype=np.float32).ravel())
    mesh.update()
    scene.render.filepath = str(dest / f'{index:02d}.png')
    bpy.ops.render.render(write_still=True)
print('PREVIEW_FRAMES_COMPLETE', len(vertices))
