"""Blender-only: render a supplied textured mesh to real conditioning images."""
import argparse
import json
import math
from pathlib import Path
import sys
import bpy
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument('--mesh', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
args.output.mkdir(parents=True, exist_ok=False)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=str(args.mesh.resolve()))
bpy.context.view_layer.update()
objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
assert objects
points = [o.matrix_world @ Vector(corner) for o in objects for corner in o.bound_box]
lo = Vector([min(v[i] for v in points) for i in range(3)])
hi = Vector([max(v[i] for v in points) for i in range(3)])
center = (lo + hi) / 2
scale = max(hi - lo)
for pos, power, size in [((2,-3,4),180,3),((-3,-1,2),120,3),((0,3,3),180,3)]:
    bpy.ops.object.light_add(type='AREA', location=center+Vector(pos)*scale)
    light=bpy.context.object;light.data.energy=power*scale*scale;light.data.size=size*scale
    light.rotation_euler=(center-light.location).to_track_quat('-Z','Y').to_euler()
bpy.ops.object.camera_add()
camera=bpy.context.object;camera.data.type='ORTHO';camera.data.ortho_scale=1.35*scale
scene=bpy.context.scene;scene.camera=camera
scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(1,1,1,1);scene.world.node_tree.nodes['Background'].inputs[1].default_value=0.7
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16
scene.render.resolution_x=512;scene.render.resolution_y=512;scene.render.resolution_percentage=100
scene.render.film_transparent=False;scene.render.image_settings.file_format='PNG'
scene.view_settings.view_transform='Standard';scene.view_settings.look='Medium High Contrast'
for index,offset in enumerate([(0,-4,0.7),(4,0,0.7),(0,4,0.7),(-4,0,0.7)]):
    camera.location=center+Vector(offset)*scale
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler()
    scene.render.filepath=str(args.output/f'view_{index}.png')
    bpy.ops.render.render(write_still=True)
(args.output/'render.json').write_text(json.dumps({'source_mesh':str(args.mesh),'renderer':'Blender 3.5.1 Cycles CPU','camera_offsets':[(0,-4,0.7),(4,0,0.7),(0,4,0.7),(-4,0,0.7)],'resolution':[512,512]},indent=2)+'\n')
