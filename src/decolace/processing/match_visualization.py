#| label: setup
#| code-fold: true
import bpy

import molecularnodes as mn

def clear_scene():
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="MESH")
    bpy.ops.object.delete()
    for node in bpy.data.node_groups:
        if node.type == "GEOMETRY":
            bpy.data.node_groups.remove(node)

def orient_camera(location, ortho_scale = 500):
    camera = bpy.data.objects['Camera']
    camera.location = location
    camera.data.type = 'ORTHO'
    camera.rotation_euler = (0,0,0)
    camera.data.ortho_scale = ortho_scale
    camera.data.clip_end = 10000

    # camera.data.dof.focus_distance = 1.2

def render_image(path, engine = 'eevee', x = 1000, y = 1000):
    # setup render engine
    if engine == "eevee":
        bpy.context.scene.render.engine = "BLENDER_EEVEE"
    elif engine == "cycles":
        
        bpy.context.scene.render.engine = "CYCLES"
        try:
            bpy.context.scene.cycles.device = "GPU"
        except:
            print("GPU Rendering not available")
    

    # Render

    bpy.context.scene.render.resolution_x = x
    bpy.context.scene.render.resolution_y = y
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    #display(Image(filename=path))

def render_aa(project, aa, mtj, filterset_name, output_path):

    bpy.ops.wm.open_mainfile(filepath="/nrs/elferich/THP1_brequinar/ribosome.blend")

    star_path = project.project_path / "Matches" /f"{aa.area_name}_{mtj.run_name}_{mtj.run_id}_filtered.star"
    if not star_path.exists():
        return
    print(star_path)
    obj = mn.star.load_star_file(file_path=str(star_path))
    print(obj.name)
    print(f"{list(bpy.data.images.keys())}")
    size = bpy.data.images[0].size

    print(f"Image size: {size[0]}")
    major = max(size)
    modifier = obj.modifiers["MolecularNodes"]
    modifier["Input_8"] = -1000.0
    modifier["Input_2"] = bpy.data.objects["Ribosome"]
    orient_camera((size[0]/20,size[1]/20,500), ortho_scale = major/10)

    render_image(str(output_path), engine='eevee',x=int(size[0]),y=int(size[1]))