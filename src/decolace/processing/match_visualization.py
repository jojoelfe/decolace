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

def render_aa(project, aa, mtj, filterset_name, output_path, blender_template):
    mn.register()
    bpy.ops.wm.open_mainfile(filepath=str(blender_template))

    star_path = project.project_path / "Matches" /f"{aa.area_name}_{mtj.run_name}_{mtj.run_id}_filtered.star"
    if not star_path.exists():
        return
    obj = mn.entities.ensemble.ui.load_starfile(file_path=str(star_path))
    
    
    bpy.data.node_groups["MN_starfile_NewStarInstances"].nodes["Starfile Instances"].inputs[1].default_value = bpy.data.objects["Ribosome"]
    bpy.data.node_groups["MN_starfile_NewStarInstances"].nodes["Starfile Instances"].inputs[5].default_value = True
    bpy.data.node_groups["MN_starfile_NewStarInstances"].nodes["Starfile Instances"].inputs[9].default_value = 0.25
    bpy.context.evaluated_depsgraph_get().update()
    
    print(f"{list(bpy.data.images.keys())}")
    size = bpy.data.images[0].size
    #size = (1000,1000)
    print(f"Image size: {size[0]}")
    major = max(size)
    bpy.data.node_groups["MN_starfile_NewStarInstances"].nodes["Starfile Instances"].inputs[5].default_value = False
    orient_camera((size[0]/20,size[1]/20,500), ortho_scale = major/10)

    render_image(str(output_path), engine='cycles',x=int(size[0]),y=int(size[1]))
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path.parent / f"{output_path.stem}.blend"))
    # Exit blender
    bpy.ops.wm.quit_blender()