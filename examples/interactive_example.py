#!/usr/bin/python3
import os
import time
import numpy as np
import transformations as tf
from PIL import Image
import pygame
from pyViewer.viewer import CScene, CPointCloud, CNode, CTransform, CEvent, CImage, CGLFWWindowManager, CPygameWindowManager
from pyViewer.geometry_makers import make_mesh, make_objects
from pyViewer.models import REFERENCE_FRAME_MESH, FLOOR_MESH

os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"


def interactive_example():

    # Obstacles parameters:
    objects_path = ["../models/table/table.obj"]  # Target objects models
    objects_pose = [[0.6, 0, -0.65, 0, 0, 0]]

    # Load n_targets random objects at random locations
    n_targets = 10
    lower = np.array([0.2, -0.4, 0.2, 0, 0, 0])  # Random object pos(x,y,z) and rot(x,y,z) boundaries
    upper = np.array([0.7, 0.4, 0.25, np.pi, np.pi, np.pi])
    object_pool_path = ["../models/duck/duck_vhacd.obj", "../models/intel_cup/intel_cup.obj"]
    for i in range(n_targets):
        objects_path.append(object_pool_path[np.random.randint(0, len(object_pool_path)-1)])
        objects_pose.append(np.random.uniform(lower, upper))

    #####################################################
    # Visualizer initialization
    #####################################################
    # Load scene
    scene = CScene(name='Intel Labs::SSR::VU Depth Renderer. javier.felip.leon@intel.com',
                   width=800, height=600,
                   window_manager=CGLFWWindowManager(), options=pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)

    # Example reference frame size 1.0
    nodes1 = CNode(geometry=make_mesh(scene.ctx, REFERENCE_FRAME_MESH, scale=1.0))
    scene.insert_graph([nodes1])

    # Example floor
    floor_node = CNode(geometry=make_mesh(scene.ctx, FLOOR_MESH, scale=1.0), transform=CTransform( tf.compose_matrix(translate=[0,0,-0.65]) ) )
    scene.insert_graph([floor_node])

    # Example point cloud
    pcnode = CNode(geometry=CPointCloud(scene.ctx))
    pcdata = np.random.rand(10000000 * 7).astype(np.float32)
    pcnode.geom.set_data(pcdata)
    pcnode.geom.size = 5
    scene.insert_graph([pcnode])

    # Object nodes
    object_nodes = make_objects(scene.ctx, objects_path, objects_pose)
    scene.insert_graph(object_nodes)

    # Example image node
    image_display = CImage(scene.ctx)
    image_display.set_texture("../textures/intel_labs.png")
    image_display.set_position((-1, 0.6), (0.4, 0.4))
    imgnode = CNode(geometry=image_display)
    scene.insert_graph([imgnode])

    #####################################################
    # Main Loop
    #####################################################
    depth_image = None
    timings = dict()
    is_done = False
    while not is_done:
        pcdata = np.random.rand(1000 * 7).astype(np.float32)
        pcnode.geom.set_data(pcdata)

        t_ini = time.time()

        for event in scene.get_events():
            if event.type == CEvent.QUIT:
                is_done = True
            scene.process_event(event)

        tic = time.time()

        scene.clear()
        # Draw scene
        scene.draw()

        timings["draw"] = time.time() - tic

        # Draw debug items before swap buffers (lines)
        scene.draw_line(np.array([0, 0, 0], np.float32), np.array([1, 1, 1], np.float32), np.array([1, 0, 0, 1], np.float32), 5)

        tic = time.time()

        # Draw debug items before swap buffers (text)
        mouse_x = int(scene.wm.get_mouse_pos()[0])
        mouse_y = int(scene.wm.get_mouse_pos()[1])
        if depth_image is not None:
            if 0 < mouse_x < depth_image.width and 0 < mouse_y < depth_image.height:
                scene.draw_text("Depth (%d, %d): %.3f" % (mouse_x, mouse_y, depth_image.getpixel((mouse_x, mouse_y))), (20, 60), scale=0.6)
                # print(" Depth (%d, %d): %.3f " % (mouse_x, mouse_y, depth_image.getpixel((mouse_x, mouse_y))))
        scene.draw_text(repr(timings), (20, 20), scale=0.6)
        timings["text"] = time.time() - tic

        scene.swap_buffers()

        tic = time.time()
        # scene.camera.alpha = time.time() * 0.2
        # scene.camera.camera_matrix = scene.camera.look_at(scene.camera.focus_point, scene.camera.up_vector)
        depth_image = scene.get_depth_image()
        # image_cm = np.uint8(cm.viridis(depth_image/scene.far) * 255.0)
        image_bw = np.uint8(np.clip((depth_image / 3.0) * 255, 0, 255))
        texture_image = Image.frombytes("L", depth_image.shape, image_bw)
        # texture_image = Image.frombytes("RGBA", depth_image.shape, image_cm)
        image_display.set_texture(texture_image)

        # Flip depth image to match screen coordinates
        depth_image = Image.frombytes("F", depth_image.shape, depth_image).transpose(Image.FLIP_TOP_BOTTOM)
        timings["read_depth"] = time.time() - tic

        timings["all"] = time.time() - t_ini
        print(timings)


if __name__ == "__main__":
    interactive_example()
