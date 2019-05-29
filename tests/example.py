#!/usr/bin/python3
import os
import time
import pybullet as pb
import numpy as np
import pyViewer.transformations as tf
from pyViewer.viewer import CScene, CPointCloud, CNode, CTransform, CEvent, CImage
from pyViewer.geometry_makers import make_mesh
from pyViewer.models import REFERENCE_FRAME_MESH, FLOOR_MESH
from pyViewer.pybullet_utils import init_physics, load_simulation, update_pybullet_nodes, make_pybullet_scene
from matplotlib import cm
from PIL import Image

os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"


def pybullet_example():

    #####################################################
    # Simulator initialization
    #####################################################
    sim_id = init_physics(False)

    # Obstacles parameters:
    model_path = "../models/human_torso/model.urdf"  # Manipulator model
    objects_path = ["../models/table/table.urdf"]  # Target objects models
    objects_pose = [[0.6, 0, -0.65]]
    objects_static = [1]

    # Load n_targets random objects at random locations
    n_targets = 10
    lower = np.array([0.2, -0.4, 0.2])  # Random object location boundaries
    upper = np.array([0.7, 0.4, 0.25])
    object_pool_path = ["../models/duck/duck_vhacd.urdf"]
    for i in range(n_targets):
        objects_path.append(object_pool_path[np.random.random_integers(0, len(object_pool_path)-1)])
        objects_pose.append(np.random.uniform(lower, upper))
        objects_static.append(0)

    model, obstacles = load_simulation(model_path, objects_path, objects_pose, objects_static)

    #####################################################
    # Visualizer initialization
    #####################################################
    # Load scene
    scene = CScene(name='IL::SSR::VU Probabilistic Computing. javier.felip.leon@intel.com')

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
    scene.insert_graph([pcnode])

    # Example image node
    image_display = CImage(scene.ctx)
    image_display.set_texture("../textures/intel_labs.png")
    image_display.set_position((-1, 0.6), (0.4, 0.4))
    imgnode = CNode(geometry=image_display)
    scene.insert_graph([imgnode])

    # Load pybullet geometry
    pybullet_nodes = make_pybullet_scene(scene.ctx, physicsClientId=sim_id)
    scene.insert_graph(pybullet_nodes)

    #####################################################
    # Main Loop
    #####################################################
    timings = dict()
    while pb.isConnected(physicsClientId=sim_id):
        pcdata = np.random.rand(100000 * 7).astype(np.float32)
        pcnode.geom.set_data(pcdata)

        t_ini = time.time()

        for event in scene.get_events():
            if event.type == CEvent.QUIT:
                quit()

            scene.process_event(event)

        tic = time.time()
        pb.stepSimulation(physicsClientId=sim_id)
        timings["sim_step"] = time.time() - tic

        tic = time.time()
        update_pybullet_nodes(pybullet_nodes, physicsClientId=sim_id)
        timings["update_poses"] = time.time() - tic

        # scene.camera.alpha = time.time() * 0.2
        # scene.camera.camera_matrix = scene.camera.look_at(scene.camera.focus_point, scene.camera.up_vector)
        depth_image = scene.get_depth_image()
        image_cm = np.uint8(cm.hot(depth_image/scene.far) * 255)
        # image_bw = np.uint8(depth_image / scene.far * 255)
        # texture_image = Image.frombytes("L", depth_image.shape, image_bw)
        texture_image = Image.frombytes("RGBA", depth_image.shape, image_cm)
        image_display.set_texture(texture_image)

        # Flip depth image to match screen coordinates
        depth_image = Image.frombytes("F", depth_image.shape, depth_image).transpose(Image.FLIP_TOP_BOTTOM)

        # Do some bit order manipulation to obtain the image in the proper screen coordinates
        # depth_shape = depth_image.shape
        # depth_image = depth_image.reshape(-1, order="C")
        # depth_image = np.flip(depth_image.reshape(depth_shape, order="F"), 1)

        tic = time.time()
        scene.clear()

        scene.draw_line(np.array([0, 0, 0], np.float32), np.array([1, 1, 1], np.float32), np.array([1, 0, 0, 1], np.float32), 5)

        # TODO: BUG. If the text draw is not called inmediately before a scene.draw() the color does not work
        mouse_x = int(scene.wm.get_mouse_pos()[0])
        mouse_y = int(scene.wm.get_mouse_pos()[1])
        if 0 < mouse_x < depth_image.width and 0 < mouse_y < depth_image.height:
            scene.draw_text("Depth (%d, %d): %f" % (mouse_x, mouse_y, depth_image.getpixel((mouse_x, mouse_y))), (20, 60), (1.0, 1.0, 0.0))
        scene.draw_text(repr(timings), (20, 20), (1.0, 1.0, 0.0))

        scene.draw()

        timings["draw"] = time.time() - tic

        timings["all"] = time.time() - t_ini
        print(timings)


if __name__ == "__main__":
    pybullet_example()
