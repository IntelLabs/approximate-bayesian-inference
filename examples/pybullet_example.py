# TODO: THERE IS A BUG WITH TEXTURES IN THE PYBULLET EXAMPLE

#!/usr/bin/python3
import time
import pybullet as pb
import numpy as np
import transformations as tf
from pyViewer.viewer import CScene, CPointCloud, CNode, CTransform, CEvent, CImage, CGLFWWindowManager
from pyViewer.geometry_makers import make_mesh
from pyViewer.models import REFERENCE_FRAME_MESH, FLOOR_MESH
from pyViewer.pybullet_utils import init_physics, load_simulation, update_pybullet_nodes, make_pybullet_scene
from PIL import Image
try:
    from matplotlib import cm
    matplotlib_enabled = True
except ModuleNotFoundError:
    matplotlib_enabled = False


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
    object_pool_path = ["../models/duck/duck_vhacd.urdf", "../models/intel_cup/intel_cup.urdf"]
    for i in range(n_targets):
        objects_path.append(object_pool_path[np.random.random_integers(0, len(object_pool_path)-1)])
        objects_pose.append(np.random.uniform(lower, upper))
        objects_static.append(0)

    model, obstacles = load_simulation(model_path, objects_path, objects_pose, objects_static)

    #####################################################
    # Visualizer initialization
    #####################################################
    # Load scene
    scene = CScene(name='Intel Labs::SSR::VU Depth Renderer. javier.felip.leon@intel.com', width=800, height=600, window_manager=CGLFWWindowManager())

    # Example reference frame size 1.0
    nodes1 = CNode(geometry=make_mesh(scene, REFERENCE_FRAME_MESH, scale=0.1))
    scene.insert_graph([nodes1])

    # Example floor
    floor_node = CNode(geometry=make_mesh(scene, FLOOR_MESH, scale=1.0),
                       transform=CTransform(tf.compose_matrix(translate=[0, 0, -0.65])))
    scene.insert_graph([floor_node])

    # Load pybullet geometry
    pybullet_nodes = make_pybullet_scene(scene, physicsClientId=sim_id, add_ref_frame=True)
    scene.insert_graph(pybullet_nodes)

    # Example image node
    image_display = CImage(scene)
    image_display.set_texture("../textures/intel_labs.png")
    image_display.set_position((-1, 0.6), (0.4, 0.4))
    imgnode = CNode(geometry=image_display)
    scene.insert_graph([imgnode])

    #####################################################
    # Main Loop
    #####################################################
    timings = dict()
    depth_image = None
    while pb.isConnected(physicsClientId=sim_id):

        t_ini = tic = time.time()
        for event in scene.get_events():
            if event.type == CEvent.QUIT:
                quit()
            scene.process_event(event)
        timings["events"] = time.time() - tic

        tic = time.time()
        pb.stepSimulation(physicsClientId=sim_id)
        timings["sim_step"] = time.time() - tic

        tic = time.time()
        update_pybullet_nodes(pybullet_nodes, physicsClientId=sim_id)
        timings["update_poses"] = time.time() - tic

        tic = time.time()
        scene.clear()
        scene.draw()
        timings["draw"] = time.time() - tic

        tic = time.time()
        # Draw debug items before swap buffers (text)
        mouse_x = int(scene.wm.get_mouse_pos()[0])
        mouse_y = int(scene.wm.get_mouse_pos()[1])
        if depth_image is not None:
            if 0 < mouse_x < depth_image.width and 0 < mouse_y < depth_image.height:
                scene.draw_text("Depth (%d, %d): %.3f" % (mouse_x, mouse_y, depth_image.getpixel((mouse_x, mouse_y))), (20, 60))
                print(" Depth (%d, %d): %.3f " % (mouse_x, mouse_y, depth_image.getpixel((mouse_x, mouse_y))))
        scene.draw_text(str({k: str(round(v * 1000.0, 3)) + "ms" if isinstance(v, float) else v for k, v in timings.items()}), (20, 20))
        timings["text"] = time.time() - tic

        tic = time.time()
        scene.swap_buffers()
        timings["swap_buffers"] = time.time() - tic

        tic = time.time()
        depth_image = scene.get_depth_image()
        if matplotlib_enabled:
            texture_image = scene.get_depth_colormap(depth_image, cm.viridis)
        else:
            image_cm = np.uint8((np.array(depth_image) / 20) * 255)
            texture_image = Image.frombytes("L", (image_cm.shape[1], image_cm.shape[0]), image_cm)
        image_display.set_texture(texture_image)
        timings["read_depth"] = time.time() - tic

        timings["all"] = time.time() - t_ini

        print({k: str(round(v * 1000.0, 3)) + "ms" if isinstance(v, float) else v for k, v in timings.items()})


if __name__ == "__main__":

    pybullet_example()
