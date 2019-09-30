#!/usr/bin/python3
import os
import time
import numpy as np
import transformations as tf
from pyViewer.viewer import CScene, CPointCloud, CNode, CTransform, CEvent, CImage, CGLFWWindowManager
from pyViewer.geometry_makers import make_mesh, make_objects
from pyViewer.models import REFERENCE_FRAME_MESH, FLOOR_MESH

os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"


def make_skeleton():
    # Skeleton definition
    skeleton_joint_names = ["head", "neck", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
                            "l_hip", "r_hip", "l_knee", "r_knee", "l_foot", "r_foot"]
    skeleton_links = [
        [0, 1],     # ["head", "neck"]
        [1, 2],     # ["neck", "l_shoulder"]
        [1, 3],     # ["neck", "r_shoulder"]
        [2, 4],     # ["l_shoulder", "l_elbow"]
        [3, 5],     # ["r_shoulder", "r_elbow"]
        [4, 6],     # ["l_elbow", "l_wrist"]
        [5, 7],     # ["r_elbow", "r_wrist"]
        [1, 8],     # ["neck", "l_hip"]
        [1, 9],     # ["neck", "r_hip"]
        [8, 10],    # ["l_hip", "l_knee"]
        [9, 11],    # ["r_hip", "r_knee"]
        [10, 12],  # ["r_hip", "r_knee"]
        [11, 13],  # ["r_hip", "r_knee"]
    ]

    skeleton_joint_pos = np.zeros((len(skeleton_joint_names), 3))

    skeleton = {"links": skeleton_links, "names": skeleton_joint_names, "pos": skeleton_joint_pos, "base": [0, 0, 0]}
    return skeleton


def render_skeleton(skeleton, scene, color=(0,0,1,1), thickness=5):
    for link in skeleton["links"]:
        link1 = link[0]
        link2 = link[1]
        link1_pos = skeleton["pos"][link1] + skeleton["base"]
        link2_pos = skeleton["pos"][link2] + skeleton["base"]
        scene.draw_line(link1_pos.astype(np.float32),
                        link2_pos.astype(np.float32), np.array(color, np.float32), thickness)


def skeleton_example():

    # Set initial joint positions for visualization. See details of the skeleton definition in make_skeleton()
    skeleton = make_skeleton()
    skeleton["base"] = np.array([1, 0, 0])
    skeleton_joint_pos = skeleton["pos"]
    skeleton_joint_pos[0] = np.array([0, 0, 1.7])       # Head
    skeleton_joint_pos[1] = np.array([0, 0, 1.5])       # Neck
    skeleton_joint_pos[2] = np.array([0, 0.3, 1.5])     # l_shoulder
    skeleton_joint_pos[3] = np.array([0, -0.3, 1.5])    # r_shoulder
    skeleton_joint_pos[4] = np.array([0, 0.4, 1.1])     # l_elbow
    skeleton_joint_pos[5] = np.array([0, -0.4, 1.1])    # r_elbow
    skeleton_joint_pos[6] = np.array([0, 0.5, 0.8])     # l_wrist
    skeleton_joint_pos[7] = np.array([0, -0.5, 0.8])    # r_wrist
    skeleton_joint_pos[8] = np.array([0, 0.2, 0.9])     # l_hip
    skeleton_joint_pos[9] = np.array([0, -0.2, 0.9])    # r_hip
    skeleton_joint_pos[10] = np.array([0, 0.25, 0.4])   # l_knee
    skeleton_joint_pos[11] = np.array([0, -0.25, 0.4])  # r_knee
    skeleton_joint_pos[12] = np.array([0, 0.25, 0])     # l_foot
    skeleton_joint_pos[13] = np.array([0, -0.25, 0])    # r_foot

    #####################################################
    # Visualizer initialization
    #####################################################
    # Load scene
    scene = CScene(name='Intel Labs::SSR::VU Depth Renderer. javier.felip.leon@intel.com', width=800, height=600, window_manager = CGLFWWindowManager())

    # Example floor
    floor_node = CNode(geometry=make_mesh(scene.ctx, FLOOR_MESH, scale=1.0), transform=CTransform( tf.compose_matrix(translate=[0,0,-0.02]) ) )
    scene.insert_graph([floor_node])

    #####################################################
    # Main Loop
    #####################################################
    timings = dict()
    is_done = False
    while not is_done:
        t_ini = time.time()

        # Process events
        for event in scene.get_events():
            if event.type == CEvent.QUIT:
                is_done = True
            scene.process_event(event)

        # Draw scene
        tic = time.time()
        scene.clear()
        scene.draw()
        timings["draw"] = time.time() - tic

        # Draw skeleton
        tic = time.time()
        render_skeleton(skeleton, scene, color=(1, 0, 1, 1), thickness=8)
        timings["draw_skeleton"] = time.time() - tic

        # Draw reference frame
        scene.draw_line(np.array([0, 0, 0], np.float32), np.array([1, 0, 0], np.float32), np.array([1, 0, 0, 1], np.float32), 4)
        scene.draw_line(np.array([0, 0, 0], np.float32), np.array([0, 1, 0], np.float32), np.array([0, 1, 0, 1], np.float32), 4)
        scene.draw_line(np.array([0, 0, 0], np.float32), np.array([0, 0, 1], np.float32), np.array([0, 0, 1, 1], np.float32), 4)

        tic = time.time()
        scene.swap_buffers()
        timings["swap"] = time.time() - tic

        timings["all"] = time.time() - t_ini
        print(timings)


if __name__ == "__main__":
    skeleton_example()
