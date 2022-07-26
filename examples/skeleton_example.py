#!/usr/bin/python3
import time
import numpy as np
import transforms3d as tf
import moderngl as mgl
from pyViewer.viewer import CScene, CPointCloud, CNode, CTransform, CEvent, CGLFWWindowManager
from pyViewer.geometry_makers import make_mesh
from pyViewer.models import REFERENCE_FRAME_MESH, FLOOR_MESH


def make_skeleton(scene):
    # Skeleton definition
    skeleton_node = CNode(geometry=CPointCloud(scene),
                          transform=CTransform(
                              tf.affines.compose(T=[0,1,0],
                                                 R=np.eye(3),
                                                 Z=np.ones(3))))
    skeleton_node.geom.draw_mode = mgl.LINES
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

    skeleton_data = np.zeros((len(skeleton_joint_names), 7))
    skeleton_node.geom.set_data(skeleton_data)

    skeleton = {"links": skeleton_links, "names": skeleton_joint_names, "pos": skeleton_joint_pos, "node": skeleton_node}
    return skeleton


def render_skeleton(skeleton, color=(0,0,1,1), thickness=5):
    color = np.array(color)
    skeleton_data = np.array([])
    skeleton_node = skeleton["node"]
    skeleton_node.geom.size = thickness
    for link in skeleton["links"]:
        link1 = link[0]
        link2 = link[1]
        link1_pos = skeleton["pos"][link1]
        link2_pos = skeleton["pos"][link2]
        skeleton_data = np.concatenate((skeleton_data, link1_pos, color)) if skeleton_data.size else np.concatenate((link1_pos, color))
        skeleton_data = np.concatenate((skeleton_data, link2_pos, color))
    skeleton_node.geom.set_data(skeleton_data.astype(np.float32))


def skeleton_example():

    #####################################################
    # Visualizer initialization
    #####################################################
    # Load scene
    scene = CScene(name='Intel Labs::SSR::VU Depth Renderer. javier.felip.leon@intel.com', width=800, height=600, window_manager = CGLFWWindowManager())

    # Set the camera position
    scene.camera.r = 5.0
    scene.camera.update()

    # Example floor
    floor_node = CNode(geometry=make_mesh(scene, FLOOR_MESH, scale=1.0),
                       transform=CTransform(tf.affines.compose(T=[0,0,-0.02],
                                                               R=np.eye(3),
                                                               Z=np.ones(3))))
    scene.insert_graph([floor_node])

    # Example reference frame size 1.0
    nodes1 = CNode(geometry=make_mesh(scene, REFERENCE_FRAME_MESH, scale=1.0))
    scene.insert_graph([nodes1])

    # Set initial joint positions for visualization. See details of the skeleton definition in make_skeleton()
    skeleton = make_skeleton(scene)
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
    scene.insert_graph([skeleton["node"]])

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
        # Add some animation to the example
        skeleton["pos"][12][0] = np.sin(time.time()) / 2
        skeleton["pos"][13][0] = -skeleton["pos"][12][0]

        # Draw skeleton
        render_skeleton(skeleton, color=(1, 0, 1, 1), thickness=8)
        timings["draw_skeleton"] = time.time() - tic

        tic = time.time()
        scene.swap_buffers()
        timings["swap"] = time.time() - tic

        timings["all"] = time.time() - t_ini
        print(timings)
        time.sleep(0.001)


if __name__ == "__main__":
    skeleton_example()
