#!/usr/bin/python3
import os
import time
from matplotlib import cm
from PIL import Image
import numpy as np

import pyViewer.transformations as tf
from pyViewer.viewer import CScene, CNode, CTransform, COffscreenWindowManager, CGLFWWindowManager
from pyViewer.geometry_makers import make_mesh

os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"


def depth_render(scene, camera_positions=[(0.7, 0.7, 2)], width=100, height=100, camera_K=None, show=False):
    #####################################################
    # Visualizer initialization
    #####################################################
    # Load scene
    window_manager = COffscreenWindowManager()
    if show:
        window_manager = CGLFWWindowManager()

    viz = CScene(name='Intel Labs::SSR::VU Depth Renderer. javier.felip.leon@intel.com', width=width, height=height, window_manager = window_manager)

    if camera_K is not None:
        viz.camera.set_intrinsics(width, height,
                                  camera_K[0,0], camera_K[1,1], camera_K[0,2], camera_K[1,2], camera_K[0,1])

    # Load objects from the object list
    object_meshes = scene["meshes"]
    object_translations = scene["translations"]
    object_rotations = scene["rotations"]
    for i in range(len(object_meshes)):

        object_node = CNode(geometry=make_mesh(viz.ctx, object_meshes[i], scale=1.0),
                            transform=CTransform(tf.compose_matrix(translate=object_translations[i], angles=object_rotations[i])))
        viz.insert_graph([object_node])

    #####################################################
    # Image render loop
    #####################################################               # FPS data on a i7-6700K CPU @ 4.00GHz + Titan X(Pascal)
    # depth_images = np.zeros((len(camera_positions), width, height))   # 4200 FPS for a batch of 10K 100x100px imgs
    depth_images = [np.zeros((width, height))] * len(camera_positions)  # 4580 FPS for a batch of 10K 100x100px imgs (256x256 795fps) (128x128 4026fps)
    # depth_images = list()                                             # 4496 FPS for a batch of 10K 100x100px imgs
    for i, cpos in enumerate(camera_positions):
        # Move camera
        cam = viz.camera
        cam.alpha = cpos[0]
        cam.beta = cpos[1]
        cam.r = cpos[2]
        cam.camera_matrix = cam.look_at(cam.focus_point, cam.up_vector)

        # Clear scene and render
        viz.clear()
        viz.draw()
        viz.swap_buffers()
        depth_images[i] = viz.get_depth_image()
        # depth_images.append(viz.get_depth_image())

    return depth_images


if __name__ == "__main__":

    # Define the scene to be rendered with a list of meshes, positions and orientations
    scene = dict()
    scene["meshes"] = ["../models/duck/duck_vhacd.obj", "../models/intel_cup/intel_cup.obj"]
    scene["translations"] = [(0, 0, 0), (0, 0.2, 0)]
    scene["rotations"] = [(0, 0, 0), (0, 0, 0.707)]

    max_dist = 1.5

    # Define the camera parameters (e.g. from a Realsense D435 camera @ VGA resolution)
    K = np.array([[613.223, 0.      , 313.568],
                  [0.     , 613.994 , 246.002],
                  [0.     , 0.      , 1.0    ]])

    # Define the list of camera positions to render the scene from
    cameras = list()
    for i in range(1000):
        cameras.append(np.random.uniform(low=(-np.pi, -np.pi, 0.1), high=(np.pi, np.pi, max_dist)))

    # Generate depth images for the defined scene, camera positions, camera parameters and resolution
    t_ini = time.time()
    images = depth_render(scene, cameras, height=480, width=640, show=True, camera_K=K)
    t_elapsed = time.time() - t_ini
    print("Generated %d images in %3.3fs | %3.3ffps" % (len(cameras), t_elapsed, len(cameras)/t_elapsed))

    # Convert images with a colormap and save
    for i, img in enumerate(images):
        image_cm = np.uint8(cm.viridis(img / max_dist) * 255)
        pil_image = Image.frombytes("RGBA", img.shape, image_cm)
        pil_image.save("../depth_images/depth_%d.png" % i, "PNG")
