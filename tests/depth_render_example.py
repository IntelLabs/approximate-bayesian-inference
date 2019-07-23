#!/usr/bin/python3
import os
import time
from matplotlib import cm
from PIL import Image
import numpy as np
import copy

import pyViewer.transformations as tf
from pyViewer.viewer import CScene, CNode, CTransform, CEvent, CImage, COffscreenWindowManager, CGLFWWindowManager
from pyViewer.geometry_makers import make_mesh

os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"


def depth_render_example(scene, camera_positions=[(0.7, 0.7, 2)], width=800, height=600, show=False):
    #####################################################
    # Visualizer initialization
    #####################################################
    # Load scene
    window_manager = COffscreenWindowManager()
    if show:
        window_manager = CGLFWWindowManager()

    viz = CScene(name='Intel Labs::SSR::VU Depth Renderer. javier.felip.leon@intel.com', width=width, height=height, window_manager = window_manager)

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
    #####################################################
    # depth_images = np.zeros((len(camera_positions), width, height))   # 4200 FPS for a batch of 10K 100x100px imags
    depth_images = [np.zeros((width, height))] * len(camera_positions)  # 4580 FPS for a batch of 10K 100x100px imags (256x256 795fps) (128x128 4026fps)
    # depth_images = list()                                             # 4496 FPS for a batch of 10K 100x100px imags
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
    scene = dict()
    # scene["meshes"] = ["models/YCB_Dataset/035_power_drill/tsdf/textured.obj"]
    scene["meshes"] = ["models/duck/duck_vhacd.obj", "models/duck/duck_vhacd.obj"]
    scene["translations"] = [(0, 0, 0), (0, 0.2, 0)]
    scene["rotations"] = [(0, 0, 0), (0, 0, 0.707)]

    max_dist = 1

    cameras = list()
    for i in range(1000):
        cameras.append(np.random.uniform(low=(-np.pi, -np.pi, 0.1), high=(np.pi, np.pi, 2.0)))

    # cameras = [(0.7, 0.7, 2), (0.7, 0.7, 1), (0.7, 0.7, 0.5), (0.7, 0.7, 0.2)]

    timings = list()
    n_exeuctions = 10
    for i in range(n_exeuctions):
        t_ini = time.time()
        images = depth_render_example(scene, cameras, height=256, width=256, show=True)
        timings.append(time.time() - t_ini)

    print("Generated %d images in %3.3fs | %3.3ffps" % (len(cameras), np.mean(timings), len(cameras)/np.mean(timings)))

    # Convert images with colormap and save
    for i, img in enumerate(images):
        image_cm = np.uint8(cm.hot(img / max_dist) * 255)
        pil_image = Image.frombytes("RGBA", img.shape, image_cm)
        pil_image.save("depth_images/depth_%d.png" % i, "PNG")
    #     print("Save image: depth_images/depth_%d.png" % i)
