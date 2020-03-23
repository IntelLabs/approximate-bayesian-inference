#!/usr/bin/python3
import time
from PIL import Image
import numpy as np
import transformations as tf

from pyViewer.viewer import CScene, CNode, CTransform, COffscreenWindowManager, CGLFWWindowManager
from pyViewer.geometry_makers import make_mesh

try:
    from matplotlib import cm
    matplotlib_enabled = True
except ModuleNotFoundError:
    matplotlib_enabled = False


def get_visibility_percent(img_full, img_ind, i, save_images=False):
    index = np.max(img_ind)
    # handle the case when the target object is not visible in the frame
    if index == 0:
        return 0.0

    total = np.sum(img_ind == index)
    partial = np.sum(img_full == index)

    if save_images:
        img_full_mark = np.copy(img_full)
        img_full_mark[img_full == index] = 0xffffffff
        pil_image = Image.frombytes("RGBA", img_full.shape[0:2], img_full_mark)
        pil_image.save("semantic_images/semantic_%d_obj_%d.png" % (i, index & 0xffffffff), "PNG")

        pil_image = Image.frombytes("RGBA", img_ind.shape[0:2], img_ind)
        pil_image.save("semantic_images/semantic_%d_obj_%d_single.png" % (i, index & 0xffffffff), "PNG")

    return partial/np.float(total)


def semantic_render_with_occlusion(scene, camera_positions=[(0.7, 0.7, 2)], width=100, height=100, camera_K=None, show=False):

    # Get the full scene renderings
    full_images = semantic_render(scene, camera_positions, width, height, camera_K, show)

    # Get individual object renderings for each full scene camera config
    individual_object_images = np.zeros((len(camera_positions), len(scene["ids"]),
                                         full_images[0].shape[0], full_images[0].shape[1], full_images[0].shape[2]),
                                        dtype=np.uint8)

    i = 0
    for mesh, trans, rot, id in zip(scene["meshes"], scene["translations"], scene["rotations"], scene["ids"]):
        scene_single = dict()
        scene_single["meshes"] = [mesh]
        scene_single["translations"] = [trans]
        scene_single["rotations"] = [rot]
        scene_single["ids"] = [id]
        imgs = semantic_render(scene_single, camera_positions, width, height, camera_K, show)

        for j in range(len(imgs)):
            individual_object_images[j, i] = np.copy(imgs[j])  # Image j and object i
        i = i + 1

    # Get occlusion
    occlusions = []
    for j, img_full in enumerate(full_images):  # For each full image
        occlusion_per_object = []
        for i in range(len(individual_object_images[j])):    # Compute occlusion for each individual object
            occ = get_visibility_percent(img_full.view(np.uint32), individual_object_images[j][i].view(np.uint32), j, save_images=False)
            occlusion_per_object.append(occ)
        occlusions.append(occlusion_per_object)

    return full_images, occlusions


def semantic_render(scene, camera_positions=[(0.7, 0.7, 2)], width=100, height=100, camera_K=None, show=False, max_retries=1000):
    #####################################################
    # Visualizer initialization
    #####################################################
    # Load scene
    window_manager = COffscreenWindowManager()
    if show:
        window_manager = CGLFWWindowManager()

    viz = CScene(name='Intel Labs::SSR::VU Depth Renderer. javier.felip.leon@intel.com', width=width, height=height, window_manager=window_manager)

    if camera_K is not None:
        viz.camera.set_intrinsics(width, height,
                                  camera_K[0,0], camera_K[1,1], camera_K[0,2], camera_K[1,2], camera_K[0,1])

    # Load objects from the object list
    object_meshes = scene["meshes"]
    object_translations = scene["translations"]
    object_rotations = scene["rotations"]
    object_ids = scene["ids"]
    for i in range(len(object_meshes)):
        object_node = CNode(geometry=make_mesh(viz, object_meshes[i], scale=1.0), id=object_ids[i],
                            transform=CTransform(tf.compose_matrix(translate=object_translations[i], angles=object_rotations[i])))
        viz.insert_graph([object_node])

    #####################################################
    # Image render loop
    #####################################################
    seg_images = np.zeros((len(camera_positions), width, height, 4), dtype=np.uint8)

    i = 0
    retries = 0
    while i < len(camera_positions) and retries < max_retries:
        cpos = camera_positions[i]

        # Move camera
        cam = viz.camera
        cam.alpha = cpos[0]
        cam.beta = cpos[1]
        cam.r = cpos[2]
        cam.update()

        # Clear scene and render
        viz.clear()
        viz.draw()
        img = np.asarray(viz.get_semantic_image())
        seg_images[i] = img.reshape(width, height, 4)
        viz.swap_buffers()

        if show:
            time.sleep(0.1)

        # Check that the image is properly generated
        if np.sum(np.array(seg_images[i])) == 0:
            print("WARNING!!!: Generated blank image!. Maybe objects are out of sight?")
            print(viz)
            # print("WARNING!!!: Generated blank image!. This is an unknown bug of the renderer. Retry. Scene params:", scene)
            # i = i-1
            # retries += 1
        i = i + 1

    viz.__del__()
    return seg_images


def depth_render(scene, camera_positions=[(0.7, 0.7, 2)], width=100, height=100, camera_K=None, show=False):
    #####################################################
    # Visualizer initialization
    #####################################################
    # Load scene
    window_manager = COffscreenWindowManager()
    if show:
        window_manager = CGLFWWindowManager()

    viz = CScene(name='Intel Labs::SSR::VU Depth Renderer. javier.felip.leon@intel.com', width=width, height=height, window_manager=window_manager)

    if camera_K is not None:
        viz.camera.set_intrinsics(width, height,
                                  camera_K[0, 0], camera_K[1, 1], camera_K[0, 2], camera_K[1, 2], camera_K[0, 1])

    # Load objects from the object list
    object_meshes = scene["meshes"]
    object_translations = scene["translations"]
    object_rotations = scene["rotations"]
    for i in range(len(object_meshes)):

        object_node = CNode(geometry=make_mesh(viz, object_meshes[i], scale=1.0),
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
        if show:
            time.sleep(0.1)

    viz.__del__()
    return depth_images


if __name__ == "__main__":

    np.random.seed(1)

    # Define the scene to be rendered with a list of meshes, positions and orientations
    scene = dict()
    scene["meshes"] = ["../models/intel_cup/intel_cup.obj",
                       "../models/duck/duck.obj",
                       "../models/ball/red_ball.obj",
                       "../models/duck/duck.obj"]

    scene["translations"] = [(0, 0, 0), (-0.05, 0.2, 0.05), (0, -0.2, 0), (0, 0, 0.2)]
    scene["rotations"] = [(0, 0, 0), (-1.57, 0, 0), (0, 0, 0), (0, 0, 0)]
    scene["ids"] = [np.random.randint(0, 2**24-1) & 0xffffffff for _ in range(4)]

    max_dist = 1.5

    # Define the camera parameters (e.g. from a Realsense D435 camera @ VGA resolution)
    K = np.array([[613.223, 0.      , 313.568],
                  [0.     , 613.994 , 246.002],
                  [0.     , 0.      , 1.0    ]])

    # Define the list of camera positions to render the scene from
    cameras = list()
    for i in range(10):
        cameras.append(np.random.uniform(low=(-np.pi, -np.pi, 0.1), high=(np.pi, np.pi, max_dist)))

    # Generate depth images for the defined scene, camera positions, camera parameters and resolution
    t_ini = time.time()
    images_depth = depth_render(scene, cameras, height=600, width=800, show=False, camera_K=K)
    t_elapsed = time.time() - t_ini
    print("Generated %d depth images in %3.3fs | %3.3ffps" % (len(cameras), t_elapsed, len(cameras)/t_elapsed))

    # Convert depth images with a colormap and save
    for i, img in enumerate(images_depth):
        if matplotlib_enabled:
            image_cm = np.uint8(cm.viridis(np.array(img) / max_dist) * 255)
            pil_image = Image.frombytes("RGBA", img.size, image_cm)
        else:
            image_cm = np.uint8((np.array(img) / max_dist) * 255)
            pil_image = Image.frombytes("L", img.size, image_cm)
        pil_image.save("depth_images/depth_%d.png" % i, "PNG")

    t_ini = time.time()
    images_sem, occlusions = semantic_render_with_occlusion(scene, cameras, height=600, width=800, show=False, camera_K=K)
    t_elapsed = time.time() - t_ini
    print("Generated %d semantic images in %3.3fs | %3.3ffps" % (len(cameras), t_elapsed, len(cameras)/t_elapsed))

    # Convert semantic images with a colormap and save
    for i, img in enumerate(images_sem):
        img_fp = img.view(np.uint8)
        pil_image = Image.frombytes("RGBA", img.shape[0:2], img_fp)
        pil_image.save("semantic_images/semantic_%d.png" % i, "PNG")

    for i,occ in enumerate(occlusions):
        print("Frame %d. Occlusions:" % i)
        for o, id, mesh in zip(occ, scene["ids"], scene["meshes"]):
            print("ID: %08d. Occlusion: %5.3f Mesh: %s" % (id, 1-o, mesh.split("/")[-1]))
