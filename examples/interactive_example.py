#!/usr/bin/python3
import time
import numpy as np
import transformations as tf
from PIL import Image

try:
    from matplotlib import cm
    matplotlib_enabled = True
except ModuleNotFoundError:
    matplotlib_enabled = False

from pyViewer.viewer import CScene, CPointCloud, CNode, CTransform, CEvent, CImage, CGLFWWindowManager, CFloatingText, CLinePlot, CBarPlot, CLines
from pyViewer.geometry_makers import make_mesh, make_objects
from pyViewer.models import REFERENCE_FRAME_MESH, FLOOR_MESH


def interactive_example():
    ###################################################################################################################
    # Example create empty scenes with windows
    scene = CScene(name='Intel Labs::SSR::VU Depth Renderer. javier.felip.leon@intel.com',
                   width=640, height=480, window_manager=CGLFWWindowManager())
    scene.set_window_pos((0, 0))
    # Example create a secondary scene
    scene2 = CScene(name='Depth Renderer. Second window example',
                    width=640, height=480, window_manager=CGLFWWindowManager())
    scene2.set_window_pos((700, 0))
    # Optional: Set fon face and sizes
    scene.set_font(font_size=64, font_color=(255, 255, 255, 255), background_color=(0, 0, 0, 0))

    # Optional: Set camera parameters (e.g. from a Realsense D435 camera @ VGA resolution)
    camera_k = np.array([[613.223, 0.00000, 313.568],
                         [0.00000, 613.994, 246.002],
                         [0.00000, 0.00000, 1.00000]])

    scene.camera.set_intrinsics(scene.width, scene.height,
                                camera_k[0, 0], camera_k[1, 1], camera_k[0, 2], camera_k[1, 2], camera_k[0, 1])

    scene2.camera.set_intrinsics(scene2.width, scene2.height,
                                 camera_k[0, 0], camera_k[1, 1], camera_k[0, 2], camera_k[1, 2], camera_k[0, 1])
    ###################################################################################################################

    ###################################################################################################################
    # Example point cloud
    pcnode = CNode(geometry=CPointCloud(scene))
    pcdata = np.random.rand(100000 * 7).astype(np.float32)
    pcnode.geom.set_data(pcdata)
    pcnode.geom.size = 2
    scene.insert_graph([pcnode])

    # Example point cloud for second window
    pcnode2 = CNode(geometry=CPointCloud(scene2))
    pcdata2 = np.random.rand(100000 * 7).astype(np.float32)
    pcnode2.geom.set_data(pcdata2)
    pcnode2.geom.size = 2
    scene2.insert_graph([pcnode2])
    ###################################################################################################################

    ###################################################################################################################
    # Example load multiple object meshes
    objects_path = ["../models/table/table.obj"]  # Target objects models
    objects_pose = [[0.6, 0, -0.65, 0, 0, 0]]

    # Load n_targets random objects at random locations
    n_targets = 10
    lower = np.array([0.2, -0.4, 0.2, 0, 0, 0])  # Random object pos(x,y,z) and rot(x,y,z) boundaries
    upper = np.array([0.7, 0.4, 0.25, np.pi, np.pi, np.pi])
    object_pool_path = ["../models/duck/duck.obj", "../models/intel_cup/intel_cup.obj"]
    for i in range(n_targets):
        objects_path.append(object_pool_path[np.random.randint(0, len(object_pool_path))])
        objects_pose.append(np.random.uniform(lower, upper))

    objects_ids = [np.random.randint(0, 2 ** 24 - 1) & 0xffffffff for _ in range(len(objects_path))]
    object_nodes = make_objects(scene, objects_path, objects_pose, objects_ids)
    scene.insert_graph(object_nodes)
    ###################################################################################################################

    ###################################################################################################################
    # Example load single object mesh
    floor_node = CNode(geometry=make_mesh(scene, FLOOR_MESH, scale=1.0),
                       transform=CTransform(tf.compose_matrix(translate=[0, 0, -0.65])))
    scene.insert_graph([floor_node])
    ###################################################################################################################

    ###################################################################################################################
    # Example camera facing text. It has to be drawn last because of the transparency order
    textnode = CNode(geometry=CFloatingText(scene), transform=CTransform(tf.compose_matrix(translate=[0, 0, 1.0])))
    textnode.geom.set_text("Camera facing text example.")
    textnode.geom.set_position((0, 0, 1))
    textnode.geom.set_height(0.1)
    textnode.geom.camera_facing = False
    scene.insert_graph([textnode])
    ###################################################################################################################

    ###################################################################################################################
    # Example image nodes
    image_display = CImage(scene)
    image_display.set_texture("../textures/intel_labs.png")
    image_display.set_position((-1, 0.6), (0.4, 0.4))
    imgnode = CNode(geometry=image_display)
    scene.insert_graph([imgnode])

    # Example image node for the semantic segmentation
    image_seg_display = CImage(scene)
    image_seg_display.set_texture("../textures/intel_labs.png")
    image_seg_display.set_position((-1, 0.2), (0.4, 0.4))
    imgseg_node = CNode(geometry=image_seg_display)
    scene.insert_graph([imgseg_node])

    # Example image node for the rgb image buffer
    image_rgb_display = CImage(scene)
    image_rgb_display.set_texture("../textures/intel_labs.png")
    image_rgb_display.set_position((-1, -0.2), (0.4, 0.4))
    imgrgb_node = CNode(geometry=image_rgb_display)
    scene.insert_graph([imgrgb_node])
    ###################################################################################################################

    ###################################################################################################################
    # Example standalone line plot
    plot_display = CLinePlot(scene)
    plot_display_node = CNode(geometry=plot_display,
                              transform=CTransform(tf.compose_matrix(translate=[0, 1, 0],
                                                                     angles=[np.pi/2, 0, np.pi/2])))
    scene.insert_graph([plot_display_node])
    plot_display.set_x_ticks(20, c=(0, 0, 0, 1))
    plot_display.set_y_ticks(5, c=(0, 0, 0, 1))
    plot_display.set_x_lines(5, c=(0, .4, .7, 0.2))
    plot_display.set_y_lines(5, c=(0, .4, .7, 0.2))
    plot_display.is_transparent = True
    plot_display.line_width = 2
    plot_display.set_x_lim(0, 100)
    plot_display.set_y_lim(-1.1, 1.1)
    ###################################################################################################################

    ###################################################################################################################
    # Example standalone bar plot
    bplot_display = CBarPlot(scene)
    bplot_display_node = CNode(geometry=bplot_display,
                               transform=CTransform(tf.compose_matrix(translate=[0, 2.1, 0],
                                                                      angles=[np.pi/2, 0, np.pi/2])))

    scene.insert_graph([bplot_display_node])
    bplot_display.set_x_ticks(20, c=(0, 0, 0, 1))
    bplot_display.set_y_ticks(5, c=(0, 0, 0, 1))
    bplot_display.set_x_lines(5, c=(0, .4, .7, 0.2))
    bplot_display.set_y_lines(5, c=(0, .4, .7, 0.2))
    bplot_display.is_transparent = True
    bplot_display.line_width = 2
    bplot_display.set_x_lim(0, 10)
    bplot_display.set_y_lim(0, 0.1)
    ###################################################################################################################

    ###################################################################################################################
    # Example lines
    lines_display = CLines(scene)
    lines_data = np.array([0, 0, 0, 1, 0, 0, 1,
                           1, 1, 1, 0, 1, 0, 0], dtype=np.float32)
    lines_display.set_data(lines_data)
    lines_display.line_width = 5
    lines_display.is_transparent = True
    lines_display_node = CNode(geometry=lines_display)
    scene.insert_graph([lines_display_node])
    ###################################################################################################################

    ###################################################################################################################
    # Main Loop
    ###################################################################################################################
    depth_image = None
    timings = dict()
    is_done = False
    t_start = time.time()

    plt_points_x = []
    plt_points_y = []

    while not is_done:
        pcdata = np.random.rand(10000 * 7).astype(np.float32)
        pcnode.geom.set_data(pcdata)
        pcnode2.geom.set_data(pcdata)

        t_ini = tic = time.time()
        for event in scene.get_events():
            if event.type == CEvent.QUIT:
                is_done = True
            scene.process_event(event)

        for event in scene2.get_events():
            if event.type == CEvent.QUIT:
                is_done = True
            scene2.process_event(event)
        timings["events"] = time.time() - tic

        # Update traced plot
        tic = time.time()
        # Update plot data
        plt_points_x.append(time.time()-t_start)
        plt_points_y.append(np.sin(time.time()-t_start))
        # Draw plot
        plot_display.plot_append(time.time()-t_start, np.sin(time.time()-t_start), c=(0, .44, .77, 1))
        bplot_display.plot(y=np.array([v for v in timings.values()], dtype=np.float32), stdev=np.zeros(len(timings)), c=(0,.44,.77,1))
        timings["traced_plot"] = time.time() - tic

        # Draw scene
        tic = time.time()
        scene.clear()
        scene.draw(use_ortho=False)
        scene2.clear((0, .44, .77, 1))
        scene2.draw(use_ortho=True)
        timings["draw"] = time.time() - tic

        # Draw debug items before swap buffers (text)
        tic = time.time()
        mouse_x = int(scene.wm.get_mouse_pos()[0])
        mouse_y = int(scene.wm.get_mouse_pos()[1])
        if depth_image is not None:
            if 0 < mouse_x < depth_image.width and 0 < mouse_y < depth_image.height:
                scene.draw_text("Depth (%d, %d): %.3f" % (mouse_x, mouse_y, depth_image.transpose(Image.FLIP_TOP_BOTTOM).getpixel((mouse_x, mouse_y))), (20, 60), scale=1)

        scene.draw_text(str({k: str(round(v*1000.0, 3))+"ms" if isinstance(v, float) else v for k, v in timings.items()}), (20, 20), scale=0.5)
        timings["text"] = time.time() - tic

        tic = time.time()
        scene.swap_buffers()
        scene2.swap_buffers()
        timings["swap_buffers"] = time.time() - tic

        # Move camera in scene 2 around the center
        scene2.camera.alpha = time.time() * 0.2
        scene2.camera.camera_matrix = scene2.camera.look_at(scene2.camera.focus_point, scene2.camera.up_vector)

        # Get the rgb image from the renderbuffer
        tic = time.time()
        rgb_image = scene.get_render_image()
        timings["read_rgb"] = time.time() - tic

        # Get the depth and semantic segmentation from the auxiliary render buffer
        tic = time.time()
        depth_image, seg_image = scene.get_depth_and_semantic_image()
        timings["read_depth_segmented"] = time.time() - tic

        tic = time.time()
        if matplotlib_enabled:
            texture_image = scene.get_depth_colormap(depth_image, cm.viridis)
        else:
            image_cm = np.uint8((np.array(depth_image) / 20) * 255)
            texture_image = Image.frombytes("L", (image_cm.shape[1], image_cm.shape[0]), image_cm)

        image_seg_display.set_texture(seg_image)
        image_display.set_texture(texture_image)
        image_rgb_display.set_texture(rgb_image)
        timings["show_rgb_depth_seg"] = time.time() - tic

        timings["all"] = time.time() - t_ini

        print({k: str(round(v * 1000.0, 3)) + "ms" if isinstance(v, float) else v for k, v in timings.items()})

        time.sleep(0.001)
    ###################################################################################################################
    ###################################################################################################################


if __name__ == "__main__":
    interactive_example()

