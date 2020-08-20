#!/usr/bin/python3
import time
import numpy as np
from pyViewer.viewer import CScene, CPointCloud, CNode, CTransform, CEvent, CImage, CGLFWWindowManager, CFloatingText, CLinePlot, CBarPlot, CLines, CCamera
from pyViewer.geometry_makers import make_mesh, make_objects
from pyViewer.models import REFERENCE_FRAME_MESH, FLOOR_MESH
import transformations as tf


def plots_example():
    ###################################################################################################################
    # Example create empty scenes with windows
    scene = CScene(name='Intel Labs::SSR::VU Depth Renderer. javier.felip.leon@intel.com',
                   width=640, height=480, window_manager=CGLFWWindowManager())
    scene.set_window_pos((0, 0))
    scene.set_font(font_size=64, font_color=(255, 255, 255, 255), background_color=(0, 0, 0, 0))

    # Example reference frame size 0.1
    nodes1 = CNode(geometry=make_mesh(scene, REFERENCE_FRAME_MESH, scale=0.1))
    scene.insert_graph([nodes1])

    ###################################################################################################################
    # Example standalone line plot
    plot_display = CLinePlot(scene)
    plot_display_node = CNode(geometry=plot_display,
                              transform=CTransform(tf.compose_matrix(translate=[0, 0, 0],
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
                               transform=CTransform(tf.compose_matrix(translate=[0, 1.1, 0],
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

    is_done = False
    timings = dict()
    plt_points_x = []
    plt_points_y = []
    t_start = time.time()

    while not is_done:
        t_ini = tic = time.time()
        for event in scene.get_events():
            if event.type == CEvent.QUIT:
                is_done = True
            scene.process_event(event)
        timings["events"] = time.time() - tic

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
        timings["draw"] = time.time() - tic

        tic = time.time()
        scene.swap_buffers()
        timings["swap_buffers"] = time.time() - tic

        timings["all"] = time.time() - t_ini
        print({k: str(round(v * 1000.0, 3)) + "ms" if isinstance(v, float) else v for k, v in timings.items()})
        time.sleep(0.001)


if __name__ == "__main__":
    plots_example()
