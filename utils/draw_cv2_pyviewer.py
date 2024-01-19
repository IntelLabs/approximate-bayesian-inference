
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT

import pyViewer
import numpy as np
import pybullet as p
import cv2

def draw_line(a, b, color=[1, 0, 0, 1], width=1, lifetime=0, physicsClientId=0, img=None, camera=None):
    if img is not None and camera is not None:
        a_img, jac = cv2.projectPoints(np.array([a]), camera.rvec, camera.tvec, camera.intrinsics, distCoeffs=None)
        b_img, jac = cv2.projectPoints(np.array([b]), camera.rvec, camera.tvec, camera.intrinsics, distCoeffs=None)

        clr = ( int(color[2]*255), int(color[1]*255), int(color[0]*255) )

        cv2.line(img=img, pt1=tuple(a_img.reshape(-1).astype(int)), pt2=tuple(b_img.reshape(-1).astype(int)),
                 color=clr, thickness=width)

    if isinstance(physicsClientId, pyViewer.CScene):
        physicsClientId.draw_line(np.array(a, np.float32),
                                  np.array(b, np.float32),
                                  color=np.array(color, np.float32),
                                  thickness=width)

    elif physicsClientId is not None:
        return p.addUserDebugLine(a, b, color, width, lifetime, physicsClientId=physicsClientId)


def draw_point(pt, color=[1, 0, 0], size=0.1, width=1, lifetime=0, physicsClientId=0, img=None, camera=None):
    res = []
    lxa = [pt[0] - (size / 2), pt[1], pt[2]]
    lxb = [pt[0] + (size / 2), pt[1], pt[2]]
    lya = [pt[0], pt[1] - (size / 2), pt[2]]
    lyb = [pt[0], pt[1] + (size / 2), pt[2]]
    lza = [pt[0], pt[1], pt[2] - (size / 2)]
    lzb = [pt[0], pt[1], pt[2] + (size / 2)]
    res.append(draw_line(lxa, lxb, color=color, width=width, lifetime=lifetime, physicsClientId=physicsClientId, img=img, camera=camera))
    res.append(draw_line(lya, lyb, color=color, width=width, lifetime=lifetime, physicsClientId=physicsClientId, img=img, camera=camera))
    res.append(draw_line(lza, lzb, color=color, width=width, lifetime=lifetime, physicsClientId=physicsClientId, img=img, camera=camera))
    return res


def draw_trajectory(traj, color=[1, 0, 0], width=1, lifetime=0, physicsClientId=0, draw_points=True, img=None, camera=None):
    lines = []
    for i in range(1, int(len(traj))):
        lines.append(draw_line(traj[i-1][0:3], traj[i][0:3], color, width, lifetime, physicsClientId, img=img, camera=camera))
        if draw_points:
            lines.extend(draw_point(traj[i][0:3], color, 0.01, width, lifetime, physicsClientId, img=img, camera=camera))

    return lines


def draw_text(text, position, visualizer):
    return p.addUserDebugText(text, textPosition=position, physicsClientId=visualizer)


def draw_trajectory_cov(traj, cov, color_traj=[1, 0, 0], width=1, lifetime=0, physicsClientId=0, color_cov=[1, 0, 0]):
    num_labels = 10

    traj = traj.view(-1, 3).detach()
    cov = cov.view(-1, 3).detach()

    label_gap = int(len(traj)/num_labels)

    lines = []
    # Draw covariance
    for i in range(0, len(cov)):
        ptraj = traj[i][0:3]
        pcov = cov[i]
        pini_x = [ptraj[0] + pcov[0], ptraj[1],           ptraj[2]]
        pend_x = [ptraj[0] - pcov[0], ptraj[1],           ptraj[2]]
        pini_y = [ptraj[0],           ptraj[1] + pcov[1], ptraj[2]]
        pend_y = [ptraj[0],           ptraj[1] - pcov[1], ptraj[2]]
        pini_z = [ptraj[0],           ptraj[1],           ptraj[2] - pcov[2]]
        pend_z = [ptraj[0],           ptraj[1],           ptraj[2] + pcov[2]]

        lines.append(draw_line(pini_x, pend_x, color_cov, 1, lifetime, physicsClientId))
        lines.append(draw_line(pini_y, pend_y, color_cov, 1, lifetime, physicsClientId))
        lines.append(draw_line(pini_z, pend_z, color_cov, 1, lifetime, physicsClientId))

    # Draw mean
    for i in range(1, len(traj)):
        lines.append(draw_line(traj[i-1][0:3], traj[i][0:3], color_traj, width, lifetime, physicsClientId))
        lines.extend(draw_box(np.array(traj[i][0:3]) - 0.001, np.array(traj[i][0:3]) + 0.001, color_traj, width*2, lifetime, physicsClientId))

        if (i % label_gap) == 0 and len(traj[i]) > 3:
            lines.append(p.addUserDebugText("%.3f" % traj[i][3], textPosition=traj[i][0:3]))


    return lines


def draw_trajectory_diff(t1, t2, color=[1, 0, 0], width=1, lifetime=0, physicsClientId=0):
    lines = []

    for i in range(0, len(t1), 3):
        lines.append(draw_line(t1[i:i+3], t2[i:i+3], color, width, lifetime, physicsClientId))
    return lines


def draw_arrow(a, b, color=[1, 0, 0], width=1, lifetime=0, physicsClientId=0):
    obj_ids = []
    a = np.array(a)
    b = np.array(b)
    dist = np.sqrt(np.sum((a-b)*(a-b)))
    obj_ids.extend(draw_point(a, color=color, size=dist/2, width=width, lifetime=lifetime, physicsClientId=physicsClientId))
    obj_ids.append(p.addUserDebugLine([a[0], a[1], a[2]], [b[0], b[1], b[2]], color, width, lifetime, physicsClientId=physicsClientId))
    return obj_ids


def draw_box(a, b, color=[1, 0, 0], width=1, lifetime=0, physicsClientId=0):
    obj_ids = []

    obj_ids.append(draw_line([a[0], a[1], a[2]], [a[0], a[1], b[2]], color, width, lifetime, physicsClientId=physicsClientId))
    obj_ids.append(draw_line([a[0], a[1], a[2]], [a[0], b[1], a[2]], color, width, lifetime, physicsClientId=physicsClientId))
    obj_ids.append(draw_line([a[0], a[1], a[2]], [b[0], a[1], a[2]], color, width, lifetime, physicsClientId=physicsClientId))

    obj_ids.append(draw_line([b[0], b[1], b[2]], [b[0], b[1], a[2]], color, width, lifetime, physicsClientId=physicsClientId))
    obj_ids.append(draw_line([b[0], b[1], b[2]], [b[0], a[1], b[2]], color, width, lifetime, physicsClientId=physicsClientId))
    obj_ids.append(draw_line([b[0], b[1], b[2]], [a[0], b[1], b[2]], color, width, lifetime, physicsClientId=physicsClientId))

    obj_ids.append(draw_line([a[0], a[1], b[2]], [b[0], a[1], b[2]], color, width, lifetime, physicsClientId=physicsClientId))
    obj_ids.append(draw_line([a[0], a[1], b[2]], [a[0], b[1], b[2]], color, width, lifetime, physicsClientId=physicsClientId))

    obj_ids.append(draw_line([a[0], b[1], a[2]], [a[0], b[1], b[2]], color, width, lifetime, physicsClientId=physicsClientId))
    obj_ids.append(draw_line([a[0], b[1], a[2]], [b[0], b[1], a[2]], color, width, lifetime, physicsClientId=physicsClientId))

    obj_ids.append(draw_line([b[0], a[1], a[2]], [b[0], a[1], b[2]], color, width, lifetime, physicsClientId=physicsClientId))
    obj_ids.append(draw_line([b[0], b[1], a[2]], [b[0], a[1], a[2]], color, width, lifetime, physicsClientId=physicsClientId))

    return obj_ids


def draw_point_cloud(points, colors, physicsClientId=0, size=0.1, width=1, img=None, camera=None):
    assert len(points) == len(colors)

    obj_ids = []

    if isinstance(physicsClientId, pyViewer.CScene):
        pts = np.array(points[:,0:3], np.float32).reshape(-1, 3)
        cols = np.array(colors, np.float32).reshape(-1, 4)
        pc = pyViewer.CPointCloud(ctx=physicsClientId.ctx)
        pc.size = size
        pc_data = np.hstack((pts, cols))
        pc.set_data(pc_data.reshape(-1))
        pc.draw(np.matmul(physicsClientId.perspective, physicsClientId.camera.camera_matrix))
        pc.__del__()

    elif physicsClientId is not None:
        for i in range(len(points)):
            draw_point(pt=points[i], color=colors[i], size=size, width=width,
                       physicsClientId=physicsClientId, img=img, camera=camera)

        return obj_ids