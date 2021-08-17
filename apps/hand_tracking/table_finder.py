
from ar_markers import detect_markers

import numpy as np
from collections import deque
import cv2
import pyrealsense2 as rs
import copy


def rs_rgb_to_depth(rgb_pixel, color_to_depth_extrin, color_intrin, depth_intrin):
    rgb_point = rs.rs2_deproject_pixel_to_point(color_intrin, rgb_pixel, 1)
    depth_point = rs.rs2_transform_point_to_point(color_to_depth_extrin, rgb_point)
    depth_pixel = rs.rs2_project_point_to_pixel(depth_intrin, depth_point)
    return depth_pixel


def rs_rgb_to_3d(rgb_pixel, depth_image, color_to_depth_extrin, color_intrin, depth_intrin, depth_scale):
    depth_pixel = rs_rgb_to_depth(rgb_pixel=rgb_pixel,
                                  color_to_depth_extrin=color_to_depth_extrin,
                                  color_intrin=color_intrin,
                                  depth_intrin=depth_intrin)

    depth_value = depth_image[int(depth_pixel[1]), int(depth_pixel[0])]
    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(depth_pixel[0]), int(depth_pixel[1])],
                                               depth_value * depth_scale)

    return point_3d


def rs_depth_to_points(depth_image, depth_intrin, depth_scale):
    points = np.zeros([depth_image.shape[0], depth_image.shape[1], 3])
    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            depth_value = depth_image[i, j]
            point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [i, j], depth_value * depth_scale)
            points[i, j] = point_3d

    return points


def fit_plane_model_svd(points, indices, debug_image=None):
    point_subset = np.array([])
    for idx in indices:
        point_subset = np.concatenate((point_subset, points[idx[0], idx[1]]))

    point_subset = point_subset.reshape((-1, 3))
    # Compute the centroid of selected points
    centroid = np.mean(point_subset, axis=0)
    point_subset = point_subset - centroid

    evals, evecs = np.linalg.eig(point_subset.T @ point_subset)

    mineval_idx = int(np.argmin(evals))

    A = evecs[mineval_idx,0]
    B = evecs[mineval_idx,1]
    C = evecs[mineval_idx,2]
    D = np.sqrt(np.dot(centroid, centroid))

    if debug_image is not None:
        camera_intrinsics = np.array([[613.223, 0., 313.568],
                                      [0., 613.994, 246.002],
                                      [0., 0., 1.0]])
        distortion = np.zeros((1,4))

        cv2.drawFrameAxes(debug_image, camera_intrinsics, distortion, evecs, centroid, 0.2, thickness=3)

    return [A, B, C, D]


def fit_plane_model_pinv(points, indices):
    point_subset = np.array([])
    for idx in indices:
        point_subset = np.concatenate((point_subset, points[idx[0], idx[1]]))

    point_subset = point_subset.reshape((-1, 3))
    # Compute the centroid of selected points
    centroid = np.mean(point_subset, axis=0)
    point_subset = point_subset - centroid

    ones = np.ones(len(point_subset)).reshape(-1, 1)
    XY = point_subset[:, 0:2]
    A = np.hstack((XY, ones))
    b = point_subset[:, 2]
    fit = np.linalg.pinv(A.T @ A) @ A.T @ b
    errors = b - A @ fit
    residual = np.linalg.norm(errors)

    return [fit[0], fit[1], 0, fit[2]]


def plane_ransac(points, inlier_tolerance=0.01, nsamples=20, npoints_x_sample=20, debug_image=None):
    # Pick blocks of random points
    # point_indices_x = np.random.randint(0, points.shape[0], nsamples * npoints_x_sample)
    # point_indices_y = np.random.randint(0, points.shape[1], nsamples * npoints_x_sample)
    point_indices_x = np.random.randint(250, 450, nsamples * npoints_x_sample)
    point_indices_y = np.random.randint(250, 450, nsamples * npoints_x_sample)
    point_indices = np.zeros((nsamples*npoints_x_sample, 2), dtype=np.int)
    point_indices[:, 0] = point_indices_x
    point_indices[:, 1] = point_indices_y
    point_indices = point_indices.reshape(nsamples, npoints_x_sample, 2)
    plane_coeffs = np.zeros([nsamples, 5])
    plane_inlier_indices = [None] * nsamples

    debug_image_input = copy.deepcopy(debug_image)

    # For each block of random points
    for i in range(len(point_indices)):
        debug_image = copy.deepcopy(debug_image_input)
        plane_coeffs[i][0:4] = fit_plane_model_svd(points, point_indices[i], debug_image)
        # plane_coeffs[i][0:4] = fit_plane_model_pinv(points, point_indices[i])

        if debug_image is not None:
            for idx in point_indices[i]:
                # print("Point [%d %d]: [%3.4f %3.4f %3.4f]" % (idx[0], idx[1], points[idx[0], idx[1]][0],points[idx[0], idx[1]][1],points[idx[0], idx[1]][2]))
                cv2.circle(debug_image, (idx[0], idx[1]), 3, (0, 255, 255), 2)
                cv2.imshow('TablePoints', debug_image)
                cv2.waitKey(1)

        if np.sum(plane_coeffs[i]*plane_coeffs[i]) > 0:
            # Compute the inliers
            inlier_indices = []
            for u in range(points.shape[0]):
                for v in range(points.shape[1]):
                    p = points[u, v]
                    dist = plane_coeffs[i][0] * p[0] + plane_coeffs[i][1] * p[1] + plane_coeffs[i][2] * p[2] + plane_coeffs[i][3]
                    if np.abs(dist) < inlier_tolerance:
                        plane_coeffs[i][4] = plane_coeffs[i][4] + 1
                        inlier_indices.append([u, v])
            plane_inlier_indices[i] = inlier_indices
            print("Sample %d: %3.4fx %3.4fy %3.4fz + %3.4f = 0 | inliers: %d " %(i, plane_coeffs[i][0], plane_coeffs[i][1], plane_coeffs[i][2], plane_coeffs[i][3], plane_coeffs[i][4]))

    # Coeffs: A B C D ninliers
    return plane_coeffs, plane_inlier_indices


class CTableFinder:
    def __init__(self, nsamples=60, marker_ids=(1, 2, 3)):
        self.marker1_points = deque()
        self.marker2_points = deque()
        self.marker3_points = deque()

        self.origin_marker_id = marker_ids[0]
        self.x_axis_marker_id = marker_ids[1]
        self.y_axis_marker_id = marker_ids[2]

        self.m1 = None
        self.m2 = None
        self.m3 = None

        self.transform = np.eye(4)
        self.offset = np.zeros(3)
        self.num_samples = nsamples

    def add_offset(self, offset):
        self.offset = self.offset + offset

    def get_transform(self):
        self.transform = self.compute_transform()
        return self.transform

    def has_transform(self):
        return len(self.marker1_points) >= self.num_samples and \
               len(self.marker2_points) >= self.num_samples and \
               len(self.marker3_points) >= self.num_samples

    def remove_extra_points(self):
        while len(self.marker1_points) > self.num_samples:
            self.marker1_points.pop()
        while len(self.marker2_points) > self.num_samples:
            self.marker2_points.pop()
        while len(self.marker3_points) > self.num_samples:
            self.marker3_points.pop()

    def compute_transform(self):
        if not self.has_transform():
            raise Exception("Not enough samples acquired")

        self.remove_extra_points()

        marker1_pos = np.median(self.marker1_points, 0)
        marker2_pos = np.median(self.marker2_points, 0)
        marker3_pos = np.median(self.marker3_points, 0)

        x_axis = marker3_pos - marker1_pos
        y_axis = marker2_pos - marker1_pos
        z_axis = np.cross(x_axis, y_axis)
        x_axis = np.cross(y_axis, z_axis)

        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        mat = np.eye(4)
        mat[0:3, 0] = x_axis
        mat[0:3, 1] = y_axis
        mat[0:3, 2] = z_axis

        mat_t = np.eye(4)
        mat_t[0:3, 3] = marker1_pos
        mat = np.matmul(mat_t, mat)

        mat_t_off = np.eye(4)
        mat_t_off[0:3, 3] = self.offset
        mat = np.matmul(mat, mat_t_off)
        self.transform = mat
        return self.transform

    def process_image(self, img, depth_image, color_to_depth, color_intrin, depth_intrin, depth_scale):
        markers = detect_markers(img)

        for mark in markers:
            mark.highlite_marker(img)
            mark_pixel = [mark.center[0], mark.center[1]]

            if mark.id == self.origin_marker_id:
                m1_pos = rs_rgb_to_3d(mark_pixel, depth_image, color_to_depth, color_intrin, depth_intrin, depth_scale)
                self.marker1_points.appendleft(m1_pos)
            elif mark.id == self.x_axis_marker_id:
                m2_pos = rs_rgb_to_3d(mark_pixel, depth_image, color_to_depth, color_intrin, depth_intrin, depth_scale)
                self.marker2_points.appendleft(m2_pos)
            elif mark.id == self.y_axis_marker_id:
                m3_pos = rs_rgb_to_3d(mark_pixel, depth_image, color_to_depth, color_intrin, depth_intrin, depth_scale)
                self.marker3_points.appendleft(m3_pos)

        self.m1 = np.median(self.marker1_points, 0)
        self.m2 = np.median(self.marker2_points, 0)
        self.m3 = np.median(self.marker3_points, 0)
        return [self.m1, self.m2, self.m3]


class CTableFinderPointCloud(CTableFinder):
    def __init__(self, nsamples=30):
        super(CTableFinderPointCloud, self).__init__(nsamples)
        self.points = []
        self.plane_tolerance = 0.1  # Initial plane RANSAC tolerance 10cm
        self.min_plane_tolerance = 0.001 # Maximum plane precision RANSAC tolerance 1mm
        self.max_table_points = 100000

    def add_offset(self, offset):
        self.offset = self.offset + offset

    def get_transform(self):
        self.transform = self.compute_transform()
        return self.transform

    def has_transform(self):
        return len(self.points) > self.max_table_points

    def remove_extra_points(self):
        if len(self.points) > self.max_table_points:
            self.points = self.filter_table_points(self.points, self.plane_tolerance)

    def compute_transform(self):
        if not self.has_transform():
            raise Exception("Not enough samples acquired")

        return self.transform

    def is_inlier(self, point, plane_model, inlier_tolerance=0.01):
        dist = plane_model[0] * point[0] + plane_model[1] * point[1] + plane_model[2] * point[2] + plane_model[3]
        return np.abs(dist) < inlier_tolerance

    def filter_table_points(self, points, tolerance, depth_image=None):
        depth_colormap = None
        if depth_image is not None:
            cv2.namedWindow('TablePoints', cv2.WINDOW_NORMAL)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_VIRIDIS)

        planes, indices = plane_ransac(points, tolerance, debug_image=depth_colormap)
        closest = 1000
        plane_model = np.array([0,0,0,0,0])
        res = np.array([])

        # Select the filtering plane
        for i in range(len(planes)):
            plane = planes[i]
            inliers = indices[i]
            # Debug. Draw inliers
            if depth_image is not None and inliers is not None:
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_VIRIDIS)
                for idx in inliers:
                    depth_colormap[idx[0], idx[1]] = [0, 0, 255]
                cv2.imshow('TablePoints', depth_colormap)
                cv2.waitKey(1)

            # Assume closest plane with more than 1000 points is the table plane
            if plane[4] > 1000 and plane[3] < closest:
                plane_model = plane
                closest = plane[3]

        # Filter outliers
        print("TABLE PLANE FOUND: %3.4fx %3.4fy %3.4fz + %3.4f = 0 | inliers: %d " % (plane_model[0], plane_model[1], plane_model[2], plane_model[3], plane_model[4]))

        for i in range(points.shape[0]):
            for j in range(points.shape[1]):
                if self.is_inlier(points[i, j], plane_model, tolerance):
                    res = np.concatenate((res, points[i, j]))

        return res.reshape(-1, 3)

    def process_image(self, img, depth_image, color_to_depth, color_intrin, depth_intrin, depth_scale):
        if self.plane_tolerance < self.min_plane_tolerance:
            return True

        point_cloud = rs_depth_to_points(depth_image, depth_intrin, depth_scale)

        table_plane_points = self.filter_table_points(point_cloud, self.plane_tolerance, depth_image)
        self.points.extend(table_plane_points)

        if len(self.points) > self.max_table_points:
            self.plane_tolerance = self.plane_tolerance / 2.0
            if self.plane_tolerance < self.min_plane_tolerance:
                self.plane_tolerance = self.min_plane_tolerance
            else:
                self.points = self.filter_table_points(self.points, self.plane_tolerance)

        return True
