#!/usr/bin/python3
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import pyrealsense2 as rs
import numpy as np
import cv2
from manipulator_planning_control.pybullet_utils import draw_trajectory
from manipulator_planning_control.pybullet_utils import draw_line
from manipulator_planning_control.pybullet_utils import draw_box
from manipulator_planning_control.pybullet_utils import init_physics
import pybullet as p
import math
from hand_tracking.table_finder import CTableFinder
from hand_tracking.table_finder import CTableFinderPointCloud
import time
import copy
import threading
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import Pose
from std_msgs.msg import Int8
from std_msgs.msg import UInt8MultiArray
from std_msgs.msg import Empty
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from collections import deque
import transforms3d as tf3d
from scipy import stats


def rs_rgb_to_depth(rgb_pixel, color_to_depth_extrin, color_intrin, depth_intrin):
    rgb_point = rs.rs2_deproject_pixel_to_point(color_intrin, rgb_pixel, 1)
    depth_point = rs.rs2_transform_point_to_point(color_to_depth_extrin, rgb_point)
    depth_pixel = rs.rs2_project_point_to_pixel(depth_intrin, depth_point)
    return depth_pixel


def rs_rgb_to_depth_batch(rgb_pixels, color_to_depth_extrin, color_intrin, depth_intrin):
    depth_pixels = np.ones((len(rgb_pixels), 2))
    for i, rgb_pixel in enumerate(rgb_pixels):
        rgb_point = rs.rs2_deproject_pixel_to_point(color_intrin, list(rgb_pixel), 1)
        depth_point = rs.rs2_transform_point_to_point(color_to_depth_extrin, rgb_point)
        depth_pixels[i] = rs.rs2_project_point_to_pixel(depth_intrin, depth_point)

    return depth_pixels


def rs_rgb_to_3d(rgb_pixel, depth_image, color_to_depth_extrin, color_intrin, depth_intrin, depth_scale):
    depth_pixel = rs_rgb_to_depth(rgb_pixel=rgb_pixel,
                                  color_to_depth_extrin=color_to_depth_extrin,
                                  color_intrin=color_intrin,
                                  depth_intrin=depth_intrin)

    depth_value = depth_image[int(depth_pixel[1]), int(depth_pixel[0])]
    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [int(depth_pixel[0]), int(depth_pixel[1])],
                                               depth_value * depth_scale)

    return point_3d


target_color_values = []
def process_click(event, x, y, flags, param):
    # grab references to the global variables
    global tracker, target_color_values

    if event == cv2.EVENT_LBUTTONUP:
        print("BGR: ", tracker.color_image[y, x])
        print("HSV: ", tracker.color_imageHSV[y, x])
        # target_color_values.append(tracker.color_imageHSV[y, x])
        target_color_values.append(tracker.color_image[y, x])
        lower_purple = np.min(target_color_values, 0)
        upper_purple = np.max(target_color_values, 0)

        tracker.lower_color = lower_purple
        tracker.upper_color = upper_purple

        print("Color Min-Max: ", lower_purple, upper_purple)


class ColorRegionTracker:
    def __init__(self, detection_radius=30, tracking_detection_radius=10, min_region_size=300, init_location=(0, 0)):
        self.detection_radius = detection_radius
        self.tracking_detection_radius = tracking_detection_radius
        self.last_track = np.array(init_location)
        self.min_region_size = min_region_size  # Minimum number of pixels in a connected component to consider it for tracking
        self.tracking = False
        self.segmented_img = None

    def track_color_region(self, color_image, lower_color, upper_color):
        mask = cv2.inRange(color_image, np.array(lower_color), np.array(upper_color))
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

        self.segmented_img = cv2.bitwise_and(color_image, color_image, mask=mask)
        if nb_components < 1:
            return None, None

        max_label = 0
        sizes = stats[:, -1]
        max_size = sizes[0]
        num_pixels = 200
        mask[:] = 0
        selected_clusters = []
        for i in range(1, nb_components):
            if sizes[i] > num_pixels:
                max_label = i
                max_size = sizes[i]
                selected_clusters.append(i)

        # Select the search radius
        if not self.tracking:
            min_dist = self.detection_radius
        else:
            min_dist = self.tracking_detection_radius

        # Select the cluster centroid closest to the last tracked pixel coordinates
        selected_cluster = -1
        for i in selected_clusters:
            dist = np.sqrt(np.sum((centroids[i] - self.last_track) * (centroids[i] - self.last_track)))
            if dist < min_dist:
                min_dist = dist
                selected_cluster = i
                centroid = centroids[i]
                self.tracking = True
                self.last_track = (int(centroid[0]), int(centroid[1]))

        # No clusters found in the search area. Tracking lost.
        if selected_cluster == -1:
            self.tracking = False
            for i in selected_clusters:
                mask[output == i] = 255
            self.segmented_img = cv2.bitwise_and(color_image, color_image, mask=mask)
            return None, None

        mask[output == selected_cluster] = 255

        res = np.asarray(np.where(mask == 255))

        for i in selected_clusters:
            mask[output == i] = 255

        self.segmented_img = cv2.bitwise_and(color_image, color_image, mask=mask)

        return res, self.last_track
        # return self.last_track


class RealsenseTrajectoryCapture:

    def __init__(self, lower_color, upper_color, start_volume_center=(0, 0, 0), start_volume_size=0.3,
                 motion_start_volume=0.05, trajectory_duration=3.0, wait_time=1.0):
        self.start_volume_center = np.array(start_volume_center)
        self.start_volume_size = start_volume_size
        self.start_volume_min = self.start_volume_center - self.start_volume_size/2.0
        self.start_volume_max = self.start_volume_center + self.start_volume_size/2.0
        self.motion_start_volume = motion_start_volume
        self.trajectory = []
        self.trajectory_pixel = []
        self.trajectory_duration = trajectory_duration
        self.track_point = None
        self.track_region = None
        self.track_pixel = None
        self.depth_frame = None
        self.color_frame = None
        self.point_data = None
        self.start_point = self.start_volume_center
        self.lower_color = lower_color
        self.upper_color = upper_color
        self.color_tracker = ColorRegionTracker(detection_radius=30, tracking_detection_radius=10,
                                                min_region_size=300, init_location=[300, 300])

        self.table_finder = CTableFinder()
        # self.table_finder = CTableFinderPointCloud()

        self.init_time = time.time()
        self.wait_time = wait_time

        # Kalman filter for tracked 3D point
        self.kalman = cv2.KalmanFilter(6, 1)
        self.time_last_frame = time.time()
        self.kalman_dt = 0.033
        self.kalman.transitionMatrix = np.array([[1,0,0,self.kalman_dt,0,0], [0,1,0,0,self.kalman_dt,0], [0,0,1,0,0,self.kalman_dt], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]], np.float32)
        self.kalman.measurementMatrix = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0]], np.float32)
        self.kalman.processNoiseCov = 1e-3 * np.eye(6, dtype=np.float32)
        self.kalman.measurementNoiseCov = 1e-3 * np.eye(3, dtype=np.float32)

        self.last_traj_nofilter = deque([], maxlen=20)
        self.last_traj_filter = deque([], maxlen=20)

        # Capture state
        # 0: Detect environment.
        # 1: Waiting for hand to move into initial volume.
        # 2: Hand in initial volume less than 1 sec
        # 3: Hand in initial volume more than 1 sec. Wating for trajectory to start
        # 4: Tracking
        self.capture_state = 0

        # Initialize camera
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)

        self.align = rs.align(rs.stream.depth)

        # Start streaming
        self.pipe_profile = self.pipeline.start(self.config)

        sensors = self.pipe_profile.get_device().query_sensors()
        sensors[1].set_option(rs.option.exposure, 80)
        sensors[1].set_option(rs.option.gain, 100)
        sensors[1].set_option(rs.option.saturation, 80)
        sensors[1].set_option(rs.option.enable_auto_exposure, False)
        sensors[1].set_option(rs.option.enable_auto_white_balance, True)

    def set_exposure(self, exposure):
        sensors = self.pipe_profile.get_device().query_sensors()
        sensors[1].set_option(rs.option.exposure, exposure)

    def set_gain(self, gain):
        sensors = self.pipe_profile.get_device().query_sensors()
        sensors[1].set_option(rs.option.gain, gain)

    def set_saturation(self, saturation):
        sensors = self.pipe_profile.get_device().query_sensors()
        sensors[1].set_option(rs.option.saturation, saturation)

    def set_auto_exposure(self, auto_exposure):
        sensors = self.pipe_profile.get_device().query_sensors()
        sensors[1].set_option(rs.option.enable_auto_exposure, auto_exposure)

    def set_auto_white_balance(self, auto_wb):
        sensors = self.pipe_profile.get_device().query_sensors()
        sensors[1].set_option(rs.option.enable_auto_white_balance, auto_wb)

    def __del__(self):
        self.pipeline.stop()

    def reset(self):
        self.trajectory = []
        self.trajectory_pixel = []
        self.table_finder = CTableFinder()
        self.track_point = None
        self.track_pixel = None
        self.depth_frame = None
        self.color_frame = None
        self.init_time = time.time()
        self.time_last_frame = time.time()
        self.last_traj_nofilter = deque([], maxlen=20)
        self.last_traj_filter = deque([], maxlen=20)
        self.capture_state = 0

    def environment_ready(self):
        return self.table_finder.has_transform()

    def get_trajectory(self):
        return self.trajectory

    def get_endpoint(self):
        return self.track_point

    def compute_track_point(self, useHSV=False):
        # Compute tracking regions using the input image RGB or HSV
        track_region, track_pixel = self.color_tracker.track_color_region(self.color_image, self.lower_color, self.upper_color)
        # track_region, track_pixel = self.color_tracker.track_color_region(self.color_imageHSV, self.lower_color, self.upper_color)

        # if track_region is not None:
        #    centroid = np.mean(track_region, 1)
        #    track_pixel = (int(centroid[1]), int(centroid[0]))

        if track_pixel is not None and self.table_finder.has_transform():
            self.track_pixel = track_pixel
            track_points = self.compute_3d_points(track_region.T, self.color_to_depth_extrin, self.color_intrin,
                                                  self.depth_intrin, self.depth_image, self.depth_scale)
            pt = np.zeros(3)
            pt[0] = track_points[:, 0].mean()
            pt[1] = track_points[:, 1].mean()
            pt[2] = np.median(track_points[:, 2])

            track_point = self.compute_3d_point(track_pixel, self.color_to_depth_extrin, self.color_intrin,
                                                self.depth_intrin, self.depth_image, self.depth_scale)
            pt[0:2] = track_point[0:2]

            self.kalman_dt = time.time() - self.time_last_frame
            self.time_last_frame = time.time()
            self.kalman.transitionMatrix = np.array([[1, 0, 0, self.kalman_dt, 0, 0], [0, 1, 0, 0, self.kalman_dt, 0], [0, 0, 1, 0, 0, self.kalman_dt], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], np.float32)
            res = self.kalman.predict()[0:3]
            self.kalman.correct(pt.astype(np.float32))

            self.last_traj_filter.append(res.flatten())
            self.last_traj_nofilter.append(pt)
            return res.flatten()
            # return pt.flatten()
        else:
            return None

    def compute_3d_point(self, color_pixel_coordinates, color_to_depth_extrin, color_intrin, depth_intrin,
                         depth_image, depth_scale):
        # Convert the centroid point in RGB to a 3D point in world coordinates
        point_3d = rs_rgb_to_3d(rgb_pixel=[int(color_pixel_coordinates[0]), int(color_pixel_coordinates[1])],
                                color_to_depth_extrin=color_to_depth_extrin,
                                color_intrin=color_intrin,
                                depth_intrin=depth_intrin,
                                depth_image=depth_image,
                                depth_scale=depth_scale)
        point_3d.append(1)
        point_3d_base = np.matmul(np.linalg.inv(self.table_finder.compute_transform()), point_3d)
        return point_3d_base[0:3]

    def compute_3d_points(self, color_pixel_coordinates, color_to_depth_extrin, color_intrin, depth_intrin,
                         depth_image, depth_scale):

        points_3d = np.ones((100, 4))

        samples = np.random.randint(0, len(color_pixel_coordinates), 100)

        # Convert a batch of points in RGB to a 3D point in world coordinates
        for i, px in enumerate(color_pixel_coordinates[samples]):
            point_3d = rs_rgb_to_3d(rgb_pixel=[int(px[1]), int(px[0])],
                                    color_to_depth_extrin=color_to_depth_extrin,
                                    color_intrin=color_intrin,
                                    depth_intrin=depth_intrin,
                                    depth_image=depth_image,
                                    depth_scale=depth_scale)
            points_3d[i, 0:3] = point_3d

        tfmat = np.linalg.inv(self.table_finder.compute_transform())
        points_3d_base = np.matmul(tfmat, points_3d.reshape(-1, 4, 1))[:, 0:3]

        return points_3d_base

    def get_pointcloud_msg(self, use_color=True, use_depth_colormap=False):
        if self.depth_frame is None:
            return PointCloud2()

        # Compute point cloud
        w, h = self.depth_intrin.width, self.depth_intrin.height
        pc = rs.pointcloud()
        points = pc.calculate(self.depth_frame)

        # Prepare point cloud color
        if use_color:
            if use_depth_colormap:
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.03), cv2.COLORMAP_BONE)
                # OpenCV works by default with BGR ordering. But the color output in the point cloud is expected in RGB
                point_colors = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)

            elif self.color_frame is not None:
                # Images from realsense are in BGR ordering. But the color output in the point cloud is expected in RGB
                point_colors = cv2.cvtColor(np.asanyarray(self.aligned_frames.get_color_frame().get_data()), cv2.COLOR_BGR2RGB)

            # Add the alpha channel to the color source and convert the color source to uint8
            point_colors = np.dstack((point_colors.astype(np.uint8), np.full((h, w, 1), 255, dtype=np.uint8)))
            point_colors = point_colors.view(dtype=np.float32)      # Reinterpret the 4 uint8 into a float32
            point_colors = point_colors.reshape((h, w, 1))          # Reshape to match the dense point cloud shape

        # Populate ROS message
        border_crop = 50
        pc_msg = PointCloud2()
        pc_msg.header.stamp = rospy.Time.now()
        pc_msg.header.frame_id = "depth_optical_frame"
        pc_msg.height = h - border_crop*2                                      # Account for cropping
        # pc_msg.height = h
        pc_msg.is_bigendian = False
        pc_msg.is_dense = True
        pc_msg.width = w - border_crop*2  # Account for cropping
        # pc_msg.width = w
        pc_field_x = PointField()
        pc_field_x.count = 1
        pc_field_x.datatype = PointField.FLOAT32
        pc_field_x.name = 'x'
        pc_field_x.offset = 0

        pc_field_y = PointField()
        pc_field_y.count = 1
        pc_field_y.datatype = PointField.FLOAT32
        pc_field_y.name = 'y'
        pc_field_y.offset = 4

        pc_field_z = PointField()
        pc_field_z.count = 1
        pc_field_z.datatype = PointField.FLOAT32
        pc_field_z.name = 'z'
        pc_field_z.offset = 8

        if use_color:
            pc_msg.point_step = 16
            pc_msg.row_step = pc_msg.point_step * pc_msg.width
            pc_field_rgb = PointField()
            pc_field_rgb.count = 1
            pc_field_rgb.datatype = PointField.FLOAT32
            pc_field_rgb.name = 'rgb'
            pc_field_rgb.offset = 12
            pc_msg.fields = [pc_field_x, pc_field_y, pc_field_z, pc_field_rgb]
            point_coords = np.asarray(points.get_vertices(2)).reshape(h, w, 3)
            # point_data = np.dstack((point_data, point_colors))
            self.point_data = np.dstack((point_coords[border_crop:h-border_crop,border_crop:w-border_crop], point_colors[border_crop:h-border_crop, border_crop:w-border_crop]))
            pc_msg.data = self.point_data.astype(np.float32).tobytes()
        else:
            pc_msg.point_step = 12
            pc_msg.row_step = pc_msg.point_step * pc_msg.width
            pc_msg.fields = [pc_field_x, pc_field_y, pc_field_z]
            point_coords = np.asarray(points.get_vertices(2)).reshape(h, w, 3)
            self.point_data = point_coords[border_crop:h-border_crop,border_crop:w-border_crop]
            pc_msg.data = point_coords[border_crop:h-border_crop,border_crop:w-border_crop].tobytes()

        return pc_msg

    def update_frames(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        self.depth_frame = frames.get_depth_frame()
        self.color_frame = frames.get_color_frame()
        if not self.depth_frame or not self.color_frame:
            return False

        # Intrinsics & Extrinsics
        self.depth_intrin = self.depth_frame.profile.as_video_stream_profile().intrinsics
        self.color_intrin = self.color_frame.profile.as_video_stream_profile().intrinsics
        self.color_to_depth_extrin = self.color_frame.profile.get_extrinsics_to(self.depth_frame.profile)

        # print("Depth intrinsics: ", self.depth_intrin)
        # print("RGB intrinsics: ", self.color_intrin)
        # print("Color-To-Depth extrinsics: ", self.color_to_depth_extrin)

        # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
        self.depth_scale = self.pipe_profile.get_device().first_depth_sensor().get_depth_scale()

        # Convert images to numpy arrays and to RGB from BGR
        self.depth_image = np.asanyarray(self.depth_frame.get_data())
        self.color_image = cv2.cvtColor(np.asanyarray(self.color_frame.get_data()), cv2.COLOR_BGR2RGB)
        self.color_imageHSV = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2HSV_FULL)

        self.aligned_frames = self.align.process(frames)
        return True

    def update(self):
        if self.update_frames():

            # State 0: Detect table and compute camera - base frame transform
            if self.capture_state == 0:
                self.table_finder.process_image(self.color_image, self.depth_image,
                                                self.color_to_depth_extrin, self.color_intrin,
                                                self.depth_intrin, self.depth_scale)
                if self.table_finder.has_transform():
                    self.capture_state = 1

            # State 1: Waiting for hand to move into initial volume.
            elif self.capture_state == 1:
                self.track_point = self.compute_track_point()
                if self.track_point is None:
                    return False
                # print("Track: ", self.track_point)
                if self.track_point is not None and np.all(self.start_volume_min < self.track_point) and np.all(self.track_point < self.start_volume_max):
                        self.capture_state = 2
                        self.init_time = time.time()

            # State 2: Waiting for hand to move into initial volume and stay in it for wait_time
            elif self.capture_state == 2:
                self.track_point = self.compute_track_point()
                if self.track_point is None:
                    return False

                if np.all(self.start_volume_min < self.track_point) and np.all(self.track_point < self.start_volume_max):
                    if time.time() - self.init_time > self.wait_time:
                        self.init_time = time.time()
                        self.capture_state = 3
                        self.start_point = self.track_point
                else:
                    self.init_time = time.time()
                    self.capture_state = 1

            # State 3: Hand in initial volume more than wait_time sec. Wating for trajectory to start.
            elif self.capture_state == 3:  # Wait for the hand to move
                self.track_point = self.compute_track_point()
                if self.track_point is None:
                    return False

                if np.any(self.start_point - self.motion_start_volume/2 > self.track_point) or \
                        np.any(self.track_point > self.start_point + self.motion_start_volume/2):
                    self.init_time = time.time()
                    self.capture_state = 4

            # State 4: Tracking. Waiting for the trajectory to finish and restart the trajectory capture
            elif self.capture_state == 4:
                self.track_point = self.compute_track_point()
                if self.track_point is None:
                    return False

                self.trajectory.append(self.track_point)
                self.trajectory_pixel.append(self.track_pixel)

                if time.time() - self.init_time > self.trajectory_duration:
                    # print "Captured trajectory: \n", self.trajectory
                    self.trajectory = []
                    self.trajectory_pixel = []
                    self.capture_state = 1

            return True


def traj_to_ros_msg(traj):
    p_array = []
    for pt in traj:
        pos = Pose()
        pos.position.x = pt[0]
        pos.position.y = pt[1]
        pos.position.z = pt[2]
        p_array.append(pos)

    return PoseArray(Header(), p_array)


def cb_color_threshold(data):
    global tracker
    tracker.lower_color[0] = ord(data.data[0])
    tracker.lower_color[1] = ord(data.data[1])
    tracker.lower_color[2] = ord(data.data[2])
    tracker.upper_color[0] = ord(data.data[3])
    tracker.upper_color[1] = ord(data.data[4])
    tracker.upper_color[2] = ord(data.data[5])
    print("Updated color threshold: ", tracker.lower_color, tracker.upper_color)


def cb_reset(data):
    global must_reset
    must_reset = True
    print("Received reset signal!")


def cb_pc_enable(data):
    global publish_points
    publish_points = not publish_points
    print("Point cloud publish status: ", publish_points)


def cb_set_auto_exposure(data):
    print("Point cloud publish status: ", publish_points)


if __name__ == "__main__":
    print("OpenCV version: ", cv2.__version__)
    print(cv2.getBuildInformation())

    use_ros = True
    publish_images = True
    showImages = True
    visualize3D = False
    publish_points = False
    must_reset = False

    if len(sys.argv) == 2:
        showImages = bool(int(sys.argv[1]))
        print("showImages = ", showImages)

    if use_ros:
        rospy.init_node('hand_traj_tracker', anonymous=True)
        pub_point = rospy.Publisher("/output/tracked_point", Point, queue_size=1)
        pub_traj  = rospy.Publisher("/output/trajectory", PoseArray, queue_size=1)
        pub_rgb = rospy.Publisher("/output/image_color", Image, queue_size=1)
        pub_depth  = rospy.Publisher("/output/image_depth", Image, queue_size=1)
        pub_seg  = rospy.Publisher("/output/image_seg", Image, queue_size=1)
        pub_state = rospy.Publisher("/output/state", Int8, queue_size=1)
        pub_camtf = rospy.Publisher("/output/camtf", Pose, queue_size=1)
        sub_color = rospy.Subscriber("/input/color_threshold", UInt8MultiArray, cb_color_threshold)
        pub_pc = rospy.Publisher("/output/points", PointCloud2, queue_size=1)
        sub_reset = rospy.Subscriber("/input/reset", Empty, cb_reset)
        sub_reset = rospy.Subscriber("/input/pc_enable", Empty, cb_pc_enable)
        rate = rospy.Rate(30)

    # Configurable tracker parameters
    lower_color = [245, 185, 146]       ## HSV Thresholds
    upper_color = [255, 255, 247]

    ## HSV ICML Thresholds
    lower_color = [0, 239, 129]     ## HSV Thresholds
    upper_color = [255, 255, 180]

    ## HSV CVPR Thresholds
    lower_color = [212, 133, 70]        ## HSV Thresholds
    upper_color = [240, 224, 233]

    ## HSV Inte HUB JFCC Thresholds
    lower_color = [0, 182, 254]         ## HSV Thresholds
    upper_color = [9, 255, 255]

    ## JFCC120 ILOpenHouse 2019
    lower_color = [0, 201,  81]        ## HSV Thresholds
    upper_color = [7, 255, 255]

    ## HSV RoboticsLab Thresholds
    lower_color = [0, 152, 155]        ## HSV Thresholds
    upper_color = [255, 219, 251]

    ## HSV RoboticsLab Thresholds Saturation 80 and Monitor as a table
    lower_color = [0, 213, 202]        ## HSV Thresholds
    upper_color = [255, 255, 250]

    lower_color = [171, 0, 0]        ## RGB Thresholds
    upper_color = [255, 40, 60]


    # lower_color = [200, 0, 0]           ## RGB Thresholds
    # upper_color = [255, 50, 50]
    start_volume_center = [0, 0.2, -0.15]
    start_volume_size = 0.1
    wait_time = 1.0
    traj_time = 3.0
    initial_track_pixel = (320, 320)
    initial_location_radius = 100
    tracking_min_dist = 60
    tracker = RealsenseTrajectoryCapture(lower_color=lower_color, upper_color=upper_color,
                                         start_volume_center=start_volume_center, start_volume_size=start_volume_size,
                                         trajectory_duration=traj_time, wait_time=wait_time)
    tracker.color_tracker.last_track = initial_track_pixel
    tracker.color_tracker.detection_radius = initial_location_radius
    tracker.color_tracker.tracking_detection_radius = tracking_min_dist
    table_offset = [-0.25, -0.2, -0.1]
    tracker.table_finder.add_offset(table_offset)

    # Image processing debug windows
    if showImages:
        cv2.namedWindow('ColorCalib', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("ColorCalib", process_click)

    # Tracker 3D Visualization using pybullet
    if visualize3D:
        sim_id = init_physics(True, 0.01, 0)
        objects_path = ["../manipulator_planning_control/models/table/table.urdf"]
        objects_pose = [[0.6, 0, -0.65]]
        for i in range(len(objects_path)):
            p.loadURDF(objects_path[i], useFixedBase=1, basePosition=objects_pose[i], physicsClientId=sim_id)

        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=90, cameraPitch=230,
                                     cameraTargetPosition=[0.0, 0.0, 0.0], physicsClientId=sim_id)

    try:
        while not rospy.is_shutdown():
            if must_reset:
                must_reset = False
                tracker.reset()
                tracker.table_finder.add_offset(table_offset)

            tini = time.time()
            tracker.update()
            t_tracker = time.time()-tini
            tini = time.time()

            # Annotate color image with debug data from the tracker
            if showImages:
                if tracker.capture_state == 1:
                    cv2.circle(tracker.color_image, tracker.track_pixel, 10, (0, 0, 255), thickness=3)
                elif tracker.capture_state == 2:
                    cv2.circle(tracker.color_image, tracker.track_pixel, 10, (255, 0, 255), thickness=3)
                elif tracker.capture_state == 3:
                    cv2.circle(tracker.color_image, tracker.track_pixel, 10, (0, 255, 0), thickness=3)
                    # for px in tracker.trajectory_pixel:
                    #     cv2.circle(tracker.color_image, px, 10, (255, 255, 255))
                elif tracker.capture_state == 4:
                    cv2.circle(tracker.color_image, tracker.track_pixel, 10, (255, 0, 0), thickness=3)
                else:
                    cv2.circle(tracker.color_image, tracker.color_tracker.last_track,
                               tracker.color_tracker.detection_radius, (255, 0, 0), thickness=3)

                # Show hand tracker debug images (RGB to BGR conversion is required because this app works in RGB order but openCV uses BGR by default)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(tracker.depth_image, alpha=0.03), cv2.COLORMAP_BONE)
                if tracker.color_tracker.segmented_img is not None:
                    cv2.imshow('Streams', np.hstack((cv2.cvtColor(tracker.color_image, cv2.COLOR_RGB2BGR), depth_colormap, cv2.cvtColor(tracker.color_tracker.segmented_img, cv2.COLOR_RGB2BGR))))
                else:
                    cv2.imshow('Streams', np.hstack((cv2.cvtColor(tracker.color_image, cv2.COLOR_RGB2BGR), depth_colormap, cv2.cvtColor(tracker.color_image, cv2.COLOR_RGB2BGR))))

                # Draw reference frame
                if tracker.table_finder.has_transform():
                    transform = tracker.table_finder.compute_transform()
                    cam_matrix = np.array([tracker.color_intrin.fx, 0.0, tracker.color_intrin.ppx,
                                           0.0, tracker.color_intrin.fy, tracker.color_intrin.ppy,
                                           0.0, 0.0, 1.0]).reshape(3, 3)

                    rvec = np.eye(3)
                    tvec = np.zeros(3)
                    marker1_pos = np.mean(tracker.table_finder.marker1_points, 0)
                    marker2_pos = np.mean(tracker.table_finder.marker2_points, 0)
                    marker3_pos = np.mean(tracker.table_finder.marker3_points, 0)

                    # cv2.drawFrameAxes(tracker.color_image, cam_matrix, np.zeros((1, 4)), rvec, tvec, 0.1, thickness=3)
                    rvec = transform[0:3, 0:3]
                    tvec = transform[0:3, 3]
                    cv2.drawFrameAxes(tracker.color_image, cam_matrix, np.zeros((1, 4)), rvec, tvec, 0.1, thickness=3)
                    cv2.drawFrameAxes(tracker.color_image, cam_matrix, np.zeros((1, 4)), rvec, marker1_pos, 0.05, thickness=3)
                    cv2.drawFrameAxes(tracker.color_image, cam_matrix, np.zeros((1, 4)), rvec, marker2_pos, 0.05, thickness=3)
                    cv2.drawFrameAxes(tracker.color_image, cam_matrix, np.zeros((1, 4)), rvec, marker3_pos, 0.05, thickness=3)

                cv2.imshow('ColorCalib', cv2.cvtColor(tracker.color_image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            # Show debug information in the 3D display
            if visualize3D:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
                p.removeAllUserDebugItems(physicsClientId=sim_id)
                # Draw markers and camera frame
                if tracker.table_finder.has_transform():
                    transform = np.linalg.inv(tracker.table_finder.compute_transform())
                    p_ini = transform[0:3, 3]
                    draw_line(p_ini, p_ini + transform[0:3, 0] * 0.1, [1, 0, 0], width=3)  # Draw X-axis with 0.1 length
                    draw_line(p_ini, p_ini + transform[0:3, 1] * 0.1, [0, 1, 0], width=3)  # Draw Y-axis with 0.1 length
                    draw_line(p_ini, p_ini + transform[0:3, 2] * 0.1, [0, 0, 1], width=3)  # Draw Z-axis with 0.1 length
                    m1 = tracker.table_finder.m1
                    m2 = tracker.table_finder.m2
                    m3 = tracker.table_finder.m3
                    if len(m1) == 3:
                        m1 = np.append(m1, 1)
                    if len(m2) == 3:
                        m2 = np.append(m2, 1)
                    if len(m3) == 3:
                        m3 = np.append(m3, 1)
                    pos1 = np.matmul(transform, m1)
                    pos2 = np.matmul(transform, m2)
                    pos3 = np.matmul(transform, m3)
                    draw_box(pos1 - 0.01, pos1 + 0.01, [1, 0, 0], width=2, physicsClientId=sim_id)
                    draw_box(pos2 - 0.01, pos2 + 0.01, [0, 1, 0], width=2, physicsClientId=sim_id)
                    draw_box(pos3 - 0.01, pos3 + 0.01, [0, 0, 1], width=2, physicsClientId=sim_id)

                # Draw current tracked position
                if tracker.track_point is not None:
                    draw_box(tracker.track_point - 0.01, tracker.track_point + 0.01, [1, 0, 0], width=2, physicsClientId=sim_id)

                # Draw current tracked trajectory
                if tracker.last_traj_nofilter is not None:
                    draw_trajectory(tracker.last_traj_nofilter, color=[0,0,1], width=1, physicsClientId=sim_id)
                if tracker.last_traj_filter is not None:
                    draw_trajectory(tracker.last_traj_filter, color=[0,1,0], width=1, physicsClientId=sim_id)

                # Draw start box
                if tracker.capture_state == 1:
                    draw_box(tracker.start_volume_min, tracker.start_volume_max, [1, 0, 0], 6)
                elif tracker.capture_state == 2:
                    draw_box(tracker.start_volume_min, tracker.start_volume_max, [1, 1, 0], 6)
                elif tracker.capture_state == 3:
                    draw_box(tracker.start_volume_min, tracker.start_volume_max, [0, 1, 0], 6)
                elif tracker.capture_state == 4:
                    draw_trajectory(traj=tracker.trajectory, color=[0, 1, 0], width=2, draw_points=True, physicsClientId=sim_id)

                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

            t_display = time.time() - tini
            tini = time.time()
            if use_ros:
                if not showImages:
                    if tracker.capture_state == 1:
                        if tracker.track_pixel is None:
                            cv2.circle(tracker.color_image, tracker.color_tracker.last_track,
                                       tracker.color_tracker.detection_radius, (255, 0, 0), thickness=3)
                        #else:
                        #    cv2.circle(tracker.color_image, tracker.track_pixel, 10, (0, 0, 255), thickness=3)
                    elif tracker.capture_state == 2:
                        cv2.circle(tracker.color_image, tracker.track_pixel, 10, (255, 255, 0), thickness=3)
                    elif tracker.capture_state == 3:
                        cv2.circle(tracker.color_image, tracker.track_pixel, 10, (0, 255, 0), thickness=3)
                    elif tracker.capture_state == 4:
                        cv2.circle(tracker.color_image, tracker.track_pixel, 10, (0, 255, 0), thickness=3)
                        # for px in tracker.trajectory_pixel:
                        #     cv2.circle(tracker.color_image, px, 10, (255, 255, 255))
                    else:
                        cv2.circle(tracker.color_image, tracker.color_tracker.last_track,
                                   tracker.color_tracker.detection_radius, (255, 0, 0), thickness=3)

                    # Show hand tracker debug images
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(tracker.depth_image, alpha=0.03),
                                                       cv2.COLORMAP_BONE)

                ros_color_img = CvBridge().cv2_to_imgmsg(tracker.color_image)
                ros_depth_img = CvBridge().cv2_to_imgmsg(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
                if tracker.color_tracker.segmented_img is not None:
                    ros_seg_img = CvBridge().cv2_to_imgmsg(tracker.color_tracker.segmented_img)
                    ros_seg_img.header.stamp = rospy.Time.now()
                    pub_seg.publish(ros_seg_img)

                ros_color_img.header.stamp = rospy.Time.now()
                ros_depth_img.header.stamp = rospy.Time.now()
                pub_rgb.publish(ros_color_img)
                pub_depth.publish(ros_depth_img)

                if tracker.trajectory is not None:
                    traj_msg = traj_to_ros_msg(tracker.trajectory)
                    pub_traj.publish(traj_msg)

                if tracker.track_point is not None:
                    point_msg = Point()
                    point_msg.x = tracker.track_point[0]
                    point_msg.y = tracker.track_point[1]
                    point_msg.z = tracker.track_point[2]
                    pub_point.publish(point_msg)

                if tracker.table_finder.has_transform():
                    # The table transform contains the position of the table frame in w.r.t camera.
                    # the output of this topic is the camera position w.r.t the table which is the inverse
                    # if the transformation computed bu the table_finder
                    transform = np.linalg.inv(tracker.table_finder.compute_transform())

                    camtf_msg = Pose()
                    camtf_msg.position.x = transform[0, 3]
                    camtf_msg.position.y = transform[1, 3]
                    camtf_msg.position.z = transform[2, 3]

                    quat = tf3d.quaternions.mat2quat(transform)
                    camtf_msg.orientation.x = quat[1]
                    camtf_msg.orientation.y = quat[2]
                    camtf_msg.orientation.z = quat[3]
                    camtf_msg.orientation.w = quat[0]
                    pub_camtf.publish(camtf_msg)
                    # print("Transform: ", transform)

                state_msg = Int8()
                state_msg.data = tracker.capture_state
                pub_state.publish(state_msg)

                if publish_points:
                    pub_pc.publish(tracker.get_pointcloud_msg())

                if rospy.is_shutdown():
                    break

            t_ros = time.time() - tini

            tini = time.time()
            if use_ros:
                rate.sleep()
            t_sleep = time.time() - tini

            # print("State: %d | Tracker: %2.4f | Display: %2.4f | ROS: %2.4f | Sleep: %2.4f | Total: %2.4f Hz" % (tracker.capture_state, t_tracker, t_display, t_ros, t_sleep, 1/(t_tracker+t_display+t_ros+t_sleep)))

    finally:
        pass
