import numpy as np


# TODO: TEST THIS
def get_twist_from_two_poses(origin, target):
    """
    :param origin: Origin pose expressed as a 3D vector and a Quaternion in a 7D array as [x y z qx qy qz qw]
    :param target: Target pose expressed as a 3D vector and a Quaternion in a 7D array as [x y z qx qy qz qw]
    :return: twist that moves from origin to target in a unit time
    """
    translation = target[0:3] - origin[0:3]
    rotation = quaternion_multiply(quaternion_inverse(origin[3:]), target[3:])

    return np.concatenate((translation, 2*rotation[0:3]))


# TODO: TEST THIS
def get_angle_from_two_quaternions(q0, q1):
    angle = np.arccos(2 * np.dot(q0, q1) * np.dot(q0, q1) - 1)
    return angle


# TODO: TEST THIS
def quaternion_inverse(quaternion):
    """Return inverse of quaternion.
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    np.negative(q[0:3], q[0:3])
    return q / np.dot(q, q)


# TODO: TEST THIS
def quaternion_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.
    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array([
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0], dtype=np.float64)


def transform_pose_with_twist(pose, twist):
    pos = pose[0:3] + twist[0:3]
    delta_q = np.concatenate((0.5 * twist[3:6], np.array([0])))
    rot = quaternion_multiply(delta_q, pose[3:])
    return np.concatenate((pos, rot))


def transform_pose(origin, transformation):
    pos = origin[0:3] + transformation[0:3]
    rot = quaternion_multiply(transformation[3:], origin[3:])
    return np.concatenate((pos, rot))


if __name__ == '__main__':
    pos1 = [1, 0, 0]
    pos2 = [0, 2, 5]

    rot1 = [0.491, -0.494, 0.644, 0.315]
    rot2 = [0, 0, -0.707, 0.707]

    identity = [0, 0, 0, 1]
    res = quaternion_multiply(quaternion_inverse(rot1), rot1)
    if np.any(identity != res):
        print("ERROR! quaternion_multiply or quaternion_inverse did not pass the test. result: ", res, " expected:", identity)
    else:
        print("quaternion_multiply and quaternion_inverse OK!")

    pose1 = np.concatenate((pos1, rot1))
    pose2 = np.concatenate((pos2, rot2))

    # Test twist from pose diff
    pose_diff = get_twist_from_two_poses(origin=pose1, target=pose2)

    pose1_to_pose2 = transform_pose_with_twist(pose=pose1, twist=pose_diff)

    if np.any(pose2 != pose1_to_pose2):
        print("ERROR! get_twist_from_two_poses and transform_pose did not pass the test. result: ", pose1_to_pose2, " expected:", pose2, " twist: ", pose_diff)
    else:
        print("get_twist_from_two_poses and transform_pose OK!")
