import pybullet as p
import numpy as np
import time
import utils.robotics_utils as ru
from PIL import Image
from matplotlib import cm
from utils.draw import draw_text


def init_physics(use_gui, timestep=0.01, physicsClientId=0):
    pType = p.GUI
    if not use_gui:
        pType = p.DIRECT

    sim_id = p.connect(pType)

    p.setGravity(0, 0, -9.81, physicsClientId=sim_id)

    p.setPhysicsEngineParameter(
        fixedTimeStep=timestep,
        # Physics engine timestep in fraction of seconds, each time you call 'stepSimulation'.Same as 'setTimeStep'
        numSolverIterations=100,  # Choose the number of constraint solver iterations.
        useSplitImpulse=0,
        # Advanced feature, only when using maximal coordinates: split the positional constraint solving and velocity
        # constraint solving in two stages, to prevent huge penetration recovery forces.
        splitImpulsePenetrationThreshold=0.01,
        # Related to 'useSplitImpulse': if the penetration for a particular contact constraint is less than this
        # specified threshold, no split impulse will happen for that contact.
        numSubSteps=1,
        # Subdivide the physics simulation step further by 'numSubSteps'. This will trade performance over accuracy.
        collisionFilterMode=0,
        # Use 0 for default collision filter: (group A&maskB) AND (groupB&maskA). Use 1 to switch to the OR collision
        # filter: (group A&maskB) OR (groupB&maskA)
        contactBreakingThreshold=0.02,
        # Contact points with distance exceeding this threshold are not processed by the LCP solver. In addition,
        # AABBs are extended by this number. Defaults to 0.02 in Bullet 2.x.
        maxNumCmdPer1ms=0,  # Experimental: add 1ms sleep if the number of commands executed exceed this threshold.
        enableFileCaching=1,  # Set to 0 to disable file caching, such as .obj wavefront file loading
        restitutionVelocityThreshold=0.01,  # If relative velocity is below this threshold, restitution will be zero.
        erp=0.9,
        # Constraint error reduction parameter (non-contact, non-friction)
        # Details: http://www.ode.org/ode-latest-userguide.html#sec_3_7_0
        contactERP=0.2,  # Contact error reduction parameter
        frictionERP=0.2,  # Friction error reduction parameter (when positional friction anchors are enabled)
        enableConeFriction=0,
        # Set to 0 to disable implicit cone friction and use pyramid approximation (cone is default)
        deterministicOverlappingPairs=0,
        # Set to 0 to disable sorting of overlapping pairs (backward compatibility setting).
        physicsClientId=sim_id
    )
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=sim_id)
    return sim_id


def send_joint_pos_command(model, q, joint_indices, physicsClientId=0):
    p.setJointMotorControlArray(model,
                                joint_indices,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=q,
                                physicsClientId=physicsClientId
                                )


def send_joint_vel_command(model, q_dot, joint_indices, physicsClientId=0):
    # lower, upper = get_actuable_joint_limits(model, physicsClientId)
    # angles = get_actuable_joint_angles(model, physicsClientId)
    # lower_vel = np.array(lower) - np.array(angles)
    # upper_vel = np.array(upper) - np.array(angles)
    # q_dot = np.clip(q_dot, lower_vel, upper_vel)

    vel_limit = np.array(get_actuable_joint_vel_limits(model, physicsClientId))
    q_dot = np.clip(q_dot, -vel_limit, vel_limit)

    p.setJointMotorControlArray(model,
                                joint_indices,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocities=q_dot,
                                physicsClientId=physicsClientId
                                )


def send_joint_eff_command(model, tau, joint_indices, physicsClientId=0):
    p.setJointMotorControlArray(model,
                                joint_indices,
                                controlMode=p.TORQUE_CONTROL,
                                forces=tau,
                                physicsClientId=physicsClientId
                                )


def get_actuable_joint_indices(model, physicsClientId=0):
    actuated_joint_list = []
    for i in range(0, p.getNumJoints(model, physicsClientId=physicsClientId)):
        info = p.getJointInfo(model, i, physicsClientId=physicsClientId)
        if not info[2] == p.JOINT_FIXED:
            actuated_joint_list.append(i)

    return actuated_joint_list


def get_joint_limits(model, physicsClientId=0):
    joint_info = []
    for i in range(0, p.getNumJoints(model, physicsClientId=physicsClientId)):
        joint_info.append(p.getJointInfo(model, i, physicsClientId=physicsClientId))  # Get all joint limits

    upper_limit = []
    lower_limit = []
    for j in joint_info:
        upper_limit.append(j[9])
        lower_limit.append(j[8])

    return lower_limit, upper_limit


def get_actuable_joint_limits(model, physicsClientId=0):
    actuated_joints = get_actuable_joint_indices(model, physicsClientId)

    joint_info = []
    for i in actuated_joints:
        joint_info.append(p.getJointInfo(model, i, physicsClientId=physicsClientId))  # Get all joint limits

    upper_limit = []
    lower_limit = []
    for j in joint_info:
        upper_limit.append(j[9])
        lower_limit.append(j[8])

    return lower_limit, upper_limit


def get_actuable_joint_vel_limits(model, physicsClientId=0):
    actuated_joints = get_actuable_joint_indices(model, physicsClientId)
    joint_info = []
    for i in actuated_joints:
        joint_info.append(p.getJointInfo(model, i, physicsClientId=physicsClientId))  # Get all joint limits

    limits = []
    for j in joint_info:
        limits.append(j[11])

    return limits


def get_joint_vel_limits(model, physicsClientId=0):
    joint_info = []
    for i in range(0, p.getNumJoints(model, physicsClientId=physicsClientId)):
        joint_info.append(p.getJointInfo(model, i, physicsClientId=physicsClientId))  # Get all joint limits

    limits = []
    for j in joint_info:
        limits.append(j[11])

    return limits


def get_actuable_joint_angles(model, physicsClientId):
    actuated_joints = get_actuable_joint_indices(model, physicsClientId)
    states = p.getJointStates(model, actuated_joints, physicsClientId)
    angles = []
    for state in states:
        angles.append(state[0])
    return angles


def get_actuable_joint_velocities(model, physicsClientId):
    actuated_joints = get_actuable_joint_indices(model, physicsClientId)
    states = p.getJointStates(model, actuated_joints, physicsClientId)
    angles = []
    for state in states:
        angles.append(state[1])
    return angles


def set_joint_angles(model, joint_vals, physicsClientId, joint_vel=None):
    joint_idx = get_actuable_joint_indices(model, physicsClientId)
    if joint_vel is None:
        joint_vel = get_actuable_joint_velocities(model, physicsClientId)
    assert len(joint_vals) == len(joint_idx)
    for i in range(len(joint_idx)):
        p.resetJointState(model, joint_idx[i], joint_vals[i], joint_vel[i], physicsClientId=physicsClientId)


def get_all_joint_angles(model, physicsClientId):
    states = p.getJointStates(model, range(0, p.getNumJoints(model, physicsClientId=physicsClientId)), physicsClientId=physicsClientId)
    angles = []
    for state in states:
        angles.append(state[0])
    return angles


def get_eef_position(model, eef_link, physicsClientId):
    res = p.getLinkState(model, eef_link, physicsClientId=physicsClientId)
    return res[0]


def set_eef_position(pose, model, eef_link, physicsClientId):
    joints = get_ik_solutions(model, eef_link, pose[0:3], physicsClientId)
    set_joint_angles(model, joints, physicsClientId)


def get_eef_orientation(model, eef_link, physicsClientId):
    res = p.getLinkState(model, eef_link, physicsClientId=physicsClientId)
    return res[1]


def get_ik_solutions(model, eeffLink, goal, physicsClientId=0):
    pos = [0, 0, 0]
    rot = [0, 0, 0, 1]
    if len(goal) == 3:
        pos = goal
    elif len(goal) == 7:
        pos = goal[0:3]
        rot = goal[3:]
    else:
        return []

    lower, upper = get_joint_limits(model, physicsClientId=physicsClientId)
    rest = [0] * len(lower)
    ranges = [10000] * len(lower)
    # for i in range(0, len(lower)):
    #     ranges.append(upper[i] - lower[i])

    return p.calculateInverseKinematics(model, eeffLink, pos,
                                        lowerLimits=lower, upperLimits=upper, jointRanges=ranges, restPoses=rest,
                                        physicsClientId=physicsClientId)


def get_random_joint_values(model, physicsClientId):
    min_val, max_val = get_actuable_joint_limits(model, physicsClientId)
    return np.random.rand(len(min_val)) * (np.array(max_val) - np.array(min_val)) + np.array(min_val)


def get_eef_pose(model, eef_link, physicsClientId):
    pos = get_eef_position(model, eef_link, physicsClientId)
    rot = get_eef_orientation(model, eef_link, physicsClientId)
    return np.concatenate((np.array(pos), np.array(rot)))


def get_ik_jac_pinv_ns(model, eef_link, target, joint_ini=[], cmd_sec=[], max_time=0.1, pos_error_threshold=0.001,
                       rot_error_threshold=0.01, physicsClientId=0, debug=False):
    """
    :param model: Unique id for the model used
    :param eef_link: End effector link index in the model
    :param target: Desired position/pose for the end effector. The solver will use either a 3D vector specifying the
     target position or a 7D vector, describing the 3D target position and its orientation in quaternion form [x,y,z,w].
     The solver will use the unconstrained DOFs to satisfy the secondary objective.
    :param joint_ini: Starting joint values for the iterative solver. If not specified a random value will be used.
    :param cmd_sec: Desired joint values defining the secondary objective.
    :param max_time: Max time allowed for the solver to generate a solution.
    :param pos_error_threshold: Acceptable distance from the end effector to the target to consider a solution found.
    :param rot_error_threshold: Acceptable angular distance from the end effector orientation to the target orientation.
    :return: Joint values that position the end effector at the desired target pose.
    """
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=physicsClientId)

    current_joint_values = get_actuable_joint_angles(model, physicsClientId=physicsClientId)

    if len(joint_ini) == 0:
        joint_ini = get_random_joint_values(model, physicsClientId=physicsClientId)

    if len(cmd_sec) == 0:
        cmd_sec = np.zeros(len(joint_ini))

    pos_err = pos_error_threshold + 10
    rot_err = rot_error_threshold + 10
    start_time = time.time()
    sol = joint_ini
    set_joint_angles(model, sol, physicsClientId=physicsClientId)
    num_it = 0
    pos_only = False

    if target is None:
        raise ValueError("target is None")
    elif len(target) == 3:
        pos_only = True
        target = np.concatenate((target, np.array([0, 0, 0, 1])))
    elif len(target) != 7:
        raise ValueError("target: Desired position/pose for the end effector. The solver will use either a 3D vector " +
                         "specifying the target position or a 7D vector, describing the 3D target position and its " +
                         "orientation in quaternion form [x,y,z,w]. The solver will use the unconstrained DOFs to " +
                         "satisfy the secondary objective.Target pose invalid dimensions: " + str(target))

    while pos_err > pos_error_threshold or (rot_err > rot_error_threshold and not pos_only):
        # Obtain the twist that moves the end effector to the target pose in a unit time
        pose = get_eef_pose(model, eef_link, physicsClientId=physicsClientId)
        twist = ru.get_twist_from_two_poses(target=target, origin=pose)

        if pos_only:
            joint_delta = CartToJntVel(model, eef_link, twist[0:3], cmd_sec, physicsClientId=physicsClientId)
        else:
            joint_delta = CartToJntVel(model, eef_link, twist, cmd_sec, physicsClientId=physicsClientId)

        sol = sol + joint_delta
        set_joint_angles(model, sol, physicsClientId=physicsClientId)
        eef_pos = get_eef_position(model, eef_link, physicsClientId=physicsClientId)
        eef_rot = get_eef_orientation(model, eef_link, physicsClientId=physicsClientId)

        pos_err = np.sqrt(np.matmul((target[0:3] - eef_pos), (target[0:3] - eef_pos).transpose()))
        rot_err = ru.get_angle_from_two_quaternions(target[3:], eef_rot)
        elapsed_time = time.time() - start_time
        num_it = num_it + 1

        if elapsed_time > max_time:
            break

    # Leave the robot state as it was before the IK computation
    set_joint_angles(model, current_joint_values, physicsClientId=physicsClientId)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=physicsClientId)
    if debug:
        print("Iters: %d | pos_err: %2.5f | rot_err: %2.5f | time: %2.4f" % (num_it, pos_err, rot_err, elapsed_time))
    return sol


def CartToJntVel(model, eef_link, cmd, cmd_sec=[], physicsClientId=0):
    """
    :param model: Unique id for the model used
    :param eef_link: End effector link index in the model
    :param cmd: Desired end-effector twist in model base frame
    :param cmd_sec: Desired joint values for the secondary objective controller
    :param physicsClientId: pybullet physics client id. For simultaneous usage of multiple simulation instances
    :return: Joint velocity commands to achieve desired velocity at the tip
    """

    # TODO: Depending on the pybullet version the calculateJacobian function needs all joint values or
    # only the joints that are actuated

    # Use all joint values
    # pos = get_all_joint_angles(model, physicsClientId)
    # vel = [0] * len(pos)
    # acc = [0] * len(pos)

    # Use only actuated joint values
    pos_act = get_actuable_joint_angles(model, physicsClientId)
    vel = [0] * len(pos_act)
    acc = [0] * len(pos_act)
    

    Jt, Jr = p.calculateJacobian(bodyUniqueId=model, linkIndex=eef_link, localPosition=[0, 0, 0],
                        objPositions=pos_act, objVelocities=vel, objAccelerations=acc,
                        physicsClientId=physicsClientId)

    Jt_pinv = np.linalg.pinv(Jt)
    q_dot_lin = np.dot(Jt_pinv, cmd[0:3])

    # Compute the velocity to move the joints to its resting state and project it to the Jacobian nullspace to ensure
    # that joint velocities do not impact the primary task controller
    if len(cmd_sec) != 0:
        J_ns = np.eye(Jt_pinv.shape[0]) - np.dot(Jt_pinv, Jt)
        q_dot_ns = np.dot(J_ns, (np.array(cmd_sec) - np.array(pos_act)))
        q_dot_lin = q_dot_lin + q_dot_ns

    # TODO: Verify the angular jacobian
    if len(cmd) == 6:
        Jr_pinv = np.linalg.pinv(Jr)
        q_dot_ang = np.dot(Jr_pinv, cmd[3:])
        return q_dot_lin + q_dot_ang
    elif len(cmd) == 3:
        return q_dot_lin
    else:
        raise Exception("Command dimensions must be 3 or 6")


# TODO: Implementation required
def CartToJntEff(model, eef_link, cmd, cmd_sec=[], physicsClientId=0):
    """
    :param model: Unique id for the commanded model
    :param cmd: Desired end-effector force/torque in model base frame
    :return: Joint torque commands to achieve desired force at the tip
    """
    pass


def test_motion_ranges(model, n_iter, sim_id=0):
    speed = 0.02
    p_gain = 0.001
    time_step = 0.01

    for i in get_actuable_joint_indices(model, sim_id):
        p.setJointMotorControl2(model, i, p.POSITION_CONTROL, 0, p_gain)

    lower, upper = get_actuable_joint_limits(model, sim_id)
    for it in range(n_iter):
        for i in get_actuable_joint_indices(model, sim_id):
            for ang in np.arange(lower[i], upper[i], speed):
                p.removeAllUserDebugItems(physicsClientId=sim_id)
                draw_text("Jnt idx: " + str(i) + " Angle: " + str(ang), [0, 0, 1], sim_id)
                p.setJointMotorControl2(model, i, p.POSITION_CONTROL, ang, p_gain)
                time_end = sim_time + 2
                while sim_time < time_end:
                    p.stepSimulation()
                    sim_time = sim_time+time_step
            p.setJointMotorControl2(model, i, p.POSITION_CONTROL, 0, p_gain)


def execute_command(model, cmd, cmd_type, cmd_sec=[], physicsClientId=0, eef_link=-1):
    joint_indices = get_actuable_joint_indices(model, physicsClientId)

    # If the command comes in cartesian space convert it to joint space
    if cmd_type == "cart_pos":
        joint_cmd = CartToJntVel(model, eef_link, cmd, cmd_sec, physicsClientId)
        send_joint_vel_command(model, joint_cmd, joint_indices, physicsClientId=physicsClientId)
    elif cmd_type == "cart_vel":
        joint_cmd = CartToJntVel(model, eef_link, cmd, cmd_sec, physicsClientId)
        send_joint_vel_command(model, joint_cmd, joint_indices, physicsClientId=physicsClientId)
    elif cmd_type == "cart_eff":
        joint_cmd = CartToJntEff(model, eef_link, cmd, cmd_sec, physicsClientId)
        send_joint_eff_command(model, joint_cmd, joint_indices, physicsClientId=physicsClientId)
    elif cmd_type == "pos":
        send_joint_pos_command(model, cmd, joint_indices, physicsClientId=physicsClientId)
    elif cmd_type == "vel":
        send_joint_vel_command(model, cmd, joint_indices, physicsClientId=physicsClientId)
    elif cmd_type == "eff":
        send_joint_eff_command(model, cmd, joint_indices, physicsClientId=physicsClientId)
    else:
        raise Exception(" Unknown control mode:" + cmd_type)


def disable_always_on_self_collisions(bodyUniqueId, physicsClientId=0, rate=0.8, iterations=1000, debug=False):
    num_joints = p.getNumJoints(bodyUniqueId, physicsClientId=physicsClientId) + 1
    ini_pos, ini_rot = p.getBasePositionAndOrientation(bodyUniqueId, physicsClientId=physicsClientId)
    ini_vel = p.getBaseVelocity(bodyUniqueId, physicsClientId=physicsClientId)
    pairs = dict()

    print("Computing collision disable for body: %d" % bodyUniqueId)

    # Initialize pairwise dictionary
    for i in range(num_joints):
        for j in range(num_joints):
            pairs[(i,j)] = 0

    for it in range(iterations):
        # Set joints to a valid random value
        joints = get_random_joint_values(bodyUniqueId, physicsClientId=physicsClientId)
        set_joint_angles(bodyUniqueId, joints,physicsClientId=physicsClientId)

        # Step simulation to perform collision detection
        p.stepSimulation()

        # Check the joints that are in contact
        contacts = p.getContactPoints(physicsClientId=physicsClientId)
        for c in contacts:
            if c[1] == c[2] and c[2] == bodyUniqueId and c[3] != -1 and c[4] != -1:
                pairs[(c[3],c[4])] = pairs[(c[3],c[4])] + 1

        p.resetBasePositionAndOrientation(bodyUniqueId, ini_pos, ini_rot)
        p.resetBaseVelocity(bodyUniqueId, ini_vel)
        set_joint_angles(bodyUniqueId, np.zeros(joints.shape), joint_vel=np.zeros(joints.shape), physicsClientId=physicsClientId)

    disable_pairs = []
    for key,value in pairs.items():
        joint_rate = value / float(iterations)
        if joint_rate >= rate:
            print("Disable collision between links (%d, %d). Collision ratio: %f" % (key[0], key[1], joint_rate))
            disable_pairs.append([key[0], key[1]])
            if debug:
                body_visual_data = p.getVisualShapeData(bodyUniqueId)
                p.changeVisualShape(bodyUniqueId, key[0], rgbaColor=(1, 0, 0, 1))
                p.changeVisualShape(bodyUniqueId, key[1], rgbaColor=(1, 0, 0, 1))
            p.setCollisionFilterPair(bodyUniqueIdA=bodyUniqueId, bodyUniqueIdB=bodyUniqueId, linkIndexA=key[0], linkIndexB=key[1], enableCollision=0, physicsClientId=physicsClientId)
            p.setCollisionFilterPair(bodyUniqueIdA=bodyUniqueId, bodyUniqueIdB=bodyUniqueId, linkIndexA=key[1], linkIndexB=key[0], enableCollision=0, physicsClientId=physicsClientId)
            if debug:
                time.sleep(1)
                for data in body_visual_data:
                    if data[1] == key[0]:
                        p.changeVisualShape(bodyUniqueId, key[0], rgbaColor=data[7])
                    if data[1] == key[1]:
                        p.changeVisualShape(bodyUniqueId, key[1], rgbaColor=data[7])

    return disable_pairs


def get_link_matrix(robot, link_name, physicsClientId=0):
    n_joints = p.getNumJoints(robot, physicsClientId=physicsClientId)
    for i in range(n_joints):
        state = p.getJointInfo(bodyUniqueId=robot, jointIndex=i,physicsClientId=physicsClientId)
        if state:
            name = state[12].decode("utf-8")
            if name == link_name:
                lstate = p.getLinkState(bodyUniqueId=robot, linkIndex=i, physicsClientId=physicsClientId)

                rot = lstate[5]
                pos = lstate[4]
                mat = tf.quaternion_matrix([rot[3], rot[0], rot[1], rot[2]])
                mat[0:3, 3] = pos

                return mat


def get_camera_images(robot, link_name, physicsClientId=0, renderer=p.ER_BULLET_HARDWARE_OPENGL):
    viewMat = get_link_matrix(robot, link_name, physicsClientId=physicsClientId)
    camPos = viewMat[0:3,3].reshape(-1)
    viewMat = p.computeViewMatrix(cameraEyePosition=camPos,cameraTargetPosition=camPos+viewMat[0:3,2], cameraUpVector=-viewMat[0:3,1],physicsClientId=physicsClientId)
    projMat = p.computeProjectionMatrixFOV(45.0, 1.0, 0.05, 1000.0)
    images = p.getCameraImage(640, 480, viewMatrix=viewMat, projectionMatrix=projMat, physicsClientId=physicsClientId, renderer=renderer)
    img_rgb = Image.frombytes(mode='RGBA', size=(images[2].shape[1],images[2].shape[0]), data=images[2]).transpose(Image.FLIP_TOP_BOTTOM)

    far = 1000.0
    near = 0.01
    depth_raw = far * near / (far - (far - near) * images[3])
    img_depth = Image.frombytes(mode='F', size=(images[3].shape[1],images[3].shape[0]), data=depth_raw).transpose(Image.FLIP_TOP_BOTTOM)

    img_seg = Image.frombytes(mode='I', size=(images[4].shape[1],images[4].shape[0]), data=images[4]).transpose(Image.FLIP_TOP_BOTTOM)
    return [img_rgb,img_depth,img_seg]


def get_depth_colormap(depth_buffer):
    depth_raw = np.frombuffer(depth_buffer.tobytes(), np.float32)
    depth_raw = np.clip(depth_raw, 0.0, 10.0) / 10.0
    depth_map = cm.gist_earth(depth_raw, bytes=True).reshape(depth_buffer.size[1],depth_buffer.size[0],4)
    return Image.fromarray(depth_map)


def get_semantic_colormap(seg_buffer, num_classes=20):
    seg_raw = np.frombuffer(seg_buffer.tobytes(), np.uint32).astype(np.float32)
    seg_raw = seg_raw/num_classes
    seg_map = cm.rainbow(seg_raw, bytes=True).reshape(seg_buffer.size[1],seg_buffer.size[0],4)
    return Image.fromarray(seg_map)

