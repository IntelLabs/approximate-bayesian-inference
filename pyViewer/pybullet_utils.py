import pybullet as p
import pyViewer.transformations as tf
from pyViewer.viewer import CTransform, CNode
from pyViewer.geometry_makers import make_mesh
from pyViewer.models import REFERENCE_FRAME_MESH
import numpy as np

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
        # Advanced feature, only when using maximal coordinates: split the positional constraint solving and velocity constraint solving in two stages, to prevent huge penetration recovery forces.
        splitImpulsePenetrationThreshold=0.01,
        # Related to 'useSplitImpulse': if the penetration for a particular contact constraint is less than this specified threshold, no split impulse will happen for that contact.
        numSubSteps=1,
        # Subdivide the physics simulation step further by 'numSubSteps'. This will trade performance over accuracy.
        collisionFilterMode=0,
        # Use 0 for default collision filter: (group A&maskB) AND (groupB&maskA). Use 1 to switch to the OR collision filter: (group A&maskB) OR (groupB&maskA)
        contactBreakingThreshold=0.02,
        # Contact points with distance exceeding this threshold are not processed by the LCP solver. In addition, AABBs are extended by this number. Defaults to 0.02 in Bullet 2.x.
        maxNumCmdPer1ms=0,  # Experimental: add 1ms sleep if the number of commands executed exceed this threshold.
        enableFileCaching=1,  # Set to 0 to disable file caching, such as .obj wavefront file loading
        restitutionVelocityThreshold=0.01,  # If relative velocity is below this threshold, restitution will be zero.
        erp=0.9,
        # Constraint error reduction parameter (non-contact, non-friction) Details: http://www.ode.org/ode-latest-userguide.html#sec_3_7_0
        contactERP=0.2,  # Contact error reduction parameter
        frictionERP=0.2,  # Friction error reduction parameter (when positional friction anchors are enabled)
        enableConeFriction=0,
        # Set to 0 to disable implicit cone friction and use pyramid approximation (cone is default)
        deterministicOverlappingPairs=0,
        # Set to 0 to disable sorting of overlapping pairs (backward compatibility setting).
        physicsClientId=sim_id
    )
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    return sim_id


def load_simulation(model_path, objects_path, objects_pose, objects_static, physicsClientId=0):
    obstacles = []
    model = -1
    if model_path != "":
        model = p.loadURDF(model_path, useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT, physicsClientId=physicsClientId)
    for i in range(len(objects_path)):
        obstacles.append(p.loadURDF(objects_path[i], useFixedBase=objects_static[i], basePosition=objects_pose[i], physicsClientId=physicsClientId))
    return model, obstacles


def update_pybullet_nodes(nodes, physicsClientId=0):
    for b in nodes:
        if b.pybullet_id is not None:
            if b.pybullet_link_id == -1:
                state = p.getBasePositionAndOrientation(b.pybullet_id, physicsClientId=physicsClientId)
                pos = state[0]
                rot = state[1]
                mat = tf.quaternion_matrix([rot[3],rot[0],rot[1],rot[2]])
                mat[0:3, 3] = pos
                b.t = CTransform(mat)
            else:
                state = p.getLinkState(b.pybullet_id, b.pybullet_link_id, physicsClientId=physicsClientId)
                # pos_com = np.array(state[0])  # Position of the local inertial frame in the world coordinates
                # rot_com = state[1]  # Position of the local inertial frame in the world coordinates
                # pos_local = np.array(state[2])  # Link position in local inertial frame
                # rot_local = state[3]  # Link orientation in local inertial frame
                pos = state[4]  # Link World absolute position
                rot = state[5]  # Link World absolute orientation
                # rot_com = tf.quaternion_matrix([rot_com[3],rot_com[0],rot_com[1],rot_com[2]])
                # rot_local = tf.quaternion_matrix([rot_local[3],rot_local[0],rot_local[1],rot_local[2]])
                mat = tf.quaternion_matrix([rot[3], rot[0], rot[1], rot[2]])
                mat[0:3, 3] = pos
                b.t = CTransform(np.matmul(b.pybullet_v_mat, mat))


def make_pybullet_scene(ctx, physicsClientId=0):
    root = CNode(0, None, CTransform())
    nodes = [root]
    if not p.isConnected(physicsClientId=physicsClientId):
        raise Exception("Not connected to pybullet server %d" % physicsClientId)

    nbodies = p.getNumBodies(physicsClientId=physicsClientId)
    for i in range(nbodies):
        node = make_pybullet_node(ctx, i, physicsClientId)
        node[0].set_parent(root)
        nodes.extend(node)
    return nodes


def make_pybullet_node(ctx, body_id, physicsClientId=0):
    if not p.isConnected(physicsClientId=physicsClientId):
        raise Exception("Not connected to pybullet server %d" % physicsClientId)
    body_name = p.getBodyInfo(body_id, physicsClientId=physicsClientId)
    nodes = []
    print("Loading body: " + str(body_name))
    for shapes in p.getVisualShapeData(body_id, physicsClientId=physicsClientId):
        link_id = shapes[1]
        mesh_file = shapes[4]
        if mesh_file:
            print("---- Link idx: %d Shape file: " % link_id + str(mesh_file))
            node = CNode(geometry=make_mesh(ctx, mesh_file))
            node_frame = CNode(geometry=make_mesh(ctx, REFERENCE_FRAME_MESH, scale=0.05))
            node_frame.set_parent(node)
            node.pybullet_id = body_id
            node.pybullet_link_id = link_id
            nodes.append(node)
            nodes.append(node_frame)

    return nodes

