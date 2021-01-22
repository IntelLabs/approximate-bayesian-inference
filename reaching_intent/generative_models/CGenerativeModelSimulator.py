import time
import pybullet as p

from common.common import *
from common.CBaseGenerativeModel import CBaseGenerativeModel
from utils.misc import resample_trajectory
import utils.pybullet_utils as pb
from utils.draw import draw_point
from utils.draw import draw_trajectory
import utils.pybullet_utils as pb
import utils.pybullet_controller as pbc
from pathlib import Path
from os import sep


def create_sim_params(sim_viz=True, sim_timestep=0.01, sim_time=5.0,
                      model_path="pybullet_models/human_torso/model.urdf", sample_rate=30):
    simulator_params = dict()
    simulator_params["robot_model_path"] = model_path
    simulator_params["visualization"] = sim_viz
    simulator_params["timestep"] = sim_timestep
    simulator_params["episode_time"] = sim_time
    simulator_params["sample_rate"] = sample_rate
    simulator_params["sim_id"] = 0
    simulator_objects = dict()
    simulator_objects["path"] = ["pybullet_models/table/table.urdf"]
    simulator_objects["pose"] = [[0.6, 0, -0.65]]
    simulator_objects["static"] = [True]
    simulator_params["objects"] = simulator_objects

    # Controller parameters
    Kp = 10
    Kd = 0
    Ki = 0.01
    iClamp = 25.0
    Krep = 0.1
    PIDcontroller = pbc.CPIDController(Kp=Kp, Kd=Kd, Ki=Ki, iClamp=iClamp)  # Controller to be used
    controller = pbc.CPotentialFieldController(Krep=Krep, ctrl=PIDcontroller)
    simulator_params["robot_controller"] = controller

    return simulator_params


class CGenerativeModelSimulator(CBaseGenerativeModel):
    def __init__(self, sim_params):  # [model, visualization=False, timestep=0.01, sim_time=5, sim_id=0]
        self.sim_id = sim_params["sim_id"]
        self.model_id = 0
        self.eef_link = 0
        self.current_time = 0
        self.model_path = sim_params["robot_model_path"]
        self.visualize = sim_params["visualization"]
        self.timestep = sim_params["timestep"]
        self.sim_time = sim_params["episode_time"]
        self.sample_rate = sim_params["sample_rate"]
        self.device = torch.device("cpu")
        self.is_differentiable = False
        self.obstacles = []
        self.objects_path = sim_params["objects"]["path"]
        self.objects_static = sim_params["objects"]["static"]
        self.objects_pose = sim_params["objects"]["pose"]
        self.controller = sim_params["robot_controller"]
        self.model = self.Model(self.generate, 4, self.sim_time * self.sample_rate * 3, self.device)
        self.coll_disable_pairs = []
        self.initialize(self.model_path, self.visualize, self.timestep, self.sim_time)

    @staticmethod
    def get_name():
        return "sim"

    def initialize(self, model, visualization=False, timestep=0.01, sim_time=5.0):
        self.init_physics(visualization, timestep)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetSimulation(physicsClientId=self.sim_id)
        self.model_id = p.loadURDF(model, useFixedBase=1, physicsClientId=self.sim_id, flags=p.URDF_USE_SELF_COLLISION)
        self.coll_disable_pairs = pb.disable_always_on_self_collisions(self.model_id, physicsClientId=self.sim_id, rate=0.8, iterations=1000, debug=False)

        # self.eef_link = p.getNumJoints(self.model_id, physicsClientId=self.sim_id) - 1
        act_joint_idx = pb.get_actuable_joint_indices(self.model_id, physicsClientId=self.sim_id)
        # self.eef_link = act_joint_idx[-1]
        self.eef_link = act_joint_idx[-1] + 1
        self.timestep = timestep
        self.sim_time = sim_time
        self.current_time = 0

        for i in range(len(self.objects_path)):
            self.obstacles.append(p.loadURDF(self.objects_path[i], useFixedBase=self.objects_static[i],
                                             basePosition=self.objects_pose[i], physicsClientId=self.sim_id))

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # Set the resting position for each joint
        joint_indices = pb.get_actuable_joint_indices(self.model_id, self.sim_id)
        # joint_rest_positions = t_tensor(torch.zeros(len(joint_indices)))
        joint_rest_positions = np.zeros(len(joint_indices))
        joint_rest_positions[5] = -.5  # Elbow at 60 deg
        joint_rest_positions[6] = 1.57   # Palm down
        self.joint_rest_positions = joint_rest_positions
        pb.set_joint_angles(self.model_id, self.joint_rest_positions, physicsClientId=self.sim_id)

    def reset(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.resetSimulation(physicsClientId=self.sim_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.sim_id)
        self.model_id = p.loadURDF(self.model_path, useFixedBase=1, physicsClientId=self.sim_id, flags=p.URDF_USE_SELF_COLLISION)
        pb.set_joint_angles(self.model_id, self.joint_rest_positions, physicsClientId=self.sim_id)
        for pair in self.coll_disable_pairs:
            p.setCollisionFilterPair(bodyUniqueIdA=self.model_id, bodyUniqueIdB=self.model_id, linkIndexA=pair[0],
                                     linkIndexB=pair[1], enableCollision=0, physicsClientId=self.sim_id)
        self.obstacles = []
        for i in range(len(self.objects_path)):
            self.obstacles.append(p.loadURDF(self.objects_path[i], useFixedBase=self.objects_static[i],
                                             basePosition=self.objects_pose[i], physicsClientId=self.sim_id))
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self.current_time = 0

    def __call__(self, params):
        return self.generate(params)

    def generate(self, z, n):
        """
        Generates a trajectory using the initial cartesian position, target cartesian position and controller
        parameters (Kp Ki Kd iClamp Krep).
        """

        assert len(z) == len(n), "Latent and nuisance number of samples (batch dimension size) must match"

        trajs = t_tensor([])

        # The parameters are passed to a neural network in mini-batches. Iterate through the minibatch
        # and generate all desired trajectories
        for i in range(len(z)):
            goal = t_tensor(z[i])        # Goal hand position (x,y,z)
            start = t_tensor(n[i, 0:3])  # Starting hand position (x,y,z)
            K = t_tensor(n[i, 3:])       # Controller parameters (Kp Ki Kd iClamp Krep)

            # Obtain initial joint values corresponding to the start cartesian position
            # joint_vals = pb.get_actuable_joint_angles(self.model_id, physicsClientId=self.sim_id) # Use this for generating ik solutions that start at the current position
            # joint_vals = torch.zeros_like(t_tensor(self.joint_rest_positions))
            joint_vals = self.joint_rest_positions
            joints_ik = pb.get_ik_jac_pinv_ns(model=self.model_id, eef_link=self.eef_link,
                                              target=start, joint_ini=joint_vals, cmd_sec=self.joint_rest_positions,
                                              physicsClientId=self.sim_id, debug=False)

            goal_threshold = 0.03
            timeout = self.sim_time
            [plan_joint, plan_cart] = self.get_plan(joints_ik, goal, self.model_id, self.eef_link, K,
                                                    goal_threshold, timeout, self.obstacles, physicsClientId=self.sim_id)

            res = resample_trajectory(plan_cart, 1/self.timestep, self.sample_rate)
            trajs = torch.cat((trajs, res))

        return trajs.view(len(z), -1)

    def gradient(self):
        raise NotImplementedError

    def generate_dataset(self, n_points, goal_sampler):
        output_cart = []
        njoints = p.getNumJoints(self.model_id, physicsClientId=self.sim_id)

        while len(output_cart) < n_points:
            time_ini = time.time()

            # Fix the torso joints to start close to zero (to avoid singular initial configuration)
            joint_ini_angles = [0.01] * p.getNumJoints(self.model_id, physicsClientId=self.sim_id)

            # Generate random parameters
            params = goal_sampler.sample(None)
            goal = params[0:3]
            controller_gain = params[3]

            self.reset()
            self.controller.reset()

            # Generate a probable control sequence that achieves the goal
            plan_cart = self.generate(params.view(1, -1))
            print("sample #", len(output_cart) + 1, " goal: ", goal, " K: ", controller_gain, " plan size:", len(plan_cart[0]/3),
                  " time: ", "%f" % (time.time() - time_ini))

            if len(plan_cart) > 0:
                # goal = plan_cart[0][int((len(plan_cart[0])/2)-3):int(len(plan_cart[0])/2)]
                output_cart.append([joint_ini_angles, goal, controller_gain, plan_cart])

        return output_cart

    def step_plan_potential_field(self, model, goal, eef_link, obstacles, controller_gains, physicsClientId=0):
        state = pb.get_eef_position(model, eef_link, physicsClientId)
        self.controller.set_model(model, eef_link, physicsClientId)
        self.controller.set_obstacles(obstacles)
        self.controller.ctrl.Kp = np.array(controller_gains[0])
        self.controller.ctrl.Ki = np.array(controller_gains[1])
        self.controller.ctrl.Kd = np.array(controller_gains[2])
        self.controller.Krep = np.array(controller_gains[3])
        self.controller.ctrl.iClamp = np.array(controller_gains[4])
        control_action = self.controller.get_command(state, goal.detach().numpy())
        pb.execute_command(model=model, cmd=control_action,
                           cmd_type=self.controller.control_type,
                           cmd_sec=self.joint_rest_positions, eef_link=eef_link, physicsClientId=physicsClientId)

        # Step the simulation and store the control action taken
        p.stepSimulation(physicsClientId=physicsClientId)
        self.current_time = self.current_time + self.timestep

    def step_plan_ik(self, model, goal, joint_pos, eef_link, K, actuated_joint_list, physicsClientId=0):
        # Set the control action to move the joints towards the target IK
        q = self.get_ik_solutions(model, eef_link, goal, physicsClientId=physicsClientId)
        q_dot = [0] * len(q)
        for i in range(0, len(q)):
            q_dot[i] = (q[i] - joint_pos[actuated_joint_list[i]]) * K

        # Update the motor control only of the actuated joints (not fixed)
        p.setJointMotorControlArray(model,
                                    actuated_joint_list,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities=q_dot,
                                    physicsClientId=physicsClientId
                                    )

        # Step the simulation and store the control action taken
        p.stepSimulation(physicsClientId=physicsClientId)
        self.current_time = self.current_time + self.timestep

    def get_plan(self, ini_state, goal, model, eef_link, K, goal_threshold=0.01, timeout=10.0, obstacles=[], physicsClientId=0):
        self.current_time = 0
        plan = []
        plan_cart = t_tensor()

        # Move the joints to the starting state
        indices = pb.get_actuable_joint_indices(model, physicsClientId=physicsClientId)
        for i,idx in enumerate(indices):
            p.resetJointState(model, idx, ini_state[i], physicsClientId=physicsClientId)

        is_goal_satisfied = False
        is_goal_reached = False

        joint_pos = [0] * p.getNumJoints(model, physicsClientId=physicsClientId)

        if self.visualize:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            debug_objs = [p.addUserDebugText("k = " + str(K), [0, 0, 1.05], physicsClientId=physicsClientId)]
            debug_objs.extend(draw_point(goal[0:3], [1,0,0], size=0.1, width=3, physicsClientId=physicsClientId))
            debug_traj = []
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # Reach for a via-point inbetween
        current = pb.get_eef_pose(model, eef_link, physicsClientId)
        via_point = t_tensor([current[0] + (goal[0] - current[0])/2, current[1] + (goal[1] - current[1])/2, goal[2]+0.2])
        via_point_satisfied = True
        real_goal = goal
        # goal = via_point
        x_err_mod = goal_threshold + 10
        while not is_goal_satisfied and p.isConnected(physicsClientId=physicsClientId):
            if x_err_mod < 0.15 and not via_point_satisfied:
                goal = real_goal
                via_point_satisfied = True

            t_ini = time.time()
            # Obtain current joint positions
            joint_state = p.getJointStates(model, range(0, p.getNumJoints(model, physicsClientId=physicsClientId)),
                                           physicsClientId=physicsClientId)
            for i in range(0, p.getNumJoints(model, physicsClientId=physicsClientId)):
                joint_pos[i] = joint_state[i][0]

            plan.append(joint_pos)

            # Obtain current end effector position
            link_state = p.getLinkState(model, eef_link, 0, 1, physicsClientId=physicsClientId)
            x = t_tensor(link_state[0])
            x_err = goal - x
            x_err_mod = torch.sqrt(torch.sum(x_err * x_err))

            if x_err_mod < goal_threshold and not is_goal_reached:
                is_goal_reached = True
                # print("Reached goal in ", self.current_time, "s")

            plan_cart = torch.cat((plan_cart, x))

            # if x_err_mod < goal_threshold or self.current_time >= timeout:
            if self.current_time >= timeout:
                    is_goal_satisfied = True

            self.step_plan_potential_field(model, goal, eef_link, obstacles, K, physicsClientId)

            if self.visualize:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
                for obj_id in debug_traj:
                    p.removeUserDebugItem(obj_id, physicsClientId=physicsClientId)
                debug_traj = draw_trajectory(plan_cart.view(-1,3), [0, 1, 0], 4, physicsClientId=physicsClientId, draw_points=False)
                if len(self.controller.ctrl.err_int) > 0:
                    debug_traj.append(p.addUserDebugText("t = %.2f err_int=[%.2f,%.2f,%.2f]" % (self.current_time, self.controller.ctrl.err_int[0],self.controller.ctrl.err_int[1],self.controller.ctrl.err_int[2]), [0, 0, 1.0], physicsClientId=physicsClientId))
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
                # elapsed_time = time.time() - t_ini
                # t_sleep = self.timestep - elapsed_time
                # if t_sleep > 0:
                #     time.sleep(self.timestep - elapsed_time)

        if self.visualize:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            # draw_line(plan_cart[0:3], goal, [0, 1, 1], width=1, physicsClientId=physicsClientId)
            # draw_point(goal, [0, 1, 0], size=0.05, width=3, physicsClientId=physicsClientId)
            for obj_id in debug_objs:
                p.removeUserDebugItem(obj_id, physicsClientId=physicsClientId)
            for obj_id in debug_traj:
                p.removeUserDebugItem(obj_id, physicsClientId=physicsClientId)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        return [plan, plan_cart]

    def init_physics(self, use_gui, timestep=0.01):
        pType = p.GUI
        if not use_gui:
            pType = p.DIRECT

        if not p.isConnected(self.sim_id):
            self.sim_id = p.connect(pType)

        p.setGravity(0, 0, -9.81, physicsClientId=self.sim_id)

        p.setPhysicsEngineParameter(
            fixedTimeStep=timestep,                 # Physics engine timestep in fraction of seconds, each time you call 'stepSimulation'.Same as 'setTimeStep'
            numSolverIterations=100,                # Choose the number of constraint solver iterations.
            useSplitImpulse=0,                      # Advanced feature, only when using maximal coordinates: split the positional constraint solving and velocity constraint solving in two stages, to prevent huge penetration recovery forces.
            splitImpulsePenetrationThreshold=0.01,  # Related to 'useSplitImpulse': if the penetration for a particular contact constraint is less than this specified threshold, no split impulse will happen for that contact.
            numSubSteps=1,                         # Subdivide the physics simulation step further by 'numSubSteps'. This will trade performance over accuracy.
            collisionFilterMode=0,                  # Use 0 for default collision filter: (group A&maskB) AND (groupB&maskA). Use 1 to switch to the OR collision filter: (group A&maskB) OR (groupB&maskA)
            contactBreakingThreshold=0.02,          # Contact points with distance exceeding this threshold are not processed by the LCP solver. In addition, AABBs are extended by this number. Defaults to 0.02 in Bullet 2.x.
            maxNumCmdPer1ms=0,                      # Experimental: add 1ms sleep if the number of commands executed exceed this threshold.
            enableFileCaching=1,                    # Set to 0 to disable file caching, such as .obj wavefront file loading
            restitutionVelocityThreshold=0.01,      # If relative velocity is below this threshold, restitution will be zero.
            erp=0.8,                                # Constraint error reduction parameter (non-contact, non-friction) Details: http://www.ode.org/ode-latest-userguide.html#sec_3_7_0
            contactERP=0.2,                         # Contact error reduction parameter
            frictionERP=0.2,                        # Friction error reduction parameter (when positional friction anchors are enabled)
            enableConeFriction=0,                   # Set to 0 to disable implicit cone friction and use pyramid approximation (cone is default)
            deterministicOverlappingPairs=0,        # Set to 0 to disable sorting of overlapping pairs (backward compatibility setting).
            physicsClientId=self.sim_id
        )

    @staticmethod
    def get_joint_limits(self, model, physicsClientId=0):
        joint_info = []
        for i in range(0, p.getNumJoints(model, physicsClientId=physicsClientId)):
            joint_info.append(p.getJointInfo(model, i, physicsClientId=physicsClientId))  # Get all joint limits

        upper_limit = []
        lower_limit = []
        for j in joint_info:
            upper_limit.append(j[8])
            lower_limit.append(j[9])

        return lower_limit, upper_limit

    def get_ik_solutions(self, model, eeffLink, goal, physicsClientId=0):
        pos = [0, 0, 0]
        rot = [0, 0, 0, 1]
        if len(goal) == 3:
            pos = goal
        elif len(goal) == 7:
            pos = goal[0:3]
            rot = goal[3:]
        else:
            return []

        lower, upper = self.get_joint_limits(model, physicsClientId=physicsClientId)
        rest = [0] * len(lower)
        ranges = [10000] * len(lower)
        # for i in range(0, len(lower)):
        #     ranges.append(upper[i] - lower[i])

        return p.calculateInverseKinematics(model, eeffLink, pos,
                                            lowerLimits=lower, upperLimits=upper, jointRanges=ranges, restPoses=rest,
                                            physicsClientId=physicsClientId)
