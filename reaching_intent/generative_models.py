from neural_emulators.common import *

import pybullet as p
import json
import time
import numpy.random as rd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.distributions.multivariate_normal import MultivariateNormal

from neural_emulators.generative_models import CGenerativeModel
from neural_emulators.generative_models import CGenerativeModelNN
from neural_emulators.generative_models import CDataset
from neural_emulators_code.utils.draw import draw_trajectory
from neural_emulators_code.utils.draw import draw_point
from neural_emulators_code.utils.draw import draw_line
from neural_emulators_code.utils.misc import resample_trajectory
from manipulator_planning_control import pybullet_utils as pb


class CGenerativeSimulator(CGenerativeModel):
    def __init__(self, sim_params):
        self.model = None
        model = CGenerativeModelSimulator(sim_params)
        self.initialize(model)

    def initialize(self, model):
        self.model = model

    def generate(self, params):
        return self.model(params)

    def gradient(self):
        return self.model.gradient()


class CGenerativeModelSimulator(object):
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

    class Model(object):
        def __init__(self, generate_f, input_dim, output_dim, device):
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.generate_f = generate_f
            self.device = device

        def __call__(self, params):
            return self.generate_f(params)

    def initialize(self, model, visualization=False, timestep=0.01, sim_time=5.0):
        self.init_physics(visualization, timestep)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
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

    def reset(self):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.resetSimulation(physicsClientId=self.sim_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.sim_id)
        self.model_id = p.loadURDF(self.model_path, useFixedBase=1, physicsClientId=self.sim_id, flags=p.URDF_USE_SELF_COLLISION)
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

    def generate(self, params):
        """
        Generates a trajectory using the initial cartesian position, target cartesian position and controller
        parameters (Kp Ki Kd Krep).
        """

        trajs = t_tensor([])

        # The parameters are passed to a neural network in mini-batches. Iterate through the minibatch
        # and generate all desired trajectories
        for i in range(len(params)):
            sample = params[i]
            start = sample[0:3]  # Starting hand position (x,y,z)
            goal = sample[3:6]   # Goal hand position (x,y,z)
            K = sample[6:]       # Controller parameters (Kp Ki Kd Krep)

            # Obtain initial joint values corresponding to the start cartesian position
            joint_vals = pb.get_actuable_joint_angles(self.model_id, physicsClientId=self.sim_id)
            joints_ik = pb.get_ik_jac_pinv_ns(model=self.model_id, eef_link=self.eef_link,
                                              target=start, joint_ini=joint_vals, cmd_sec=self.joint_rest_positions,
                                              physicsClientId=self.sim_id, debug=False)

            goal_threshold = 0.03
            timeout = self.sim_time
            [plan_joint, plan_cart] = self.get_plan(joints_ik, goal, self.model_id, self.eef_link, K, goal_threshold, timeout, self.obstacles, physicsClientId=self.sim_id)

            res = resample_trajectory(plan_cart, 1/self.timestep, self.sample_rate)
            trajs = torch.cat((trajs,res))
            trajs = torch.cat((trajs, torch.ones(len(res)).double()))  # WARNING: Add the vector of standard deviations. Set manually to one.

        return trajs.view(len(params), -1)

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
        self.controller.Kp = np.array(controller_gains[0])
        self.controller.Ki = np.array(controller_gains[1])
        self.controller.Kd = np.array(controller_gains[2])
        self.controller.Krep = np.array(controller_gains[3])
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

        #Reach for a via-point inbetween
        current = pb.get_eef_pose(model, eef_link, physicsClientId)
        via_point = t_tensor([current[0] + (goal[0] - current[0])/2, current[1] + (goal[1] - current[1])/2, goal[2]+0.2])
        via_point_satisfied = False
        real_goal = goal
        goal = via_point
        x_err_mod = goal_threshold + 10
        while not is_goal_satisfied and p.isConnected(physicsClientId=physicsClientId):
            if x_err_mod < 0.15 and not via_point_satisfied:
                goal = real_goal
                via_point_satisfied = True

            t_ini = time.time()
            # Obtain current joint positions
            joint_state = p.getJointStates(model, range(0, p.getNumJoints(model, physicsClientId=physicsClientId)), physicsClientId=physicsClientId)
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
                debug_traj.append(p.addUserDebugText("t = %.2f" % self.current_time, [0, 0, 1.0], physicsClientId=physicsClientId))
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
                # elapsed_time = time.time() - t_ini
                # t_sleep = self.timestep - elapsed_time
                # if t_sleep > 0:
                #     time.sleep(self.timestep - elapsed_time)

        if self.visualize:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            draw_line(plan_cart[0:3], goal, [0, 1, 1], width=1, physicsClientId=physicsClientId)
            draw_point(goal, [0, 1, 0], size=0.05, width=3, physicsClientId=physicsClientId)
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

    def get_random_state(self, model):
        joint_info = []
        for i in range(0, p.getNumJoints(model, physicsClientId=self.sim_id)):
            joint_info.append(p.getJointInfo(model, i, physicsClientId=self.sim_id))  # Get all joint limits

        joint_values = []

        lower_limit, upper_limit = self.get_joint_limits(model, physicsClientId=self.sim_id)

        for j in range(0, len(lower_limit)):
            joint_values.append(rd.uniform(lower_limit[i], upper_limit[i]))

        return joint_values

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

    def get_random_goal(self, a, b):
        res = [0, 0, 0]
        for i in range(0, 3):
            res[i] = rd.uniform(a[i], b[i])
        return res

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


class CReachingNeuralEmulatorNN(CGenerativeModelNN):
    def __init__(self, input_dim, output_dim, learning_rate=0.01, nlayers=4, debug=False, device="cpu", activation=F.relu, criterion=F.mse_loss):
        super(CReachingNeuralEmulatorNN, self).__init__()
        self.is_differentiable = True

        self.device = device
        device = torch.device(self.device)

        self.activation = activation
        self.criterion = criterion
        self.arch = ""
        self.dropout = nn.Dropout(p=0.2)

        self.layers = [None] * nlayers

        self.layers[0] = nn.Linear(input_dim, output_dim * 2).double().to(device)
        self.arch = self.arch + "fc{%d_%d}-" % (input_dim, output_dim * 2)
        for i in range(1, nlayers):
            self.layers[i] = nn.Linear(output_dim * 2, output_dim * 2).double().to(device)  # Outputs mu and sigma. Therefore the ouput dims are * 2
            self.arch = self.arch + "fc{%d_%d}-" % (output_dim * 2, output_dim * 2)

        self.output_dim = output_dim
        self.input_dim = input_dim
        print("=============================================")
        print("Neural Emulator weights")
        print("=============================================")
        i = 0
        for l in self.layers:
            print("==== LAYER %d ===============================" % i)
            print(l.weight)
            i = i + 1
        print("=============================================")
        print("=============================================")

        self.debug = debug

    def move_to_device(self, device):
        self.to(device)
        for i in range(len(self.layers)):
            self.layers[i].to(device)

        self.device = device

    def parameters(self):
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params

    def forward(self, x, dropout=None):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            if dropout is not None:
                x = dropout(x)
            x = self.activation(x)

        # Do not use activation in the last layer
        x = self.layers[-1](x)

        # Enforce the positive values for the covariance with an abs function
        y = x.clone()
        y[:, self.output_dim:] = torch.abs(y[:, self.output_dim:].clone())
        # y[:, self.output_dim:] = torch.max(y[:, self.output_dim:].clone(), t_tensor([0.001]).to(self.device))
        # x[:, self.output_dim:] = (x[:, self.output_dim:]*x[:, self.output_dim:]).clone()

        # if torch.isnan(torch.sum(x)):
        #     print("WARNING! NN output contains NaNs!")

        return y

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def test(self, dataset):
        device = torch.device(self.device)

        testloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        with torch.no_grad():
            for i, (inputs, ground_truth) in enumerate(testloader, 0):
                inputs = inputs.to(device)
                ground_truth = ground_truth.to(device)

                # forward pass and loss computation
                outputs = self.forward(inputs, dropout=None)
                loss, loss_terms = self.criterion(outputs, ground_truth)

        return loss.mean().item(), [loss_terms[0].mean().item(), loss_terms[1].mean().item(), loss_terms[2].mean().item(), loss_terms[3].mean().item()]

    def train(self, dataset, epochs, learning_rate=0.01, minibatch_size=1024):

        device = torch.device(self.device)

        # create your optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # file = open("training_evolution.txt", 'a')

        for epoch in range(epochs):  # loop over the dataset multiple times
            time_ini = time.time()

            trainloader = torch.utils.data.DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

            for i, (inputs, ground_truth) in enumerate(trainloader, 0):
                inputs = inputs.to(device)
                ground_truth = ground_truth.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs, nn.Dropout(p=0.2))
                loss, loss_terms = self.criterion(outputs, ground_truth)
                loss.mean().backward()  # https://discuss.pytorch.org/t/loss-backward-raises-error-grad-can-be-implicitly-created-only-for-scalar-outputs/12152
                optimizer.step()

                # print statistics
                if self.debug:
                    print('[%2d, %6d] loss: %.5f time: %.2f device: %s' % (epoch + 1, i + 1, loss.mean(), time.time()-time_ini, self.device))
                    print('           terms: %3.5f\t%3.5f\t%3.5f\t%3.5f' % (loss_terms[0].mean().item(), loss_terms[1].mean().item(), loss_terms[2].mean().item(), loss_terms[3].mean().item()) )

            # print('[%2d, %6d] loss: %.5f time: %.2f device: %s' % (epoch + 1, i + 1, running_loss / 1000, time.time() - time_ini, self.device))
            # print('           terms: ', running_loss_terms[0] / 1000, running_loss_terms[1] / 1000, running_loss_terms[2] / 1000, running_loss_terms[3] / 1000)


class CReachingDataset(CDataset):
    def __init__(self, filename, noise_sigma=0.0, dataset_sample_rate=30, output_sample_rate=30, ndims=3):
        self.filename = filename
        self.samples = []
        self.dataset_load(noise_sigma, ndims, dataset_sample_rate, output_sample_rate)

    def dataset_load(self, noise_sigma=0.001, ndims=3, dataset_sample_rate=30, output_sample_rate=30,
                     prefix_samples=4, trajectory_duration=5.0):

        try:
            file = open(self.filename, 'r')
        except FileNotFoundError:
            return self.samples
        lines = file.readlines()

        # Load samples into a list
        for l in lines:
            sample = json.loads(l)

            traj_len = len(sample[1]) / 2 # Crop the stdev part of the trajectory

            out_params = resample_trajectory(t_tensor(sample[1][0:int(traj_len)]), dataset_sample_rate, output_sample_rate)

            in_params = out_params[0:ndims * prefix_samples]    # Use the trajectory prefix as the input data

            in_params = torch.cat( (in_params, out_params[-ndims:])) # Add the last point of the trajectory as the goal

            if ndims * trajectory_duration * output_sample_rate > len(out_params):
                padding = out_params[-ndims:]
                while ndims * trajectory_duration * output_sample_rate > len(out_params):
                    out_params = torch.cat((out_params, padding))
            else:
                out_params = out_params[0:ndims * trajectory_duration * output_sample_rate]


            # Add optional noise to the trajectory
            if noise_sigma != 0:
                noise_cov = torch.diag(torch.ones_like(out_params) * noise_sigma * noise_sigma)
                noise_dist = MultivariateNormal(torch.zeros_like(out_params), covariance_matrix=noise_cov)
                noise = noise_dist.sample()
                out_params = out_params + noise.double()

            self.samples.append([in_params, out_params])

        print("Loaded %d trajectories" % len(self.samples))
        return self.samples


