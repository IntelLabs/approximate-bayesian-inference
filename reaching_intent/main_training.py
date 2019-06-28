#!/usr/bin/python3

#################
# SYSTEM IMPORTS
#################
import torch
import torch.nn
import time
import pybullet as p
import numpy as np

##############################
# APPLICATION SPECIFIC IMPORTS (import from your application specific module)
##############################
from generative_models import CReachingNeuralEmulatorNN as CNeuralEmulatorNN
##############################


#######################################
# GENERIC IMPORTS (no need to edit)
#######################################
from neural_emulators.common import *
from generative_models import CReachingDataset
from neural_emulators.generative_models import CGenerativeModelNeuralEmulator
from generative_models import CGenerativeModelSimulator
from neural_emulators.loss_functions import loss_MSE
##############################


#######################################
# DEBUG DRAWING IMPORTS
#######################################
from utils.draw import draw_trajectory
from utils.draw import draw_trajectory_diff
from mss import mss

###################################
# APPLICATION SPECIFIC PARAMETERS (add/remove/edit parameters to fit your implementation)
###################################
sample_rate             = 30 # Sensor sampling rate

sim_time                = 5  # Prediction time window

n_prefix_points         = 4  # Number of points in the prefix trajectory used for the generative model

n_dims                  = 3  # Point dimensionality

n_points                = sample_rate * sim_time
###################################


###################################
# GENERIC NN PARAMETERS
###################################
input_dim               = n_prefix_points * n_dims + n_dims  # Prefix trajectory plus the goal point

output_dim              = n_points * n_dims  # x = [xt,xt+1,..., xt+k]

train_loss_threshold    = -3000

train_epochs            = 1

train_learning_rate     = 1e-3

minibatch_size          = 64

activation              = torch.nn.functional.tanh

train_percentage        = 0.9

nn_layers               = 5

loss_f                  = loss_MSE

noise_sigma             = 0.001  # Sigma of the multivariate normal used to add noise to the ground truth position read from the simulator

load_existing_model = False

nn_model_path = "models/continous_table10K.pt"

dataset_path  = "datasets/continous_table10K.dat"

debug = False

viz_debug = True

###############
# GENERIC CODE
###############
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"

neNEmulator = CGenerativeModelNeuralEmulator(nn_model_path)  # Neural emulator for the likelihood function

# Check if there is a neural emulator model loaded. Create a new one otherwise.
if not neNEmulator.model or not load_existing_model:
    neNEmulator.model = CNeuralEmulatorNN(input_dim, output_dim, train_learning_rate, nn_layers, debug, device, activation, loss_f)
else:
    neNEmulator.model.move_to_device(device)

neNEmulator.model.activation = activation
neNEmulator.model.criterion = loss_f


if viz_debug:
    # Load simulator
    simulator_params = dict()
    simulator_params["robot_model_path"] = "models/human_torso/model.urdf"
    simulator_params["visualization"] = True
    simulator_params["timestep"] = 0.001
    simulator_params["episode_time"] = sim_time
    simulator_params["sample_rate"] = sample_rate
    simulator_params["sim_id"] = 0
    simulator_params["robot_controller"] = None
    simulator_objects = dict()
    simulator_objects["path"] = ["models/table/table.urdf"]
    simulator_objects["pose"] = [[0.6, 0, -0.65]]
    simulator_objects["static"] = [True]
    simulator_params["objects"] = simulator_objects
    neSimulator = CGenerativeModelSimulator(simulator_params)
    p.resetDebugVisualizerCamera(cameraDistance=1.3, cameraYaw=30, cameraPitch=210, cameraTargetPosition=[0.5, -0.3, -0.1],
                                 physicsClientId=neSimulator.sim_id)

# Check if the NeuralEmulator is trained. Train it otherwise.
print("=============================================")
print("LOAD AND TRAIN Neural Emulator")
print("Loading dataset...")
tic = time.time()
dataset = CReachingDataset(filename=dataset_path, output_sample_rate=sample_rate, dataset_sample_rate=sample_rate,
                           noise_sigma=noise_sigma)

train_dataset = CReachingDataset("")
test_dataset = CReachingDataset("")
for i in range(len(dataset)):
    if i < len(dataset)*train_percentage:
        train_dataset.samples.append(dataset[i])
    else:
        test_dataset.samples.append(dataset[i])
print("Loaded dataset with %d data points. took: %.3f" % (len(dataset), time.time() - tic))

# Show some random trajectories from the dataset
if viz_debug:
    for i in range(10):
        idx = int((torch.rand(1) * len(dataset)).item())
        dataset_traj = dataset.samples[idx][1]
        draw_trajectory(dataset_traj.view(-1,3), color=[1, 1, 0], width=2, physicsClientId=neSimulator.sim_id, draw_points=True)

train_time = time.time()
current_epoch = 0
current_loss = train_loss_threshold + 1
while current_loss > train_loss_threshold:
    train_time = time.time()
    neNEmulator.model.train(train_dataset, train_epochs, train_learning_rate, minibatch_size)
    train_time = time.time() - train_time
    current_epoch = current_epoch + 1
    current_loss, loss_terms = neNEmulator.model.test(test_dataset)
    current_loss_train, loss_terms_train = neNEmulator.model.test(train_dataset)
    print("Epoch: ", current_epoch, " train loss: ", current_loss_train, " test_loss: ", current_loss, " time:",train_time)
    # print('   loss terms: %3.5f\t%3.5f\t%3.5f\t%3.5f' % (loss_terms[0], loss_terms[1], loss_terms[2], loss_terms[3]))

    if current_epoch % 10 == 0:
        torch.save(neNEmulator.model, nn_model_path)
        torch.save(neNEmulator.model, nn_model_path+"epoch_%d" % current_epoch)

        if viz_debug:
            # Compute and show a trajectory from the test dataset
            sample_idx = np.random.random_integers(0, len(test_dataset)-1)
            z = test_dataset.samples[sample_idx][0]
            p.removeAllUserDebugItems()
            traj_gen = neNEmulator.generate(z.to(neNEmulator.model.device).view(1,-1))[0]
            traj_std = 1 / torch.sqrt(traj_gen[int(len(traj_gen)/2):])
            traj_gen = traj_gen[0:int(len(traj_gen)/2)].view(-1,3)

            traj_gt = test_dataset.samples[sample_idx][1].view(-1,3)
            draw_trajectory(traj_gen, color=[1, 0, 0], width=2, physicsClientId=neSimulator.sim_id, draw_points=True)
            draw_trajectory(traj_gt, color=[0, 1, 0], width=2, physicsClientId=neSimulator.sim_id, draw_points=True)
            draw_trajectory_diff(traj_gen, traj_gt, color=[0, 0, 1], width=2, physicsClientId=neSimulator.sim_id)

            # Compute and show a trajectory from the trains dataset
            sample_idx = np.random.random_integers(0, len(train_dataset)-1)
            z = train_dataset.samples[sample_idx][0]
            traj_gen = neNEmulator.generate(z.to(neNEmulator.model.device).view(1,-1))[0]
            traj_std = 1 / torch.sqrt(traj_gen[int(len(traj_gen)/2):])
            traj_gen = traj_gen[0:int(len(traj_gen)/2)].view(-1,3)

            traj_gt = train_dataset.samples[sample_idx][1].view(-1,3)
            draw_trajectory(traj_gen, color=[0.5, 0, 1], width=2, physicsClientId=neSimulator.sim_id, draw_points=True)
            draw_trajectory(traj_gt, color=[0, 0.5, 0], width=2, physicsClientId=neSimulator.sim_id, draw_points=True)
            draw_trajectory_diff(traj_gen, traj_gt, color=[1, 0, 1], width=2, physicsClientId=neSimulator.sim_id)

            # # Take a screenshot each 100 epochs
            # if current_epoch % 100 == 0:
            with mss() as sct:
                sct.shot(mon=2, output="training_video/frame_%d.png" % current_epoch)


print("emulator loss:", current_loss, " training time: ", time.time() - train_time)
print("=============================================")

print("=============================================")
print("Neural Emulator weights")
print("=============================================")
i = 0
for l in neNEmulator.model.layers:
    print("==== LAYER %d ===============================" % i)
    print(l.weight)
    i = i + 1
print("=============================================")
