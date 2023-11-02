#!/usr/bin/python3

#################
# SYSTEM IMPORTS
#################
from common.common import *
import torch.nn
import time
import datetime
import pybullet as p
import numpy as np

##############################
# APPLICATION SPECIFIC IMPORTS (import from your application specific module)
##############################
from neural_emulators.CNeuralEmulatorNN import CNeuralEmulatorNN
##############################


#######################################
# GENERIC IMPORTS (no need to edit)
#######################################
from reaching_intent.generative_models.CReachingDataset import CReachingDataset
from neural_emulators.CGenerativeModelNeuralEmulator import CGenerativeModelNeuralEmulator
from reaching_intent.generative_models.CGenerativeModelSimulator import CGenerativeModelSimulator
from reaching_intent.generative_models.CGenerativeModelSimulator import create_sim_params
from neural_emulators.loss_functions import loss_MSE
##############################


#######################################
# DEBUG DRAWING IMPORTS
#######################################
from utils.draw import draw_trajectory
from utils.draw import draw_trajectory_diff
from utils.draw import draw_point
from utils.draw import draw_text
from reaching_intent.generative_models.CGenerativeModelSimulator import scene_with_table
from reaching_intent.generative_models.CGenerativeModelSimulator import scene_with_ur5table

# import mss  # For screenshots


###################################
# APPLICATION SPECIFIC PARAMETERS (add/remove/edit parameters to fit your implementation)
###################################
sample_rate = 30    # Sensor sampling rate
sim_time = 3.2      # Prediction time window in seconds
n_dims = 3          # Point dimensionality
n_points = int(sample_rate * sim_time)  # Number of points in sequence that compose a trajectory
debug = False       # Print verbose debug to stdout
viz_debug = True   # Show 3D visualization debug of the training process
###################################

###################################
# GENERIC NEURAL SURROGATE ARCHITECTURE PARAMS
###################################
# Neural surrogate input dimensions: Start point, End point, controller params
input_dim = n_dims + n_dims + 5

# Select the parameters that are considered nuisance and the parameters that are considered interesting
latent_mask = t_tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]) == 1  # We are interested in the end position
nuisance_mask = latent_mask == 0  # The rest of the params are considered nuisance

# Dimensionality of the output. In this case it depends on the number of points that describe a trajectory (n_points)
# and the dimensionality of each trajectory point (n_dims)
output_dim = n_points * n_dims

# Neural Surrogate architecture description
activation = torch.relu
# nn_layers = 4
# nn_layer_dims = [torch.count_nonzero(latent_mask), 32, 64, 64, output_dim]
nn_layers = 3
nn_layer_dims = [torch.count_nonzero(latent_mask), 32, 64, output_dim]
loss_f = loss_MSE

# Model filename
nn_model_path = "pytorch_models/ne_fc%d_10k2D_MSE_in%d_out%d.pt" % (nn_layers, input_dim, output_dim)


###################################
# GENERIC TRAINING PARAMETERS
###################################
# True = Load an existing model (if exists) and resume training.
load_existing_model = False

# Dataset to use
dataset_path = "datasets/dataset1K_2D_ur5_96p.dat"
# Portion (0. - 1.) of the dataset used for training. The remainder will be used for testing.
train_percentage = 0.9

# Training stops once the loss is below the threshold or the max_train_epochs are reached
train_loss_threshold = 0.0001
max_train_epochs = 300

# controls the number of times that the training dataset is gone over on each train call
train_epochs = 1

train_learning_rate = 1e-4

minibatch_size = 16

# Sigma of the multivariate normal used to add noise to the ground truth position read from the simulator
noise_sigma = 0.0001


###############
# GENERIC CODE
###############
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

neNEmulator = CGenerativeModelNeuralEmulator(nn_model_path, device)  # Neural emulator for the likelihood function

# Check if there is a neural emulator model loaded. Create a new one otherwise.
if neNEmulator.model is None or not load_existing_model:
    neNEmulator.model = CNeuralEmulatorNN(torch.sum(latent_mask), output_dim, nn_layers, nn_layer_dims,
                                          debug, device, activation, loss_f)
else:
    neNEmulator.model.move_to_device(device)

neNEmulator.model.latent_mask = latent_mask
neNEmulator.model.activation = activation
neNEmulator.model.criterion = loss_f
neNEmulator.model.debug = False
neNEmulator.model.output_dim = output_dim
neNEmulator.model.input_dim = input_dim
neNEmulator.output_dims = output_dim
neNEmulator.input_dims = input_dim


if viz_debug:
    # Load simulator for visualization purposes only
    simulator_params = create_sim_params(sim_time=sim_time, sample_rate=sample_rate)
    # scene_with_table(simulator_params)
    scene_with_ur5table(simulator_params)
    neSimulator = CGenerativeModelSimulator(simulator_params)

# Check if the NeuralEmulator is trained. Train it otherwise.
print("=============================================")
print("LOAD AND TRAIN Neural Emulator")
print("Loading dataset...")
tic = time.time()
dataset = CReachingDataset(filename=dataset_path, output_sample_rate=sample_rate, dataset_sample_rate=sample_rate,
                           noise_sigma=noise_sigma, n_datapoints=-1, traj_duration=sim_time)

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
    for i in range(5):
        idx = int((torch.rand(1) * len(dataset)).item())
        dataset_traj = dataset.samples[idx][1]
        draw_trajectory(dataset_traj.view(-1, 3), color=[1, 1, 0], width=2, draw_points=True,
                        physicsClientId=neSimulator.sim_id)

train_time = time.time()
train_ini_time = time.time()
current_epoch = 0
current_loss = train_loss_threshold + 1
while current_loss > train_loss_threshold and max_train_epochs > current_epoch:
    train_time = time.time()
    neNEmulator.model.train(train_dataset, train_epochs, train_learning_rate, minibatch_size)
    train_time = time.time() - train_time
    current_epoch = current_epoch + train_epochs
    current_loss, loss_terms = neNEmulator.model.test(test_dataset)
    current_loss_train, loss_terms_train = neNEmulator.model.test(train_dataset)
    print("Epoch: %d  train loss: %7.5f  test_loss: %7.5f  epoch_time: %3.3f total_time: %s" %
          (current_epoch, current_loss_train, current_loss, train_time, str(datetime.timedelta(seconds=(time.time()-train_ini_time)))))
    # print('   loss terms: %3.5f\t%3.5f\t%3.5f\t%3.5f' % (loss_terms[0], loss_terms[1], loss_terms[2], loss_terms[3]))
    str_status = "%d %7.5f %7.5f %3.3f %3.3f \n" % (current_epoch, current_loss_train, current_loss, train_time, time.time()-train_ini_time)
    with open(nn_model_path + ".train_report.txt", "a+") as fp:
        fp.write(str_status)

    if current_epoch % 100 == 0:
        torch.save(neNEmulator.model, nn_model_path)
        torch.save(neNEmulator.model, nn_model_path+"epoch_%d" % current_epoch)

        if viz_debug:
            # Compute and show a trajectory from the test dataset
            sample_idx = np.random.randint(0, len(test_dataset)-1)
            z = test_dataset.samples[sample_idx][0][latent_mask]
            p.removeAllUserDebugItems()
            traj_gen = neNEmulator.generate(z.to(neNEmulator.model.device).view(1, -1), n=None)[0]
            traj_gt = test_dataset.samples[sample_idx][1].view(-1, 3).to(device)
            draw_trajectory(traj_gen.view(-1, 3), color=[1, 0, 0], width=2, physicsClientId=neSimulator.sim_id, draw_points=True)
            draw_trajectory(traj_gt, color=[0, 1, 0], width=2, physicsClientId=neSimulator.sim_id, draw_points=True)
            draw_trajectory_diff(traj_gen.view(-1, 3), traj_gt, color=[0, 0, 1], width=1, physicsClientId=neSimulator.sim_id)
            draw_point(z, [0, 0, 1], size=0.05, width=5, physicsClientId=neSimulator.sim_id)
            loss, loss_terms = neNEmulator.model.criterion(traj_gen.view(-1, 3), traj_gt)
            draw_text("Test Gen: loss: %7.5f" % torch.sqrt(loss).mean().item(), traj_gen.view(-1, 3)[-1], neSimulator.sim_id, [1, 0, 0])
            draw_text("Test GT", traj_gt[-1], neSimulator.sim_id, [0, 1, 0])

            # Compute and show a trajectory from the train dataset
            sample_idx = np.random.randint(0, len(train_dataset)-1)
            z = train_dataset.samples[sample_idx][0][latent_mask]
            traj_gen = neNEmulator.generate(z.to(neNEmulator.model.device).view(1, -1), n=None)[0]

            traj_gt = train_dataset.samples[sample_idx][1].view(-1, 3).to(device)
            draw_trajectory(traj_gen.view(-1, 3), color=[0.5, 0, 1], width=2, physicsClientId=neSimulator.sim_id, draw_points=True)
            draw_trajectory(traj_gt, color=[0, 0.5, 0], width=2, physicsClientId=neSimulator.sim_id, draw_points=True)
            draw_trajectory_diff(traj_gen.view(-1, 3), traj_gt, color=[1, 0, 1], width=1, physicsClientId=neSimulator.sim_id)
            loss, loss_terms = neNEmulator.model.criterion(traj_gen.view(-1, 3), traj_gt)
            draw_text("Train Gen: loss: %7.5f" % torch.sqrt(loss).mean().item(), traj_gen.view(-1, 3)[-1], neSimulator.sim_id, [0.5, 0, 0])
            draw_text("Train GT", traj_gt[-1], neSimulator.sim_id, [0, 0.5, 0])

            # # Take a screenshot each 100 epochs
            # if current_epoch % 100 == 0:
            # with mss() as sct:
            #     sct.shot(mon=2, output="training_video/frame_%d.png" % current_epoch)


print("emulator loss:", current_loss, " training time: ", (time.time() - train_time))
print("=============================================")
torch.save(neNEmulator.model, nn_model_path)

print("=============================================")
print("Neural Emulator weights")
print("=============================================")
i = 0
for l in neNEmulator.model.layers:
    print("==== LAYER %d ===============================" % i)
    print(l.weight)
    i = i + 1
print("=============================================")
