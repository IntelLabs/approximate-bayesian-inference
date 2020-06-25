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
from neural_emulators.CBayesianNeuralEmulatorNN import CBayesianNeuralEmulatorNN
##############################


#######################################
# GENERIC IMPORTS (no need to edit)
#######################################
from reaching_intent.generative_models.CReachingDataset import CReachingDataset
from neural_emulators.CGenerativeModelBayesianNeuralEmulator import CGenerativeModelBayesianNeuralEmulator
from reaching_intent.generative_models.CGenerativeModelSimulator import CGenerativeModelSimulator
from reaching_intent.generative_models.CGenerativeModelSimulator import create_sim_params
from neural_emulators.loss_functions import loss_MSE
from neural_emulators.loss_functions import neg_log_likelihood
##############################


#######################################
# DEBUG DRAWING IMPORTS
#######################################
from utils.draw import draw_trajectory
from utils.draw import draw_trajectory_diff
from utils.draw import draw_point
from utils.draw import draw_text
# import mss  # For screenshots


###################################
# APPLICATION SPECIFIC PARAMETERS (add/remove/edit parameters to fit your implementation)
###################################
sample_rate = 30    # Sensor sampling rate
sim_time = 5        # Prediction time window in seconds
n_dims = 3          # Point dimensionality
n_points = sample_rate * sim_time
###################################


###################################
# GENERIC NN PARAMETERS
###################################
input_dim               = n_dims + n_dims + 5  # Start point, End point, controller params

# Select the parameters that are considered nuisance and the parameters that are considered interesting
latent_mask = t_tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]) == 1  # We are interested in the end position
nuisance_mask = latent_mask == 0  # The rest of the params are considered nuisance

output_dim              = n_points * n_dims  # x = [xt,xt+1,..., xt+k]

train_loss_threshold    = -3000

train_epochs            = 1

train_learning_rate     = 1e-4

minibatch_size          = 16

activation              = torch.tanh

train_percentage        = 0.9

nn_layers               = 4

loss_f                  = loss_MSE

noise_sigma             = 0.001  # Sigma of the multivariate normal used to add noise to the ground truth position read from the simulator

load_existing_model = False

nn_model_path = "pytorch_models/ne_bfc4_10k_MSE_in%d_out%d.pt" % (input_dim, output_dim)

dataset_path = "datasets/default.dat"  # Small dataset for testing the approach

debug = False

viz_debug = True

###############
# GENERIC CODE
###############
device = "cuda:0" if torch.cuda.is_available() else "cpu"

neNEmulator = CGenerativeModelBayesianNeuralEmulator(nn_model_path)  # Neural emulator for the likelihood function

# Check if there is a neural emulator model loaded. Create a new one otherwise.
if not neNEmulator.model or not load_existing_model:
    neNEmulator.model = CBayesianNeuralEmulatorNN(torch.sum(latent_mask), output_dim, nn_layers, debug, device, activation, loss_f)
    neNEmulator.output_dims = output_dim
    neNEmulator.input_dims = int(torch.sum(latent_mask).numpy())
else:
    neNEmulator.model.move_to_device(device)

neNEmulator.model.latent_mask = latent_mask
neNEmulator.model.activation = activation
neNEmulator.model.criterion = loss_f
neNEmulator.model.debug = False


if viz_debug:
    # Load simulator for visualization purposes only
    simulator_params = create_sim_params(sim_time=sim_time, sample_rate=sample_rate)
    neSimulator = CGenerativeModelSimulator(simulator_params)

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
# if viz_debug:
#     for i in range(10):
#         idx = int((torch.rand(1) * len(dataset)).item())
#         dataset_traj = dataset.samples[idx][1]
#         draw_trajectory(dataset_traj.view(-1,3), color=[1, 1, 0], width=2, physicsClientId=neSimulator.sim_id, draw_points=True)

train_time = time.time()
train_ini_time = time.time()
current_epoch = 0
current_loss = train_loss_threshold + 1
while current_loss > train_loss_threshold:
    train_time = time.time()
    neNEmulator.model.train(train_dataset, train_epochs, train_learning_rate, minibatch_size)
    train_time = time.time() - train_time
    current_epoch = current_epoch + 1
    current_loss, loss_terms = neNEmulator.model.test(test_dataset)
    current_loss_train, loss_terms_train = neNEmulator.model.test(train_dataset)
    print("Epoch: %d  train loss: %3.3f  test_loss: %3.3f  epoch_time: %3.3f total_time: %s" %
          (current_epoch, current_loss_train, current_loss, train_time, str(datetime.timedelta(seconds=(time.time()-train_ini_time)))))
    # print('   loss terms: %3.5f\t%3.5f\t%3.5f\t%3.5f' % (loss_terms[0], loss_terms[1], loss_terms[2], loss_terms[3]))
    str_status = "%d %3.3f %3.3f %3.3f %3.3f \n" % (current_epoch, current_loss_train, current_loss, train_time, time.time()-train_ini_time)
    with open(nn_model_path + ".train_report.txt", "a+") as fp:
        fp.write(str_status)

    if current_epoch % 100 == 0:
        torch.save(neNEmulator.model, nn_model_path)
        torch.save(neNEmulator.model, nn_model_path+"epoch_%d" % current_epoch)

        if viz_debug:
            # Compute and show a trajectory from the test dataset
            sample_idx = np.random.random_integers(0, len(test_dataset)-1)
            z = test_dataset.samples[sample_idx][0][latent_mask]
            p.removeAllUserDebugItems()
            traj_gen = neNEmulator.generate(z.to(neNEmulator.model.device).view(1,-1), n=None)[0]
            traj_gt = test_dataset.samples[sample_idx][1].view(-1,3)
            draw_trajectory(traj_gen.view(-1, 3), color=[1, 0, 0], width=2, physicsClientId=neSimulator.sim_id, draw_points=True)
            draw_trajectory(traj_gt, color=[0, 1, 0], width=2, physicsClientId=neSimulator.sim_id, draw_points=True)
            draw_trajectory_diff(traj_gen.view(-1, 3), traj_gt, color=[0, 0, 1], width=1, physicsClientId=neSimulator.sim_id)
            draw_point(z, [0, 0, 1], size=0.05, width=5, physicsClientId=neSimulator.sim_id)
            loss, loss_terms = neNEmulator.model.criterion(traj_gen.view(-1, 3), traj_gt)
            draw_text("Test Gen: loss: %5.3f" % torch.sqrt(loss).mean().item(), traj_gen.view(-1, 3)[-1], neSimulator.sim_id, [1, 0, 0])
            draw_text("Test GT", traj_gt[-1], neSimulator.sim_id, [0, 1, 0])

            # Compute and show a trajectory from the train dataset
            sample_idx = np.random.random_integers(0, len(train_dataset)-1)
            z = train_dataset.samples[sample_idx][0][latent_mask]
            traj_gen = neNEmulator.generate(z.to(neNEmulator.model.device).view(1,-1), n=None)[0]

            traj_gt = train_dataset.samples[sample_idx][1].view(-1,3)
            draw_trajectory(traj_gen.view(-1, 3), color=[0.5, 0, 1], width=2, physicsClientId=neSimulator.sim_id, draw_points=True)
            draw_trajectory(traj_gt, color=[0, 0.5, 0], width=2, physicsClientId=neSimulator.sim_id, draw_points=True)
            draw_trajectory_diff(traj_gen.view(-1, 3), traj_gt, color=[1, 0, 1], width=1, physicsClientId=neSimulator.sim_id)
            loss, loss_terms = neNEmulator.model.criterion(traj_gen.view(-1, 3), traj_gt)
            draw_text("Train Gen: loss: %5.3f" % torch.sqrt(loss).mean().item(), traj_gen.view(-1, 3)[-1], neSimulator.sim_id, [0.5, 0, 0])
            draw_text("Train GT", traj_gt[-1], neSimulator.sim_id, [0, 0.5, 0])

            # # Take a screenshot each 100 epochs
            # if current_epoch % 100 == 0:
            # with mss() as sct:
            #     sct.shot(mon=2, output="training_video/frame_%d.png" % current_epoch)


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
