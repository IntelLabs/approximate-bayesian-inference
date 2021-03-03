import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from reaching_intent.generative_models.CGenerativeModelSimulator import scene_with_table

from common.common import *


class CNeuralEmulatorNN(nn.Module):
    def __init__(self, input_dim, output_dim, nlayers=4, nlayers_dims=[],
                 debug=False, device="cpu", activation=F.relu, criterion=F.mse_loss):
        super(CNeuralEmulatorNN, self).__init__()
        self.is_differentiable = True

        self.device = device
        device = torch.device(self.device)

        self.activation = activation
        self.criterion = criterion
        self.arch = ""
        self.dropout = nn.Dropout(p=0.2)

        self.layers = [None] * nlayers

        if len(nlayers_dims) > 0:
            assert len(nlayers_dims) == nlayers + 1
        else:
            nlayers_dims = list()
            nlayers_dims.append(input_dim)
            for i in range(1, nlayers):
                nlayers_dims.append(output_dim)

        for i in range(0, nlayers):
            in_dim = nlayers_dims[i]
            out_dim = nlayers_dims[i+1]
            self.layers[i] = nn.Linear(nlayers_dims[i], nlayers_dims[i+1]).to(device)  # Outputs mu and sigma. Therefore the ouput dims are * 2
            self.arch = self.arch + "fc%d_%d-" % (in_dim, out_dim)

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
        self.latent_mask = t_tensor(torch.ones(int(input_dim)))

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

        return x

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
                inputs = inputs.to(device)[:, self.latent_mask]
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
                inputs = inputs.to(device)[:, self.latent_mask]
                ground_truth = ground_truth.to(device)

                # zero the parameter gradientsK
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs, nn.Dropout(p=0.2))  # Use dropout when training to avoid overfitting
                # outputs = self.forward(inputs, None)
                loss, loss_terms = self.criterion(outputs, ground_truth)
                # https://discuss.pytorch.org/t/loss-backward-raises-error-grad-can-be-implicitly-created-only-for-scalar-outputs/12152
                loss.mean().backward()
                optimizer.step()

                # print statistics
                if self.debug:
                    print('=============================')
                    print('[%2d, %6d] loss: %.5f time: %.2f device: %s' % (epoch + 1, i + 1, loss.mean(), time.time()-time_ini, self.device))
                    print('           terms: %3.5f\t%3.5f\t%3.5f\t%3.5f' % (loss_terms[0].mean().item(), loss_terms[1].mean().item(), loss_terms[2].mean().item(), loss_terms[3].mean().item()) )
                    for j, param in enumerate(self.parameters()):
                        if len(param.shape) == 2:
                            print('Layer Weights %d [%d x %d]  grads: ' % (j, param.shape[0], param.shape[1]), param.grad)
                        # if len(param.shape) == 1:
                        #     print('Layer Bias %d [%d]  grads: ' % (j, param.shape[0]), param.grad)
                    print('=============================')

            # print('[%2d, %6d] loss: %.5f time: %.2f device: %s' % (epoch + 1, i + 1, running_loss / 1000, time.time() - time_ini, self.device))
            # print('           terms: ', running_loss_terms[0] / 1000, running_loss_terms[1] / 1000, running_loss_terms[2] / 1000, running_loss_terms[3] / 1000)




