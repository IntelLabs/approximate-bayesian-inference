import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from common.common import *
from torch_vi.layers.variational_layers.linear_variational import LinearReparameterization


class CBayesianNeuralEmulatorNN(nn.Module):
    def __init__(self, input_dim, output_dim, nlayers=4, debug=False, device="cpu", activation=F.relu, criterion=F.mse_loss):
        super(CBayesianNeuralEmulatorNN, self).__init__()
        self.is_differentiable = True

        self.device = device
        device = torch.device(self.device)

        self.activation = activation
        self.criterion = criterion
        self.arch = ""

        self.layers = [None] * nlayers

        self.layers[0] = LinearReparameterization(0, 1, 0, -3, input_dim, output_dim).to(device)
        self.arch = self.arch + "bfc{%d_%d}-" % (input_dim, output_dim)
        for i in range(1, nlayers):
            self.layers[i] = LinearReparameterization(0, 1, 0, -3, output_dim, output_dim).to(device)
            self.arch = self.arch + "bfc{%d_%d}-" % (output_dim, output_dim)

        self.output_dim = output_dim
        self.input_dim = input_dim
        print("=============================================")
        print("Neural Emulator weights")
        print("=============================================")
        i = 0
        for l in self.layers:
            print("==== LAYER %d MU ===============================" % i)
            print(l.mu_weight)
            print("==== LAYER %d RHO ===============================" % i)
            print(l.rho_weight)
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
        kl_sum = 0
        for i in range(len(self.layers)-1):
            x, kl = self.layers[i](x)
            kl_sum += kl
            x = self.activation(x)

        # Do not use activation in the last layer.
        x, kl = self.layers[-1](x)
        kl_sum += kl

        return x, kl_sum

    # @staticmethod
    # def num_flat_features(x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features

    def test(self, dataset):
        device = torch.device(self.device)

        testloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        with torch.no_grad():
            for i, (inputs, ground_truth) in enumerate(testloader, 0):
                inputs = inputs.to(device)[:, self.latent_mask]
                ground_truth = ground_truth.to(device)

                # forward passes and loss computation
                outputs, kl = self.forward(inputs)
                scaled_kl = kl / len(inputs)
                loss, loss_terms = self.criterion(outputs, ground_truth)

        return loss.mean().item() + scaled_kl, [loss_terms[0].mean().item(), loss_terms[1].mean().item(), loss_terms[2].mean().item(), loss_terms[3].mean().item(), scaled_kl]

    def train(self, dataset, epochs, learning_rate=0.01, minibatch_size=1024):

        device = torch.device(self.device)

        # create your optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):  # loop over the dataset multiple times
            time_ini = time.time()

            trainloader = torch.utils.data.DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

            for i, (inputs, ground_truth) in enumerate(trainloader, 0):
                inputs = inputs.to(device)[:, self.latent_mask]
                ground_truth = ground_truth.to(device)

                # zero the parameter gradientsK
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs, kl = self.forward(inputs)  # Use dropout when training to avoid overfitting
                scaled_kl = kl / minibatch_size
                loss, loss_terms = self.criterion(outputs, ground_truth)

                final_loss = loss.mean() + scaled_kl
                # https://discuss.pytorch.org/t/loss-backward-raises-error-grad-can-be-implicitly-created-only-for-scalar-outputs/12152
                final_loss.backward()
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
