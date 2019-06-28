from common import *
from scipy.spatial import distance
import torch.autograd as ag
from math import pi as PI


def loss_MLE(z, x, ne):
    """
    :param x: Batch of tensors with observed input (first dimension is batch number)
    :param z: Batch of tensors with latent parameters (first dimension is batch number)
    :param ne: Neural emulator that maps latent variables to observation space f(z) -> x and inverse covariance g(z) -> diag(sigma_1, sigma_2,... sigma_k)
    :return loss value (Autograd compatible, contains gradients embedded)
    """
    batch_size = len(x)

    k = len(x[0])           # Observation dimensions

    nn_out = ne(z)          # Run a forward pass of the network with the provided latent variables

    fz = nn_out[:, 0:k]     # Extract the vector of means from the nn output

    gz = nn_out[:, k:]      # Extract the vector of 1/stdev^2 from the nn output

    # gz = t_tensor([1]*k).to(ne.device)  # Use a vector of 1s for sanity checking the algorithm

    term1 = t_tensor([(k * 0.5) * math.log(2 * PI)]).to(ne.device)

    term2 = torch.sum(gz.log(), 1).to(ne.device)

    diff = (x - fz)
    term3 = torch.sum(diff * diff * gz, 1) / 2

    loss = term1 + term2 + term3

    return loss, [torch.mean(term1), torch.mean(term2), torch.mean(term3), torch.mean(torch.sqrt(torch.sum(diff*diff, 1)))]


def loss_MSE(z, x):
    """
    :param x: Batch of tensors with observed/ground-truth (first dimension is batch number)
    :param z: Batch of tensors with predicted values (first dimension is batch number)
    :return: loss value (Autograd compatible, contains gradients embedded)
    """
    k = len(x[0])   # Observation dimensions

    fz = z[:, 0:k]  # Extract the vector of means from the nn output

    diff = (x - fz)
    # term3 = diff.view(1, k) @ Sigma @ diff.view(k, 1)  # diff.view(k, 1) = diff.transpose() @ = matrix product

    term3 = torch.sum((diff*diff), 1)

    loss = term3

    return loss, [t_tensor([0]), t_tensor([0]), torch.mean(term3), torch.sqrt(torch.mean(term3))]


def neg_log_likelihood(z, x, ne):

    k = len(x)   # Observation dimensions

    l = ne.output_dim  # Emulator dimensions

    if k > l:
        k = l
        x = x[0:k]

    z = ag.Variable(z, requires_grad=True).double().to(ne.device)

    nn_out = ne(z.view(1, -1))  # Run a forward pass of the network with the provided latent variables

    fz = nn_out[0, 0:k]  # Extract the vector of means from the nn output for the number of observed samples

    gz = nn_out[0, l:l+k]  # Extract the vector of 1/stdev from the nn output for the number of observed samples

    term1 = t_tensor([(k * 0.5) * math.log(2 * PI)]).to(ne.device)

    term2 = -torch.sum(gz.log()).to(ne.device)

    diff = (x - fz)

    term3 = torch.sum((diff*diff*gz))

    loss = term1 + 0.5 * (term2 + term3)

    gradients = None

    if ne.is_differentiable:
        loss.backward()

        gradients = z.grad

    return loss, gradients


def log_likelihood_slacks(x, trajs, batch_size=1, slack=t_tensor([0.01])):
    k = len(x)   # Observation dimensions

    l = len(trajs[0])  # Emulator dimensions

    if k > l:
        k = l
        x = x[0:k]


    x = x.cpu().numpy()

    slack = slack.numpy().reshape(-1,1)

    term1 = np.array([-(k / 2) * math.log(2 * PI)])

    diff = distance.cdist(x.reshape((1,-1)), trajs[:,0:k], 'sqeuclidean')

    term2 = k * np.log(1/(slack*slack)) / 2

    term3 = - diff / (slack*slack) / 2

    loss = term1 + term2 + term3

    gradients = None

    return loss, gradients


def log_likelihood(z, x, ne, batch_size=1, slack=t_tensor([0.01])):

    k = len(x)   # Observation dimensions

    l = ne.output_dim  # Emulator dimensions

    if k > l:
        k = l
        x = x[0:k]

    z = ag.Variable(z, requires_grad=True).double().to(ne.device)

    nn_out = ne(z.view(batch_size, -1))  # Run a forward pass of the network with the provided latent variables

    fz = nn_out[:, 0:k]  # Extract the vector of means from the nn output for the number of observed samples

    gz = nn_out[:, l:l+k]  # Extract the vector of 1/stdev from the nn output for the number of observed samples

    slack = slack.to(ne.device)
    gz = torch.ones([batch_size, k]).double().to(ne.device) * (1/(slack*slack)) # Prepare the sigma with the slack term

    term1 = -(k / 2) * math.log(2 * PI)

    term1 = t_tensor([term1]).to(ne.device)

    term2 = torch.sum(gz.log()).to(ne.device) / 2

    diff = (x.unsqueeze(0).expand(batch_size, k) - fz)  # reshape x to match the batch size

    # term3 = -(diff.view(1, -1) @ torch.diag(gz) @ diff.view(-1, 1)) / 2
    term3 = -torch.sum(diff*diff*gz, 1) / 2

    loss = term1 + term2 + term3

    gradients = None

    ########################################
    # DEBUG CODE AREA!! Only for debug purposes. Remove after usage
    ########################################
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0,physicsClientId=1)
    # tmp_ids = []
    # tmp_ids.extend(draw_trajectory(traj=fz, color=[1, 0, 0], width=5, physicsClientId=1, show_points=True))
    # tmp_ids.extend(draw_trajectory(traj=x, color=[0, 0, 1], width=5, physicsClientId=1, show_points=True))
    # for i in range(0, len(fz), 3):
    #     a = [fz[i], fz[i+1], fz[i+2]]
    #     b = [x[i], x[i + 1], x[i + 2]]
    #     tmp_ids.append(draw_line(a, b, [0, 1, 1], 3, physicsClientId=1))
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=1)
    #
    # t_text = "%E" % torch.exp(loss)
    # p_text = [fz[0], fz[1], fz[2]-0.05]
    # tmp_ids.append(draw_text(t_text, p_text, 1))
    #
    # time.sleep(0.1)
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=1)
    # for id in tmp_ids:
    #     p.removeUserDebugItem(id, physicsClientId=1)
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=1)
    ########################################
    # END DEBUG CODE AREA!!
    ########################################

    if ne.is_differentiable:
        loss.mean().backward()

        gradients = z.grad

    return loss, gradients


def likelihood(z, x, ne, batch_size=1, slack=t_tensor([0.01])):
    """
    This function implements p(x|z)
    :param x: Tensor with observed input
    :param z: Tensor with latent parameters
    :param ne: Neural emulator that maps latent variables to observation space f(z) -> x and inverse covariance g(z) -> diag(sigma_1, sigma_2,... sigma_k)
    :return: likelihood value (Autograd compatible, contains gradients embedded)
    """

    loss, gradients = log_likelihood(z, x, ne, batch_size, slack)

    return torch.exp(loss), gradients


def closest_dist(traj1, traj2, dims=3, n_samples=30):
    res = t_tensor([0]*len(traj1))

    for i in range(0, len(traj1)-dims+1, dims):
        a = t_tensor([traj1[i], traj1[i+1], traj1[i+2]] * n_samples)
        c = a.view(n_samples, 3) - traj2.view(n_samples, 3)
        dists = (c * c).sum(1)
        dist_min = dists.min()
        for d in range(dims):
            res[i+d] = dist_min
    return res
