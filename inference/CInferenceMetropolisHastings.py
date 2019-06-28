import time
import numpy.random as rd

from common import *
from inference import CBaseInferenceAlgorithm
from utils.draw import draw_point


class CInferenceMetropolisHastings(CBaseInferenceAlgorithm):
    def __init__(self, params=None):
        super(CInferenceMetropolisHastings, self).__init__(params)
        self.proposal_dist = None

    def inference(self, obs, nsamples, prior_sampler, device, visualizer=None, proposal_sampler=None,
                  slacks=t_tensor([1E-4]), img = None, camera = None, burn_in_samples=100):
        """ Document this algorithm properly with docstrings making the associations between the variables
            and the mathematical notation in here: https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
        """
        assert self.likelihood_f is not None
        assert self.generative_model is not None

        params = prior_sampler.sample(None)
        params_ml = params
        max_likelihood = 0
        n_evals = 0
        samples = t_tensor([]).to(device)
        last_likelihood = - float_epsilon
        timeout = 5.0
        time_ini = time.time()
        while len(samples) < nsamples and timeout > (time.time() - time_ini):
            n_evals = n_evals + 1
            params_hat = proposal_sampler.sample(params)       # Sample from the proposal distribution a new value for the parameters
            gen_obs = self.generative_model.model(params_hat.view(1,-1)).cpu().detach().numpy()  # TODO: Check how to remove the inline casts
            likelihood, grad = self.likelihood_f(obs, gen_obs, len(gen_obs), slack=slacks)

            alpha = likelihood - last_likelihood                    # Compute acceptance likelihood (subtract loglikelihooods)
            # print("MCMC accepted samples:", len(samples))
            u = rd.uniform(0, 1)                                    # Sample the acceptance likelihood
            if alpha > np.log(u):                                           # If sample is accepted run the MH update step
                params = params_hat
                last_likelihood = likelihood
                samples = torch.cat((samples, params.view(1, self.particle_dims)))

                if likelihood > max_likelihood:
                    params_ml = params_hat
                    max_likelihood = likelihood

                if visualizer is not None:
                    if len(samples) > burn_in_samples:
                        draw_point(params_hat[3:6], color=[0, 0, 1], size=0.03, width=1, physicsClientId=visualizer)
                    else:
                        draw_point(params_hat[3:6], color=[1, 0, 0], size=0.01, width=1, physicsClientId=visualizer)

        if len(samples) <= burn_in_samples:
            return torch.mean(samples,0), slacks[0], n_evals
        else:
            return torch.mean(samples[burn_in_samples:], 0), slacks[0], n_evals
