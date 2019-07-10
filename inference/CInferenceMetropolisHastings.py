import time
import numpy.random as rd

from common.common import *
from common.CBaseInferenceAlgorithm import CBaseInferenceAlgorithm
from utils.draw import draw_point
from utils.draw import draw_trajectory


class CInferenceMetropolisHastings(CBaseInferenceAlgorithm):
    def __init__(self):
        super(CInferenceMetropolisHastings, self).__init__()

    def inference(self, obs, nuisance, proposal, gen_model, likelihood_f, slacks, params):
        """
        Document this algorithm properly with docstrings making the associations between the variables
        and the mathematical notation in here: https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm

        Method that performs approximate bayesian computations to provide the posterior distribution over
        the latent space values 'z' given the current observation 'obs'

        :param obs: Input observation.
        :param nuisance: Nuisance parameters of the generative model.
        :param gen_model: Generative model g(z,n) -> \hat{o}
        :param likelihood_f: Likelihood function used to evaluate a proposal p(g(z,n)|o)
        :param proposal: Initial proposal sample for the latent parameters. Usually sampled from the prior: \hat{z}~p(z)
        :param slacks: Slack values to evaluate.
        :param params: Dictionary with custom parameter values to tune the inference.
        :return: - samples: tensor with the resulting samples from the inference process (describe the posterior)
                 - likelihoods: tensor with likelihood values corresponding to each sample (describe the posterior)
                 - slacks: tensor with slack values corresponding to each likelihood
                 - stats: dictionary with statistics (for statistics and debug):
                    - nevals: number of likelihood function p(\hat{o}|o) evaluations
                    - nsamples: number of sampling \hat{z} ~ p(z|z_t-1) operations
                    - ngens: number of generative \hat{z} ~ p(z|z_t-1) operations
                    - tevals: time taken by likelihood function p(\hat{o}|o) evaluations
                    - tsamples: time taken by sampling \hat{z} ~ p(z|z_t-1) operations
                    - tgens: time taken by generative \hat{z} ~ p(z|z_t-1) operations
        """

        assert likelihood_f is not None
        assert gen_model is not None

        nsamples = params["nsamples"]
        burn_in_samples = params["burn_in"]
        proposal_sampler = params["proposal_dist"]
        timeout = params["timeout"]
        visualizer = params["visualizer"]
        z_min = params["z_min"]
        z_max = params["z_max"]

        stats = dict()

        z = proposal
        n_evals = 0
        samples = t_tensor([])
        likelihoods = np.array([])
        last_likelihood = - float_epsilon

        stats["tsamples"] = 0
        stats["tgens"] = 0
        stats["tevals"] = 0
        time_ini = time.time()
        while len(samples) < nsamples and timeout > (time.time() - time_ini):
            n_evals = n_evals + 1
            # Sample from the proposal distribution a new value for the parameters
            tic = time.time()
            z_hat = z + proposal_sampler.sample(nsamples=1, params=None)
            # Limit the sample
            z_hat = torch.max(z_hat, z_min)
            z_hat = torch.min(z_hat, z_max)
            stats["tsamples"] += time.time() - tic

            # Generate an observation for the proposal latent values
            tic = time.time()
            gen_obs = gen_model.generate(z_hat, nuisance).detach()
            stats["tgens"] += time.time() - tic

            # Compute the likelihood of the proposal (for all slacks)
            tic = time.time()
            likelihood, grad = likelihood_f(obs, gen_obs, len(gen_obs), slack=slacks)
            stats["tevals"] += time.time() - tic

            # Select the best likelihood among the slacks and use it as the last likelihood
            likelihood_best = np.max(likelihood)

            # Compute acceptance likelihood (subtract loglikelihooods)
            alpha = likelihood_best - last_likelihood

            # Sample the acceptance likelihood
            tic = time.time()
            u = rd.uniform(0, 1)
            stats["tsamples"] += time.time() - tic

            # If sample is accepted run the MH update step
            if alpha > np.log(u):
                z = z_hat
                last_likelihood = likelihood_best
                samples = torch.cat((samples, z.view(1, -1)))
                likelihoods = np.vstack((likelihoods, likelihood.reshape(1, -1))) if likelihoods.size else likelihood.reshape(1, -1)
                draw_point(z_hat.view(-1), [0, 1, 0], 0.02, physicsClientId=visualizer)
                draw_trajectory(gen_obs.view(-1, 3), [0, 1, 0], physicsClientId=visualizer, draw_points=False)
            # else:
            #     draw_point(z_hat.view(-1), [1, 0, 0], 0.01, physicsClientId=visualizer)
            # print("MCMC accepted samples:", len(samples))

        # print("MCMC stats: Total samples: %d || Accepted: %d || Accept Ratio: %3.2f%% || Avg. sample time: %3.3fs" %
        #       (n_evals, len(samples), (len(samples)/float(n_evals))*100.0, stats["tsamples"] / stats["nsamples"]))

        stats["nsamples"] = len(samples)
        stats["nevals"] = n_evals * len(slacks)
        stats["ngens"] = n_evals

        if len(samples) <= burn_in_samples:
            return samples, torch.from_numpy(likelihoods), stats
        else:
            return samples[burn_in_samples:], torch.from_numpy(likelihoods[burn_in_samples:, :]), stats
