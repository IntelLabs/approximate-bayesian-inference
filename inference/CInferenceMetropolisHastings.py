import time
import numpy.random as rd

from common.common import *
from common.CBaseInferenceAlgorithm import CBaseInferenceAlgorithm
from utils.draw import draw_point
from utils.draw import draw_trajectory


class CInferenceMetropolisHastings(CBaseInferenceAlgorithm):
    def __init__(self):
        super(CInferenceMetropolisHastings, self).__init__()

    def inference(self, obs, nuisance, proposal, gen_model, likelihood_f, params, device=torch.device("cpu"), visualizer=None,
                  slacks=t_tensor([1E-4]), img=None, camera=None):
        """ Document this algorithm properly with docstrings making the associations between the variables
            and the mathematical notation in here: https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
        """
        assert likelihood_f is not None
        assert gen_model is not None

        nsamples = params["nsamples"]
        burn_in_samples = params["burn_in"]
        proposal_sampler = params["proposal_dist"]

        z = proposal
        n_evals = 0
        samples = t_tensor([]).to(device)
        last_likelihood = - float_epsilon
        timeout = 5.0
        time_ini = time.time()
        time_samples = np.array([])
        while len(samples) < nsamples and timeout > (time.time() - time_ini):
            stime_ini = time.time()
            n_evals = n_evals + 1
            z_hat = z + proposal_sampler.sample(nsamples=1, params=None)   # Sample from the proposal distribution a new value for the parameters
            gen_obs = gen_model.generate(z_hat, nuisance).detach()
            likelihood, grad = likelihood_f(obs, gen_obs, len(gen_obs), slack=slacks)

            alpha = likelihood - last_likelihood    # Compute acceptance likelihood (subtract loglikelihooods)
            u = rd.uniform(0, 1)                    # Sample the acceptance likelihood
            if alpha > np.log(u):                   # If sample is accepted run the MH update step
                z = z_hat
                last_likelihood = likelihood
                samples = torch.cat((samples, z.view(1, -1)))
                draw_point(z_hat.view(-1), [0, 1, 0], 0.02, physicsClientId=visualizer)
                # draw_trajectory(gen_obs.view(-1, 3), [0, 1, 0], physicsClientId=visualizer, draw_points=False)
            # else:
            #     draw_point(z_hat.view(-1), [1, 0, 0], 0.01, physicsClientId=visualizer)
            # print("MCMC accepted samples:", len(samples))

            time_samples = np.hstack((time_samples, np.array([time.time()-stime_ini])))

        print("MCMC stats: Total samples: %d || Accepted: %d || Accept Ratio: %3.2f || Avg. sample time: %3.3fs" %
              (n_evals, len(samples), (len(samples)/float(n_evals))*100.0, np.mean(time_samples)))
        if len(samples) <= burn_in_samples:
            return samples, slacks[0], n_evals
        else:
            return samples[burn_in_samples:], slacks[0], n_evals
