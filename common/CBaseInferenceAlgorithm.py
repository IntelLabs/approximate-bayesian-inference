
class CBaseInferenceAlgorithm(object):
    def __init__(self):
        pass

    @staticmethod
    def get_name():
        raise NotImplementedError

    def inference(self, obs, nuisance, proposal, gen_model, likelihood_f, slacks, params):
        """
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
                 - likelihoods: NxM tensor with likelihood values corresponding to each sample M with slack N
                 - stats: dictionary with statistics (for statistics and debug):
                    - nevals: number of likelihood function p(\hat{o}|o) evaluations
                    - nsamples: number of sampling \hat{z} ~ p(z|z_t-1) operations
                    - ngens: number of generative \hat{z} ~ p(z|z_t-1) operations
                    - tevals: time taken by likelihood function p(\hat{o}|o) evaluations
                    - tsamples: time taken by sampling \hat{z} ~ p(z|z_t-1) operations
                    - tgens: time taken by generative \hat{z} ~ p(z|z_t-1) operations
        """
        # Generate candidates

        # Compute candidate likelihood

        # Perform update step

        # Draw debug

        # Compute result from samples / slacks

        # Return: samples, likelihoods, stats
        raise NotImplementedError
