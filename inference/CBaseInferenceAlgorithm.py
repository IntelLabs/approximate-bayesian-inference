class CBaseInferenceAlgorithm(object):
    def __init__(self, params):
        self.params = params

    def inference(self, obs, gen_model, likelihood_f, proposal_distribution, slacks):
        # Generate candidates

        # Compute candidate likelihood

        # Perform update step

        # Draw debug

        # Compute result from samples / slacks
        raise NotImplementedError
