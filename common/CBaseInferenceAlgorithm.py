
class CBaseInferenceAlgorithm(object):
    def __init__(self):
        pass

    def inference(self, obs, nuisance, gen_model, likelihood_f, proposal_distribution, slacks):
        # Generate candidates

        # Compute candidate likelihood

        # Perform update step

        # Draw debug

        # Compute result from samples / slacks
        raise NotImplementedError
