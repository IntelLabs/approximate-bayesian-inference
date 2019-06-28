class CObservationModel(object):
    def __init__(self, params):
        pass

    def __del__(self):
        pass

    def get_observation(self):
        """ Return current observation """
        raise NotImplementedError

    def get_goal(self):
        """ Return current observation """
        raise NotImplementedError

    def step(self):
        """ Advance the observation model to the next observation. Used for simulated observation models """
        raise NotImplementedError

    def is_ready(self):
        """ Return True or False """
        raise NotImplementedError
