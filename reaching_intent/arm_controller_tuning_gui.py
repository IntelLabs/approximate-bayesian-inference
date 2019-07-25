#!/usr/bin/python3

###################################
# GENERIC IMPORTS
###################################
from common.common import *
from samplers.CSamplerUniform import CSamplerUniform

###################################
# APPLICATION SPECIFIC IMPORTS (import from your application specific module)
###################################
import pybullet as p
from reaching_intent.generative_models.CGenerativeModelSimulator import CGenerativeModelSimulator
from reaching_intent.generative_models.CGenerativeModelSimulator import create_sim_params
###################################


if __name__ == "__main__":
    # Generative model parameter limits: start volume (x,y,z), end volume (x,y,z), controller(Kp,Ki,Kd,Krep,iClamp)
    param_limits_min = t_tensor([-0.05, 0.30, -0.10, 0.25, -0.4, 0.20, 5, 0.005, 0, 0.10, 20])
    param_limits_max = t_tensor([-0.04, 0.31, -0.09, 0.90, 0.4, 0.21, 20, 0.010, 0, 0.11, 30])

    # Sampler to sample parameter values to generate trajectories
    param_sampler = CSamplerUniform({"min": param_limits_min, "max": param_limits_max})

    # Select the parameters that are considered nuisance and the parameters that are considered interesting
    latent_mask = t_tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]) == 1  # We are interested in the end position
    nuisance_mask = t_tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]) == 0  # The rest of the params are considered nuisance

    # Get simulator parameters from the app specific import
    simulator_params = create_sim_params()

    # Simulator used to generate synthetic data
    neSimulator = CGenerativeModelSimulator(simulator_params)

    # Simulated arm controller parameters
    controller_params = dict()
    Kp, Ki, Kd, Krep, iClamp = 20, 0.010, 0, 0.11, 30
    controller_params["Kp"] = p.addUserDebugParameter(paramName="Kp", rangeMin=0.0, rangeMax=100.0, startValue=Kp, physicsClientId=neSimulator.sim_id)
    controller_params["Kd"] = p.addUserDebugParameter(paramName="Kd", rangeMin=0.0, rangeMax=10.0, startValue=Kd, physicsClientId=neSimulator.sim_id)
    controller_params["Ki"] = p.addUserDebugParameter(paramName="Ki", rangeMin=0.0, rangeMax=10.0, startValue=Ki, physicsClientId=neSimulator.sim_id)
    controller_params["iClamp"] = p.addUserDebugParameter(paramName="iClamp", rangeMin=0.0, rangeMax=100.0, startValue=iClamp, physicsClientId=neSimulator.sim_id)
    controller_params["Krep"] = p.addUserDebugParameter(paramName="Krep", rangeMin=0.0, rangeMax=10.0, startValue=Krep, physicsClientId=neSimulator.sim_id)

    while p.isConnected(physicsClientId=neSimulator.sim_id):
        Kp = p.readUserDebugParameter(controller_params["Kp"], physicsClientId=neSimulator.sim_id)
        Kd = p.readUserDebugParameter(controller_params["Kd"], physicsClientId=neSimulator.sim_id)
        Ki = p.readUserDebugParameter(controller_params["Ki"], physicsClientId=neSimulator.sim_id)
        iClamp = p.readUserDebugParameter(controller_params["iClamp"], physicsClientId=neSimulator.sim_id)
        Krep = p.readUserDebugParameter(controller_params["Krep"], physicsClientId=neSimulator.sim_id)

        # Sample a random value in the parameter space (latent + nuisance)
        params = param_sampler.sample(1, None)

        # Fix the nuisance parameters to the GUI input parameters
        params[:, nuisance_mask] = [Kp, Ki, Kd, Krep, iClamp]

        # Generate a trajectory
        generated = neSimulator.generate(params[:, latent_mask], params[:, nuisance_mask])
