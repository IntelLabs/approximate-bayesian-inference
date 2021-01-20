Problem definition
------------------
Predict the 3D goal position of a single-arm reaching movement performed by a human using a partially observed 
hand trajectory.

Getting started
------------------
The follow the next steps in order to have the neural surrogate generative model inference pipeline ready. All the
main files have a header function with tunable parameters. 

1- Generate a trajectory dataset. (Default dataset is a toy dataset, a 10K+ dataset is recommended)
    1.1 - Optional. Tune the arm controller limits and explore the effects of the different controller parameters 
            with arm_controller_tuining_gui.py
    1.2 - Generate a dataset with the simulator using main_datageneration.py. It is possible to run multiple instances
            to generate trajectories concurrently and merge the datasets. See scripts/generate_10K_multithread.sh
    1.3 - Optional. Check that the dataset trajectories look alright with main_viewer_dataset.py                
      
2- Train the neural surrogate. 
    2.1 Using main_training.py. Configure network architecture and observation parameters to match the generated data.
    2.2 Optional: Check trajectory generation makes sense with main_viewer_neural_emulator.py

3- Run inference. Can be configured with: 
    3.1 - Generative model to use. Currently avaliable: simulator, neural surrogate.
    3.2 - Sampling algorithm. Currently avaliable: Grid or MCMC Metropolis-Hastings.
    3.3 - Observation model. Currently available: 
        3.3.1 - Simulator: Samples a random point in the parameter space and generates a trajectory.
        3.3.1 - Random sample a trajectory from a dataset.
    3.4 - Prior distribution. This can be any distribution that implements sampling and log_prob methods.
    3.5 - Parameter domain boundaries for target quantities z ∋ ℝ^3 and and nuisance quantities n ∋ ℝ^8. Other
        relevant parameters like slack, prediction time window and sampling rate.

Definitions
-----------
- z ∋ ℝ^3: reaching intent final position (x,y,z). This is what we want to infer. 
- n ∋ ℝ^8: initial position (x,y,z) arm controller parameters (Kp, Ki, Kd, Kr, iClamp). These are extra 
           variables required by the generative model that we are not interested in inferring.
- x ∋ ?: unobservable state space, not used in this application.
- o ∋ ℝ^3*t: observed trajectory, projection of the state space into the observation space. A trajectory is a 
             timestamp ordered sequence of positions. i.e. a list of tuples (x,y,z,t) ordered by t 
- t ∋ ℕ+: number of observed points of the moving hand.
- ε ∋ ℝ^30: 30 discretized slack values.


Generative model
----------------
Describe the simulator. In the meantime see the paper.


Likelihood function
-------------------
Describe how observations and synthetic observations are compared. In the meantime see the paper.


Future work
-----------
- Generalize for any initial position
    - Generate dataset from different starting positions with different controller parameters.
    - Change NN topology as follows and prepare training dataset
        - 30 inputs (past 0.3 sec trajectory @ 30Hz)
        - 90 outputs (next 1 sec trajectory @ 30Hz)
    - Perform inference by unrolling the state for an arbitrary time window

- Generalize for any trajectory length and prediction time window
