Problem definition
==================
Predict the 3D goal position of a single-arm reaching movement performed by a human using the current observed part of the motion.

Definitions
===========
- Define/identify state variables (See an example in reaching_intent/README.md)
    - z \in R^3: reaching intent final position (x,y,z). This is what we want to infer. 
    - n \in R^8: initial position (x,y,z) arm controller parameters (Kp, Ki, Kd, Kr, iClamp). These are extra variables required by the generative model that we are not interested in inferring.
    - x \in ?: unobservable state space, not used in this application.
    - o \in R^3*t: observed trajectory, projection of the state space into the observation space. A trajectory is a timestamp ordered sequence of positions. i.e. a list of tuples (x,y,z,t) ordered by t 
    - t \in N+: number of observed points of the moving hand. 
    - ε \in R^30: 30 discretized slack values.

Future work
===========
- Generalize for any initial position
    - Generate dataset from different starting positions with different controller parameters.
    - Change NN topology as follows and prepare training dataset
        - 30 inputs (past 0.3 sec trajectory @ 30Hz)
        - 90 outputs (next 1 sec trajectory @ 30Hz)
    - Perform inference by unrolling the state for an arbitrary time window

- Generalize for any trajectory length and prediction time window
