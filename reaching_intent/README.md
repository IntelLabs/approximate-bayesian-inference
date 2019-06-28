FUTURE WORK:

- Generalize for any initial position
    - Generate dataset from different starting positions with different controller parameters.
    - Change NN topology as follows and prepare training dataset
        - 30 inputs (past 0.3 sec trajectory @ 30Hz)
        - 90 outputs (next 1 sec trajectory @ 30Hz)
    
    - Perform inference by unrolling the state for an arbitrary time window

- Generalize for any trajectory length and prediction time window

Problem definition:
Predict the 3D goal position of a single-arm reaching movement performed by a human using the current observed part of the motion

Definitions:
position: a triplet (x,y,z)
trajectory: a timestamp ordered sequence of positions. i.e. a list of tuples (x,y,z,t) ordered by t
goal: the final position of a trajectory
start: the initial position of a trajectory
observation: a part of a trajectory may or may not contain the start or the goal



Mapping to the neural emulators paper variables
goal                -> z \in R^3
controller_gain     -> n \in R
time_window         -> k = 30
observation         -> x \in R^{120}
likelihood          -> p(x|z,n) implemented as the likelihood_function


