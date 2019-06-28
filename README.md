DEPENDENCIES
============
- General
    - pytorch
    - numpy
    - json
- Reaching intent app
    - pybullet   
- Plot results
    - matplotlib


NOTATION
========
- x      : state space
- o      : observation space
- z      : latent space
- n      : nuisance space. Used to model variables that are required by the generative model
but are not relevant for the inference process. Such that the inference will marginalize them out.
- ε      : slack term.
- g(z,n) : Generative model that maps the latent space to either the state space or the observation space. This can
be a deterministic o = g(z,n) or stochastic o ~ p(o|z,n) generative model.
- L(x,x',ε): Surrogate likelihood (ABC). Provides a likelihood value using two observations or states. Usually an 
observation from the observation model is compared with a generated observation: L(x,g(z,n),ε) 
- Estimations are denoted by '. For example the estimated state space is denoted as x'.
- The tilde "~" can be read as "distributed as". For example, the fact that a sample x is obtained from a prior 
distribution p(x) can be written as: x ~ p(x). 


ARCHITECTURE
============
- Space. Represents a multidimensional space. Has a sampler attached that allows generating valid sample from the space.

- Sampler. Provides a sample from a space. Used to abstract the implementation of sampling from distributions on spaces.
 Examples are: prior distributions p(x), proposal distributions p(x_t|x_{t-1}), conditional distributions p(x|y). This 
 can also be used to implement deterministic mappings x' = f(x).

- Observation model (stochastic or deterministic mapping from state space "x" to observation space "o")
    - o ~ p(o|x). If the state space is fully observed, the observation model is the identity operation: x = o
    
- Generative model
    - maps latent space and nuisance space to state space
    - maps latent space and nuisance space to observation space
    
- Surrogate likelihood function. Used to perform inference
    - computes L = p(g(z,n)|x,ε)
    
- Inference algorithm: Samples the PDF of the latent parameters z given the observations.
    - Has access to generative model, observation model and likelihood
    - Provides posterior PDF: p(z|x) or p(z|o) 


IMPLEMENTING A NEW NEURAL EMULATOR INFERENCE
============================================
1- Define/identify state variables (See reaching_intent_estimation/README.md)
    - x: latent state
    - n: nuisance (extra variables for the generative model)
    - z: observable variables
    - epsilon: slack
    
2- Implement the CGenerativeModel interface (See reaching_intent_estimation.generative_models.CGenerativeModelSimulator)

3- Implement the CGenerativeModelNN interface (See reaching_intent_estimation.generative_models.CReachingNeuralEmulatorNN)

4- Implement the dataset_load(filename, k) and dataset_save(dataset, filename): custom functions (See reaching_intent_estimation.generative_models.dataset_load, dataset_save )

5- Implement the CObservationModel interface (See reaching_intent_estimation.observation_models.CObservationModel)

6- Optional: Implement your custom likelihood_function or use provided (See neural_emulators.loss_functions)

7- Implement the main inference script (See reaching_intent_estimation.main_inference.py, copy and make changes where required)

8- Run main_{yourmodule}.py and have fun.


Author
======
- Javier Felip Leon (javier.felip.leon@intel.com)


Contributors
============
