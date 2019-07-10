Dependencies
============
- General
    - pytorch
    - numpy
    - json
- Reaching intent app
    - pybullet   
- Plot results
    - matplotlib


Notation
========
- x      : state space
- o      : observation space.
- z      : latent space.
- n      : nuisance space. Used to model variables that are required by the generative model
but are not relevant for the inference process. Such that the inference will marginalize them out.
- ε      : slack term. Used to model the gap from the generative model to the observations.
- g(z,n) : Generative model that maps the latent space to either the state space or the observation space. This can
be a deterministic o = g(z,n) or stochastic o ~ p(o|z,n) generative model.
- L(x,x',ε): Surrogate likelihood (ABC). Provides a likelihood value using two observations or states. Usually an 
observation from the observation model is compared with a generated observation: L(o,g(z,n),ε) 
- Estimations are denoted by '. For example the estimated state space is denoted as x' and an estimated latent value z'.
- The tilde "~" can be read as "distributed as". For example, the fact that a sample x is obtained from a prior 
distribution p(x) can be written as: x ~ p(x).


Architecture
============
- Space. Represents a multidimensional space. Has a sampler attached that allows generating valid sample from the space.

- Sampler. Provides a sample from a space. Used to abstract the implementation of sampling from distributions on spaces.
 Examples are: prior distributions p(x), proposal distributions p(x_t|x_{t-1}), conditional distributions p(x|y). This 
 can also be used to implement deterministic mappings x' = f(x).

- Observation model (stochastic or deterministic mapping from state space "x" to observation space "o")
    - o ~ p(o|x). If the state space is fully observed, the observation model is the identity operation: x = o
    
- Generative model
    - maps latent space and nuisance space to observation space
    
- Surrogate likelihood function. Used to perform inference
    - computes the likelihood of a generated observation given an observation and the slack term: L = p(g(z,n)|o,ε)
    
- Inference algorithm: Samples the PDF of the latent parameters z given the observations.
    - Has access to generative model and likelihood function
    - Provides the posterior PDF "p(z|o)", as a set of importance weighted samples.


Implementing a new inference algorithm
======================================
- Implement a class derived from common.CBaseInferenceAlgorithm. 
- See working examples in inference.CInferenceGrid and inference.CInferenceMetropolisHastings
- The inference algorithm can be tested by adding it to the example reaching intent application in 
reaching_intent.main_inference.py and comparing the results with some of the algorithms already there.


Implementing a new application with neural emulators
====================================================
1- Define/identify/map the variables defined in the Notation section to the specific problem (See an example in reaching_intent/README.md)
    
2- Implement the CGenerativeModel interface (See reaching_intent.generative_models.CGenerativeModelSimulator)

3- Implement the CGenerativeModelNN interface (See reaching_intent.generative_models.CReachingNeuralEmulatorNN)

4- Implement the dataset_load(filename, k) and dataset_save(dataset, filename): custom functions (See reaching_intent.generative_models.dataset_load, dataset_save )

5- Implement the CObservationModel interface (See reaching_intent.observation_models.CObservationModel)

6- Optional: Implement your custom likelihood_function or use provided (See neural_emulators.loss_functions)

7- Implement the different main scripts, copy and make changes where required:
    - Dataset generation (e.g. reaching_intent.main_datageneration.py )
    - Neural emulation training (e.g. reaching_intent.main_training.py)
    - Inference (e.g. reaching_intent.main_inference.py)


Author
======
- Javier Felip Leon (javier.felip.leon@intel.com)


Contributors
============

References
==========
- Felip, J., Ahuja, N., Gómez-Gutiérrez, D., Tickoo, O., & Mansinghka, V. (2019). Real-time Approximate Bayesian Computation for Scene Understanding. arXiv preprint arXiv:1905.13307.

Last updated: Jul 2019