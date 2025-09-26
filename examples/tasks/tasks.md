# model_discovery
## description
Can you find the equations of motions for this system? Try to find the qualitative model and its numerical parameters.
## result_tool
save_result_find_eom

# phase_diagram
## description
Can you establish and finally plot the approximate phase diagram for this system, by
varying the system parameters to explore it?
## result_tool
save_phase_diagram

# model_distinction_v2
## description
Can you find out whether this system is described by:

(a) A 2D harmonic oscillator (isotropic, undamped)
(b) A 2D anharmonic oscillator (isotropic, undamped)
(c) A 2D harmonic oscillator (anisotropic, undamped)
(d) A 2D anharmonic oscillator (anisotropic, undamped)
(e) A 2D gravitational potential
(f) A 2D Lennard-Jones potential
(g) A charged particle in a magnetic field
(h) A damped 2D harmonic oscillator (isotropic)
(i) A damped 2D harmonic oscillator (anisotropic)
(j) A particle in a 2D double-well potential

In the end, save the equation of motion you suspect the system to have.

## result_tool
save_result_find_eom


# parameters
## description
Can you figure out the unknown parameters of this system?
## result_tool
save_estimated_parameters

# regimes
## description
This system can exhibit two qualitatively different behaviors.
Can you figure out which qualitative behaviours are possible and find the border between the two regimes?
## result_tool
save_result_qualitative_regimes

# model_distinction_1d
## description
Figure out which of the following models describes this system:
    1. Harmonic oscillator
    2. Pendulum
    3. Particle in double well potential
    4. Parametric oscillator
## result_tool
save_result_model_distinction_1d

# model_distinction_2d
## description
Figure out which of the following models describes this system:
    1. Particle in two dimensional gravitational potential
    2. Two coupled harmonic oscillators
    3. Double pendulum
## result_tool
save_result_model_distinction_2d

# non_linear
## description
I want to figure out whether this experimental system follows a linear or a nonlinear equation.
## result_tool
save_result_linear

# model_discovery_Hamiltonian
## description
Can you find the Hamiltonian for this spin system?
## result_tool
announce_Hamiltonian



# target_region
## description
Can you find initial conditions to get the system to the target region? The system only needs to reach to target region during the trajectory and is not required to stop there until the end.
## result_tool
save_target_distance

# target_minima
## description
This system has multiple potential minima. Can you find initial conditions that settle into each of them?
## result_tool
save_different_regimes_inits


# model_discovery_hidden
## description
Can you find the equations of motions for this system? Try to find the qualitative model and its numerical parameters for both the observed and hidden coordinates.
## result_tool
save_result_find_eom_hidden_degrees