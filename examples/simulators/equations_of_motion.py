import jax
import numpy as np
import scipy
import jax.numpy as jnp
from functools import partial
import ast
from jax import vmap

import matplotlib.pyplot as plt

from sciexplorer.physics_eom import ode_solve



class ODEsolver:
    def __init__(self, solver_steps_per_timestep=1):
        self.rhs={}
        self.description="You have access to an ODE solver."
        self.toolboxes = {'ODEsolver': [self.set_rhs, self.solve_ode]}
        self.solver_steps_per_timestep = solver_steps_per_timestep
        
    def set_rhs(self, rhs_label:str, rhs: str) -> dict:
        """
        Define the right-hand side of an ordinary differential
        equation. 

        You pass a python code that must be of the form

        def rhs(X,t,params):
            <code that calculates rhs and returns the result>
        
        The function must return a jax.Array of the same shape as X.

        Here X is a jax.numpy array with the variables,
        t will be the time, and params is another jax.numpy
        array containing any parameters needed, of fixed length.

        Use jax.numpy syntax in the form "jnp.sin(...)".
        
        Args:
            rhs_label:str the label the funciton will be stored under.
            rhs:str the python code defining the right-hand side function.
    
        Returns:
            Message indicating wether the function was set successfully set.
        """
        global_dict={"rhs": None, 'jnp': jnp, 'jax': jax, 'np': np, 'scipy': scipy}
        try:
            exec(rhs, global_dict)
        except Exception as e:
            return {'error': f"Error in executing provided code: {e}"}
        if global_dict["rhs"] is None:
            return {'error': "Error: rhs function not defined in code."}
        self.rhs[rhs_label] = global_dict["rhs"]
        return {'result_str': f"Success: Set rhs function successfully and saved it under the label: {rhs_label}."}#{'to_model': f"Success: Set rhs function successfully and saved it under the label: {rhs_label}."}
    
    def solve_ode(self, rhs_label: str, 
                  X0: str, params: str, T: float, dt: float) -> dict:
        """
        Solve the ordinary differential equation that was previouslydefined via set_rhs. 
        X0 and params must be arrays of the right dimension.


        Args:
            rhs_label: the label for the previously defined right-hand side
            X0: the initial condition, in the form of a list, for example "[0.3,0.2,0.5]"
            params: the parameters, in the form of a list, for example "[0.1,0.8]"
            T: the time until which the equation should be solved
            dt: the time step size for the solution.

        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'X0':jax.Array of shape [dimension] with the initial condition,
            'array':jax.Array of shape [nsteps,dimension] with the solution,
            'params':jax.Array of shape [nparams] with the parameters.
        """
        #print(f"solve_ode X0 for {rhs_label}: {X0} params: {params}")
        try:
            X0=jnp.array( ast.literal_eval( X0 ) )
        except Exception as e:
            return {"error": f"Error in parsing initial condition: {e}"}
        try:
            params=jnp.array( ast.literal_eval( params ) )
        except Exception as e:
            return {"error": f"Error in parsing parameters: {e}"}
        #ts=jnp.linspace(0.0, T, int(T/dt)+1)
        try:
            rhs = self.rhs[rhs_label]
        except KeyError:
            return {"error": f"Error: rhs function with label '{rhs_label}' not found."}
        try:
            Xs = ode_solve( X0, rhs, params, dt, T, solver_steps_per_timestep=self.solver_steps_per_timestep)
            ts = jnp.arange(0, T+dt, dt)
        except Exception as e:
            return {"error": f"Error in solving ODE: {e}"}
        solution={"ts":ts, "X0": jnp.copy(X0),
                  "array":jnp.copy(Xs), "params": jnp.copy(params)}
        return solution
    

# Parent class for all mechanical systrems
class MysterySystem():
    '''Parent class for mystery systems that are governed by an ordinary differential equation.'''
    def __init__(self, 
                 ode_rhs:str='', # the right-hand side of the ODE, in the form of a string.
                 seed:int=0, # seed for random runs
                 min_random=-1.0, # minimum value for random initial conditions
                 max_random=1.0, # maximum value for random initial conditions
                 T=20., # the time until which the equation should be solved
                 dt=0.1, # the time step size for the solution
                 solver_steps_per_timestep=100, # the number of steps per time step to use in the solver
                 ode_dim=2,  # the dimension of the ODE, e.g. 2 for a pendulum with angle and angular velocity
                 true_distinguish_func=None,  # a function that distinguishes between two qualitative regimes of the system, e.g. bound and traveling states.
                 eval_seed=42, # seed for evaluation, used for random initial conditions
                 description=None, # description of the system
                 params=None,
                 model_type=None,
                 max_experiments=None, # maximum number of experiments to run, if None, no limit
                 ):
        self.max_experiments = max_experiments
        if max_experiments is None:
            self.max_experiments = int(1e6) # set a very high number if no limit is given
        self.current_experiments = 0
        self.ode_solver = ODEsolver(solver_steps_per_timestep)
        self.key = jax.random.key(seed)
        if not ode_rhs:
            raise ValueError("ode_rhs must be provided")
        self.ode_rhs = ode_rhs
        self.ode_solver.set_rhs("mystery_rhs", ode_rhs)
        self.toolboxes = {}
        
        self.min_random = min_random
        self.max_random = max_random

        self.T = T
        self.dt = dt
        self.ode_dim = ode_dim
        self.true_distinguish_func = true_distinguish_func
        self.eval_key = jax.random.key(eval_seed)
        assert eval_seed != seed, 'Eval seed and seed must be different.'
        if not description:
            raise ValueError("Description must be provided")
        else:
            self.description = description

        self.params=params
        self.model_type = model_type

        if max_experiments:
            if max_experiments > 1:
                self.description += f"\nYou can observe a maximum of {max_experiments} evolutions."
            else:
                self.description += f"\nYou can observe a maximum of {max_experiments} evolution."
        
    def observe_evolution(self) -> dict:
        raise NotImplementedError("This method should be implemented in subclasses.")

    
    def solve_ode(self, rhs_label: str, 
                  X0: str, params: str, T: float, dt: float) -> dict:
        """
        Solve the ordinary differential equation that was previously defined via set_rhs. 
        X0 and params must be arrays of the right dimension.


        Args:
            rhs_label: the label for the previously defined right-hand side
            X0: the initial condition, in the form of a list, for example "[0.3,0.2,0.5]"
            params: the parameters, in the form of a list, for example "[0.1,0.8]"
            T: the time until which the equation should be solved
            dt: the time step size for the solution.

        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'X0':jax.Array of shape [dimension] with the initial condition,
            'array':jax.Array of shape [nsteps,dimension] with the solution,
            'params':jax.Array of shape [nparams] with the parameters.
        """
        return self.ode_solver.solve_ode(rhs_label, X0, params, T, dt)
    
    def set_rhs(self, rhs_label: str, rhs: str) -> dict:
        """
        Define the right-hand side of an ordinary differential equation. 

        You pass a python code that must be of the form

        def rhs(X,t,params):
            <code that calculates rhs and returns the result>
        
        The function must return a jax.Array of the same shape as X.

        Here X is a jax.numpy array with the variables,
        t will be the time, and params is another jax.numpy
        array containing any parameters needed, of fixed length.

        Use jax.numpy syntax in the form "jnp.sin(...)".
        
        Args:
            rhs_label:str the label the funciton will be stored under.
            rhs:str the python code defining the right-hand side function.
    
        Returns:
            Message indicating wether the function was set successfully set.
        """

        return self.ode_solver.set_rhs(rhs_label, rhs)
    
    def observe_multiple_random_evolutions(self) -> dict:
        """
        Observe multiple evolutions of the mystery system with random but reasonable initial conditions.
        Args:
            None
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, dimension] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """

        n_trajs = 3

        steps = len(jnp.linspace(0.0, self.T, int(self.T/self.dt)+1))
                    
        Xs_solutions = jnp.zeros((n_trajs, steps, self.ode_dim))  # shape [n_evolutions, nsteps, dimension]
        ts_solutions = jnp.zeros((n_trajs, steps))  # shape [n_evolutions, nsteps]

        for i in range(n_trajs):
            self.ode_solver.set_rhs("mystery_rhs", self.ode_rhs)
            solution=self.observe_random_evolution()
            if 'error' in solution.keys():
                self.current_experiments -= i
                return {'error': solution['error']}
            Xs_solutions = Xs_solutions.at[i].set(solution['array'])
            ts_solutions = ts_solutions.at[i].set(solution['ts'])

        solutions = {
            'tss': ts_solutions,
            'arrays': Xs_solutions,
            }
        # if self.current_experiments > self.max_experiments:
        #     self.current_experiments -= n_trajs
        #     return {'error': f"Error: Maximum number of experiments ({self.max_experiments}) exceeded. Only {self.max_experiments - self.current_experiments} are available."}
        return solutions
    
    
    
    def observe_random_evolution(self) -> dict:
        """
        Observe a random evolution of the mystery system with random but reasonable initial conditions.
        Args:
            None
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,dimension] with the solution,
                    array[:, 0] holds the coordinate and array[:, 1] its velocity.
        """
        self.key, subkey = jax.random.split(self.key)
        inits = jax.random.uniform(subkey, shape=(self.ode_dim,), minval=self.min_random, maxval=self.max_random)
        if self.current_experiments > self.max_experiments:
            return {'error': f"Error: Maximum number of experiments ({self.max_experiments}) exceeded."}
        return self.observe_evolution(*[float(i) for i in inits])
    
    def observe_multiple_evolutions_1d(self, q_inits: str, q_dot_inits: str) -> dict:
        """
        SHOULD ONLY BE CALLED IN 1D SYSTEM FUNCTION NOT AS TOOL.
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        Args:
            q_inits: list of initial generalized coordinates, in the form '[value1, value2, ...]'
            q_dot_inits: list of initial generalized velocities, in the form '[value1, value2, ...]'
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, dimension] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        # evaluate str list:
        try:
            q_inits = jnp.array(ast.literal_eval(q_inits))
            q_dot_inits = jnp.array(ast.literal_eval(q_dot_inits))
        except Exception as e:
            return {'error': f"Error in parsing initial conditions: {e}"}
        if len(q_inits) != len(q_dot_inits):
            return {'error': "Error: The number of initial coordinates and velocities must match."}
        n_trajs = len(q_inits)
        if n_trajs > 5:
            return {'error': "Error: The maximum number of evolutions is 5."}   
        self.ode_solver.set_rhs("mystery_rhs", self.ode_rhs)
        ts_solutions = jnp.zeros((n_trajs, int(self.T/self.dt)+1))  # shape [n_evolutions, nsteps]
        Xs_solutions = jnp.zeros((n_trajs, int(self.T/self.dt)+1, self.ode_dim))  # shape [n_evolutions, nsteps, dimension]
        for i in range(n_trajs):
            solution=self.observe_evolution(q_inits[i], q_dot_inits[i])
            if 'error' in solution.keys():
                self.current_experiments -= (i + 1)
                return {'error': solution['error']}
            ts_solutions = ts_solutions.at[i].set(solution['ts'])
            Xs_solutions = Xs_solutions.at[i].set(solution['array'])
        solutions = {
            'tss': ts_solutions,
            'arrays': Xs_solutions,
            }
        # if self.current_experiments > self.max_experiments:
        #     self.current_experiments -= n_trajs
        #     return {'error': f"Error: Maximum number of experiments ({self.max_experiments}) exceeded. Only {self.max_experiments - self.current_experiments} are available."}
        return solutions
    

    def observe_multiple_evolutions_2d(self, q0s: str, q1s: str, q0_dots: str, q1_dots: str) -> dict:
        """
        SHOULD ONLY BE CALLED IN 1D SYSTEM FUNCTION NOT AS TOOL.
        """
        # evaluate str list:
        try:
            q0_inits = jnp.array(ast.literal_eval(q0s))
            q0_dot_inits = jnp.array(ast.literal_eval(q0_dots))
            q1_inits = jnp.array(ast.literal_eval(q1s))
            q1_dot_inits = jnp.array(ast.literal_eval(q1_dots))
        except Exception as e:
            return {'error': f"Error in parsing initial conditions: {e}"}
        n_trajs = len(q0_inits)
        if n_trajs > 5:
            return {'error': "Error: The maximum number of evolutions is 5."}   
        self.ode_solver.set_rhs("mystery_rhs", self.ode_rhs)
        ts_solutions = jnp.zeros((n_trajs, int(self.T/self.dt)+1))  # shape [n_evolutions, nsteps]
        Xs_solutions = jnp.zeros((n_trajs, int(self.T/self.dt)+1, self.ode_dim))  # shape [n_evolutions, nsteps, dimension]
        for i in range(n_trajs):
            solution=self.observe_evolution(float(q0_inits[i]), float(q1_inits[i]), float(q0_dot_inits[i]), float(q1_dot_inits[i]))
            if 'error' in solution.keys():
                self.current_experiments -= (i + 1)
                return {'error': solution['error']}
            ts_solutions = ts_solutions.at[i].set(solution['ts'])
            Xs_solutions = Xs_solutions.at[i].set(solution['array'])
        solutions = {
            'tss': ts_solutions,
            'arrays': Xs_solutions,
            }
        # if self.current_experiments > self.max_experiments:
        #     self.current_experiments -= n_trajs  # revert the count if we hit the limit
        #     return {'error': f"Error: Maximum number of experiments ({self.max_experiments}) exceeded. Only {self.max_experiments - self.current_experiments} are available."}
        return solutions\
        

    def observe_multiple_evolutions_4d(self, q0s: str, q1s: str, q2s:str, q3s:str,
                                        q0_dots: str, q1_dots: str, q2_dots:str, q3_dots:str) -> dict:
        """
        SHOULD ONLY BE CALLED IN 1D SYSTEM FUNCTION NOT AS TOOL.
        """
        # evaluate str list:
        try:
            q0_inits = jnp.array(ast.literal_eval(q0s))
            q0_dot_inits = jnp.array(ast.literal_eval(q0_dots))
            q1_inits = jnp.array(ast.literal_eval(q1s))
            q1_dot_inits = jnp.array(ast.literal_eval(q1_dots))
            q2_inits = jnp.array(ast.literal_eval(q2s))
            q2_dot_inits = jnp.array(ast.literal_eval(q2_dots))
            q3_inits = jnp.array(ast.literal_eval(q3s))
            q3_dot_inits = jnp.array(ast.literal_eval(q3_dots))
        except Exception as e:
            return {'error': f"Error in parsing initial conditions: {e}"}
        n_trajs = len(q0_inits)
        if n_trajs > 5:
            return {'error': "Error: The maximum number of evolutions is 5."}   
        self.ode_solver.set_rhs("mystery_rhs", self.ode_rhs)
        ts_solutions = jnp.zeros((n_trajs, int(self.T/self.dt)+1))  # shape [n_evolutions, nsteps]
        Xs_solutions = jnp.zeros((n_trajs, int(self.T/self.dt)+1, self.ode_dim))  # shape [n_evolutions, nsteps, dimension]
        for i in range(n_trajs):
            solution=self.observe_evolution(float(q0_inits[i]), float(q1_inits[i]), float(q2_inits[i]), float(q3_inits[i]),
                                            float(q0_dot_inits[i]), float(q1_dot_inits[i]), float(q2_dot_inits[i]), float(q3_dot_inits[i]))
            if 'error' in solution.keys():
                self.current_experiments -= (i + 1)
                return {'error': solution['error']}
            ts_solutions = ts_solutions.at[i].set(solution['ts'])
            Xs_solutions = Xs_solutions.at[i].set(solution['array'])
        solutions = {
            'tss': ts_solutions,
            'arrays': Xs_solutions,
            }
        # if self.current_experiments > self.max_experiments:
        #     self.current_experiments -= n_trajs  # revert the count if we hit the limit
        #     return {'error': f"Error: Maximum number of experiments ({self.max_experiments}) exceeded. Only {self.max_experiments - self.current_experiments} are available."}
        return solutions
    
    def observe_evolution_1d(self, q_init: float, q_dot_init: float) -> dict:
        """
        DONT USE AS TOOL
        """
        self.ode_solver.set_rhs("mystery_rhs", self.ode_rhs)
        solution=self.ode_solver.solve_ode('mystery_rhs', str([float(q_init), float(q_dot_init)]),
                     str(self.params), self.T, self.dt)
        if 'error' in solution.keys():
            return {'error': solution['error']}
        solution_subset = {
            'ts': solution['ts'],
            'array': solution['array']}
        self.current_experiments += 1
        if self.current_experiments > self.max_experiments:
            return {'error': f"Error: Maximum number of evolutions ({self.max_experiments}) exceeded."}
        return solution_subset
    
    def observe_evolution_2d(self, q0: float, q1: float, q0_dot: float, q1_dot: float) -> dict:
        """
        DONT USE AS TOOL
        """
        self.ode_solver.set_rhs("mystery_rhs", self.ode_rhs)
        solution = self.ode_solver.solve_ode(
            'mystery_rhs',
            str([float(q0), float(q1), float(q0_dot), float(q1_dot)]),
            str(self.params),
            self.T,
            self.dt
        )
        if 'error' in solution.keys():
            return {'error': solution['error']}
        self.current_experiments += 1
        if self.current_experiments > self.max_experiments:
            return {'error': f"Error: Maximum number of evolutions ({self.max_experiments}) exceeded."}
        return {
            'ts': solution['ts'],
            'array': solution['array']
        }    
    
    def observe_evolution_4d(self, q0: float, q1: float, q2:float, q3:float, 
                             q0_dot: float, q1_dot: float, q2_dot:float, q3_dot:float) -> dict:
        """
        DONT USE AS TOOL
        """
        self.ode_solver.set_rhs("mystery_rhs", self.ode_rhs)
        solution = self.ode_solver.solve_ode(
            'mystery_rhs',
            str([float(q0), float(q1), float(q2), float(q3), 
                 float(q0_dot), float(q1_dot), float(q2_dot), float(q3_dot)]),
            str(self.params),
            self.T,
            self.dt
        )
        if 'error' in solution.keys():
            return {'error': solution['error']}
        self.current_experiments += 1
        if self.current_experiments > self.max_experiments:
            return {'error': f"Error: Maximum number of evolutions ({self.max_experiments}) exceeded."}
        return {
            'ts': solution['ts'],
            'array': solution['array']
        }    
    
    def save_result_qualitative_regimes(self, distinguish_func:str, plot_result=False) -> dict:
        '''Save a function that can distinguish between different qualitative regimes of the system given the initial conditions.
        The function should take the initial conditions as input and return 0 for regime 1, and 1 for regime 2. 
        The order of the regimes is not important, as long as the function is consistent.
        This tool should only be used to provide the final result. It can only be called once per experiment.
        Args:
            distinguish_func: The function that distinguishes between the two regimes. 
            Should be a string representation of a python function of the form:
            'def distinguish_func(X0: jax.Array) -> int:
                <code that distinguishes between the two regimes and returns 0 or 1>'
                return 0 if regime_1, and 1 if regime_2.
            where X0 is a jax.Array of shape [ode_dimension] containing the initial conditions as specified in the observe_evolution function.
            the function may use the following packages:
            jax: jax for numerical operations.
            jnp: jax.numpy for numerical operations.
            np: numpy for numerical operations.
            scipy: scipy for numerical operations including optimization and solving differential equations.
        Returns:
            save_message:str A message that the prediction has been saved.
        '''

        globals_dict = {'distinguish_func':None, 'jax':jax, 'jnp': jnp, 'np':np, 'scipy':scipy}
        assert self.true_distinguish_func is not None, "You must define a true_distinguish_func in the subclass to use this tool."
        
        self.eval_key, subkey = jax.random.split(self.eval_key)
        n_random_inits = 1000
        random_inits = jax.random.uniform(subkey, shape=(n_random_inits, self.ode_dim), minval=self.min_random, maxval=self.max_random)
        true_regimes = jnp.array([self.true_distinguish_func(init) for init in random_inits])

        
        try:
            exec(distinguish_func, globals_dict)
            distinguish_func_pred = globals_dict["distinguish_func"]
            predicted_regimes = jnp.array([distinguish_func_pred(init) for init in random_inits])
            predicted_regimes_flipped = 1 - predicted_regimes  # flip the regimes nad compare both

            accuracy = jnp.mean(predicted_regimes == true_regimes)
            accuracy_flipped = jnp.mean(predicted_regimes_flipped == true_regimes)
            higher_accuracy = jnp.max(jnp.array([accuracy, accuracy_flipped]))

            self.last_result = {'predicted_regime_function': distinguish_func,
                                'accuracy': higher_accuracy,
                                'error': None}
            
            # better predicted regime function
            if accuracy_flipped > accuracy:
                predicted_regimes = predicted_regimes_flipped
            if plot_result:
            # plot result
                plt.scatter(random_inits[:, 0], random_inits[:, 1], c=predicted_regimes, cmap='coolwarm', alpha=1., label='predicted')
                plt.scatter(random_inits[:, 0], random_inits[:, 1], c=true_regimes, 
                            cmap='coolwarm', marker='x', alpha=1., edgecolors='black', label='true')
                plt.title(f'Predicted Regimes (Accuracy: {higher_accuracy:.2f})')
                plt.xlabel('Initial Coordinate (q)')
                plt.ylabel('Initial Velocity (q_dot)')
                plt.colorbar(label='Predicted Regime')
                plt.legend()
                plt.show()

            
        except Exception as e:
            self.last_result = {'predicted_regime_function': distinguish_func,
                                'accuracy': jnp.nan,
                                'error': str(e)}
        
        
        return {'save_message': 'Saved predicted regime function. You can now stop the experiment.'}
    

    def save_result_find_eom(self, rhs: str) -> dict:
        """
        Compare the provided right-hand side of the ordinary differential equation with the true right-hand side governing the differential equation of the system.
        The loss is computed as the mean squared error between the true rhs and predicted rhs at some randomly sampled points.
        This tool should only be used to provide the final result. It can only be called once per experiment.
        Args:
            rhs: Define the right-hand side of an ordinary differential equation. 
                You pass a python code that must be of the form
                    def rhs(X:jax.Array, t:float) -> jax.Array:
                        ''' Calulates the right-hand side of the ODE. 
                            Args:
                                X:jax.Array containing the generalized coordinates/coordinate followed by their velocities/its velocity. E.g. jnp.array([q, q_dot]).
                                t:float The time variable. Might be used in case system is time-dependent.
                            Returns:
                                The right-hand side of the ODE, a jax.Array of shape n_coordinates * 2.'''
                    rhs may use jnp syntax in the form "jnp.sin(...)".
        Returns:
            save_message:str A message that the prediction has been saved.
        """
        n_random_inits = 1000
        self.eval_key, subkey = jax.random.split(self.eval_key)
        random_inits = jax.random.uniform(subkey, shape=(n_random_inits, self.ode_dim), minval=2*self.min_random, maxval=2*self.max_random)
        self.eval_key, subkey = jax.random.split(self.eval_key)
        random_times = jax.random.uniform(subkey, shape=(n_random_inits,), minval=0., maxval=self.T)
        true_rhss_all = jnp.array([self.ode_solver.rhs['mystery_rhs'](init, float(t),  jnp.array(self.params)) 
                               for init, t in zip(random_inits, random_times)])
        # fiilter out nans of true_rhss
        true_non_nan_idcs = jnp.isfinite(true_rhss_all)
        try:
            self.set_rhs("predicted_rhs", rhs)
            predicted_rhss_all = jnp.array([self.ode_solver.rhs['predicted_rhs'](init, float(t)) for init, t in zip(random_inits, random_times)])
            # use only where true rhs is finite
            true_rhss = true_rhss_all[true_non_nan_idcs]
            predicted_rhss = predicted_rhss_all[true_non_nan_idcs]
            
            R2 = 1 - jnp.mean((true_rhss - predicted_rhss)**2) / (jnp.std(true_rhss)**2 + 1e-8)
            # calc R2 for velocity pred in acceleration ored separatly
            R2_list = []
            for i in range(true_rhss_all.shape[1]):
                R2_i = 1 - jnp.mean((true_rhss_all[:, i][true_non_nan_idcs[:, i]] - predicted_rhss_all[:, i][true_non_nan_idcs[:, i]])**2) / (jnp.std(true_rhss_all[:, i][true_non_nan_idcs[:, i]])**2 + 1e-8)
                R2_list.append(R2_i)
            # R2_qd = 1 - jnp.mean((true_rhss[:, 0] - predicted_rhss[:, 0])**2) / (jnp.std(true_rhss[:, 0])**2 + 1e-8)
            # R2_qdd = 1 - jnp.mean((true_rhss[:, 1] - predicted_rhss[:, 1])**2) / (jnp.std(true_rhss[:, 1])**2 + 1e-8)
            R2 = jnp.mean(jnp.array(R2_list))

            self.last_result = {'R2': R2, 'predicted_rhs': rhs,
                                'R2_list': R2_list,
                                'error': None,}
        except Exception as e:
            self.last_result = {'R2': jnp.nan, 'R2_list': [],
                                'predicted_rhs': rhs,
                                'error': str(e),}
            
        return {'save_message': 'Saved predicted rhs and params. You can now stop the experiment.'}
    



        

    

    def save_result_find_eom_hidden_degrees_old(self, rhs: str) -> dict:
        """
        Compare the provided right-hand side of the ordinary differential equation with the true right-hand side governing the differential equation of the system.
        The loss is computed as the mean squared error between the true rhs and predicted rhs at some randomly sampled points.
        This tool should only be used to provide the final result. It can only be called once per experiment.
        Args:
            rhs: Define the right-hand side of an ordinary differential equation. 
                You pass a python code that must be of the form
                    def rhs(X:jax.Array, t:float) -> jax.Array:
                        ''' Calulates the right-hand side of the ODE. 
                            Make sure to also include the hidden dimensions in X.
                            X is of shape (n_visible + n_hidden) * 2 where the first half are the coordinates and the second half the velocities.
                            E.g. for 1 visible and 1 hidden dimension, X is of shape (2*2,) and of the form jnp.array([q0, q1, q0_dot, q1_dot]).
                            Make sure to adhere to this format.
                            Args:
                                X:jax.Array containing the generalized coordinates (including the hidden coordinates) followed by their velocities. E.g. jnp.array([q0, q1, q0_dot, q1_dot]).
                                t:float The time variable. Might be used in case system is time-dependent.
                            Returns:
                                The right-hand side of the ODE, a jax.Array of shape n_coordinates * 2.'''
                    rhs may use jnp syntax in the form "jnp.sin(...)".
        Returns:    
            save_message:str A message that the prediction has been saved.
        """
        # for now this works only for 1d observed systems with hidden dimensions
        n_random_inits = 1000
        self.eval_key, subkey = jax.random.split(self.eval_key)
        random_inits = jax.random.uniform(subkey, shape=(n_random_inits, self.ode_dim), minval=2*self.min_random, maxval=2*self.max_random)
        #random_inits = jnp.concatenate([random_inits_observed[:, 0][:, None], qs_hidden, random_inits_observed[:, 1][:, None], qdots_hidden], axis=1)
        self.eval_key, subkey = jax.random.split(self.eval_key)
        random_times = jax.random.uniform(subkey, shape=(n_random_inits,), minval=0., maxval=self.T)
        true_rhss_all = jnp.array([self.ode_solver.rhs['mystery_rhs'](init, float(t),  jnp.array(self.params)) 
                               for init, t in zip(random_inits, random_times)])
        # fiilter out nans of true_rhss
        true_non_nan_idcs = jnp.isfinite(true_rhss_all)
        try:
            self.set_rhs("predicted_rhs", rhs)
            predicted_rhss_all = jnp.array([self.ode_solver.rhs['predicted_rhs'](init, float(t)) for init, t in zip(random_inits, random_times)])
            # use only where true rhs is finite
            true_rhss = true_rhss_all[true_non_nan_idcs]
            predicted_rhss = predicted_rhss_all[true_non_nan_idcs]
            
            #R2_q = 1 - jnp.mean((true_rhss[:, 0] - predicted_rhss[:, 0])**2) / (jnp.std(true_rhss[:, 0])**2 + 1e-8)
            #R2_q_dot = 1 - jnp.mean((true_rhss[:, self.ode_dim//2] - predicted_rhss[:, self.ode_dim//2])**2) / (jnp.std(true_rhss[:, self.ode_dim//2])**2 + 1e-8)
            # calc R2 for velocity pred in acceleration ored separatly
            R2_list = []
            for i in range(true_rhss_all.shape[1]):
                R2_i = 1 - jnp.mean((true_rhss_all[:, i][true_non_nan_idcs[:, i]] - predicted_rhss_all[:, i][true_non_nan_idcs[:, i]])**2) / (jnp.std(true_rhss_all[:, i][true_non_nan_idcs[:, i]])**2 + 1e-8)
                R2_list.append(R2_i)
            # R2_qd = 1 - jnp.mean((true_rhss[:, 0] - predicted_rhss[:, 0])**2) / (jnp.std(true_rhss[:, 0])**2 + 1e-8)
            # R2_qdd = 1 - jnp.mean((true_rhss[:, 1] - predicted_rhss[:, 1])**2) / (jnp.std(true_rhss[:, 1])**2 + 1e-8)
            R2_q = R2_list[0]
            R2_q_dot = R2_list[self.ode_dim//2]
            R2 = jnp.mean(jnp.array([R2_q, R2_q_dot]))

            self.last_result = {'R2': R2, 'predicted_rhs': rhs,
                                'R2_list': R2_list,
                                'R2_q': R2_q,
                                'R2_q_dot': R2_q_dot,
                                'error': None,}
        except Exception as e:
            self.last_result = {'R2': jnp.nan, 'R2_list': [],
                                'R2_q': jnp.nan,
                                'R2_q_dot': jnp.nan,
                                'predicted_rhs': rhs,
                                'error': str(e),}
            
        return {'save_message': 'Saved predicted rhs and params. You can now stop the experiment.'}

    #  This needs to be consistent with all specific models and the task!
    def save_result_model_distinction_1d(self, model_type:str) -> dict:
        """
        Save the model type that the agent has distinguished.
        This tool should only be used to provide the final result. It can only be called once per experiment.
        Args:
            model_type: The model type that the agent has distinguished. 
                Should exactly match one of the following:
                'Harmonic oscillator'
                'Pendulum'
                'Particle in double well potential'
                'Parametric oscillator'
        Returns:
            save_message:str A message that the prediction has been saved.
        """
        if model_type.strip() == self.model_type:
            correct = True
        else:
            correct = False
        self.last_result = {'correct': correct, 
                            'predicted_model_type': model_type, 
                            'true_model_type': self.model_type,
                            'error': None}
        return {'save_message': 'Saved model type. You can now stop the experiment.'}
    
    #  This needs to be consistent with all specific models and the task!
    def save_result_model_distinction_2d(self, model_type:str) -> dict:
        """
        Save the model type that the agent has distinguished.
        This tool should only be used to provide the final result. It can only be called once per experiment.
        Args:
            model_type: The model type that the agent has distinguished. 
                Should exactly match one of the following:
                'Particle in two dimensional gravitational potential'
                'Two coupled harmonic oscillators'
                'Double pendulum'
        Returns:
            save_message:str A message that the prediction has been saved.
        """
        if model_type.strip() == self.model_type:
            correct = True
        else:
            correct = False
        self.last_result = {'correct': correct, 
                            'predicted_model_type': model_type, 
                            'true_model_type': self.model_type,
                            'error': None}
        return {'save_message': 'Saved model type. You can now stop the experiment.'}
    

    def observe_multiple_evolutions_6d(self,
                                       q0s: str, q1s: str, q2s: str, q3s: str, q4s: str, q5s: str,
                                       q0_dots: str, q1_dots: str, q2_dots: str, q3_dots: str, q4_dots: str, q5_dots: str) -> dict:
        """
        SHOULD ONLY BE CALLED IN 2D THREE-PARTICLE SYSTEM FUNCTION, NOT AS TOOL.
        """
        try:
            q0_inits = jnp.array(ast.literal_eval(q0s))
            q0_dot_inits = jnp.array(ast.literal_eval(q0_dots))
            q1_inits = jnp.array(ast.literal_eval(q1s))
            q1_dot_inits = jnp.array(ast.literal_eval(q1_dots))
            q2_inits = jnp.array(ast.literal_eval(q2s))
            q2_dot_inits = jnp.array(ast.literal_eval(q2_dots))
            q3_inits = jnp.array(ast.literal_eval(q3s))
            q3_dot_inits = jnp.array(ast.literal_eval(q3_dots))
            q4_inits = jnp.array(ast.literal_eval(q4s))
            q4_dot_inits = jnp.array(ast.literal_eval(q4_dots))
            q5_inits = jnp.array(ast.literal_eval(q5s))
            q5_dot_inits = jnp.array(ast.literal_eval(q5_dots))
        except Exception as e:
            return {'error': f"Error in parsing initial conditions: {e}"}
        
        n_trajs = len(q0_inits)
        if n_trajs > 5:
            return {'error': "Error: The maximum number of evolutions is 5."}
        
        self.ode_solver.set_rhs("mystery_rhs", self.ode_rhs)
        ts_solutions = jnp.zeros((n_trajs, int(self.T/self.dt)+1))
        Xs_solutions = jnp.zeros((n_trajs, int(self.T/self.dt)+1, self.ode_dim))
        
        for i in range(n_trajs):
            solution = self.observe_evolution(float(q0_inits[i]), float(q1_inits[i]), float(q2_inits[i]), float(q3_inits[i]), float(q4_inits[i]), float(q5_inits[i]),
                                              float(q0_dot_inits[i]), float(q1_dot_inits[i]), float(q2_dot_inits[i]), float(q3_dot_inits[i]), float(q4_dot_inits[i]), float(q5_dot_inits[i]))
            if 'error' in solution.keys():
                self.current_experiments -= (i + 1)
                return {'error': solution['error']}
            ts_solutions = ts_solutions.at[i].set(solution['ts'])
            Xs_solutions = Xs_solutions.at[i].set(solution['array'])
        
        return {
            'tss': ts_solutions,
            'arrays': Xs_solutions,
        }
    

    def observe_multiple_evolutions_3d(self, q0s: str, q1s: str, q2s: str,
                                        q0_dots: str, q1_dots: str, q2_dots: str) -> dict:
        """
        SHOULD ONLY BE CALLED IN 1D SYSTEM FUNCTION NOT AS TOOL.
        Evolves multiple trajectories of a 6D system (three oscillators).
        """
        try:
            q0_inits = jnp.array(ast.literal_eval(q0s))
            q0_dot_inits = jnp.array(ast.literal_eval(q0_dots))
            q1_inits = jnp.array(ast.literal_eval(q1s))
            q1_dot_inits = jnp.array(ast.literal_eval(q1_dots))
            q2_inits = jnp.array(ast.literal_eval(q2s))
            q2_dot_inits = jnp.array(ast.literal_eval(q2_dots))
        except Exception as e:
            return {'error': f"Error in parsing initial conditions: {e}"}

        n_trajs = len(q0_inits)
        if n_trajs > 5:
            return {'error': "Error: The maximum number of evolutions is 5."}   

        self.ode_solver.set_rhs("mystery_rhs", self.ode_rhs)

        ts_solutions = jnp.zeros((n_trajs, int(self.T/self.dt)+1))  
        Xs_solutions = jnp.zeros((n_trajs, int(self.T/self.dt)+1, self.ode_dim))  

        for i in range(n_trajs):
            solution = self.observe_evolution_3d(
                float(q0_inits[i]), float(q1_inits[i]), float(q2_inits[i]),
                float(q0_dot_inits[i]), float(q1_dot_inits[i]), float(q2_dot_inits[i])
            )
            if 'error' in solution.keys():
                self.current_experiments -= (i + 1)
                return {'error': solution['error']}

            ts_solutions = ts_solutions.at[i].set(solution['ts'])
            Xs_solutions = Xs_solutions.at[i].set(solution['array'])

        return {
            'tss': ts_solutions,
            'arrays': Xs_solutions,
        }
    def observe_evolution_full(self, initial_conditions: list) -> dict:
        """
        DONT USE AS TOOL
        Evolve one trajectory in a system of arbitrary dimension.
        """
        if len(initial_conditions) != self.ode_dim:
            return {'error': f"Error: Initial conditions must be of length {self.ode_dim}."}
        
        self.ode_solver.set_rhs("mystery_rhs", self.ode_rhs)

        solution = self.ode_solver.solve_ode(
            'mystery_rhs',
            str([float(ic) for ic in initial_conditions]),
            str(self.params),
            self.T,
            self.dt
        )

        if 'error' in solution.keys():
            return {'error': solution['error']}

        self.current_experiments += 1
        if self.current_experiments > self.max_experiments:
            return {'error': f"Error: Maximum number of evolutions ({self.max_experiments}) exceeded."}

        return {
            'ts': solution['ts'],
            'array': solution['array']
        }
    def observe_multiple_evolutions_full(self, initial_conditions_list: list) -> dict:
        """
        SHOULD ONLY BE CALLED IN SYSTEM FUNCTION NOT AS TOOL.
        Evolve multiple trajectories in a system of arbitrary dimension.
        """
        n_trajs = len(initial_conditions_list)
        if n_trajs > 5:
            return {'error': "Error: The maximum number of evolutions is 5."}   

        self.ode_solver.set_rhs("mystery_rhs", self.ode_rhs)

        ts_solutions = jnp.zeros((n_trajs, int(self.T/self.dt)+1))  
        Xs_solutions = jnp.zeros((n_trajs, int(self.T/self.dt)+1, self.ode_dim))  

        for i in range(n_trajs):
            initial_conditions = initial_conditions_list[i]
            if len(initial_conditions) != self.ode_dim:
                self.current_experiments -= i
                return {'error': f"Error: Initial conditions must be of length {self.ode_dim}."}
            
            solution = self.observe_evolution_full(initial_conditions)
            if 'error' in solution.keys():
                self.current_experiments -= (i + 1)
                return {'error': solution['error']}

            ts_solutions = ts_solutions.at[i].set(solution['ts'])
            Xs_solutions = Xs_solutions.at[i].set(solution['array'])

        return {
            'tss': ts_solutions,
            'arrays': Xs_solutions,
        }
    def observe_evolution_3d(self, q0: float, q1: float, q2: float,
                             q0_dot: float, q1_dot: float, q2_dot: float) -> dict:
        """
        DONT USE AS TOOL
        Evolve one trajectory in a 6D system (three oscillators).
        """
        self.ode_solver.set_rhs("mystery_rhs", self.ode_rhs)

        solution = self.ode_solver.solve_ode(
            'mystery_rhs',
            str([float(q0), float(q1), float(q2),
                 float(q0_dot), float(q1_dot), float(q2_dot)]),
            str(self.params),
            self.T,
            self.dt
        )

        if 'error' in solution.keys():
            return {'error': solution['error']}

        self.current_experiments += 1
        if self.current_experiments > self.max_experiments:
            return {'error': f"Error: Maximum number of evolutions ({self.max_experiments}) exceeded."}

        return {
            'ts': solution['ts'],
            'array': solution['array']
        }

    

    def observe_evolution_6d(self,
                             q0: float, q1: float, q2: float, q3: float, q4: float, q5: float,
                             q0_dot: float, q1_dot: float, q2_dot: float, q3_dot: float, q4_dot: float, q5_dot: float) -> dict:
        """
        DONT USE AS TOOL
        """
        self.ode_solver.set_rhs("mystery_rhs", self.ode_rhs)
        solution = self.ode_solver.solve_ode(
            'mystery_rhs',
            str([float(q0), float(q1), float(q2), float(q3), float(q4), float(q5),
                 float(q0_dot), float(q1_dot), float(q2_dot), float(q3_dot), float(q4_dot), float(q5_dot)]),
            str(self.params),
            self.T,
            self.dt
        )
        if 'error' in solution.keys():
            return {'error': solution['error']}
        self.current_experiments += 1
        if self.current_experiments > self.max_experiments:
            return {'error': f"Error: Maximum number of evolutions ({self.max_experiments}) exceeded."}
        return {
            'ts': solution['ts'],
            'array': solution['array']
        }


        
class VelocityPosCoupling(MysterySystem):
    """This system is governed by the ode \ddot{x} = a * x^2\dot{x} - b * x"""
    def __init__(self, 
                 a:float=1.0, # parameter a
                 b:float=1.0, # parameter b
                 seed:int=0, # seed for random initial conditions
                 give_model_info:bool = False, # Tell LLM that this is pendulum
                 give_potential_info: bool = False, # Tell LLM that this is moving in conservative potential
                 eval_seed=42, # seed for evaluation
                 min_random = -1, # minimum for random inits
                 max_random = 1, # maximum for rnadom minits
                 T = 20., # the time until which the equation should be solved
                 dt = 0.1, # the time step size for the solution
                 solver_steps_per_timestep=100, # the number of steps per time step to use in the solver
                 max_experiments=None, # maximum number of experiments to run, if None, no limit
                    ):
        self.a = float(a)
        self.b = float(b)

        self.toolboxes = {"all": [self.observe_evolution]}

        ode_rhs = f"""def rhs(X,t,params):
    a, b = params
    return jnp.array([X[1], -a * X[0]**2 * X[1] - b * X[0]])"""
        
        description = "This is a physical system governed by an ordinary differential equation."
        if give_model_info:
            description = "This mystery system is a system with velocity-position coupling. You don't know the following parameters: [a, b]. " \
            "The system is governed by the equation: \ddot{x} = a * x^2\dot{x} - b * x."
        

        super().__init__(ode_rhs=ode_rhs,
                 seed=seed,
                 min_random=min_random,
                 max_random=max_random,
                 T=T,
                 dt=dt,
                 solver_steps_per_timestep=solver_steps_per_timestep,
                 ode_dim=2,
                 true_distinguish_func=None,  # No true distinguish function for this system
                 eval_seed=eval_seed,
                 description=description,
                 params=[self.a, self.b],
                 model_type='Velocity-Position Coupling',
                 max_experiments=max_experiments
                 )
    def observe_evolution(self, q_init: float, q_dot_init: float) -> dict:
        """
        Observe the evolution of the mystery system. You can control the initial conditions of the ordinary differential equation.
        A reasonable coordinate range is e.g. (-1, 1).
        Args:
            q_init: initial generalized coordinate
            q_dot_init: initial generalized velocity
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,dimension] with the solution
                    array[:, 0] holds the coordinate and array[:, 1] its velocity.
        """

        return self.observe_evolution_1d(q_init, q_dot_init)
    
    def observe_multiple_evolutions(self, q_inits: str, q_dot_inits: str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. (-3, 3).
        Args:
            q_inits: list of initial generalized coordinates, in the form '[value1, value2, ...]'
            q_dot_inits: list of initial generalized velocities, in the form '[value1, value2, ...]'
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, dimension] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        return self.observe_multiple_evolutions_1d(q_inits, q_dot_inits)

    
    


class MysteryPendulumInit(MysterySystem):
    """
This mystery system is a pendulum. The angent can control the initial conditions of the ODE.
A suitable task is to ask the agent to dioscover the systems eom."""

    def __init__(self, 
                 alpha:float=1.0, #g/L parameter of the pendulum, which is the ratio of gravitational acceleration g and length L.
                 seed:int=0, # seed for random initial conditions
                 omega:int=1.0, # frequency of sin potential
                 gamma:float=0.0, # damping parameter
                 give_model_info:bool = False, # Tell LLM that this is pendulum
                 give_potential_info: bool = False, # Tell LLM that this is moving in conservative potential
                 eval_seed=42, # seed for evaluation
                 min_random = -3, # minimum for random inits
                 max_random = 3, # maximum for rnadom minits
                 T = 20., # the time until which the equation should be solved
                 dt = 0.1, # the time step size for the solution
                 solver_steps_per_timestep=100, # the number of steps per time step to use in the solver
                 max_experiments=None, # maximum number of experiments to run, if None, no limit
                 ):
        
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.gamma = float(gamma)
        self.toolboxes = {"all": [self.observe_evolution]}
                
        ode_rhs = f"""
def rhs(X,t,params):
    alpha, omega, gamma = params
    return jnp.array([X[1], -alpha*jnp.sin(omega * X[0]) - gamma * X[1]])
"""
        
        def true_distinguish_func(X0: jax.Array) -> int:
            """
            Calculates the energy of a pendulum to distinguish between bound and traveling states.

            Parameters:
            X0 (jax.Array): A 2-element array [theta, omega].
            alpha (float): The value of g/l.

            Returns:
            int: 0 if bound (oscillating), 1 if traveling (rotating).
            """
            theta, omega = X0
            energy = 0.5 * omega**2 - self.alpha * jnp.cos(theta)

            return int(energy > self.alpha)  # 0 = bound, 1 = traveling
        
        description = "This is a physical system governed by an ordinary differential equation." 
        if give_potential_info:
            description = "This is a physical system moving in an unkown but time-independent potential."
        if give_model_info:
            description = "This mystery system is a pendulum. You don't know the following parameters: [length]. " \
            " It may be damped by a term proportional to the velocity." \
            "The gravitiational acceleration and mass are set to 1.0."
            assert omega == 1.0, "The omega parameter must be set to 1.0 if you want to discover pendulum length"
        
        super().__init__(ode_rhs=ode_rhs,
                 seed=seed,
                 min_random=min_random,
                 max_random=max_random,
                 T=T,
                 dt=dt,
                 solver_steps_per_timestep=solver_steps_per_timestep,
                 ode_dim=2,
                 true_distinguish_func=true_distinguish_func,
                 eval_seed=eval_seed, 
                 description=description,
                 params = [self.alpha, self.omega, self.gamma],
                 model_type='Pendulum',
                 max_experiments=max_experiments
                 )
        
    def observe_multiple_evolutions(self, q_inits: str, q_dot_inits: str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. (-3, 3).
        Args:
            q_inits: list of initial generalized coordinates, in the form '[value1, value2, ...]'
            q_dot_inits: list of initial generalized velocities, in the form '[value1, value2, ...]'
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, dimension] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        return self.observe_multiple_evolutions_1d(q_inits, q_dot_inits)
        
        
        
    def observe_evolution(self, q_init: float, q_dot_init: float) -> dict:
        """
        Observe the evolution of the mystery system. You can control the initial conditions of the ordinary differential equation.
        A reasonable coordinate range is e.g. (-3, 3).
        Args:
            q_init: initial generalized coordinate
            q_dot_init: initial generalized velocity
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,dimension] with the solution
                    array[:, 0] holds the coordinate and array[:, 1] its velocity.
        """

        return self.observe_evolution_1d(q_init, q_dot_init)
    
    def save_estimated_parameters(self, length: float) -> dict:
        """
        Compare the provided length with the true length of the pendulum.
        The loss is computed as the mean squared error between the two lengths.
        This tool should only be used to provide the final result. It can only be called once per experiment.
        Args:
            length: The length of the pendulum as predicted by the agent.
        Returns:
            'save_message':str A message that the prediction has been saved.
        """
        try:
            mse = (1 / self.alpha - float(length)) ** 2
        except ValueError:
            mse = jnp.nan

        self.last_result = {'MSE': mse, 'predicted_length': float(length), 'true_length': 1/self.alpha}

        return {'save_message': 'Saved predicted length. You can now stop the experiment.'}

 
class MysteryAsymmetricDoubleWellInit(MysterySystem):
    """
    NOT YET FULLY IMPLEMENTED!
This mystery system is an asymmetric double well. The angent can control the initial conditions of the ODE.
A suitable task is to ask the agent to dioscover the systems eom."""

    def __init__(self, a: float = 1.0, # positive prefactor of the quartic term in the potential, 
                 b: float=0, # prefactor of the quadratic term in the potential.
                 c: float=0, # prefactor of the linear term
                 gamma: float=0.0, # damping parameter
                 seed=0, # seed for initial conditions
                 min_random=-1.0, # minimum value for random initial conditions
                 max_random=1.0, # maximum value for random initial conditions
                 T=20., # the time until which the equation should be solved
                 dt=0.1, # the time step size for the solution
                 true_distinguish_func=None,  # a function that distinguishes between two qualitative regimes of the system, e.g. bound and traveling states.
                 eval_seed=42, # seed for evaluation, used for random initial conditions
                 give_potential_info: bool = False,
                 give_model_info:bool =False,
                 solver_steps_per_timestep=100,
                 max_experiments=None, # maximum number of experiments to run, if None, no limit
                 ):
        
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.gamma = float(gamma)
        ode_rhs = f"""
def rhs(X,t,params):
    return jnp.array([X[1], -4*params[0]*X[0]**3 + 2*params[1]*X[0] + params[2] - params[3]*X[1]])
"""
        self.toolboxes = {"all": [self.observe_evolution]}
        
        description = "This is a physical system governed by an ordinary differential equation." 
        if give_potential_info:
            self.description = "This is a particle moving in an unkown but time-independent potential."   
        if give_model_info:
            description = "This mystery system is a particle moving in a double well potential of the form V(x) = a * x**4 - b * x**2 - c." \
            " It may be damped by a term proportional to the velocity." \
            " You don't know the following parameters: [a, b, c]. " \
            "The mass of the particle is set to 1.0."
        
        super().__init__(ode_rhs=ode_rhs,
                 seed=seed,
                 min_random=min_random,
                 max_random=max_random,
                 T=T, # the time until which the equation should be solved
                 dt=dt, # the time step size for the solution
                 solver_steps_per_timestep=solver_steps_per_timestep,
                 ode_dim=2,  # the dimension of the ODE, e.g. 2 for a pendulum with angle and angular velocity
                 true_distinguish_func=true_distinguish_func,  # a function that distinguishes between two qualitative regimes of the system, e.g. bound and traveling states.
                 eval_seed=eval_seed, # seed for evaluation, used for random initial conditions
                 description=description, # description of the system )
                 params = [self.a, self.b, self.c, self.gamma],
                 model_type='Particle in asymmetric double well potential',
                 max_experiments=max_experiments, # maximum number of experiments to run, if None, no limit
        )
        
    def observe_evolution(self, q_init: float, q_dot_init: float) -> dict:
        """
        Observe the evolution of the mystery system. You can control the initial conditions of the ordinary differential equation.
        A reasonable range for the coordinate and velocity is e.g. [-1, 1].
        Args:
            q_init: initial generalized coordinate
            q_dot_init: initial generalized velocity
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,dimension] with the solution,
                    array[:, 0] holds the coordinate and array[:, 1] its velocity.
        """
        return self.observe_evolution_1d(q_init, q_dot_init)
    
    def observe_multiple_evolutions(self, q_inits, q_dot_inits):
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. [-1, 1].
        Args:
            q_inits: list of initial generalized coordinates, in the form '[value1, value2, ...]'
            q_dot_inits: list of initial generalized velocities, in the form '[value1, value2, ...]'
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, dimension] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        return self.observe_multiple_evolutions_1d(q_inits, q_dot_inits)
    
    def save_estimated_parameters(self, a: float, b:float, c:float) -> dict:
        """
        Compare the provided values of a and b with the true parameters.
        The loss is computed as the mean squared error between the true and predicted parameterts.
        This tool should only be used to provide the final result. It can only be called once per experiment.
        Args:
            a: The (positive) prefactor of the quartic terms in the potential V(x) = a * x**4 - b * x**2 - c * x.
            b: The (positive) prefactor of the quadratic terms in the potential V(x) = a * x**4 - b * x**2 - c * x.
            c: The prefactor of the linear term in the potential V(x) = a * x**4 - b * x**2 - c * x.
        Returns:
            'save_message':str A message that the prediction has been saved.
        """
        try:
            mse = ((float(a) - self.a)**2 + (float(b) - self.b)**2 + (float(c) - self.c)**2) / 3
        except ValueError:
            mse = jnp.nan

        self.last_result = {'MSE': mse, 'predicted_a': float(a), 'predicted_b': float(b), 'predicted_c': float(c),}

        return {'save_message': 'Saved predicted a, b, and c. You can now stop the experiment.'}
    


class MysteryDoubleWellInit(MysterySystem):
    """
This mystery system is a double well. The angent can control the initial conditions of the ODE.
A suitable task is to ask the agent to dioscover the systems eom."""

    def __init__(self, a: float = 1.0, # positive prefactor of the quartic term in the potential, 
                 b: float=0, # prefactor of the quadratic term in the potential.
                 gamma: float=0.0, # damping parameter
                 seed=0, # seed for initial conditions
                 min_random=-1.0, # minimum value for random initial conditions
                 max_random=1.0, # maximum value for random initial conditions
                 T=20., # the time until which the equation should be solved
                 dt=0.1, # the time step size for the solution
                 solver_steps_per_timestep=100, # the number of steps per time step to use in the solver
                 true_distinguish_func=None,  # a function that distinguishes between two qualitative regimes of the system, e.g. bound and traveling states.
                 eval_seed=42, # seed for evaluation, used for random initial conditions
                 give_potential_info: bool = False,
                 give_model_info:bool =False,
                 max_experiments=None, # maximum number of experiments to run, if None, no limit
                 ):
        
        self.a = float(a)
        self.b = float(b)
        self.gamma = float(gamma)
        ode_rhs = f"""
def rhs(X,t,params):
    return jnp.array([X[1], -4*params[0]*X[0]**3 + 2*params[1]*X[0] - params[2]*X[1]])
"""
        self.toolboxes = {"all": [self.observe_evolution]}
        
        def true_distinguish_func(X0: jax.Array) -> int:
            """
            Calculates the energy to distinguish between bound and traveling states.

            Parameters:
            X0 (jax.Array): A 2-element array [theta, omega].
            alpha (float): The value of g/l.

            Returns:
            int: 0 if bound (oscillating), 1 if traveling (rotating).
            """
            x, v = X0
            energy = 0.5 * v**2 + self.a * x**4 - self.b * x**2
            return int(energy > 0.0)
        
        description = "This is a physical system governed by an ordinary differential equation." 
        if give_potential_info:
            self.description = "This is a particle moving in an unkown but time-independent potential."   
        if give_model_info:
            description = "This mystery system is a particle moving in a double well potential of the form V(x) = a * x**4 - b * x**2." \
            " It may be damped by a term proportional to the velocity." \
            " You don't know the following parameters: [a, b]. " \
            "The mass of the particle is set to 1.0."
        
        super().__init__(ode_rhs=ode_rhs,
                 seed=seed,
                 min_random=min_random,
                 max_random=max_random,
                 T=T, # the time until which the equation should be solved
                 dt=dt, # the time step size for the solution
                 solver_steps_per_timestep=solver_steps_per_timestep,
                 ode_dim=2,  # the dimension of the ODE, e.g. 2 for a pendulum with angle and angular velocity
                 true_distinguish_func=true_distinguish_func,  # a function that distinguishes between two qualitative regimes of the system, e.g. bound and traveling states.
                 eval_seed=eval_seed, # seed for evaluation, used for random initial conditions
                 description=description, # description of the system )
                 params = [self.a, self.b, self.gamma],
                 model_type='Particle in double well potential',
                 max_experiments=max_experiments, # maximum number of experiments to run, if None, no limit
        )
        
    def observe_evolution(self, q_init: float, q_dot_init: float) -> dict:
        """
        Observe the evolution of the mystery system. You can control the initial conditions of the ordinary differential equation.
        A reasonable range for the coordinate and velocity is e.g. [-1, 1].
        Args:
            q_init: initial generalized coordinate
            q_dot_init: initial generalized velocity
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,dimension] with the solution,
                    array[:, 0] holds the coordinate and array[:, 1] its velocity.
        """
        return self.observe_evolution_1d(q_init, q_dot_init)
    
    def observe_multiple_evolutions(self, q_inits: str, q_dot_inits: str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. [-1, 1]..
        Args:
            q_inits: list of initial generalized coordinates, in the form '[value1, value2, ...]'
            q_dot_inits: list of initial generalized velocities, in the form '[value1, value2, ...]'
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, dimension] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        return self.observe_multiple_evolutions_1d(q_inits, q_dot_inits)
    
    def save_estimated_parameters(self, a: float, b:float) -> dict:
        """
        Compare the provided values of a and b with the true parameters.
        The loss is computed as the mean squared error between the true and predicted parameterts.
        This tool should only be used to provide the final result. It can only be called once per experiment.
        Args:
            a: The (positive) prefactor of the quartic terms in the potential V(x) = a * x**4 - b * x**2.
            b: The (positive) prefactor of the quadratic terms in the potential V(x) = a * x**4 - b * x**2.
        Returns:
            'save_message':str A message that the prediction has been saved.
        """
        try:
            mse = ((float(a) - self.a)**2 + (float(b) - self.b)**2) / 2
        except ValueError:
            mse = jnp.nan

        self.last_result = {'MSE': mse, 'predicted_a': float(a), 'predicted_b': float(b)}

        return {'save_message': 'Saved predicted a and b. You can now stop the experiment.'}

        
class DoublePendulumInit(MysterySystem):
    """
This mystery system is a double pendulum. The angent can control the initial conditions of the ODE.
A suitable task is to ask the agent to dioscover the systems eom."""

    def __init__(self, 
                 l1:float=1.0, # length of upper pendulum
                 l2:float=1.0, # length of lower pendulum
                 gamma:float=0.0, # damping parameter
                 seed:int=0, # seed for random initial conditions
                 give_model_info:bool = False, # Tell LLM that this is pendulum
                 eval_seed=42, # seed for evaluation
                 min_random = -2, # minimum for random inits
                 max_random = 2, # maximum for rnadom minits
                 T = 20., #20., # the time until which the equation should be solved
                 dt = 0.1, # 0.0001, # the time step size for the solution
                 solver_steps_per_timestep=100, # the number of steps per time step to use in the solver
                 max_experiments=None, # maximum number of experiments to run, if None, no limit
                 ):
        
        #self.m1 = float(m1)
        self.m1 = 1.0
        self.m2 = 1.0
        self.g = 1.0

        self.l1 = float(l1)
        self.l2 = float(l2)
        self.gamma = float(gamma)

        # index 1 is the upper pendulum, index 2 is the lower pendulum
        ode_rhs = '''def rhs(X, t, params):

    theta1, theta2, omega1, omega2 = X
    m1, m2, l1, l2, g, gamma = params

    delta = theta1 - theta2
    
    den1 = l1 * (2 * m1 + m2 - m2 * jnp.cos(2 * delta))
    den2 = l2 * (2 * m1 + m2 - m2 * jnp.cos(2 * delta))

    domega1 = (-g * (2 * m1 + m2) * jnp.sin(theta1) -
               m2 * g * jnp.sin(theta1 - 2 * theta2) -
               2 * jnp.sin(delta) * m2 * (omega2 ** 2 * l2 + omega1 ** 2 * l1 * jnp.cos(delta))) / den1
    
    domega2 = (2 * jnp.sin(delta) * (omega1 ** 2 * l1 * (m1 + m2) + g * (m1 + m2) * jnp.cos(theta1) +
               omega2 ** 2 * l2 * m2 * jnp.cos(delta))) / den2

    return jnp.array([omega1, omega2, domega1 - gamma * omega1, domega2- gamma * omega2])'''
        
        self.toolboxes = {"all": [self.observe_evolution]}
        
        description = "This is a physical system governed by an ordinary differential equation." 
        if give_model_info:
            description = "This system is a double pendulum. You don't know the following parameters: [length_upper_pendulum, length_lower_pendulum]. " \
            "The gravitational acceleration is set to 1.0. The mass of the first and the second pendulum is also set to 1.0." \
            "The coordinates in the observe_evolution function are: " \
            "q0: initial angle of the upper pendulum, q1: initial angle of the lower pendulum, " \
            "q0_dot: initial angular velocity of the upper pendulum, q1_dot: initial angular velocity of the lower pendulum." \
            "The system might be damped by a term proportional to the velocity." \
        
        def true_distinguish_func(X0: jax.Array) -> int:
            # TODO: This is maybe not super accurate yet
            theta_upper, theta_lower, theta_dot_upper, theta_dot_lower = X0.tolist()
            #theta_dot_upper = 0.
            #theta_dot_lower = 0.
            threshold_diff = 0.1

            theta_upper_pert = theta_upper+0.01
            theta_lower_pert = theta_lower+0.01
            theta_dot_upper_pert = theta_dot_upper+0.01
            theta_dot_lower_pert = theta_dot_lower+0.01

            res_orig = self.observe_evolution(theta_upper, theta_lower, theta_dot_upper, theta_dot_lower,) 
            res_pert = self.observe_evolution(theta_upper_pert, theta_lower_pert,
                                               theta_dot_upper_pert, theta_dot_lower_pert,) 
            
            array_orig = res_orig['array']
            array_pert = res_pert['array']

            max_diff = jnp.max(jnp.abs(array_orig - array_pert))

            if max_diff > threshold_diff:
                return 1
            else:
                return 0


        super().__init__(ode_rhs=ode_rhs,
                 seed=seed,
                 min_random=min_random,
                 max_random=max_random,
                 T=T,
                 dt=dt,
                 solver_steps_per_timestep=solver_steps_per_timestep,
                 ode_dim=4,
                 true_distinguish_func=None,
                 eval_seed=eval_seed, 
                 description=description,
                 params = [self.m1, self.m2, self.l1, self.l2, self.g, self.gamma],
                 model_type='Double pendulum',
                 max_experiments=max_experiments, # maximum number of experiments to run, if None, no limit
                 )
        
    def observe_evolution(self, q0: float, q1: float, q0_dot:float, q1_dot:float,) -> dict:
        """
            
        Observe a trajectory of the system given initial conditions.
        A reasonable coordinate range is e.g. (-3, 3).    
        Args:
            q0: first generalized coordinate
            q1: second generalized coordinate
            q0_dot: first generalized velocity
            q1_dot: second generalized velocity
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,dimension] with the solution, 
                    array[:, 0] holds the first generalized coordinate
                    array[:, 1] holds the second generalized coordinate
                    array[:, 2] holds the first generalized velocity
                    array[:, 3] holds the second generalized velocity
        """
        #result_label:str, rhs_label: str, 
        return self.observe_evolution_2d(q0, q1, q0_dot, q1_dot)
    
    def observe_multiple_evolutions(self, q0_inits: str, q1_inits: str, q0_dot_inits: str, q1_dot_inits: str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. (-3, 3).
        Args:
            q0_inits: list of initial generalized coordinates, in the form '[value1, value2, ...]'
            q1_inits: list of initial generalized coordinates, in the form '[value1, value2, ...]'
            q0_dot_inits: list of initial generalized velocities, in the form '[value1, value2, ...]'
            q1_dot_inits: list of initial generalized velocities, in the form '[value1, value2, ...]'
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, dimension] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        return self.observe_multiple_evolutions_2d(q0_inits, q1_inits, q0_dot_inits, q1_dot_inits)
    
    def save_estimated_parameters(self, length_upper_pendulum, length_lower_pendulum) -> dict:
        """
        Compare the predicted lengths with the true lengths of the pendulums.
        The loss is computed as the mean squared error between the lengths.
        This tool should only be used to provide the final result. It can only be called once per experiment.
        Args:
            length_upper_pendulum: The predicted length of the upper pendulum
            length_lower_pendulum: The predicted length of the lower pendulum
        Returns:
            'save_message':str A message that the prediction has been saved.
        """
        try:
            mse = ((float(length_upper_pendulum) - self.l1)**2 + (float(length_lower_pendulum) - self.l2)**2) / 2
        except ValueError:
            mse = jnp.nan

        self.last_result = {'MSE': mse, 
                            'predicted_length_upper_pendulum': float(length_upper_pendulum), 
                            'length_lower_pendulum': float(length_lower_pendulum), 
                            'true_length_upper_pendulum':self.l1,
                            'true_length_lower_pendulum':self.l2,
                            }

        return {'save_message': 'Saved predicted length. You can now stop the experiment.'}
    



class CoupledOscillatorInit(MysterySystem):
    """
This mystery system are two coupled oscillators. The angent can control the initial conditions of the ODE.
A suitable task is to ask the agent to dioscover the systems eom."""

    def __init__(self, 
                 k1:float=1.0, 
                 k2:float=1.0,
                 kc:float=0.1, # coupling constant
                 gamma:float=0.0, # damping parameter
                 seed:int=0, # seed for random initial conditions
                 give_model_info:bool = False, # Tell LLM that this is pendulum
                 eval_seed=42, # seed for evaluation
                 min_random = -1., # minimum for random inits
                 max_random = 1., # maximum for rnadom minits
                 T = 20., #20., # the time until which the equation should be solved
                 dt = 0.1, # 0.0001, # the time step size for the solution
                 solver_steps_per_timestep=100, # the number of steps per time step to use in the solver
                 max_experiments=None, # maximum number of experiments to run, if None, no limit
                 ):
        
        #self.m1 = float(m1)
        self.m1 = 1.0
        self.m2 = 1.0

        self.k1 = float(k1)
        self.k2 = float(k2)
        self.kc = float(kc)
        self.gamma = float(gamma)

        ode_rhs = '''def rhs(X, t, params):
        k1, k2, kc, gamma = params
        q0, q1, q0_dot, q1_dot = X

        dq0_dot = -k1 * q0 + kc * (q1 - q0) - gamma * q0_dot
        dq1_dot = -k2 * q1 + kc * (q0 - q1) - gamma * q1_dot

        return jnp.array([q0_dot, q1_dot, dq0_dot, dq1_dot])
'''
        
        description = "This is a physical system governed by an ordinary differential equation." 
        if give_model_info:
            description = "This system is two harmonic oscillators coupled by a spring. You don't know the following parameters: [k1, k2, kc]. " \
            "k1 is the spring constant of the first harmonic oscillator. k2 is the spring constant of the second harmonic oscillator. " \
            "kc is the spring constant of the coupling term. " \
            "The masses of the first and the second oscillator are set to 1.0." \
    
        self.toolboxes = {"all": [self.observe_evolution]}
        

        super().__init__(ode_rhs=ode_rhs,
                 seed=seed,
                 min_random=min_random,
                 max_random=max_random,
                 T=T,
                 dt=dt,
                 solver_steps_per_timestep=solver_steps_per_timestep,
                 ode_dim=4,
                 true_distinguish_func=None,
                 eval_seed=eval_seed, 
                 description=description,
                 params = [self.k1, self.k2, self.kc, self.gamma],
                 model_type='Two coupled harmonic oscillators',
                 max_experiments=max_experiments, # maximum number of experiments to run, if None, no limit
                 )
        
    def observe_evolution(self, q0: float, q1: float, q0_dot:float, q1_dot:float,) -> dict:
        """
            
        Observe a trajectory of the system given initial conditions.
        A reasonable coordinate range is e.g. (-1, 1). 
        Args:
            q0: first generalized coordinate
            q1: second generalized coordinate
            q0_dot: first generalized velocity
            q1_dot: second generalized velocity
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,dimension] with the solution, 
                    array[:, 0] holds the first generalized coordinate
                    array[:, 1] holds the second generalized coordinate
                    array[:, 2] holds the first generalized velocity
                    array[:, 3] holds the second generalized velocity
        """
        #result_label:str, rhs_label: str, 
        return self.observe_evolution_2d(q0, q1, q0_dot, q1_dot)
    
    def observe_multiple_evolutions(self, q0_inits: str, q1_inits: str, q0_dot_inits: str, q1_dot_inits: str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. (-1, 1).
        Args:
            q0_inits: list of initial generalized coordinates, in the form '[value1, value2, ...]'
            q1_inits: list of initial generalized coordinates, in the form '[value1, value2, ...]'
            q0_dot_inits: list of initial generalized velocities, in the form '[value1, value2, ...]'
            q1_dot_inits: list of initial generalized velocities, in the form '[value1, value2, ...]'
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, dimension] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        return self.observe_multiple_evolutions_2d(q0_inits, q1_inits, q0_dot_inits, q1_dot_inits)
    
    def save_estimated_parameters(self, k1, k2, kc) -> dict:
        """
        Compare the predicted lengths with the true lengths of the pendulums.
        The loss is computed as the mean squared error between the lengths.
        This tool should only be used to provide the final result. It can only be called once per experiment.
        Args:
            k1: The predicted spring constant of the first harmonic oscillator.
            k2: The predicted spring constant of the second harmonic oscillator.
            kc: The predicted spring constant of the coupling term.
        Returns:
            'save_message':str A message that the prediction has been saved.
        """
        try:
            mse = ((float(k1) - self.k1)**2 + (float(k2) - self.k2)**2 + (float(kc) - self.kc)**2) / 3
        except ValueError:
            mse = jnp.nan

        self.last_result = {'MSE': mse, 
                            'predicted_k1': float(k1), 
                            'predicted_k2': float(k2), 
                            'predicted_kc': float(kc), 
                            'true_k1':self.k1,
                            'true_k2':self.k2,
                            'true_kc':self.kc,
                            }

        return {'save_message': 'Saved predicted length. You can now stop the experiment.'}
    
class ParticleIn2DGravity(MysterySystem):
    """
    A particle moving in a 2D gravitational potential created by a fixed point mass.
    The agent can control the initial conditions of the system.
    A suitable task is to ask the agent to discover the equations of motion or infer the location of the mass.
    """

    def __init__(self,
                 mass: float = 1.0,  # mass of the point mass at (xp0, xp1)
                 xp0: float = 0.0,   # x-coordinate of the point mass
                 xp1: float = 0.0,   # y-coordinate of the point mass
                 sun_radius = 0.01, # return nans if the particle is within this radius of the point mass to avoid divergence
                 seed: int = 0,
                 eval_seed: int = 42,
                 min_random: float = -2.,
                 max_random: float = 2.,
                 T: float = 20.0,
                 dt: float = 0.1,
                 solver_steps_per_timestep: int = 100,
                 give_model_info: bool = False,
                 give_2d_info: bool = True,
                 max_experiments=None,  # maximum number of experiments to run, if None, no limit
                 ):


        self.mass = float(mass)
        self.xp0 = float(xp0)
        self.xp1 = float(xp1)

        self.sun_radius = float(sun_radius)  # radius around the point mass where we return NaNs to avoid divergence

        ode_rhs = f'''def rhs(X, t, params):
    m, xp0, xp1, radius = params
    x0, x1, v0, v1 = X
    dx = x0 - xp0
    dy = x1 - xp1
    r = jnp.sqrt(dx**2 + dy**2)
    r3 = (dx**2 + dy**2)**1.5 + 1e-8
    a0 = - m * dx / r3
    a1 = - m * dy / r3
    result = jnp.array([v0, v1, a0, a1])
    return jnp.where(r < radius, jnp.nan, result)
'''
        self.toolboxes = {"all": [self.observe_evolution]}
        
        if give_2d_info:
            description = "This is single physical system moving in two spatial dimensions with the coordinates: " \
                          "q0: the real space x-coordinate of the moving particle, " \
                          "q1: the real space y-coordinate of the moving particle, " \
                          "q0_dot: the real space x-velocity of the moving particle, " \
                          "q1_dot: the real space y-velocity of the moving particle."
        else:
            description = "This is a physical system governed by an ordinary differential equation."
        if give_model_info:
            description = "A particle moves in a 2D gravitational potential due to a fixed point mass at an unknown location. " \
                          "You do not know the parameters: [mass, xp0, xp1], " \
                          "where mass is the fixed point mass and xp0 and xp1 its coordinates." \
                          "The gravitational constant is 1.0  and the particle's mass is also set to 1.0. " \
                          "The coordinates in the observe_evolution function are: " \
                          "q0: the x-coordinate of the moving particle ," \
                          "q1: the y-coordinate of the moving particle, " \
                          "q0_dot: the x-velocity of the moving particle, " \
                          "q1_dot: the y-velocity of the moving particle."
            
        def true_distinguish_func(X0: jax.Array) -> int:
            '''Return 1 if the particle escapes to infinity (unbound), else 0.'''
            x0, x1, v0, v1 = X0
            dx = x0 - self.xp0
            dy = x1 - self.xp1
            r = jnp.sqrt(dx**2 + dy**2) + 1e-8  # softening
            kinetic_energy = 0.5 * (v0**2 + v1**2)
            potential_energy = - self.mass / r
            total_energy = kinetic_energy + potential_energy
            if total_energy > 0:
                return 1
            else:
                return 0

        super().__init__(
            ode_rhs=ode_rhs,
            seed=seed,
            min_random=min_random,
            max_random=max_random,
            T=T,
            dt=dt,
            solver_steps_per_timestep=solver_steps_per_timestep,
            ode_dim=4,
            true_distinguish_func=true_distinguish_func,
            eval_seed=eval_seed,
            description=description,
            params=[self.mass, self.xp0, self.xp1, self.sun_radius],
            model_type='Particle in two dimensional gravitational potential',
            max_experiments=max_experiments,  # maximum number of experiments to run, if None, no limit
        )

    def observe_evolution(self, q0: float, q1: float, q0_dot: float, q1_dot: float) -> dict:
        """
        Observe a trajectory of the system given initial conditions.
        A reasonable coordinate range is e.g. (-2., 2.).  
        Args:
            q0: first generalized coordinate
            q1: second generalized coordinate
            q0_dot: first generalized velocity
            q1_dot: second generalized velocity
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,dimension] with the solution, 
                    array[:, 0] holds the first generalized coordinate
                    array[:, 1] holds the second generalized coordinate
                    array[:, 2] holds the first generalized velocity
                    array[:, 3] holds the second generalized velocity
        """
        return self.observe_evolution_2d(q0, q1, q0_dot, q1_dot)

    def observe_multiple_evolutions(self, q0_inits: str, q1_inits: str, q0_dot_inits: str, q1_dot_inits: str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. (-2, 2).
        Args:
            q0_inits: list of initial generalized coordinates, in the form '[value1, value2, ...]'
            q1_inits: list of initial generalized coordinates, in the form '[value1, value2, ...]'
            q0_dot_inits: list of initial generalized velocities, in the form '[value1, value2, ...]'
            q1_dot_inits: list of initial generalized velocities, in the form '[value1, value2, ...]'
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, dimension] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        return self.observe_multiple_evolutions_2d(q0_inits, q1_inits, q0_dot_inits, q1_dot_inits)

    def save_estimated_parameters(self, mass, xp0, xp1) -> dict:
        """
        Save and compare the predicted gravitational parameters.

        Args:
            mass: estimated mass
            xp0: estimated x-position of the point mass
            xp1: estimated y-position of the point mass

        Returns:
            'save_message': Confirmation message
        """
        try:
            mse = ((float(mass) - self.mass) ** 2 +
                   (float(xp0) - self.xp0) ** 2 +
                   (float(xp1) - self.xp1) ** 2) / 3
        except ValueError:
            mse = jnp.nan

        self.last_result = {
            'MSE': mse,
            'predicted_mass': float(mass),
            'predicted_xp0': float(xp0),
            'predicted_xp1': float(xp1),
            'true_mass': self.mass,
            'true_xp0': self.xp0,
            'true_xp1': self.xp1,
        }

        return {'save_message': 'Saved predicted parameters. You can now stop the experiment.'}

class DrivenOsci(MysterySystem):
    """
    This mystery system is a driven oscillator. 
    The agent can control the initial conditions of the ODE.
    A suitable task is to ask the agent to discover the system's equation of motion.
    """

    def __init__(self, 
                 omega: float = 1.0,  # natural frequency
                 gamma: float = 0.1,  # damping coefficient
                 epsilon: float = 1.,  # amplitude of drive
                 drive_freq: float = 1.04,  # frequency of drive
                 seed: int = 0,  # random seed for initial conditions
                 eval_seed: int = 42,
                 min_random: float = -1.0,
                 max_random: float = 1.0,
                 T: float = 20.0,
                 dt: float = 0.1,
                 solver_steps_per_timestep: int = 100,  # number of steps per time
                 give_model_info: bool = False,
                 max_experiments=None,  # maximum number of experiments to run, if None, no limit
                 ):

        self.omega = float(omega)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.drive_freq = float(drive_freq)

        ode_rhs = f'''def rhs(X, t, params):
    omega, gamma, epsilon, drive_freq = params
    x, v = X
    a = - omega**2 * x - gamma * v + epsilon * jnp.cos(drive_freq * t)
    return jnp.array([v, a])
'''

        self.toolboxes = {"all": [self.observe_evolution]}
        
        description = "This is a physical system governed by an ordinary differential equation."
        if give_model_info:
            description = (
                "This system is a driven oscillator described by an equation of the form:\n"
                "a = omega**2 * x - gamma * v + epsilon * jnp.cos(drive_freq * t) \n"
                "You don't know the values of the parameters [omega, gamma, epsilon, drive_freq]."
            )

        super().__init__(
            ode_rhs=ode_rhs,
            seed=seed,
            min_random=min_random,
            max_random=max_random,
            T=T,
            dt=dt,
            solver_steps_per_timestep=solver_steps_per_timestep,
            ode_dim=2,
            true_distinguish_func=None,
            eval_seed=eval_seed,
            description=description,
            params=[self.omega, self.gamma, self.epsilon, self.drive_freq],
            model_type='Driven oscillator',
            max_experiments=max_experiments,  # maximum number of experiments to run, if None, no limit
        )

    def observe_evolution(self, q_init: float, q_dot_init: float) -> dict:
        """
        Observe a trajectory of the system given initial conditions.
        A reasonable coordinate range is e.g. [-1, 1].  
        Args:
            q_init: Initial position
            q_dot_init: Initial velocity
        Returns:
            Dict with time series 'ts' and state trajectory 'array'
        """
        return self.observe_evolution_1d(q_init, q_dot_init)
    
    def observe_multiple_evolutions(self, q_inits: str, q_dot_inits: str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. [-1, 1]..
        Args:
            q_inits: list of initial generalized coordinates, in the form '[value1, value2, ...]'
            q_dot_inits: list of initial generalized velocities, in the form '[value1, value2, ...]'
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, dimension] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        return self.observe_multiple_evolutions_1d(q_inits, q_dot_inits)

class ParametricOsci(MysterySystem):
    """
    This mystery system is a parametrically driven oscillator. 
    The agent can control the initial conditions of the ODE.
    A suitable task is to ask the agent to discover the system's equation of motion.
    """

    def __init__(self, 
                 omega: float = 1.0,  # natural frequency
                 gamma: float = 0.1,  # damping coefficient
                 epsilon: float = 1.,  # amplitude of parametric drive
                 drive_freq: float = 1.04,  # frequency of parametric modulation
                 seed: int = 0,  # random seed for initial conditions
                 eval_seed: int = 42,
                 min_random: float = -1.0,
                 max_random: float = 1.0,
                 T: float = 20.0,
                 dt: float = 0.1,
                 solver_steps_per_timestep: int = 100,  # number of steps per time
                 give_model_info: bool = False,
                 max_experiments=None,  # maximum number of experiments to run, if None, no limit
                ):

        self.omega = float(omega)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.drive_freq = float(drive_freq)

        ode_rhs = f'''def rhs(X, t, params):
    omega, gamma, epsilon, drive_freq = params
    x, v = X
    a = - (omega**2 + epsilon * jnp.cos(drive_freq * t)) * x - gamma * v
    return jnp.array([v, a])
'''

        self.toolboxes = {"all": [self.observe_evolution]}
        
        description = "This is a physical system governed by an ordinary differential equation."
        if give_model_info:
            description = (
                "This system is a parametrically driven oscillator described by an equation of the form:\n"
                "d²x/dt² + gamma dx/dt + [omega**2 + epsilon cos(drive_freq t)] x = 0.\n"
                "You don't know the values of the parameters [omega, gamma, epsilon, drive_freq]."
            )

        super().__init__(
            ode_rhs=ode_rhs,
            seed=seed,
            min_random=min_random,
            max_random=max_random,
            T=T,
            dt=dt,
            solver_steps_per_timestep=solver_steps_per_timestep,
            ode_dim=2,
            true_distinguish_func=None,
            eval_seed=eval_seed,
            description=description,
            params=[self.omega, self.gamma, self.epsilon, self.drive_freq],
            model_type='Parametric oscillator',
            max_experiments=max_experiments,  # maximum number of experiments to run, if None, no limit
        )

    def observe_evolution(self, q_init: float, q_dot_init: float) -> dict:
        """
        Observe a trajectory of the system given initial conditions.
        A reasonable coordinate range is e.g. [-1, 1].  
        Args:
            q_init: Initial position
            q_dot_init: Initial velocity
        Returns:
            Dict with time series 'ts' and state trajectory 'array'
        """
        return self.observe_evolution_1d(q_init, q_dot_init)
    
    def observe_multiple_evolutions(self, q_inits: str, q_dot_inits: str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. [-1, 1]..
        Args:
            q_inits: list of initial generalized coordinates, in the form '[value1, value2, ...]'
            q_dot_inits: list of initial generalized velocities, in the form '[value1, value2, ...]'
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, dimension] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        return self.observe_multiple_evolutions_1d(q_inits, q_dot_inits)

    def save_estimated_parameters(self, omega: float, gamma: float, epsilon: float, drive_freq: float) -> dict:
        """
        Save and evaluate estimated parameters.
        Args:
            omega: Estimated natural frequency
            gamma: Estimated damping coefficient
            epsilon: Estimated amplitude of parametric drive
            drive_freq: Estimated frequency of parametric modulation
        Returns message that the parameters have been saved.
        """
        try:
            mse = (
                ((float(omega) - self.omega) ** 2 +
                 (float(gamma) - self.gamma) ** 2 +
                 (float(epsilon) - self.epsilon) ** 2 +
                 (float(drive_freq) - self.drive_freq) ** 2) / 4
            )
        except ValueError:
            mse = jnp.nan

        self.last_result = {
            'MSE': mse,
            'predicted_omega': float(omega),
            'predicted_gamma': float(gamma),
            'predicted_epsilon': float(epsilon),
            'predicted_drive_freq': float(drive_freq),
            'true_omega': self.omega,
            'true_gamma': self.gamma,
            'true_epsilon': self.epsilon,
            'true_drive_freq': self.drive_freq
        }

        return {'save_message': 'Saved predicted parameters. You can now stop the experiment.'}
    

class ParticleInComplicated2DGravity(MysterySystem):
    """
    A particle moving in a 2D gravitational potential created by multiple fixed point mass.
    The agent can control the initial velocity of the particle.
    A suitable task is to ask the agent to move the particle from it's start to a goal region.
    """

    def __init__(self,
                 params:list, # list holding (mass, x-coordinate, y-coordinate, radius) of the planets
                 xp0_init: float = 0.0,   # initial x-coordinate of the point mass
                 xp1_init: float = 0.0,   # initial y-coordinate of the point mass
                 x0_target: float = 0.0,  # target x-coordinate of the particle
                 y0_target: float = 0.0,  # target y-coordinate of the
                 target_distance: float = 0.1,  # distance to the target position to consider it reached
                 seed: int = 0,
                 eval_seed: int = 42,
                 min_random: float = -2.,
                 max_random: float = 2.,
                 T: float = 20.0,
                 dt: float = 0.1,
                 solver_steps_per_timestep: int = 1000,
                 give_model_info: bool = False,
                 give_2d_info: bool = True,
                 max_experiments=None,  # maximum number of experiments to run, if None, no limit
                 ):

        self.xp0_init = float(xp0_init)
        self.xp1_init = float(xp1_init)

        self.x0_target = float(x0_target)
        self.y0_target = float(y0_target)

        self.target_distance = float(target_distance)

        ode_rhs = f'''def rhs(X, t, params):
    x, y, vx, vy = X
    dx = x - params[:, 1]
    dy = y - params[:, 2]
    dist2 = dx**2 + dy**2
    radii2 = params[:, 3]**2

    # Collision if distance^2 < radius^2
    collision = jnp.any(dist2 < radii2)

    r3 = (dist2 + 1e-8)**1.5
    ax = -jnp.sum(params[:, 0] * dx / r3)
    ay = -jnp.sum(params[:, 0] * dy / r3)

    return jnp.array([vx * ~collision,
                      vy * ~collision,
                      ax * ~collision,
                      ay * ~collision,])
'''

        self.toolboxes = {"all": [self.observe_evolution]}
        
        if give_2d_info:
            description = "This is single physical system moving in two spatial dimensions." \
                        f"The target position for the particle is ({self.x0_target}, {self.y0_target})." \
                        f"You should reach a minimum distance of {self.target_distance} to the target position."
                    
        else:
            description = f"This is a physical system governed by an ordinary differential equation." \
                        f"The target position for the particle is ({self.x0_target}, {self.y0_target})." \
                        f"You should reach a minimum distance of {self.target_distance} to the target position."
        if give_model_info:
            description = "A particle moves in a 2D gravitational potential due to multiple point masses at an unknown locations. " \
                          "The gravitational constant is 1.0  and the particle's mass is also set to 1.0. " \
                          "The coordinates in the observe_evolution function are: " \
                          "q0: the x-coordinate of the moving particle ," \
                          "q1: the y-coordinate of the moving particle, " \
                          "q0_dot: the x-velocity of the moving particle, " \
                          "q1_dot: the y-velocity of the moving particle." \
                         f"The target position for the particle is ({self.x0_target}, {self.y0_target})." \
                         f"You should reach a minimum distance of {self.target_distance} to the target position."
        
        super().__init__(
            ode_rhs=ode_rhs,
            seed=seed,
            min_random=min_random,
            max_random=max_random,
            T=T,
            dt=dt,
            solver_steps_per_timestep=solver_steps_per_timestep,
            ode_dim=4,
            true_distinguish_func=None,
            eval_seed=eval_seed,
            description=description,
            params=params,
            model_type='Particle in complicated two dimensional gravitational potential',
            max_experiments=max_experiments,  # maximum number of experiments to run, if None, no limit
        )

    def observe_evolution(self, q0_dot: float, q1_dot: float) -> dict:
        """
        Observe a trajectory of the system given initial conditions.
        A reasonable coordinate range is e.g. (-2., 2.).  
        Args:
            q0_dot: first generalized velocity, maximum absolute value is 2.
            q1_dot: second generalized velocity, maximum absolute value is 2.
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,dimension] with the solution, 
                    array[:, 0] holds the first generalized coordinate
                    array[:, 1] holds the second generalized coordinate
                    array[:, 2] holds the first generalized velocity
                    array[:, 3] holds the second generalized velocity
        """
        assert abs(q0_dot) <= 2.0, "q0_dot must be in the range [-2, 2]"
        assert abs(q1_dot) <= 2.0, "q1_dot must be in the range [-2, 2]"
        self.ode_solver.set_rhs("mystery_rhs", self.ode_rhs)
        solution = self.ode_solver.solve_ode(
            'mystery_rhs',
            str([self.xp0_init, self.xp1_init, q0_dot, q1_dot]),
            str(self.params),
            self.T,
            self.dt
        )
        if 'error' in solution.keys():
            return {'error': solution['error']}
        return {
            'ts': solution['ts'],
            'array': solution['array']
        }
    
    def save_target_distance(self, q0_dot: float, q1_dot: float) -> dict:
        """
        Save the whether the minimum distance to the target position was reached with the given initial velocities.
        Args:
            q0_dot: first generalized velocity
            q1_dot: second generalized velocity
        Returns:
            'save_message': Confirmation message
        """
        solution = self.observe_evolution(q0_dot, q1_dot)
        if 'error' in solution.keys():
            return {'error': solution['error']}
        
        # Calculate the minimum distance to the target position during the trajectory
        array = solution['array']
        distances = jnp.sqrt((array[:, 0] - self.x0_target) ** 2 + (array[:, 1] - self.y0_target) ** 2)
        min_distance = jnp.min(distances)
        if min_distance <= self.target_distance:
            success = True
        else:
            success = False

        self.last_result = {
            'min_distance': min_distance,
            'target_distance': self.target_distance,
            'success': success,
            'initial_velocity_x': q0_dot,
            'initial_velocity_y': q1_dot
        }
        return {
            'save_message': f'Saved result. You can now stop the experiment.'
        }
    

    def observe_random_evolution(self) -> dict:
        """
        Observe a random evolution of the mystery system with random but reasonable initial conditions.
        Args:
            None
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,dimension] with the solution,
                    array[:, 0] holds the coordinate and array[:, 1] its velocity.
        """
        self.key, subkey = jax.random.split(self.key)
        inits = jax.random.uniform(subkey, shape=(self.ode_dim//2,), minval=self.min_random, maxval=self.max_random)

        return self.observe_evolution(*[float(i) for i in inits])

class MexicanHat(MysterySystem):
    """ A particle moving in a 2D Mexican hat potential.
    The agent can control the initial velocities and position of the system.
    """

    def __init__(self,
                 a: float = 1.0,  # parameter of the potential
                 b: float = 1.0,  # parameter of the potential
                 gamma: float = 0.,  # damping coefficient
                 give_2d_info: bool = True,
                 seed: int = 0,
                 eval_seed: int = 42,
                 min_random: float = -1.,
                 max_random: float = 1.,
                 T: float = 20.0,
                 dt: float = 0.1,
                 solver_steps_per_timestep: int = 100,
                 give_model_info: bool = False,
                 max_experiments=None,  # maximum number of experiments to run, if None, no limit
    ):  
        self.a = float(a)
        self.b = float(b)
        self.gamma = float(gamma)
        
        ode_rhs = f'''def rhs(X, t, params):
            x, y, vx, vy = X
            a, b, gamma = params  

            r2 = x**2 + y**2  # r squared
            factor = 2 * a - 4 * b * r2

            fx = factor * x - gamma * vx
            fy = factor * y - gamma * vy

            return jnp.array([vx, vy, fx, fy])'''

        self.toolboxes = {"all": [self.observe_evolution]}
        
        if give_2d_info:
            description = "This is a physical system in two dimensions governed by an ordinary differential equation." \
                                "The first generalized coordinate q0 is the x-coordinate of the particle, " \
                                "the second generalized coordinate q1 is the y-coordinate of the particle, " \
                                "the first generalized velocity q0_dot is the x-velocity of the particle, " \
                                "the second generalized velocity q1_dot is the y-velocity of the particle."
        else:
            description = "This is a physical system governed by an ordinary differential equation."
        if give_model_info:
            description = (
                "This system is a particle moving in a 2D Mexican hat potential. "
                "The potential is given by V(x, y) = -a * (x^2 + y^2) + b * (x^4 + y^4). You don't know the following parameters: [a, b]. "
                "The particle's mass is set to 1.0. "
                "You can control the initial position and velocities of the particle."
            )

        super().__init__(
            ode_rhs=ode_rhs,
            seed=seed,
            min_random=min_random,  
            max_random=max_random,
            T=T,
            dt=dt,
            solver_steps_per_timestep=solver_steps_per_timestep,
            ode_dim=4,
            true_distinguish_func=None,
            eval_seed=eval_seed,
            description=description,
            params=[self.a, self.b, self.gamma],
            model_type='Particle in Mexican hat potential',
            max_experiments=max_experiments,  # maximum number of experiments to run, if None, no limit
        )

    def observe_evolution(self, q0: float, q1: float, q0_dot: float, q1_dot: float) -> dict:
        """
        Observe a trajectory of the system given initial conditions.
        A reasonable coordinate range is e.g. (-1., 1.).  
        Args:
            q0: first generalized coordinate
            q1: second generalized coordinate
            q0_dot: first generalized velocity
            q1_dot: second generalized velocity
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,dimension] with the solution, 
                    array[:, 0] holds the first generalized coordinate
                    array[:, 1] holds the second generalized coordinate
                    array[:, 2] holds the first generalized velocity
                    array[:, 3] holds the second generalized velocity
        """
        return self.observe_evolution_2d(q0, q1, q0_dot, q1_dot)
    
    def observe_multiple_evolutions(self, q0s:str, q1s:str, q0_dots:str, q1_dots:str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. (-1, 1).
        Args:
            q0s: list of first generalized coordinates in the form '[value1, value2, ...]'
            q1s: list of second generalized coordinates in the form '[value1, value2, ...]'
            q0_dots: list of first generalized velocities in the form '[value1, value2, ...]'
            q1_dots: list of second generalized velocities in the form '[value1, value2, ...]'    
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, dimension] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        return self.observe_multiple_evolutions_2d(q0s, q1s, q0_dots, q1_dots)
        
    
        

class ThreePotentialMinima(MysterySystem):
    """ A particle moving in a 1D potential with three minima.
    The agent can control the initial velocities of the system.
    """

    def __init__(self,
                 v_params:list, # list holding (a, b, c) of the potential
                 gamma: float = 0.,  # damping coefficient
                 x0_init: float = 0.0,   # initial x-coordinate of the particle
                 seed: int = 0,
                 eval_seed: int = 42,
                 min_random: float = -5.,
                 max_random: float = 5.,
                 T: float = 20.0,
                 dt: float = 0.02,
                 solver_steps_per_timestep: int = 1000,
                 give_model_info: bool = False,
                 max_experiments=None,  # maximum number of experiments to run, if None, no limit
                 ):

        self.x0_init = float(x0_init)

        self.v_params = [float(param) for param in v_params]  # (a, b, c) parameters of the potential
        self.gamma = float(gamma)

        ode_rhs = f'''def rhs(X, t, params):
    x, v = X
    a, b, c, gamma = params

    dVdx = 6 * a * x**5 - 4 * b * x**3 + 2 * c * x
    force = -dVdx - gamma * v

    return jnp.array([v, force])
    '''

        self.toolboxes = {"all": [self.observe_evolution]}
        
        description = "This is a physical system governed by an ordinary differential equation."
        if give_model_info:
            description = (
                "This system is a particle moving in a 1D potential with three minima. "
                "The potential is given by V(x) = a * x^6 - b * x^4 + c * x^2. "
                "The particle may be subject to a damping force proportional to its velocity with coefficient gamma. "
                "The particle's mass is set to 1.0. "
                "You don't know the following parameters: [a, b, c]. "
                "You can control the initial velocity of the particle. "
            )

        # find the positions of the three minima of the potential:
        a, b, c = self.v_params
        A = 3 * a
        B = -2 * b
        C = c

        discriminant = B**2 - 4 * A * C

        if discriminant < 0:
            raise ValueError("The potential does not have three minima. Please check the parameters a, b, c.")
        else:
            sqrt_disc = np.sqrt(discriminant)
            u1 = (-B + sqrt_disc) / (2 * A)
            u2 = (-B - sqrt_disc) / (2 * A)

            # Accept only positive roots for u = x^2
            u_roots = [u for u in [u1, u2] if u > 0]

            # Step 2: Compute x = ±sqrt(u), add x = 0
            critical_points = [0.0]
            for u in u_roots:
                sqrt_u = np.sqrt(u)
                critical_points.extend([sqrt_u, -sqrt_u])

            # Step 3: Classify critical points using second derivative
            def d2Vdx2(x):
                return 30 * a * x**4 - 12 * b * x**2 + 2 * c
            is_minima = []
            for x in sorted(critical_points):
                d2 = d2Vdx2(x)
                is_minima.append(d2 > 0)
            critical_points = sorted(critical_points) 
            self.minima = sorted([x for x, is_min in zip(critical_points, is_minima) if is_min])
            self.maxima = sorted([x for x, is_min in zip(critical_points, is_minima) if not is_min])
            if len(self.minima) != 3:
                raise ValueError("The potential does not have three minima. Please check the parameters a, b, c.")
            

        super().__init__(
            ode_rhs=ode_rhs,
            seed=seed,
            min_random=min_random,
            max_random=max_random,
            T=T,
            dt=dt,
            solver_steps_per_timestep=solver_steps_per_timestep,
            ode_dim=2,
            true_distinguish_func=None,
            eval_seed=eval_seed,
            description=description,
            params=self.v_params + [self.gamma],
            model_type='Particle in three potential minima',
            max_experiments=max_experiments,  # maximum number of experiments to run, if None, no limit
        )

    def observe_evolution(self, q0_dot: float) -> dict:
        """
        Observe a trajectory of the system given initial conditions.
        A reasonable coordinate range is e.g. (-5., 5.).  
        Args:
            q0_dot: generalized velocity, maximum absolute value is 10.
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,dimension] with the solution, 
                    array[:, 0] holds the first generalized coordinate
                    array[:, 1] holds the first generalized velocity
        """
        assert abs(q0_dot) <= 10.0, "q0_dot must be in the range [-10, 10]"
        return self.observe_evolution_1d(self.x0_init, q0_dot)
    def observe_random_evolution(self):
        """
        Observe a random evolution of the mystery system with random but reasonable initial conditions.
        Args:
            None
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,dimension] with the solution,
                    array[:, 0] holds the coordinate and array[:, 1] its velocity.
        """
        #return self.observe_evolution_random()
        self.key, subkey = jax.random.split(self.key)
        inits = jax.random.uniform(subkey, shape=(self.ode_dim//2,), minval=self.min_random, maxval=self.max_random)

        return self.observe_evolution(*[float(i) for i in inits])
    
    def save_estimated_parameters(self, a: float, b: float, c: float) -> dict:
        """
        Save and evaluate estimated parameters.
        Args:
            a: Estimated parameter a of the potential V(x) = a * x^6 - b * x^4 + c * x^2.
            b: Estimated parameter b of the potential V(x) = a * x^6 - b * x^4 + c * x^2.
            c: Estimated parameter c of the potential V(x) = a * x^6 - b * x^4 + c * x^2.
        Returns:
            'save_message': Confirmation message
        """
        try:
            mse = ((float(a) - self.v_params[0]) ** 2 +
                   (float(b) - self.v_params[1]) ** 2 +
                   (float(c) - self.v_params[2]) ** 2) / 3
        except Exception:
            mse = jnp.nan

        self.last_result = {
            'MSE': mse,
            'predicted_a': float(a),
            'predicted_b': float(b),
            'predicted_c': float(c),
            'true_a': self.v_params[0],
            'true_b': self.v_params[1],
            'true_c': self.v_params[2],
        }

        return {'save_message': 'Saved predicted parameters. You can now stop the experiment.'}
    
    def save_different_regimes_inits(self, q_dot_0: float, q_dot_1:float, q_dot_2:float) -> dict:
        """
        This system has three potential minima.
        Your task is to provide three different initial velocities that lead to settling into three different potential minima.
        Args:
            q_dot_0: generalized velocity for settling in the first stable point
            q_dot_1: generalized velocity for settling in the second stable point
            q_dot_2: generalized velocity for settling in the third stable point
        Returns:
            'save_message': Confirmation message
        """
        # double time to get long-term behavior
        self.T = 2 * self.T
        solution_0 = self.observe_evolution(q_dot_0)
        solution_1 = self.observe_evolution(q_dot_1)
        solution_2 = self.observe_evolution(q_dot_2)
        self.T = self.T / 2  # reset T to original value

        if 'error' in solution_0.keys():
            return {'error': solution_0['error']}
        if 'error' in solution_1.keys():
            return {'error': solution_1['error']}
        if 'error' in solution_2.keys():
            return {'error': solution_2['error']}

        # Calculate the final positions
        final_pos_0 = solution_0['array'][-1, 0]
        final_pos_1 = solution_1['array'][-1, 0]
        final_pos_2 = solution_2['array'][-1, 0]

        # calculate corresponding minima to the final positions
        def find_nearest_minima(final_pos):
            """Find the nearest minimum to the final position."""
            return min(self.minima, key=lambda x: abs(x - final_pos))
        final_min_0 = find_nearest_minima(final_pos_0)
        final_min_1 = find_nearest_minima(final_pos_1)
        final_min_2 = find_nearest_minima(final_pos_2)


        # check that trajectory is in minimum for second half of the trajectory
        def is_in_minimum(solution, final_min):
            """Check if the trajectory is in the minimum for the second half of the trajectory."""
            second_half = solution['array'][len(solution['array']) // 2:, 0]
            # check that trajectory is not crossing any maximum during the second half
            minimum_idx = self.minima.index(final_min)
            if minimum_idx == 0:
                return jnp.all(second_half <= self.maxima[0])  # first minimum is leftmost
            elif minimum_idx == 1:
                return jnp.all(jnp.logical_and( (second_half >= self.maxima[0]), (second_half <= self.maxima[1]) ))
            elif minimum_idx == 2:
                return jnp.all(second_half >= self.maxima[1])
            
        in_min_0 = is_in_minimum(solution_0, final_min_0)
        in_min_1 = is_in_minimum(solution_1, final_min_1)
        in_min_2 = is_in_minimum(solution_2, final_min_2)

        # check whether minima are different
        different_minima = (final_min_0 != final_min_1) and (final_min_0 != final_min_2) and (final_min_1 != final_min_2)

        if bool(different_minima and in_min_0 and in_min_1 and in_min_2):
            n_stable_found = 3
        # check if two different minima were found
        elif ((final_min_0 != final_min_1) and (in_min_0 and in_min_1) or
              (final_min_0 != final_min_2) and (in_min_0 and in_min_2) or
              (final_min_1 != final_min_2) and (in_min_1 and in_min_2)):
            n_stable_found = 2
        elif (in_min_0 or in_min_1 or in_min_2):
            n_stable_found = 1
        else:
            n_stable_found = 0
        
        
        self.last_result = {
            'success': n_stable_found == 3,
            'n_stable_found': n_stable_found,
            'initial_velocity_0': q_dot_0,
            'initial_velocity_1': q_dot_1,
            'initial_velocity_2': q_dot_2,
            'final_position_0': final_pos_0,
            'final_position_1': final_pos_1,
            'final_position_2': final_pos_2,
            'final_minimum_0': final_min_0,
            'final_minimum_1': final_min_1,
            'final_minimum_2': final_min_2,
            'in_minimum_0': in_min_0,
            'in_minimum_1': in_min_1,
            'in_minimum_2': in_min_2,
        }
        return {'save_message': 'Saved initial velocities and their corresponding final positions.'}

        


class TwoParticlesIn2DGravity(MysterySystem):
    """
    Two particles moving in a 2D gravitational potential.
    The agent can control the initial conditions of the system.
    A suitable task is to ask the agent to discover the equations of motion.
    """

    def __init__(self,
                 mass1: float = 1.0,  # mass of the point mass at (xp0, xp1)
                 mass2: float = 1.0,  # mass of the point mass at (xp0, xp1)
                 particle_radius = 0.01, # return nans if the particle is within this radius of the point mass to avoid divergence
                 seed: int = 0,
                 eval_seed: int = 42,
                 min_random: float = -1.,
                 max_random: float = 1.,
                 T: float = 20.0,
                 dt: float = 0.1,
                 solver_steps_per_timestep: int = 100,
                 give_model_info: bool = False,
                 give_2d_info: bool = True,
                 max_experiments=None,  # maximum number of experiments to run, if None, no limit
                 ):


        self.m1 = float(mass1)
        self.m2 = float(mass2)

        self.particle_radius = float(particle_radius)  # radius around the point mass where we return NaNs to avoid divergence

        ode_rhs = f'''def rhs(X, t, params):
    m1, m2, radius = params
    x0, y1, x2, y2, vx0, vy1, vx2, vy2 = X
    dx = x0 - x2
    dy = y1 - y2
    r = jnp.sqrt(dx**2 + dy**2)
    r3 = (dx**2 + dy**2)**1.5 + 1e-8
    ax1 = - m2 * dx / r3
    ay1 = - m2 * dy / r3
    ax2 = m1 * dx / r3
    ay2 = m1 * dy / r3
    result = jnp.array([ vx0, vy1, vx2, vy2, ax1, ay1, ax2, ay2])
    return jnp.where(r < radius, jnp.nan, result)
'''
        self.toolboxes = {"all": [self.observe_evolution]}
        
        description = "This is a physical system governed by an ordinary differential equation." \
        "The system consists of two particles moving in two dimensions."
            
        def true_distinguish_func(X0: jax.Array) -> int:
            """
            Return 1 if the two particles are unbound (escape to infinity), else 0.
            
            X0 = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
            m1, m2 = masses of the particles
            """
            x1, y1, vx1, vy1, x2, y2, vx2, vy2 = X0

            # Relative position
            dx = x1 - x2
            dy = y1 - y2
            r = jnp.sqrt(dx**2 + dy**2) + 1e-8  # softening to avoid div by zero

            # Relative velocity (for reduced-mass system)
            dvx = vx1 - vx2
            dvy = vy1 - vy2
            v_rel2 = dvx**2 + dvy**2

            # Reduced mass
            mu = (self.m1 * self.m2) / (self.m1 + self.m2)

            # Energy in center-of-mass frame
            kinetic_energy = 0.5 * mu * v_rel2
            potential_energy = - (self.m1 * self.m2) / r  # G=1 assumed
            total_energy = kinetic_energy + potential_energy

            return jnp.where(total_energy > 0, 1, 0)

        super().__init__(
            ode_rhs=ode_rhs,
            seed=seed,
            min_random=min_random,
            max_random=max_random,
            T=T,
            dt=dt,
            solver_steps_per_timestep=solver_steps_per_timestep,
            ode_dim=8,
            true_distinguish_func=true_distinguish_func,
            eval_seed=eval_seed,
            description=description,
            params=[self.m1, self.m2, self.particle_radius],
            model_type='Particle in two dimensional gravitational potential',
            max_experiments=max_experiments,  # maximum number of experiments to run, if None, no limit
        )

    def observe_evolution(self, x1: float, y1: float, x2:float, y2:float, vx1 :float, vy1:float, vx2:float, vy2:float) -> dict:
        """
        Observe a trajectory of the system given initial conditions.
        A reasonable coordinate range is e.g. (-1., 1.).  
        Args:
            x1: first particle's initial x-coordinate
            y1: first particle's initial y-coordinate
            x2: second particle's initial x-coordinate
            y2: second particle's initial y-coordinate
            vx1: first particle's initial x-velocity
            vy1: first particle's initial y-velocity
            vx2: second particle's initial x-velocity
            vy2: second particle's initial y-velocity
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,dimension] with the solution, 
                    array[:, 0] holds the first particle's x-coordinate
                    array[:, 1] holds the first particle's y-coordinate
                    array[:, 2] holds the second particle's x-coordinate
                    array[:, 3] holds the second particle's y-coordinate
                    array[:, 4] holds the first particle's x-velocity
                    array[:, 5] holds the first particle's y-velocity
                    array[:, 6] holds the second particle's x-velocity
                    array[:, 7] holds the second particle's y-velocity
        """
        return self.observe_evolution_4d(x1, y1, x2, y2, vx1, vy1, vx2, vy2)

    def observe_multiple_evolutions(self, x1s:str, y1s:str, x2s:str, y2s:str, vx1s:str, vy1s:str, vx2s:str, vy2s:str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. (-1, 1).
        Args:
            x1s: list of first particle's initial x-coordinates in the form '[value1, value2, ...]'
            y1s: list of first particle's initial y-coordinates in the form '[value1, value2, ...]'
            x2s: list of second particle's initial x-coordinates in the form '[value1, value2, ...]'
            y2s: list of second particle's initial y-coordinates in the form '[value1, value2, ...]'
            vx1s: list of first particle's initial x-velocities in the form '[value1, value2, ...]'
            vy1s: list of first particle's initial y-velocities in the form '[value1, value2, ...]'
            vx2s: list of second particle's initial x-velocities in the form '[value1, value2, ...]'
            vy2s: list of second particle's initial y-velocities in the form '[value1, value2, ...]'    
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, dimension] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        return self.observe_multiple_evolutions_4d(x1s, y1s, x2s, y2s, vx1s, vy1s, vx2s, vy2s)


class Arbitrary2dRHS(MysterySystem):
    """ A particle moving in a 2D potential.
    The agent can control the initial velocities and position of the system.
    """

    def __init__(self,
                 rhs:str,  # string representation of the potential function a(x, y, xd, yd) -> tuple(ax, ay)
                 seed: int = 0,
                 eval_seed: int = 42,
                 min_random: float = -2.,
                 max_random: float = 2.,
                 T: float = 20.0,
                 dt: float = 0.1,
                 solver_steps_per_timestep: int = 100,
                 give_model_info: bool = False,
                 max_experiments=None,  # maximum number of experiments to run, if None, no limit
    ):  

        ode_rhs = f'''def rhs(X, t, params):
            x, y, vx, vy = X
            ax, ay = {rhs}(x, y, vx, vy)
            return jnp.array([vx, vy, ax, ay])'''

        self.toolboxes = {"all": [self.observe_evolution]}
        
        description = "This is a physical system governed by an ordinary differential equation."
        # if give_model_info:
        #     description = (
        #         "This system is a particle moving in a 2D Mexican hat potential. "
        #         "The potential is given by V(x, y) = -a * (x^2 + y^2) + b * (x^4 + y^4). You don't know the following parameters: [a, b]. "
        #         "The particle's mass is set to 1.0. "
        #         "You can control the initial position and velocities of the particle."
        #     )

        super().__init__(
            ode_rhs=ode_rhs,
            seed=seed,
            min_random=min_random,  
            max_random=max_random,
            T=T,
            dt=dt,
            solver_steps_per_timestep=solver_steps_per_timestep,
            ode_dim=4,
            true_distinguish_func=None,
            eval_seed=eval_seed,
            description=description,
            params=[self.a, self.b],
            model_type='Particle in Mexican hat potential',
            max_experiments=max_experiments,  # maximum number of experiments to run, if None, no limit
        )

    def observe_evolution(self, q0: float, q1: float, q0_dot: float, q1_dot: float) -> dict:
        """
        Observe a trajectory of the system given initial conditions.
        A reasonable coordinate range is e.g. (-2., 2.).  
        Args:
            q0: first generalized coordinate
            q1: second generalized coordinate
            q0_dot: first generalized velocity
            q1_dot: second generalized velocity
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,dimension] with the solution, 
                    array[:, 0] holds the first generalized coordinate
                    array[:, 1] holds the second generalized coordinate
                    array[:, 2] holds the first generalized velocity
                    array[:, 3] holds the second generalized velocity
        """
        return self.observe_evolution_2d(q0, q1, q0_dot, q1_dot)
    
    def observe_multiple_evolutions(self, q0s:str, q1s:str, q0_dots:str, q1_dots:str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. (-2, 2).
        Args:
            q0s: list of first generalized coordinates in the form '[value1, value2, ...]'
            q1s: list of second generalized coordinates in the form '[value1, value2, ...]'
            q0_dots: list of first generalized velocities in the form '[value1, value2, ...]'
            q1_dots: list of second generalized velocities in the form '[value1, value2, ...]'    
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, dimension] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        return self.observe_multiple_evolutions_2d(q0s, q1s, q0_dots, q1_dots)


class Arbitrary2dPot(MysterySystem):
    """ A particle moving in a 2D potential.
    The agent can control the initial velocities and position of the system.
    """

    def __init__(self,
                 potential:str,  # string representation of the potential function. E.g. lambda x = x[0]**2 + x[1]**2
                 give_2d_info: bool = True, # if True, the description will contain information about the 2D nature of the systems
                 seed: int = 0,
                 eval_seed: int = 42,
                 min_random: float = -2.,
                 max_random: float = 2.,
                 T: float = 20.0,
                 dt: float = 0.1,
                 solver_steps_per_timestep: int = 100,
                 give_model_info: bool = False,
                 max_experiments=None,  # maximum number of experiments to run, if None, no limit
    ):  
        ode_rhs = f'''def rhs(X, t, params):
            x, y, vx, vy = X
            ax, ay = -jnp.array(jax.grad({potential})((x, y)))
            return jnp.array([vx, vy, ax, ay])'''

        self.toolboxes = {"all": [self.observe_evolution]}
        
        description = "This is a physical system governed by an ordinary differential equation."
        # if give_model_info:
        #     description = (
        #         "This system is a particle moving in a 2D Mexican hat potential. "
        #         "The potential is given by V(x, y) = -a * (x^2 + y^2) + b * (x^4 + y^4). You don't know the following parameters: [a, b]. "
        #         "The particle's mass is set to 1.0. "
        #         "You can control the initial position and velocities of the particle."
        #     )
        if give_2d_info:
            description = "This is single physical system moving in two spatial dimensions with the coordinates: " \
                          "q0: the real space x-coordinate of the moving particle, " \
                          "q1: the real space y-coordinate of the moving particle, " \
                          "q0_dot: the real space x-velocity of the moving particle, " \
                          "q1_dot: the real space y-velocity of the moving particle."

        super().__init__(
            ode_rhs=ode_rhs,
            seed=seed,
            min_random=min_random,  
            max_random=max_random,
            T=T,
            dt=dt,
            solver_steps_per_timestep=solver_steps_per_timestep,
            ode_dim=4,
            true_distinguish_func=None,
            eval_seed=eval_seed,
            description=description,
            params=[],
            model_type='Particle in Mexican hat potential',
            max_experiments=max_experiments,  # maximum number of experiments to run, if None, no limit
        )

    def observe_evolution(self, q0: float, q1: float, q0_dot: float, q1_dot: float) -> dict:
        """
        Observe a trajectory of the system given initial conditions.
        A reasonable coordinate range is e.g. (-2., 2.).  
        Args:
            q0: first generalized coordinate
            q1: second generalized coordinate
            q0_dot: first generalized velocity
            q1_dot: second generalized velocity
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,dimension] with the solution, 
                    array[:, 0] holds the first generalized coordinate
                    array[:, 1] holds the second generalized coordinate
                    array[:, 2] holds the first generalized velocity
                    array[:, 3] holds the second generalized velocity
        """
        return self.observe_evolution_2d(q0, q1, q0_dot, q1_dot)
    
    def observe_multiple_evolutions(self, q0s:str, q1s:str, q0_dots:str, q1_dots:str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. (-2, 2).
        Args:
            q0s: list of first generalized coordinates in the form '[value1, value2, ...]'
            q1s: list of second generalized coordinates in the form '[value1, value2, ...]'
            q0_dots: list of first generalized velocities in the form '[value1, value2, ...]'
            q1_dots: list of second generalized velocities in the form '[value1, value2, ...]'    
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, dimension] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        return self.observe_multiple_evolutions_2d(q0s, q1s, q0_dots, q1_dots)
    



class Arbitrary1dPot(MysterySystem):
    """ A particle moving in a 1D potential.
    The agent can control the initial velocities and position of the system.
    """

    def __init__(self,
                 potential:str,  # string representation of the potential function. E.g. lambda x : x**2
                 seed: int = 0,
                 eval_seed: int = 42,
                 min_random: float = -5.,
                 max_random: float = 5.,
                 T: float = 20.0,
                 dt: float = 0.1,
                 solver_steps_per_timestep: int = 100,
                 give_model_info: bool = False,
                 max_experiments=None,  # maximum number of experiments to run, if None, no limit
    ):  
        ode_rhs = f'''def rhs(X, t, params):
            x, v = X
            a = -jnp.array(jax.grad({potential})(x))
            return jnp.array([v, a])'''

        self.toolboxes = {"all": [self.observe_evolution]}
        
        description = "This is a physical system governed by an ordinary differential equation."
        # if give_model_info:
        #     description = (
        #         "This system is a particle moving in a 2D Mexican hat potential. "
        #         "The potential is given by V(x, y) = -a * (x^2 + y^2) + b * (x^4 + y^4). You don't know the following parameters: [a, b]. "
        #         "The particle's mass is set to 1.0. "
        #         "You can control the initial position and velocities of the particle."
        #     )

        super().__init__(
            ode_rhs=ode_rhs,
            seed=seed,
            min_random=min_random,  
            max_random=max_random,
            T=T,
            dt=dt,
            solver_steps_per_timestep=solver_steps_per_timestep,
            ode_dim=2,
            true_distinguish_func=None,
            eval_seed=eval_seed,
            description=description,
            params=[],
            model_type='arb_1d',
            max_experiments=max_experiments,  # maximum number of experiments to run, if None, no limit
        )

    def observe_evolution(self, q_init: float, q_dot_init: float) -> dict:
        """
        Observe a trajectory of the system given initial conditions.
        A reasonable coordinate range is e.g. [-3, 3].  
        Args:
            q_init: Initial position
            q_dot_init: Initial velocity
        Returns:
            Dict with time series 'ts' and state trajectory 'array'
        """
        return self.observe_evolution_1d(q_init, q_dot_init)
    
    def observe_multiple_evolutions(self, q_inits: str, q_dot_inits: str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. [-3, 3].
        Args:
            q_inits: list of initial generalized coordinates, in the form '[value1, value2, ...]'
            q_dot_inits: list of initial generalized velocities, in the form '[value1, value2, ...]'
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, dimension] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        return self.observe_multiple_evolutions_1d(q_inits, q_dot_inits)


class ThreeParticlesIn2DGravity(MysterySystem):
    """
    Three particles moving in a 2D gravitational potential.
    The agent can control the initial conditions of the system.
    A suitable task is to ask the agent to discover the equations of motion.
    """

    def __init__(self,
                 mass1: float = 1.0,  # mass of particle 1
                 mass2: float = 1.0,  # mass of particle 2
                 mass3: float = 1.0,  # mass of particle 3
                 particle_radius = 0.01,  # return nans if any inter-particle distance < radius (avoid divergence)
                 seed: int = 0,
                 eval_seed: int = 42,
                 min_random: float = -2.,
                 max_random: float = 2.,
                 T: float = 20.0,
                 dt: float = 0.1,
                 solver_steps_per_timestep: int = 100,
                 give_model_info: bool = False,
                 give_2d_info: bool = True,
                 max_experiments=None,  # maximum number of experiments to run, if None, no limit
                 ):

        self.m1 = float(mass1)
        self.m2 = float(mass2)
        self.m3 = float(mass3)

        # radius around any particle where we return NaNs to avoid divergence
        self.particle_radius = float(particle_radius)

        # State ordering is EXACTLY analogous to the 2-body class, just extended:
        # X = [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
        ode_rhs = f'''def rhs(X, t, params):
    m1, m2, m3, radius = params
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = X

    # pairwise displacements
    dx12 = x1 - x2; dy12 = y1 - y2
    dx13 = x1 - x3; dy13 = y1 - y3
    dx23 = x2 - x3; dy23 = y2 - y3

    # distances and r^3 (with softening)
    r12 = jnp.sqrt(dx12**2 + dy12**2)
    r13 = jnp.sqrt(dx13**2 + dy13**2)
    r23 = jnp.sqrt(dx23**2 + dy23**2)

    r12_3 = (dx12**2 + dy12**2)**1.5 + 1e-8
    r13_3 = (dx13**2 + dy13**2)**1.5 + 1e-8
    r23_3 = (dx23**2 + dy23**2)**1.5 + 1e-8

    # accelerations (G = 1)
    # a1 = - m2 * r12 / |r12|^3 - m3 * r13 / |r13|^3
    ax1 = - m2 * dx12 / r12_3 - m3 * dx13 / r13_3
    ay1 = - m2 * dy12 / r12_3 - m3 * dy13 / r13_3

    # a2 = + m1 * r12 / |r12|^3 - m3 * r23 / |r23|^3   (note r23 = r2 - r3)
    ax2 = + m1 * dx12 / r12_3 - m3 * dx23 / r23_3
    ay2 = + m1 * dy12 / r12_3 - m3 * dy23 / r23_3

    # a3 = + m1 * r13 / |r13|^3 + m2 * r23 / |r23|^3   (since r13 = r1 - r3, r23 = r2 - r3)
    ax3 = + m1 * dx13 / r13_3 + m2 * dx23 / r23_3
    ay3 = + m1 * dy13 / r13_3 + m2 * dy23 / r23_3

    result = jnp.array([
        vx1, vy1, vx2, vy2, vx3, vy3,
        ax1, ay1, ax2, ay2, ax3, ay3
    ])

    too_close = (r12 < radius) | (r13 < radius) | (r23 < radius)
    return jnp.where(too_close, jnp.nan, result)
'''

        self.toolboxes = {"all": [self.observe_evolution]}

        description = "This is a physical system governed by an ordinary differential equation." \
        "The system consists of three particles moving in two dimensions."

        def true_distinguish_func(X0: jax.Array) -> int:
            """
            Return 1 if the three-body system is energetically unbound (E > 0), else 0.

            X0 = [x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3]
            m1, m2, m3 = masses of the particles
            """
            x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = X0

            # pairwise distances (softened)
            dx12 = x1 - x2; dy12 = y1 - y2
            dx13 = x1 - x3; dy13 = y1 - y3
            dx23 = x2 - x3; dy23 = y2 - y3
            r12 = jnp.sqrt(dx12**2 + dy12**2) + 1e-8
            r13 = jnp.sqrt(dx13**2 + dy13**2) + 1e-8
            r23 = jnp.sqrt(dx23**2 + dy23**2) + 1e-8

            # kinetic energy
            ke = 0.5*self.m1*(vx1**2 + vy1**2) \
               + 0.5*self.m2*(vx2**2 + vy2**2) \
               + 0.5*self.m3*(vx3**2 + vy3**2)

            # potential energy (G = 1)
            pe = - self.m1*self.m2 / r12 - self.m1*self.m3 / r13 - self.m2*self.m3 / r23

            E = ke + pe
            return jnp.where(E > 0, 1, 0)

        super().__init__(
            ode_rhs=ode_rhs,
            seed=seed,
            min_random=min_random,
            max_random=max_random,
            T=T,
            dt=dt,
            solver_steps_per_timestep=solver_steps_per_timestep,
            ode_dim=12,  # 3 particles: 6 positions + 6 velocities
            true_distinguish_func=true_distinguish_func,
            eval_seed=eval_seed,
            description=description,
            params=[self.m1, self.m2, self.m3, self.particle_radius],
            model_type='Three particles in two dimensional gravitational potential',
            max_experiments=max_experiments,
        )

    def observe_evolution(self,
                          x1: float, y1: float,
                          x2: float, y2: float,
                          x3: float, y3: float,
                          vx1: float, vy1: float,
                          vx2: float, vy2: float,
                          vx3: float, vy3: float) -> dict:
        """
        Observe a trajectory of the system given initial conditions.
        A reasonable coordinate range is e.g. (-2., 2.).
        Args:
            x1, y1: particle 1 initial position
            x2, y2: particle 2 initial position
            x3, y3: particle 3 initial position
            vx1, vy1: particle 1 initial velocity
            vx2, vy2: particle 2 initial velocity
            vx3, vy3: particle 3 initial velocity
        Returns:
            'ts': jax.Array of shape [nsteps] with the time steps
            'array': jax.Array of shape [nsteps, dimension] with the solution,
                     array[:, 0] -> x1
                     array[:, 1] -> y1
                     array[:, 2] -> x2
                     array[:, 3] -> y2
                     array[:, 4] -> x3
                     array[:, 5] -> y3
                     array[:, 6] -> vx1
                     array[:, 7] -> vy1
                     array[:, 8] -> vx2
                     array[:, 9] -> vy2
                     array[:,10] -> vx3
                     array[:,11] -> vy3
        """
        return self.observe_evolution_6d(x1, y1, x2, y2, x3, y3,
                                         vx1, vy1, vx2, vy2, vx3, vy3)

    def observe_multiple_evolutions(self,
                                    x1s: str, y1s: str,
                                    x2s: str, y2s: str,
                                    x3s: str, y3s: str,
                                    vx1s: str, vy1s: str,
                                    vx2s: str, vy2s: str,
                                    vx3s: str, vy3s: str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. (-2, 2).
        Args:
            x1s, y1s: list of particle 1 initial positions '[v1, v2, ...]'
            x2s, y2s: list of particle 2 initial positions '[v1, v2, ...]'
            x3s, y3s: list of particle 3 initial positions '[v1, v2, ...]'
            vx1s, vy1s: list of particle 1 initial velocities
            vx2s, vy2s: list of particle 2 initial velocities
            vx3s, vy3s: list of particle 3 initial velocities
        Returns:
            'tss': jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays': jax.Array of shape [n_evolutions, nsteps, dimension] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        return self.observe_multiple_evolutions_6d(
            x1s, y1s, x2s, y2s, x3s, y3s, vx1s, vy1s, vx2s, vy2s, vx3s, vy3s
        )


class ThreeCoupledOscillatorInit(MysterySystem):
    """
This mystery system is three coupled oscillators with possibly different couplings between neighbors.
The agent can control the initial conditions of the ODE.
A suitable task is to ask the agent to discover the system's equations of motion.
    """

    def __init__(self, 
                 k1: float = 1.0,
                 k2: float = 1.0,
                 k3: float = 1.0,
                 kc12: float = 0.1,  # coupling constant between oscillator 1 and 2
                 kc23: float = 0.1,  # coupling constant between oscillator 2 and 3
                 kc13: float = 0.1,  # no direct coupling between oscillator 1 and 3
                 gamma: float = 0.0,  # damping parameter
                 seed: int = 0,  # seed for random initial conditions
                 give_model_info: bool = False,  # Tell LLM that this is oscillators
                 eval_seed: int = 42,  # seed for evaluation
                 min_random: float = -1.,
                 max_random: float = 1.,
                 T: float = 20.,
                 dt: float = 0.1,
                 solver_steps_per_timestep: int = 100,
                 max_experiments=None,
                 ):

        # masses (all set to 1.0)
        self.m1 = 1.0
        self.m2 = 1.0
        self.m3 = 1.0

        # spring constants
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.k3 = float(k3)
        self.kc12 = float(kc12)
        self.kc23 = float(kc23)
        self.kc13 = float(kc13)
        self.gamma = float(gamma)

        # Right-hand side for 3 coupled oscillators with asymmetric couplings
        ode_rhs = '''def rhs(X, t, params):
        k1, k2, k3, kc12, kc23, kc13, gamma = params
        q0, q1, q2, q0_dot, q1_dot, q2_dot = X

        dq0_dot = -k1 * q0 + kc12 * (q1 - q0) + kc13 * (q2 - q0) - gamma * q0_dot
        dq1_dot = -k2 * q1 + kc12 * (q0 - q1) + kc23 * (q2 - q1) - gamma * q1_dot
        dq2_dot = -k3 * q2 + kc23 * (q1 - q2) + kc13 * (q0 - q2) - gamma * q2_dot

        return jnp.array([q0_dot, q1_dot, q2_dot, dq0_dot, dq1_dot, dq2_dot])
'''

        description = "This is a physical system governed by an ordinary differential equation." 
        if give_model_info:
            description = (
                "This system is three harmonic oscillators coupled in a chain by springs. "
                "You don't know the following parameters: [k1, k2, k3, kc12, kc23, kc13]. "
                "k1, k2, k3 are the spring constants of the three oscillators. "
                "kc12 is the spring constant coupling oscillator 1 and 2, "
                "kc23 is the spring constant coupling oscillator 2 and 3. "
                "The masses of the oscillators are set to 1.0."
            )

        self.toolboxes = {"all": [self.observe_evolution]}

        super().__init__(ode_rhs=ode_rhs,
                 seed=seed,
                 min_random=min_random,
                 max_random=max_random,
                 T=T,
                 dt=dt,
                 solver_steps_per_timestep=solver_steps_per_timestep,
                 ode_dim=6,
                 true_distinguish_func=None,
                 eval_seed=eval_seed, 
                 description=description,
                 params=[self.k1, self.k2, self.k3, self.kc12, self.kc23, self.kc13, self.gamma],
                 model_type='Three coupled harmonic oscillators',
                 max_experiments=max_experiments,
                 )
        
    def observe_multiple_evolutions(self, q0s: str, q1s: str, q2s: str,
                                    q0_dots: str, q1_dots: str, q2_dots: str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. [-1, 1].
        
        Args:
            q0s, q1s, q2s: str — Python-style lists of initial positions in the form '[value1, value2, ...]'
            q0_dots, q1_dots, q2_dots: str — Python-style lists of initial velocities in the form '[value1, value2, ...]'

        Returns:
            dict with keys:
                'tss': shape [n_trajs, nsteps]
                'arrays': shape [n_trajs, nsteps, ode_dim]
        """
        return self.observe_multiple_evolutions_3d(q0s, q1s, q2s, q0_dots, q1_dots, q2_dots)

    def observe_evolution(self, q0: float, q1: float, q2: float,
                          q0_dot: float, q1_dot: float, q2_dot: float) -> dict:
        """
        Observe a trajectory of the system given initial conditions.
        A reasonable coordinate range is e.g. [-1., 1.].


        Args:
            q0, q1, q2: floats — initial generalized positions
            q0_dot, q1_dot, q2_dot: floats — initial generalized velocities

        Returns:
            dict with keys:
                'ts': shape [nsteps]
                'array': shape [nsteps, ode_dim]
        """
        return self.observe_evolution_3d(q0, q1, q2, q0_dot, q1_dot, q2_dot)

        
    
class HiddenOscillators(MysterySystem):
    """
    This mystery system is an array of N coupled oscillators with 1D nearest-neighbour couplings.
    Only the coordinate and velocity of the *first* oscillator are observable and controllable.
    The other N-1 oscillators evolve hidden but influence the dynamics.
    We have periodic boundary conditions.
    """

    def __init__(self, 
                 N: int = 3,  # number of oscillators
                 ks: list = None,   # spring constants [k1, k2, ..., kN]
                 kcs: list = None,  # coupling constants [kc1, kc2, ..., kc_{N}]
                 init_pos: list = None,  # initial positions of hidden oscillators (first oscillator position is controllable)
                 init_vel: list = None,  # initial velocities of hidden oscillators (first oscillator
                 gamma: float = 0.0,  # damping
                 seed: int = 0,
                 give_model_info: bool = False,
                 give_dimension_info: bool = True,
                 hide_N: bool = False,
                 eval_seed: int = 42,
                 min_random: float = -2.,
                 max_random: float = 2.,
                 T: float = 20.,
                 dt: float = 0.1,
                 solver_steps_per_timestep: int = 100,
                 max_experiments=None):
        
        self.N = int(N)
        self.ks = ks if ks is not None else [1.0] * self.N
        self.kcs = kcs if kcs is not None else [0.1] * self.N 
        self.gamma = float(gamma)
        if init_pos is None:
            self.init_pos = [0.0] * (self.N -1)  # Initial positions fixed at 0.0
        else:
            self.init_pos = init_pos
        if init_vel is None:
            self.init_vel = [0.0] * (self.N -1) # Initial velocities fixed at 0.0
        else:
            self.init_vel = init_vel
        assert len(self.init_pos) == self.N -1, "init_pos must have length N-1"
        assert len(self.init_vel) == self.N -1, "init_vel must have length N-1"
        assert len(self.ks) == self.N, "ks must have length N"
        assert len(self.kcs) == self.N, "kcs must have length N"

        # Masses fixed at 1.0
        self.ms = [1.0] * self.N  

        # Build RHS dynamically for N oscillators
        ode_rhs = f"""def rhs(X, t, params):
    ks = params[0:{self.N}]
    kcs = params[{self.N}:2*{self.N}]
    gamma = params[2*{self.N}]
    N = params[2*{self.N}+1]

    qs = X[:{self.N}]
    q_dots = X[{self.N}:]

    # Forces due to spring + damping
    forces = -ks * qs - gamma * q_dots

    # Coupling forces
    left = jnp.roll(qs, 1)  # periodic left neighbor
    right = jnp.roll(qs, -1)  # periodic right neighbor
    coupling = kcs * (right - qs) + jnp.roll(kcs, 1) * (left - qs)

    forces += coupling

    return jnp.concatenate([q_dots, forces])
"""

        description = "This is a physical system governed by an ordinary differential equation. "
        if give_dimension_info:
            if hide_N:
                description = "This system consists of multiple particles moving in one dimension. "
            else:
                description = f"This system consists of {N} particles moving in one dimension. "
            description += "However, you can only observe the position and velocity of the first particle."
            description += "The initial positions and velocities of the hidden particles are the same for each observed evolution."
        else:
            description += "You can only observe one degree of freedom of the system. "
            description += "However, other degrees of freedom might influence the dynamics."
        if give_model_info:
            description = (
                f"This system consists of {N} harmonic oscillators coupled by nearest-neighbour springs. "
                "You only observe the first oscillator (its position and velocity). "
                "The other oscillators are hidden but influence the dynamics."
            )

        # Only observed variables: q0, q0_dot
        self.toolboxes = {"all": [self.observe_evolution]}

        super().__init__(ode_rhs=ode_rhs,
                         seed=seed,
                         min_random=min_random,
                         max_random=max_random,
                         T=T,
                         dt=dt,
                         solver_steps_per_timestep=solver_steps_per_timestep,
                         ode_dim=2*self.N,
                         true_distinguish_func=None,
                         eval_seed=eval_seed,
                         description=description,
                         params=self.ks + self.kcs + [self.gamma, self.N],
                         model_type=f"Hidden {N}-oscillator chain",
                         max_experiments=max_experiments)
        
    def observe_evolution(self, q0: float, q0_dot: float) -> dict:
        """
        Observe a trajectory of the system given initial conditions.
        A reasonable coordinate range is e.g. (-2., 2.).  
        Args:
            q0: first generalized coordinate 
            q0_dot: first generalized velocity
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,2] with the solution, 
                    array[:, 0] holds the first generalized coordinate
                    array[:, 1] holds the first generalized velocity
        """
        init_conditions = [q0] + self.init_pos + [q0_dot] + self.init_vel
        full_traj = self.observe_evolution_full(init_conditions)
        if "error" in full_traj:
            return {"error": full_traj["error"]}
        x_vals = full_traj['array'][:, 0]
        v_vals = full_traj['array'][:, self.N]
        combined = jnp.stack([x_vals, v_vals], axis=1)
        # Return only the observed part (first oscillator)
        return {
            'ts': full_traj['ts'],
            'array': combined
        }
    
    def observe_multiple_evolutions(self, q0s: str, q0_dots: str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. (-2, 2).
        Args:
            q0s: list of first generalized coordinates in the form '[value1, value2, ...]'
            q0_dots: list of first generalized velocities in the form '[value1, value2, ...]'    
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, 2] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        try:
            q0_list = ast.literal_eval(q0s)
            q0_dot_list = ast.literal_eval(q0_dots)
        except Exception as e:
            raise ValueError("Invalid input format. Please provide lists in the form '[value1, value2, ...]'")
        if not (isinstance(q0_list, list) and isinstance(q0_dot_list, list)):
            raise ValueError("Inputs must be lists.")
        if len(q0_list) != len(q0_dot_list):
            raise ValueError("Input lists must have the same length.")
        
        initial_conditions = []
        for q0, q0_dot in zip(q0_list, q0_dot_list):
            init_cond = [q0] + self.init_pos + [q0_dot] + self.init_vel
            initial_conditions.append(init_cond)
        res = self.observe_multiple_evolutions_full(initial_conditions)

        res_array = res["arrays"]

        x_vals = res_array[:, :, 0]
        v_vals = res_array[:, :, self.N]
        combined = jnp.stack([x_vals, v_vals], axis=2)
        return {
            'tss': res['tss'],
            'arrays': combined
        }
    
    def save_result_find_eom_hidden_degrees(self, rhs: str, hidden_initial_qs:str, hidden_initial_q_dots:str) -> dict:
        """
        Compare the provided right-hand side of the ordinary differential equation with the true right-hand side governing the differential equation of the system.
        The loss is computed as the mean squared error between the true rhs and predicted rhs at some randomly sampled points.
        This tool should only be used to provide the final result. It can only be called once per experiment.
        Args:
            rhs: Define the right-hand side of an ordinary differential equation. 
                You pass a python code that must be of the form
                    def rhs(X:jax.Array, t:float) -> jax.Array:
                        ''' Calulates the right-hand side of the ODE. 
                            Make sure to also include the hidden dimensions in X.
                            X is of shape (n_visible + n_hidden) * 2 where the first half contains the coordinates and the second half contains the velocities.
                            E.g. for 1 visible and 1 hidden dimension, X is of shape (2*2,) and of the form jnp.array([q0, q1, q0_dot, q1_dot]), where q0 and q0_dot are the observed quantities.
                            Make sure to adhere to this format. 
                            Args:
                                X:jax.Array containing the generalized coordinates (including the hidden coordinates) followed by their velocities. E.g. jnp.array([q0, q1, q0_dot, q1_dot]).
                                t:float The time variable. Might be used in case system is time-dependent.
                            Returns:
                                The right-hand side of the ODE, a jnp.array of shape n_coordinates * 2.'''
                    rhs may use jnp syntax in the form "jnp.sin(...)". rhs must be jax-jittable!
            hidden_initial_qs: A string representation of a list of initial values for the hidden (not the observed!) generalized coordinates, e.g. '[0.1, ...]' with length n_hidden.
            hidden_initial_q_dots: A string representation of a list of initial values for the hidden (not the observed!) generalized velocities, e.g. '[0.0, ...]' with length n_hidden.
        Returns:    
            save_message:str A message that the prediction has been saved.
        """
        # for now this works only for 1d observed systems with hidden dimensions
        n_random_inits = 100
        T=20
        try:
            hidden_initial_qs = jnp.ravel(jnp.array(ast.literal_eval(hidden_initial_qs)))
            hidden_initial_q_dots = jnp.ravel(jnp.array(ast.literal_eval(hidden_initial_q_dots)))
        except Exception as e:
            self.last_result = {'R2': jnp.nan, 'predicted_rhs': rhs,
                                'R2_list': [],
                                'error': str(e),}
            return {'save_message': 'Saved predicted rhs and params. You can now stop the experiment.'}
        
        self.eval_key, subkey = jax.random.split(self.eval_key)
        random_inits_observed = jax.random.uniform(subkey, shape=(n_random_inits, 2), minval=self.min_random, maxval=self.max_random)
        #random_inits = jnp.concatenate([random_inits_observed[:, 0][:, None], hidden_initial_qs[None, :], random_inits_observed[:, 1][:, None], hidden_initial_q_dots[None, :]], axis=0)
        try:
            self.set_rhs("predicted_rhs", rhs)
            def pred_rhs_wrapper(X, params, t):
                return self.ode_solver.rhs['predicted_rhs'](X, t)
            R2_list = []
            for init in random_inits_observed:
                pred_init = jnp.concatenate([init[0:1], hidden_initial_qs, init[1:2], hidden_initial_q_dots], axis=0)
                true_init = jnp.concatenate([init[0:1], jnp.array(self.init_pos), init[1:2], jnp.array(self.init_vel)], axis=0)

                solutions_pred = ode_solve(pred_init, pred_rhs_wrapper, self.params, self.dt, T,)
                solutions_true = ode_solve(true_init, self.ode_solver.rhs['mystery_rhs'], jnp.array(self.params), self.dt, T)
                pos_list_true = solutions_true[:, 0]
                pos_list_pred = solutions_pred[:, 0]
                non_nan_idcs = jnp.isfinite(pos_list_true)
                # calculate r2 for non-nan values
                R2_i = 1 - jnp.mean((pos_list_true[non_nan_idcs] - pos_list_pred[non_nan_idcs])**2) / (jnp.std(pos_list_true[non_nan_idcs])**2 + 1e-8)
                R2_list.append(float(np.clip(np.nan_to_num(R2_i), 0., 1.)))
            R2 = float(jnp.mean(jnp.array(R2_list)))
            self.last_result = {'R2': R2, 'predicted_rhs': rhs,
                                'R2_list': R2_list,
                                'error': None,}
        except Exception as e:
            self.last_result = {'R2': jnp.nan, 'R2_list': [],
                                'predicted_rhs': rhs,
                                'error': str(e),}
        
        
        return {'save_message': 'Saved predicted rhs and params. You can now stop the experiment.'}


        


import ast
import jax.numpy as jnp

class HiddenNBody2DGravity(MysterySystem):
    """
    N-body 2D gravitational system (G=1).
    Only the first particle (x1, y1, vx1, vy1) is observed and controllable.
    Hidden particles have configurable initial conditions.
    """

    def __init__(self,
                 N: int = 5,
                 masses: list = None,        # list of N masses
                 hidden_init: list = None,   # optional explicit hidden initial conditions
                 particle_radius: float = 0.01,
                 seed: int = 0,
                 eval_seed: int = 42,
                 min_random: float = -3.0,
                 max_random: float = 3.0,
                 T: float = 20.0,
                 dt: float = 0.1,
                 solver_steps_per_timestep: int = 100,
                 hide_N: bool = False,
                 give_model_info: bool = False,
                 give_2d_info: bool = True,
                 max_experiments=None):
        self.N = int(N)
        if self.N < 1:
            raise ValueError("N must be >= 1")

        # masses
        self.ms = masses if masses is not None else [1.0] * self.N
        if len(self.ms) != self.N:
            raise ValueError("masses must be length N")

        self.particle_radius = float(particle_radius)

        # default hidden initial conditions: [x2, y2, ..., xN, yN, vx2, vy2, ..., vxN, vyN]
        if hidden_init is not None:
            expected_len = 4 * (self.N - 1)
            if len(hidden_init) != expected_len:
                raise ValueError(f"hidden_init must be length {expected_len} (got {len(hidden_init)})")
            self.hidden_init = hidden_init
        else:
            self.hidden_init = [0.0] * (4 * (self.N - 1))

        # Build RHS
        ode_rhs = f'''def rhs(X, t, params):
    masses = params[0:{self.N}]
    radius = params[{self.N}]
    N = {self.N}

    pos = X[:{2*self.N}].reshape((N, 2))
    vel = X[{2*self.N}:].reshape((N, 2))

    diffs = pos[:, None, :] - pos[None, :, :]
    r2 = jnp.sum(diffs**2, axis=2)
    r = jnp.sqrt(r2 + 1e-12)

    too_close = jnp.any((r < radius) & (~jnp.eye(N, dtype=bool)))

    r2_3 = r2**1.5 + 1e-8
    inv_r3 = 1.0 / r2_3
    inv_r3 = inv_r3 * (1.0 - jnp.eye(N))

    accs = - jnp.sum((masses[None, :, None] * diffs) * inv_r3[:, :, None], axis=1)

    result = jnp.concatenate([vel.reshape(-1), accs.reshape(-1)])
    return jnp.where(too_close, jnp.nan, result)
'''

        description = f"This is a physical system governed by an ordinary differential equation. "
        if give_2d_info:
            if hide_N:
                description = "This system consists of multiple particles in 2D. " \
                          "Only the first particle (x,y,vx,vy) is observed; the rest are hidden."
            else:
                description = f"This system consists of {self.N} particles in 2D. " \
                          "Only the first particle (x,y,vx,vy) is observed; the rest are hidden."
        if give_model_info:
            description = f"{self.N}-body gravitational system in 2D (G=1). " \
                          "Only the first particle is observed; hidden particles evolve but are not shown."

        self.toolboxes = {"all": [self.observe_evolution]}

        super().__init__(
            ode_rhs=ode_rhs,
            seed=seed,
            min_random=min_random,
            max_random=max_random,
            T=T,
            dt=dt,
            solver_steps_per_timestep=solver_steps_per_timestep,
            ode_dim=4 * self.N,
            true_distinguish_func=None,
            eval_seed=eval_seed,
            description=description,
            params=self.ms + [self.particle_radius],
            model_type=f"One-observed of {self.N}-body 2D gravity",
            max_experiments=max_experiments,
        )

    def observe_evolution(self, x1: float, y1: float, vx1: float, vy1: float) -> dict:
        """
        Observe a trajectory of the system given initial conditions.
        A reasonable coordinate range is e.g. [-3., 3.].  
        Args:
            x1, y1: first particle initial position
            vx1, vy1: first particle initial velocity
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps,4] with the solution, 
                    array[:, 0] holds x1
                    array[:, 1] holds y1
                    array[:, 2] holds vx1
                    array[:, 3] holds vy1
        """
        # hidden_init = [x2,y2,...,xN,yN, vx2,vy2,...,vxN,vyN]
        hidden_pos = self.hidden_init[:2*(self.N-1)]
        hidden_vel = self.hidden_init[2*(self.N-1):]

        init_positions = [x1, y1] + hidden_pos
        init_velocities = [vx1, vy1] + hidden_vel
        init_conditions = init_positions + init_velocities

        full_traj = self.observe_evolution_full(init_conditions)
        if "error" in full_traj:
            return {"error": full_traj["error"]}

        arr = full_traj["array"]
        x_vals = arr[:, 0]
        y_vals = arr[:, 1]
        vx_vals = arr[:, 2 * self.N]
        vy_vals = arr[:, 2 * self.N + 1]

        combined = jnp.stack([x_vals, y_vals, vx_vals, vy_vals], axis=1)
        return {"ts": full_traj["ts"], "array": combined}

    def observe_multiple_evolutions(self,
                                    x1s: str, y1s: str,
                                    vx1s: str, vy1s: str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. [-3, 3].
        Args:
            x1s: list of first particle initial x-positions in the form '[value1, value2, ...]'
            y1s: list of first particle initial y-positions in the form '[value1, value2, ...]'
            vx1s: list of first particle initial x-velocities in the form '[value1, value2, ...]'
            vy1s: list of first particle initial y-velocities in the form '[value1, value2, ...]'    
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, ode_dim] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        try:
            x1_list = ast.literal_eval(x1s)
            y1_list = ast.literal_eval(y1s)
            vx1_list = ast.literal_eval(vx1s)
            vy1_list = ast.literal_eval(vy1s)
        except Exception:
            raise ValueError("Invalid input format. Use strings like '[0.0, 0.1]'")

        n = len(x1_list)
        if not (len(y1_list) == n == len(vx1_list) == len(vy1_list)):
            raise ValueError("All input lists must have the same length.")
        if n > 5:
            raise ValueError("Maximum number of evolutions is 5.")

        hidden_pos = self.hidden_init[:2*(self.N-1)]
        hidden_vel = self.hidden_init[2*(self.N-1):]

        initial_conditions = []
        for x1, y1, vx1, vy1 in zip(x1_list, y1_list, vx1_list, vy1_list):
            init_positions = [x1, y1] + hidden_pos
            init_velocities = [vx1, vy1] + hidden_vel
            initial_conditions.append(init_positions + init_velocities)

        res = self.observe_multiple_evolutions_full(initial_conditions)
        res_array = res["arrays"]

        x_vals = res_array[:, :, 0]
        y_vals = res_array[:, :, 1]
        vx_vals = res_array[:, :, 2 * self.N]
        vy_vals = res_array[:, :, 2 * self.N + 1]

        combined = jnp.stack([x_vals, y_vals, vx_vals, vy_vals], axis=2)
        return {"tss": res["tss"], "arrays": combined}
    


    def save_result_find_eom_hidden_degrees(self, rhs: str, hidden_initial_pos:str, hidden_initial_vels:str) -> dict:
        """
        Compare the provided right-hand side of the ordinary differential equation with the true right-hand side governing the differential equation of the system.
        The loss is computed as the mean squared error between the true rhs and predicted rhs at some randomly sampled points.
        This tool should only be used to provide the final result. It can only be called once per experiment.
        Args:
            rhs: Define the right-hand side of an ordinary differential equation. 
                You pass a python code that must be of the form
                    def rhs(X:jax.Array, t:float) -> jax.Array:
                        ''' Calulates the right-hand side of the ODE. 
                            Make sure to also include the hidden dimensions in X.
                            X is of shape (n_visible_particles + n_hidden_particles) * 4 where the first half contains the coordinates and the second half contains the velocities.
                            E.g. for 1 visible and 1 hidden particle, X is of shape (2*4,) and of the form jnp.array([x1, y1, x2, y2, vx1, vy1, vx2, vy2]), where x1, y1, vx1, vy1, are the observed quantities.
                            Make sure to adhere to this format. 
                            Args:
                                X:jax.Array containing the generalized coordinates (including the hidden coordinates) followed by their velocities. E.g. jnp.array([q0, q1, q0_dot, q1_dot]).
                                t:float The time variable. Might be used in case system is time-dependent.
                            Returns:
                                The right-hand side of the ODE, a jax.Array of shape n_coordinates * 2.'''
                    rhs may use jnp syntax in the form "jnp.sin(...)". rhs must be jax-jittable!
            hidden_initial_pos: A string representation of a list of initial values for the hidden (not the observed!) coordinates,
            with values [x2, y2, x3, y3, ...], e.g. '[0.1, 0.2, 0.3, -0.2, ...]' with length 2 * n_hidden_particles.
            hidden_initial_vels: A string representation of a list of initial values for the hidden (not the observed!) velocities, 
            with values [vx2, vy2, vx3, vy3, ...], e.g. '[0.1, 0.2, 0.3, -0.2, ...]' with length 2 * n_hidden_particles.
        Returns:    
            save_message:str A message that the prediction has been saved.
        """
        # for now this works only for 1d observed systems with hidden dimensions
        n_random_inits = 100
        try:
            hidden_initial_qs = jnp.ravel(jnp.array(ast.literal_eval(hidden_initial_pos)))
            hidden_initial_q_dots = jnp.ravel(jnp.array(ast.literal_eval(hidden_initial_vels)))
        except Exception as e:
            self.last_result = {'R2': jnp.nan, 'predicted_rhs': rhs,
                                'R2_list': [],
                                'error': str(e),}
            return {'save_message': 'Saved predicted rhs and params. You can now stop the experiment.'}
        
        self.eval_key, subkey = jax.random.split(self.eval_key)
        random_inits_observed = jax.random.uniform(subkey, shape=(n_random_inits, 4), minval=self.min_random, maxval=self.max_random)
        #random_inits = jnp.concatenate([random_inits_observed[:, 0][:, None], hidden_initial_qs[None, :], random_inits_observed[:, 1][:, None], hidden_initial_q_dots[None, :]], axis=0)
        try:
            self.set_rhs("predicted_rhs", rhs)
            def pred_rhs_wrapper(X, params, t):
                return self.ode_solver.rhs['predicted_rhs'](X, t)
            R2_list = []
            for init in random_inits_observed:
                pred_init = jnp.concatenate([init[:2], hidden_initial_qs, init[2:], hidden_initial_q_dots], axis=0)
                true_init = jnp.concatenate([init[:2], jnp.array(self.hidden_init[:(self.N-1)*2]), init[2:], jnp.array(self.hidden_init[(self.N-1)*2:])], axis=0)

                solutions_pred = ode_solve(pred_init, pred_rhs_wrapper, self.params, self.dt, self.T,)
                solutions_true = ode_solve(true_init, self.ode_solver.rhs['mystery_rhs'], jnp.array(self.params), self.dt, self.T)
                for pos_idx in [0, 1]:  # x and y positions
                    xpos_list_true = solutions_true[:, pos_idx]
                    xpos_list_pred = solutions_pred[:, pos_idx]
                    xnon_nan_idcs_true = jnp.isfinite(xpos_list_true)
                    xnon_nan_idcs_pred = jnp.isfinite(xpos_list_pred)
                    xnon_nan_idcs = xnon_nan_idcs_true & xnon_nan_idcs_pred
                    # calculate r2 for non-nan values
                    R2_x = 1 - jnp.mean((xpos_list_true[xnon_nan_idcs] - xpos_list_pred[xnon_nan_idcs])**2) / (jnp.std(xpos_list_true[xnon_nan_idcs])**2 + 1e-8)
                    R2_list.append(float(np.clip(np.nan_to_num(R2_x), 0., 1.)))
            R2 = float(jnp.mean(jnp.array(R2_list)))
            self.last_result = {'R2': R2, 'predicted_rhs': rhs,
                                'R2_list': R2_list,
                                'error': None,}
        except Exception as e:
            self.last_result = {'R2': jnp.nan, 'R2_list': [],
                                'predicted_rhs': rhs,
                                'error': str(e),}
        
        
        return {'save_message': 'Saved predicted rhs and params. You can now stop the experiment.'}
    def save_result_find_eom_hidden_degrees_old(self, rhs: str) -> dict:
        """
        Compare the provided right-hand side of the ordinary differential equation with the true right-hand side governing the differential equation of the system.
        The loss is computed as the mean squared error between the true rhs and predicted rhs at some randomly sampled points.
        This tool should only be used to provide the final result. It can only be called once per experiment.
        Args:
            rhs: Define the right-hand side of an ordinary differential equation. 
                You pass a python code that must be of the form
                    def rhs(X:jax.Array, t:float) -> jax.Array:
                        ''' Calulates the right-hand side of the ODE. 
                            Make sure to also include the hidden dimensions in X.
                            X is of shape (n_visible_particles + n_hidden_particles) * 4 where the first half are the coordinates and the second half the velocities.
                            E.g. for 1 visible and 1 hidden particle, X is of shape (2*4,) and of the form jnp.array([x1, y1, x2, y2, vx1, vy1, vx2, vy2]).
                            Make sure to adhere to this format.
                            Args:
                                X:jax.Array containing the coordinates (including the hidden coordinates) followed by their velocities.
                                t:float The time variable. Might be used in case system is time-dependent.
                            Returns:
                                The right-hand side of the ODE, a jax.Array of shape n_coordinates * 2.'''
                    rhs may use jnp syntax in the form "jnp.sin(...)".
        Returns:    
            save_message:str A message that the prediction has been saved.
        """
        # for now this works only for 1d observed systems with hidden dimensions
        n_random_inits = 1000
        self.eval_key, subkey = jax.random.split(self.eval_key)
        random_inits = jax.random.uniform(subkey, shape=(n_random_inits, self.ode_dim), minval=2*self.min_random, maxval=2*self.max_random)
        #random_inits = jnp.concatenate([random_inits_observed[:, 0][:, None], qs_hidden, random_inits_observed[:, 1][:, None], qdots_hidden], axis=1)
        self.eval_key, subkey = jax.random.split(self.eval_key)
        random_times = jax.random.uniform(subkey, shape=(n_random_inits,), minval=0., maxval=self.T)
        true_rhss_all = jnp.array([self.ode_solver.rhs['mystery_rhs'](init, float(t),  jnp.array(self.params)) 
                               for init, t in zip(random_inits, random_times)])
        # fiilter out nans of true_rhss
        true_non_nan_idcs = jnp.isfinite(true_rhss_all)
        try:
            self.set_rhs("predicted_rhs", rhs)
            predicted_rhss_all = jnp.array([self.ode_solver.rhs['predicted_rhs'](init, float(t)) for init, t in zip(random_inits, random_times)])
            # use only where true rhs is finite
            true_rhss = true_rhss_all[true_non_nan_idcs]
            predicted_rhss = predicted_rhss_all[true_non_nan_idcs]
            
            #R2_q = 1 - jnp.mean((true_rhss[:, 0] - predicted_rhss[:, 0])**2) / (jnp.std(true_rhss[:, 0])**2 + 1e-8)
            #R2_q_dot = 1 - jnp.mean((true_rhss[:, self.ode_dim//2] - predicted_rhss[:, self.ode_dim//2])**2) / (jnp.std(true_rhss[:, self.ode_dim//2])**2 + 1e-8)
            # calc R2 for velocity pred in acceleration ored separatly
            R2_list = []
            for i in range(true_rhss_all.shape[1]):
                R2_i = 1 - jnp.mean((true_rhss_all[:, i][true_non_nan_idcs[:, i]] - predicted_rhss_all[:, i][true_non_nan_idcs[:, i]])**2) / (jnp.std(true_rhss_all[:, i][true_non_nan_idcs[:, i]])**2 + 1e-8)
                R2_list.append(float(R2_i))
            # R2_qd = 1 - jnp.mean((true_rhss[:, 0] - predicted_rhss[:, 0])**2) / (jnp.std(true_rhss[:, 0])**2 + 1e-8)
            # R2_qdd = 1 - jnp.mean((true_rhss[:, 1] - predicted_rhss[:, 1])**2) / (jnp.std(true_rhss[:, 1])**2 + 1e-8)
            R2_x = R2_list[0]
            R2_y = R2_list[1]
            R2_vx = R2_list[self.ode_dim//2]
            R2_vy = R2_list[self.ode_dim//2 + 1]
            R2 = float(jnp.mean(jnp.array([R2_x, R2_y, R2_vx, R2_vy])))

            self.last_result = {'R2': R2, 'predicted_rhs': rhs,
                                'R2_list': R2_list,
                                'R2_x': R2_x,
                                'R2_y': R2_y,
                                'R2_vx': R2_vx,
                                'R2_vy': R2_vy,
                                'error': None,}
        except Exception as e:
            self.last_result = {'R2': jnp.nan, 'R2_list': [],
                                'R2_x': jnp.nan,
                                'R2_y': jnp.nan,
                                'R2_vx': jnp.nan,
                                'R2_vy': jnp.nan,
                                'predicted_rhs': rhs,
                                'error': str(e),}
            
        return {'save_message': 'Saved predicted rhs and params. You can now stop the experiment.'}
    


# NOT READY YET!
class NBodyPairwisePotential(MysterySystem):
    """
    n_body system in 2D with pairwise potential.
    """

    def __init__(self,
                 N: int = 5,
                 potential_str = 'def potential(r):\n    return -jnp.exp(-r)',  # string defining the pairwise potential function of distance r
                 masses: list = None,        # list of N masses
                 eval_seed: int = 42,
                 min_random: float = -3.0,
                 max_random: float = 3.0,
                 T: float = 20.0,
                 dt: float = 0.1,
                 seed: int = 0,
                 solver_steps_per_timestep: int = 100,
                 max_experiments=None):
        self.N = int(N)
        if self.N < 1:
            raise ValueError("N must be >= 1")

        # masses
        self.ms = masses if masses is not None else [1.0] * self.N
        if len(self.ms) != self.N:
            raise ValueError("masses must be length N")


        # Build RHS
        ode_rhs = f'''def rhs(X, t, params):
    masses = params[0:{self.N}]
    N = {self.N}

    pos = X[:{2*self.N}].reshape((N, 2))
    vel = X[{2*self.N}:].reshape((N, 2))

    diffs = (pos[:, None, :] - pos[None, :, :])
    r2 = jnp.sum(diffs**2, axis=2)
    r = jnp.sqrt(r2 + 1e-12)
    
    Fr = jax.vmap(jax.vmap(jax.grad({potential_str})))(r)# shape (N, N)

    accs = - jnp.sum((masses[None, :, None] * diffs) * Fr[:, :, None] / r[:, :, None], axis=1)

    result = jnp.concatenate([vel.reshape(-1), accs.reshape(-1)])
    return result
'''

       # description = f"This is a physical system governed by an ordinary differential equation. "
        description = f"This system consists of {self.N} particles in 2D. " \
        # if give_2d_info:
        #     if hide_N:
        #         description = "This system consists of multiple particles in 2D. " \
        #                   "Only the first particle (x,y,vx,vy) is observed; the rest are hidden."
        #     else:
        #         description = f"This system consists of {self.N} particles in 2D. " \
        #                   "Only the first particle (x,y,vx,vy) is observed; the rest are hidden."
        # if give_model_info:
        #     description = f"{self.N}-body gravitational system in 2D (G=1). " \
        #                   "Only the first particle is observed; hidden particles evolve but are not shown."

        self.toolboxes = {"all": [self.observe_evolution]}

        super().__init__(
            ode_rhs=ode_rhs,
            seed=seed,
            min_random=min_random,
            max_random=max_random,
            T=T,
            dt=dt,
            solver_steps_per_timestep=solver_steps_per_timestep,
            ode_dim=4 * self.N,
            true_distinguish_func=None,
            eval_seed=eval_seed,
            description=description,
            params=self.ms,
            model_type=f"One-observed of {self.N}-body 2D gravity",
            max_experiments=max_experiments,
        )

    def observe_evolution(self, x_list: str, y_list: str, vx_list: str, vy_list: str) -> dict:
        """Observe a trajectory of the system given initial conditions.
        A reasonable coordinate range is e.g. [-3., 3.].  
        Args:
            x1_list: list of initial x-positions in the form '[value1, value2, ...]'
            y1_list: list of initial y-positions in the form '[value1, value2, ...]'
            vx1_list: list of initial x-velocities in the form '[value1, value2, ...]'
            vy1_list: list of initial y-velocities in the form '[value1, value2, ...]'
        Returns:
            'ts':jax.Array of shape [nsteps] with the time steps
            'array':jax.Array of shape [nsteps, 4 * N_particles] with the solution, 
                    array[:, 0] holds the first particle x-position
                    array[:, 1] holds the first particle y-position
                    ...
                    array[:, 2 * N_particles] holds the first particle x-velocity
                    array[:, 2 * N_particles + 1] holds the first particle y-velocity
                    ...
        """
        try:
            x_list = ast.literal_eval(x_list)
            y_list = ast.literal_eval(y_list)
            vx_list = ast.literal_eval(vx_list)
            vy_list = ast.literal_eval(vy_list)
        except Exception:
            raise ValueError("Invalid input format. Use strings like '[0.0, 0.1]'")
        n = len(x_list)
        if not (len(y_list) == n == len(vx_list) == len(vy_list)):
            raise ValueError("All input lists must have the same length.")
        if n != self.N:
            raise ValueError(f"Input lists must have length N={self.N}.")
        X_init = jnp.concatenate([jnp.array(x_list)[:, None], jnp.array(y_list)[:, None], jnp.array(vx_list)[:, None], jnp.array(vy_list)[:, None]], axis=1).reshape(-1)
        full_traj = self.observe_evolution_full(X_init)
        return full_traj
    def observe_multiple_evolutions(self, x_lists: str, y_lists: str, vx_lists: str, vy_lists: str) -> dict:
        """
        Observe multiple evolutions of the mystery system with given initial conditions.
        The maximum number of evolutions is 5.
        A reasonable coordinate range is e.g. [-3, 3].
        Args:
            x_lists: list of lists of initial x-positions in the form '[[value1, value2, ...], [...], ...]'
            y_lists: list of lists of initial y-positions in the form '[[value1, value2, ...], [...], ...]'
            vx_lists: list of lists of initial x-velocities in the form '[[value1, value2, ...], [...], ...]'
            vy_lists: list of lists of initial y-velocities in the form '[[value1, value2, ...], [...], ...]'    
        Returns:
            'tss':jax.Array of shape [n_evolutions, nsteps] with the time steps
            'arrays':jax.Array of shape [n_evolutions, nsteps, 4 * N_particles] with the solution.
            The dimension index is the same as in the 'observe_evolution' tool.
        """
        try:
            x_lists = ast.literal_eval(x_lists)
            y_lists = ast.literal_eval(y_lists)
            vx_lists = ast.literal_eval(vx_lists)
            vy_lists = ast.literal_eval(vy_lists)
        except Exception:
            raise ValueError("Invalid input format. Use strings like '[[0.0, 0.1], [0.2, -0.1]]'")
        n = len(x_lists)
        if not (len(y_lists) == n == len(vx_lists) == len(vy_lists)):
            raise ValueError("All input lists must have the same length.")
        if n > 5:
            raise ValueError("Maximum number of evolutions is 5.")
        for x_list in x_lists:
            if len(x_list) != self.N:
                raise ValueError(f"Each input list must have length N={self.N}.")
        for y_list in y_lists:
            if len(y_list) != self.N:
                raise ValueError(f"Each input list must have length N={self.N}.")
        for vx_list in vx_lists:
            if len(vx_list) != self.N:
                raise ValueError(f"Each input list must have length N={self.N}.")
        for vy_list in vy_lists:
            if len(vy_list) != self.N:
                raise ValueError(f"Each input list must have length N={self.N}.")
        initial_conditions = []
        for x_list, y_list, vx_list, vy_list in zip(x_lists, y_lists, vx_lists, vy_lists):
            X_init = jnp.concatenate([jnp.array(x_list)[:, None], jnp.array(y_list)[:, None], jnp.array(vx_list)[:, None], jnp.array(vy_list)[:, None]], axis=1).reshape(-1)
            initial_conditions.append(X_init)
        res = self.observe_multiple_evolutions_full(initial_conditions)
        return res
