import jax
import jax.numpy as jnp
import scipy as scipy
import numpy as np
import optax
import flax

import sciexplorer.general_analysis_tools as analysis
from sciexplorer.physics_eom  import ode_solve
import jax.numpy as jnp
from  sciexplorer.general_analysis_tools import plot_single_1d, plot_single_2d, plot_single_1d_vst

def execute_code(_fields:dict, code: str) -> dict:
    """ Evaluate python code.
        This code can be e.g. be used to transform the previously saved fields or to calculate or save new fields.
        You cannot use this code to generate plots or images.
        You cannot see print statements or other output of this code. Instead, you can use the result variable to save the results of the code.
        The following variables are available during evaluation:
            all previosuly saved fields with their previously stated result_labels as variable names (as local variables).
            jax: jax for numerical operations.
            jnp: jax.numpy for numerical operations.
            np: numpy for numerical operations.
            scipy: scipy for numerical operations including optimization and solving differential equations.
            optax: optax for efficient gradient based optimizers like adam, rmsprop, etc.
        Additionally, you have access to an ode solver with the following signature:
            def ode_solve(X0:jax.Array, rhs: callable, params:jax.Array, dt:float, T:float) -> jnp.array:
                '''Solve the differential equation dX/dt=rhs(X,t,params), up to time T using the Runge-Kutta method, with time step dt,
                and initial condition X0. The right-hand side of the differential equation is given by the function rhs(X,t,params).
                Args:
                    X0 (jnp.array): the initial condition, in the form of an array, for example "jnp.array([0.3,0.2,0.5])"
                    rhs (callable): the right-hand side of the differential equation, must be JIT-compilable, in the following form:
                        def rhs(X:jax.Array, t:float, params:jax.Aray) -> jax.Array:
                            <code that calculates rhs and returns the result>
                        The function must return a jax.Array of the same shape as X
                    params: the parameters, in the form of an array, for example "jnp.array([0.1,0.8])"
                    dt (float): the time step size, for example 0.001
                    T (float): the final time, for example 20.0,
                Returns:
                    Xs (jnp.array): the solution, an array of shape (len(ts), *X0.shape)'''
            You can vmap the ode_solve function to solve multiple initial conditions at once.
        The code must set the variable 'result' to a dictionary in the end which should contain the newly generated data. E.g. result={'<result_key>': <data>, ...}.
        Args:
            code: python code that sets the result variable to a dictionary containing some newly generated data.
        Returns:
            the result dictionary.
        """
    globals_dict = _fields | {'result':None, 'jax':jax, 'jnp': jnp, 'np':np, 'scipy':scipy, 
                              'optax':optax, 'ode_solve': ode_solve}
    exec(code, globals_dict)
    if globals_dict['result'] is None:
        raise ValueError("The code did not set the result variable. It must be set to a dictionary containing the newly generated data. E.g. result={'<result_key>': <data>, ...}")
    if not isinstance(globals_dict['result'], dict):
        raise ValueError("The result variable must be a dictionary. E.g. result={'<result_key>': <data>, ...}")
    return globals_dict['result']


toolboxes = {'coding': [
            analysis.approx_equal, 
            analysis.plot_from_code,
            execute_code,],
            'no_approx_equal': [
            analysis.plot_from_code,
            execute_code,],
            'text_only': [
            analysis.approx_equal, 
            execute_code,],
            'only_plots': [
            analysis.only_plot,],
            'no_analysis': [],}