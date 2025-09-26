import jax
import jax.numpy as jnp
import numpy
import numpy as np
import diffrax as df
from functools import partial
import scipy as sp
import scipy
import jax
import numpy as np
import scipy
import jax.numpy as jnp
from functools import partial
import ast
from jax import vmap

import matplotlib.pyplot as plt

def runge_kutta_step_with_params(X,params,t,rhs,dt):
    """
    One step of the standard Runge-Kutta fourth order.
    This assumes rhs can be called like rhs(X,t,params)
    """
    k1=rhs(X,t,params)
    k2=rhs(X+0.5*dt*k1,t+0.5*dt,params)
    k3=rhs(X+0.5*dt*k2,t+0.5*dt,params)
    k4=rhs(X+dt*k3,t+dt,params)
    return( (dt/6.)*(k1+2*k2+2*k3+k4) )

@partial(jax.jit,static_argnames=['rhs'])
def runge_kutta_solve(X0,rhs,ts,params):
    """
    Solve the differential equation dX/dt=rhs(X,t), for all (equally spaced) times in ts,
    with initial value X(t=0)=X0.
    
    Here X0 can have ANY shape as an array, and rhs(X,t) must return the same shape.
    
    This is convenient in case you want to, say, do many trajectories in parallel,
    or reserve one index for the particle number, etc. You do not need to assume
    X to be a 1d array, as you would have to do for some other Runge-Kutta implementations.
    It is also nice to split the positions and velocities, e.g. X[0,:] for the positions
    of all particles, and X[1,:] for the velocities.
    
    Returns: Xs,ts
    
    where the shape of the solution Xs is the same as that of X, except we add another dimension
    at the beginning, of size 'nsteps'. 'ts' just is a 1d array denoting the time steps for
    the solution.
    
    Plotting the solution may work like this, if X was a 1d array:
    
    plt.plot(ts,Xs[:,5])
    
    ...or like this, if e.g. X[1,:] were all the velocities of all particles:
    
    plt.plot(ts,Xs[:,1,3]) # plotting velocity of particle number 3
    
    (code by Florian Marquardt 2020, 2024)
    """
    dt=ts[1]-ts[0]

    def loop_body(x,t):
        x+=runge_kutta_step_with_params(x,params,t,rhs,dt)
        return x,x
    
    _,Xs=jax.lax.scan(loop_body,X0,xs=ts)
    # append X0 at the beginning:
    Xs = jnp.concatenate((jnp.expand_dims(X0, axis=0), Xs), axis=0)
    # remove last time step from Xs, to make it the same shape as ts:
    Xs = Xs[:-1]
    return Xs,ts



def ode_solve(X0:jax.Array, rhs: callable, params:jax.Array, dt:float, T:float, solver_steps_per_timestep=10) -> tuple:
    """
    Solve the differential equation dX/dt=rhs(X,t,params), up to time T using the Runge-Kutta method, with time step dt,
    and initial condition X0. The right-hand side of the differential equation is given by the function rhs(X,t,params).    
    Args:
        X0 (jnp.array): the initial condition, in the form of an array, for example "jnp.array([0.3,0.2,0.5])"
        rhs (callable): the right-hand side of the differential equation, in the following form:
            def rhs(X:jax.Array, t:float, params:jax.Aray) -> jax.Array:
                <code that calculates rhs and returns the result>
            The function must return a jax.Array of the same shape as X
        params: the parameters, in the form of an array, for example "jnp.array([0.1,0.8])"
        dt (float): the time step size, for example 0.001
        T (float): the final time, for example 20.0
        solver_steps_per_timestep (int): the number of steps per time step to use in the solver.
    Returns:
        Xs (jnp.array): the solution, an array of shape (len(ts), *X0.shape,)
    """
    ts = jnp.arange(0, T, dt/solver_steps_per_timestep)  # time steps from 0 to T with step size dt
    dt_solver=ts[1]-ts[0]

    def loop_body(x,t):
        x+=runge_kutta_step_with_params(x,params,t,rhs,dt_solver)
        return x,x
    
    _,Xs=jax.lax.scan(loop_body,X0,xs=ts)
    # append X0 at the beginning:
    Xs = jnp.concatenate((jnp.expand_dims(X0, axis=0), Xs[::solver_steps_per_timestep]), axis=0)
    return Xs