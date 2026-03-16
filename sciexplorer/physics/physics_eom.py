import jax
import jax.numpy as jnp

def runge_kutta_step_with_params(X:jax.Array,params:jax.Array,t:float,rhs:callable,dt:float) -> jax.Array:
    """
    One step of the standard Runge-Kutta fourth order.
    This assumes rhs can be called like rhs(X,t,params)
    """
    k1=rhs(X,t,params)
    k2=rhs(X+0.5*dt*k1,t+0.5*dt,params)
    k3=rhs(X+0.5*dt*k2,t+0.5*dt,params)
    k4=rhs(X+dt*k3,t+dt,params)
    return( (dt/6.)*(k1+2*k2+2*k3+k4) )


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
    Returns a tuple:
        Xs (jnp.array): the solution, an array of shape (len(ts), *X0.shape,)
        ts (jnp.array): the time steps, an array of shape (len(ts),)
    """
    ts = jnp.arange(0, T, dt/solver_steps_per_timestep)  # time steps from 0 to T with step size dt
    dt_solver=ts[1]-ts[0]

    def loop_body(x,t):
        x+=runge_kutta_step_with_params(x,params,t,rhs,dt_solver)
        return x,x
    
    _,Xs=jax.lax.scan(loop_body,X0,xs=ts)
    # append X0 at the beginning:
    Xs = jnp.concatenate((jnp.expand_dims(X0, axis=0), Xs), axis=0)[::solver_steps_per_timestep]
    ts = ts[::solver_steps_per_timestep]
    if len(ts) != len(Xs):
        ts = jnp.concatenate((ts, jnp.array([T])), axis=0)
    return Xs, ts