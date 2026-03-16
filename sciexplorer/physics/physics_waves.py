import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Callable
from sciexplorer.tools.tool_utils import exec_with_error_line

    
@partial(jax.jit,static_argnames=['U_kinetic','U_potential'])
def wave_solve(phi0:jax.Array,U_kinetic:Callable,U_potential:Callable,x:jax.Array,k:jax.Array,ts:jax.Array)->tuple[jax.Array,jax.Array]:
    """
    Solve a (possibly nonlinear) wave equation on a lattice via Trotter decomposition,
    doing the kinetic term in Fourier space.
    
    Args:
        phi0: jnp.array of shape (n_x,), initial wave
        U_kinetic(phi_k,k,t,dt): callable, returns one evolution step for the kinetic terms in Fourier space
        U_potential(phi,x,t,dt): callable, returns the (maybe nonlinear) evolution of phi for a time step dt, in real space.
        x: jnp.array of shape (n_x,) is the x-grid in real space
        k: jnp.array of shape (n_x,) is the k-grid in Fourier space
        dt: float, the time step size
        
    Returns: 
        phis: jnp.array of shape (len(ts), len(x)) with the solution
        ts: jnp.array of time points (same as input)
    """
    dtype=phi0.dtype
    t_idcs = jnp.arange(len(ts)-1)
    def loop_body(carry, t_idx):
        phi, ts = carry
        t=ts[t_idx]
        dt = ts[t_idx+1]-ts[t_idx]

        phi = U_potential(phi, x, t, dt / 2).astype(dtype)
        phi = jnp.fft.ifft( U_kinetic( jnp.fft.fft(phi), k, t, dt ) ).astype(dtype)
        phi = U_potential(phi, x, t, dt / 2).astype(dtype)

        return (phi, ts), phi
    
    _,phis=jax.lax.scan(loop_body, (phi0, ts), xs=t_idcs)

    # append initial condition at the beginning
    phis = jnp.vstack([phi0[None,:], phis])
    
    return phis,ts

# ================================================================
# Split-step solver with 3/2-rule dealiasing
# ================================================================

def pad(phi:jax.Array, pad_factor:float=1.5):
    """
    Zero-pad FFT to pad_factor times resolution (1D).
    phi: array (N,)
    return: phi_padded (M = 3N/2,)
    """
    N = phi.shape[0]
    M = int(pad_factor * N)

    phi_k = jnp.fft.fft(phi)
    phi_k_shift = jnp.fft.fftshift(phi_k)

    pad = jnp.zeros(M, dtype=jnp.complex64)

    a = (M - N) // 2
    pad = pad.at[a : a + N].set(phi_k_shift)
    pad = jnp.fft.ifftshift(pad)
    phi_padded = jnp.fft.ifft(pad) * (M / N)
    return phi_padded


def unpad(phi_padded:jax.Array, N:int):
    """
    Truncate padded field back to size N.
    """
    M = phi_padded.shape[0]
    phi_k = jnp.fft.fft(phi_padded)
    phi_k_shift = jnp.fft.fftshift(phi_k)

    trunc = phi_k_shift[(M - N) // 2 : (M - N) // 2 + N]
    return jnp.fft.ifft(jnp.fft.ifftshift(trunc)) * (N / M)


def pad_x_grid(x, pad_factor:float=1.5):
    """Return padded x-grid"""
    N = x.shape[0]
    M = int(pad_factor * N)
    L = x[-1] - x[0] + (x[1] - x[0])
    return jnp.linspace(x[0], x[-1], M) #x[0] + dx_pad * jnp.arange(M) #  #


# ================================================================
# Split-step with 3/2-rule nonlinearity (with padded x)
# ================================================================

#@partial(jax.jit, static_argnames=['U_kinetic','U_potential'])
def wave_solve_dealias(phi0: jax.Array,
               U_kinetic: callable,
               U_potential: callable,
               x: jax.Array,
               k: jax.Array,
               ts: jax.Array,
               pad_factor: float = 1.5) -> tuple[jax.Array, jax.Array]:
    

    N = phi0.shape[0]
    x_pad = pad_x_grid(x, pad_factor=pad_factor)            # static padded x-grid
    t_idcs = jnp.arange(len(ts) - 1)

    def loop_body(carry, t_idx):
        phi, ts = carry
        t = ts[t_idx]
        dt = ts[t_idx+1] - ts[t_idx]

        # --- Half nonlinear step on padded grid ---
        phi_pad = pad(phi, pad_factor=pad_factor)
        phi_pad = U_potential(phi_pad, x_pad, t, dt/2)
        phi = unpad(phi_pad, N)

        # --- Full kinetic step in Fourier space ---
        phi = jnp.fft.ifft(
            U_kinetic(jnp.fft.fft(phi), k, t, dt)
        )

        # --- Half nonlinear step again ---
        phi_pad = pad(phi, pad_factor=pad_factor)
        phi_pad = U_potential(phi_pad, x_pad, t, dt/2)
        phi = unpad(phi_pad, N)

        return (phi, ts), phi

    _, phis = jax.lax.scan(loop_body, (phi0, ts), xs=t_idcs)
    phis = jnp.vstack([phi0[None, :], phis])

    return phis, ts
    

class WaveBasics:
    def __init__(self, x: jnp.array=None, ts: jnp.array=None):
        self.x=x
        self.ts=ts
        if x is None:
            self.k = None 
        else:
            self.k = jnp.fft.fftfreq(len(x),d=x[1]-x[0])*2*jnp.pi
            self.k = self.k.astype('complex64')
        self.rhs={}

    def set_simulator_rhs(self, rhs_label:str, code: str) -> dict:
        """
        [THIS IS A TOOL TO CONTROL A SIMULATOR]
        Define the right-hand side of a complex field equation to be simulated. 
        The field equation will be simulated using the split-step method,
        applying potential terms in real space and kinetic terms (from spatial derivatives) in Fourier space.

        You pass a python code that must be of the form:

        def U_potential(phi,x,t,dt):
            # jax code that calculates the evolution of the
            # complex field phi for a time step dt at time t
            # and returns the result. This evolution here
            # only accounts for the terms of the field partial differential equation
            # that do not involve spatial derivatives (those
            # will be handled separately). x is a 1D array for
            # the real-space grid points.
            # Example: return jnp.exp(-1j*dt*0.1*jnp.sin(x))*phi

        def U_kinetic(phi_k,k,t,dt):
            # jax code that calculates the evolution of phi_k
            # from the spatial-derivative terms in the field equation,
            # for a time step dt at time t and returns the result.
            # phi_k is the field in Fourier space. k is a 1D array
            # for the Fourier space grid points.
            # Example: return jnp.exp(-1j*(1-jnp.cos(k))*dt)*phi_k

        Use jax.numpy syntax in the form "jnp.exp(...)".
        
        Args:
            rhs_label:str the label the function will be stored under.
            code:str the python code defining the evolution functions.
    
        Returns a dict of the form:
            result_str: Message indicating that the functions were set successfully.
        """
        global_dict={"U_potential": None, "U_kinetic": None, 
                     'jnp': jnp, 'jax': jax, 'x': self.x, 'k': self.k,}
        global_dict = exec_with_error_line(code, global_dict)
        if global_dict["U_potential"] is None or global_dict["U_kinetic"] is None:
            raise ValueError("Error: Both U_potential and U_kinetic must be defined in the code.")
        self.rhs[rhs_label] = { "U_potential": global_dict["U_potential"], "U_kinetic": global_dict["U_kinetic"] }
        return {'result_str': f"Success: Set rhs function successfully and saved it under the label: {rhs_label}."}

    # just replacing wave by field in the documentation:
    def run_simulator(self, rhs_label: str, initial_condition_code: str) -> dict:
        """
        [THIS IS A TOOL TO CONTROL A SIMULATOR]
        Run a single field simulation, for the field evolution equation defined previously under rhs_label (using set_simulator_rhs), 
        and for the given initial condition.
        Args:
            rhs_label: label used previously in a call to set_simulator_rhs
            initial_condition_code: a formula in x that uses jax syntax, i.e. jnp.sin(...) etc. 
                It must define phi0, which is a complex array of the same shape as x. 

        Returns a dict of the form:
            ts: jnp.array of shape (n_ts,), the time points
            x: jnp.array of shape (n_x,), the x grid
            phis: jnp.array of shape (n_ts, n_x), the complex field solution
        """
        try:
            rhs = self.rhs[rhs_label]
        except KeyError:
            raise ValueError(f"rhs with label '{rhs_label}' not found.")

        global_dict={"x": self.x,
                     'jnp': jnp, "phi0": None, 'jax': jax, 'np': np, 'numpy': np}
        try:
            exec_with_error_line(initial_condition_code, global_dict)
        except Exception as e:
            raise Exception(f"Error in initial condition code:\n{e}")
        

        if global_dict["phi0"] is None:
            raise ValueError("Code initial_condition_code must set phi0.")
          
        try:
            phi0=jnp.array(global_dict["phi0"], dtype="complex64")
            assert phi0.shape == self.x.shape, f"phi0 must have shape {self.x.shape}, but got {phi0.shape}"
            phis, _ = wave_solve(phi0.astype('complex64'), rhs["U_kinetic"],rhs["U_potential"],self.x,self.k,self.ts)
        except Exception as e:
            raise Exception(f"Error during simulation:\n{e}")
            
        solution={"ts":self.ts, "x":self.x, "phis": jnp.copy(phis)}
        return solution
    
    def run_simulator_controlled_res(self, rhs_label: str, initial_condition_code: str, 
                                    x_max: float=50, n_x: int=1024, t_max: float=20., n_t: int=2048, 
                                    precision:str='32') -> dict: #, dealias_pad_factor: float = 1.
        """
        [THIS IS A TOOL TO CONTROL A SIMULATOR]
        Run a single field simulation, for the field evolution equation defined previously under rhs_label (using set_simulator_rhs), 
        and for the given initial condition.
        Args:
            rhs_label: label used previously in a call to set_simulator_rhs
            initial_condition_code: a formula in x that uses jax syntax, i.e. jnp.sin(...), jax.random..., etc. 
                It must define phi0, which is a complex array of the same shape as x. 
            x_max: the maximum x value (the position  will go from -x_max to x_max), default 50
            n_x: the number of x grid points used for discretizing space, default 1024
            t_max: the maximum time value (the time grid will go from 0 to t_max), default 20.
            n_t: the number of time steps used for discretizing time, default 2048
            precision: '32' or '64', the numerical precision to use

        Returns a dict of the form:
            ts: jnp.array of shape (n_ts,), the time points
            x: jnp.array of shape (n_x,), the x grid
            phis: jnp.array of shape (n_ts, n_x), the complex field solution
        """
        #    dealias_pad_factor: the padding factor for dealiasing, if set to 1., no dealiasing is done.

        try:
            rhs = self.rhs[rhs_label]
        except KeyError:
            raise ValueError(f"rhs with label '{rhs_label}' not found.")
        
        if precision=='32':
            dtype = jnp.complex64
            float_dtype = jnp.float32
        elif precision=='64':
            dtype = jnp.complex128
            float_dtype = jnp.float64
        else:
            raise ValueError(f"precision must be '32' or '64', was: {precision}")
        
        x = jnp.linspace(-x_max, x_max, n_x, dtype=float_dtype)
        k = (jnp.fft.fftfreq(len(x),d=jnp.real(x[1]-x[0]))*2*jnp.pi).astype(dtype)
        ts = jnp.linspace(0, t_max, n_t, dtype=float_dtype)

        global_dict={"x": x, "jnp": jnp, "jax": jax, "phi0": None}
        try:
            exec_with_error_line(initial_condition_code, global_dict)
        except Exception as e:
            raise Exception(f"Error in initial condition code:\n{e}")

        if global_dict["phi0"] is None:
            raise ValueError("Code initial_condition_code must set phi0.")


        try:
            phi0=jnp.array(global_dict["phi0"], dtype=dtype)
            #if dealias_pad_factor==1.:
            phis, _ = wave_solve(phi0, rhs["U_kinetic"],rhs["U_potential"],x,k,ts)
            #else:
            #    phis, _ =   wave_solve_dealias(phi0, rhs["U_kinetic"],rhs["U_potential"],x,k,ts,dealias_pad_factor)
            # concat first time point
            phis = jnp.vstack([phi0[None,:], phis[:-1,:]])
        except Exception as e:
            raise Exception(f"Error during simulation:\n{e}")
            
        solution={"ts":ts, "x":x, "phis": jnp.copy(phis)}
        return solution
        
        