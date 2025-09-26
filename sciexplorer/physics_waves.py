import jax
import jax.numpy as jnp
import numpy
import numpy as np
from functools import partial
import scipy as sp
import scipy

def safe_evaluate(formula, x):
    # Convert string to sympy expression
    sym_x = sp.Symbol('x')
    expr = sp.sympify(formula)
    
    # Convert to numpy-compatible lambda function
    f = sp.lambdify(sym_x, expr, modules=['numpy'])
    return f(x)
    
@partial(jax.jit,static_argnames=['U_kinetic','U_potential'])
def wave_solve(phi0,U_kinetic,U_potential,x,k,ts,dx):
    """
    Solve a (possibly nonlinear) wave equation
    on a lattice via Trotter decomposition, doing the kinetic term
    in Fourier space.
    
    phi0: initial wave (complex 1D array)

    U_kinetic(phi_k,t,dt) returns one evolution step for the kinetic terms,
        in Fourier space
        
    U_potential(phi,t,dt) returns the (maybe nonlinear)
        evolution of phi for a time step dt, in real space.

    x is the x-grid in real space
    k is the k-grid in Fourier space
    ts is an equally spaced array of times
    dx is the lattice spacing in real space
        
    Returns: phis,ts, where phis[j] is the solution at time ts[j+1] (!)
    
    Remarks: we might also use a symmetric decomposition, sqrt(U_nonlin) U_kin sqrt(U_nonlin)
    """
    dt=ts[1]-ts[0]
    
    def loop_body(phi,t):
        phi = jnp.fft.ifft( U_kinetic( jnp.fft.fft(U_potential(phi,x,t,dt)), k,t,dt ) )
        return phi,phi
    
    _,phis=jax.lax.scan(loop_body,phi0,xs=ts)
    
    return phis,ts

class WaveBasics:
    def __init__(self, x: jnp.array, ts: jnp.array):
        self.x=x
        self.ts=ts
        self.k = jnp.fft.fftfreq(len(x),d=x[1]-x[0])*2*jnp.pi
        self.rhs={}
        
    def set_rhs(self, rhs_label:str, code: str) -> dict:
        """
        Define the right-hand side of a complex wave equation to be simulated. 
        The wave equation will be simulated using the split-step method,
        applying potential terms in real space and kinetic terms (from
        spatial derivatives) in Fourier space.

        You pass a python code that must be of the form:

        def U_potential(phi,x,t,dt):
            # jax code that calculates the evolution of the
            # complex field phi for a time step dt at time t
            # and returns the result. This evolution here
            # only accounts for the terms of the wave equation
            # that do not involve spatial derivatives (those
            # will be handled separately). x is a 1D array for
            # the real-space grid points.
            # Example: return jnp.exp(-1j*dt*0.1*jnp.sin(x))*phi

        def U_kinetic(phi_k,k,t,dt):
            # jax code that calculates the evolution of phi_k
            # from the spatial-derivative terms in the wave equation,
            # for a time step dt at time t and returns the result.
            # phi_k is the wave field in Fourier space. k is a 1D array
            # for the Fourier space grid points.
            # Example: return jnp.exp(-1j*(1-jnp.cos(k))*dt)*phi_k

        Use jax.numpy syntax in the form "jnp.exp(...)".
        
        Args:
            rhs_label:str the label the function will be stored under.
            code:str the python code defining the evolution functions.
    
        Returns:
            Message indicating wether the functions were set successfully.
        """
        global_dict={"U_potential": None, "U_kinetic": None, 
                     'jnp': jnp}
        try:
            exec(code, global_dict)
        except Exception as e:
            return {'error': f"Error in executing provided code: {e}"}
        if global_dict["U_potential"] is None or global_dict["U_kinetic"] is None:
            return {'error': "Error: Both U_potential and U_kinetic must be defined in the code."}
        self.rhs[rhs_label] = { "U_potential": global_dict["U_potential"], "U_kinetic": global_dict["U_kinetic"] }
        return {'result_str': f"Success: Set rhs function successfully and saved it under the label: {rhs_label}."}#{'to_model': f"Success: Set rhs function successfully and saved it under the label: {rhs_label}."}

    # same, but with 'field' instead of wave, as a label
    def set_field_rhs(self, rhs_label:str, code: str) -> dict:
        """
        Define the right-hand side of a complex field equation to be simulated. 
        The field equation will be simulated using the split-step method,
        applying potential terms in real space and kinetic terms (from
        spatial derivatives) in Fourier space.

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
    
        Returns:
            Message indicating wether the functions were set successfully.
        """
        return self.set_rhs(rhs_label, code)
        
    def run_wave_simulation(self, rhs_label: str, initial_condition_code: str) -> dict:
        """
        Run a single wave simulation, for the wave equation defined previously
        under rhs_label (using set_rhs), and for the given initial condition.

        Args:
            rhs_label: label used previously in a call to set_rhs
            initial_condition_code: a formula in x that uses jax syntax, i.e.
                jnp.sin(...) etc. It must define phi0, which is a complex 
                array of the same shape as x. 

        Returns:
            solution as a dictionary with entries
            - ts: the time points (1D array)
            - x: the x grid (1D array)
            - phis: the complex wave field solution, 
                a 2D jax.Array of shape [n_ts,n_x]
        """
        try:
            rhs = self.rhs[rhs_label]
        except KeyError:
            return {"error": f"Error: rhs with label '{rhs_label}' not found."}

        global_dict={"x": self.x,
                     'jnp': jnp, "phi0": None}
        try:
            exec(initial_condition_code, global_dict)
        except Exception as e:
            return {'error': f"Error in executing provided code: {e}"}

        if global_dict["phi0"] is None:
            return {"error": "Code initial_condition_code must set phi0."}
        
        # try:
        #     phi0 = safe_evaluate(initial_condition_formula, self.x)
        # except Exception as e:
        #     return {"error": f"Error in evaluating initial condition: {e}"}
            
        try:
            dx=self.x[1]-self.x[0]
            phi0=jnp.array(global_dict["phi0"], dtype="complex64")
            phis, _ = wave_solve(phi0, rhs["U_kinetic"],rhs["U_potential"],self.x,self.k,self.ts,dx)
        except Exception as e:
            return {"error": f"Error in solving wave equation: {e}"}
            
        solution={"ts":self.ts, "x":self.x, "phis": jnp.copy(phis)}
        return solution

    # just replacing wave by field in the documentation:
    def run_field_simulation(self, rhs_label: str, initial_condition_code: str) -> dict:
        """
        Run a single field simulation, for the field evolution equation defined previously
        under rhs_label (using set_field_rhs), and for the given initial condition.

        Args:
            rhs_label: label used previously in a call to set_field_rhs
            initial_condition_code: a formula in x that uses jax syntax, i.e.
                jnp.sin(...) etc. It must define phi0, which is a complex 
                array of the same shape as x. 

        Returns:
            solution as a dictionary with entries
            - ts: the time points (1D array)
            - x: the x grid (1D array)
            - phis: the complex field solution, 
                a 2D jax.Array of shape [n_ts,n_x]
        """
        return self.run_wave_simulation(rhs_label, initial_condition_code)
        