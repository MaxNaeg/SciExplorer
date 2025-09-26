# A new wave simulator, based on the physics_waves toolkit

import sciexplorer.physics_waves as waves
import jax
import jax.numpy as jnp

predefined_wave_systems = {
    "nonlinSEQ_1": 
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    return jnp.exp(-1j*dt*jnp.abs(phi)**2) * phi       
        """, "NONLINEAR"],

    "linSEQ_1":
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    return phi 
        """, "LINEAR"],

    "nonlinSEQ_2": 
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*0.5*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    return jnp.exp(-1j*dt*0.25*jnp.abs(phi)**2) * phi       
        """, "NONLINEAR"],

    "linSEQ_2": # next-nearest-neighbor hopping
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*(.5-jnp.cos(k)+0.5*jnp.cos(2*k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    return phi 
        """, "LINEAR"],

    "simple_GL": # normal Ginzburg Landau in 1D, dphi/dt = - gamma*(-phi+g|phi|^2 - (1/2) d^2phi/dx^2)
    # gamma=1.5, g=2
    ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-.5*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    return jnp.exp(-.5*(2*jnp.abs(phi)**2 - 1)*dt) * phi       
        """,
    "NONLINEAR"],

    "linSEQ_periodic":
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    A=0.2
    return jnp.exp(-1j*A*jnp.cos(10*2*jnp.pi*(x/jnp.max(x)))*dt)*phi 
        """, "LINEAR"],

    "complex_GL": # complex Ginzburg Landau in 1D
    ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-.5*(0.5+1j)*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    return jnp.exp(-.5*((1-0.7j)*2*jnp.abs(phi)**2 - 1)*dt) * phi       
        """,
    "NONLINEAR"],

    "linSEQ_potential":
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    return jnp.exp(+1j*.05*jnp.cos(jnp.pi*x/jnp.max(x)))*phi 
        """, "LINEAR"],

    "nonlinSEQ_potential":
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    g=0.5
    return jnp.exp(+1j*.05*jnp.cos(jnp.pi*x/jnp.max(x)) - 1j*g*jnp.abs(phi)**2)*phi 
        """, "LINEAR"],

    "nonlinSEQ_potential_g_low":
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    g=0.1
    return jnp.exp(+1j*.05*jnp.cos(jnp.pi*x/jnp.max(x)) - 1j*g*jnp.abs(phi)**2)*phi 
        """, "LINEAR"],

    "nonlinSEQ_potential_g_zero":
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    g=0.0
    V=.05
    return jnp.exp(+1j*V*jnp.cos(jnp.pi*x/jnp.max(x)) - 1j*g*jnp.abs(phi)**2)*phi 
        """, "LINEAR"],

    "nonlinSEQ_potential_V_zero_g_low":
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    g=0.1
    V=0.0
    return jnp.exp(+1j*V*jnp.cos(jnp.pi*x/jnp.max(x)) - 1j*g*jnp.abs(phi)**2)*phi 
        """, "LINEAR"],

    "nonlinSEQ_potential_V_medium_g_low":
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    g=0.1
    V=0.1
    return jnp.exp(+1j*V*jnp.cos(jnp.pi*x/jnp.max(x)) - 1j*g*jnp.abs(phi)**2)*phi 
        """, "LINEAR"],

    "nonlinSEQ_potential_V_medium_g_zero":
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    g=0.0
    V=0.1
    return jnp.exp(+1j*V*jnp.cos(jnp.pi*x/jnp.max(x)) - 1j*g*jnp.abs(phi)**2)*phi 
        """, "LINEAR"],

# FINALIZED SETTINGS FOR PRODUCTION RUNS (reasonable evolution in experimental time window)

    "LIN_SEQ": # linear SEQ (on the underlying tight binding lattice)
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    return phi 
        """, "LINEAR"],

    "LIN_SEQ_NNN": # next-nearest-neighbor hopping
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*(.5-jnp.cos(k)+0.5*jnp.cos(2*k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    return phi 
        """, "LINEAR"],
    
    "LIN_SEQ_POTENTIAL": # lin SEQ with potential
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    return jnp.exp(+1j*dt*.5*jnp.cos(jnp.pi*x/jnp.max(x)))*phi 
        """, "LINEAR"],

    "LIN_SEQ_PERIODIC": # lin SEQ in periodic potential
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    A=0.2
    return jnp.exp(-1j*A*jnp.cos(10*2*jnp.pi*(x/jnp.max(x)))*dt)*phi 
        """, "LINEAR"],
    
    "NONLIN_SEQ": # nonlin SEQ
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*0.5*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    return jnp.exp(-1j*dt*0.25*jnp.abs(phi)**2) * phi       
        """, "NONLINEAR"],
    
    "NONLIN_SEQ_POTENTIAL": # nonlin SEQ with potential
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    g=1.
    V=1.
    return jnp.exp(+1j*dt*V*jnp.cos(jnp.pi*x/jnp.max(x)) - 1j*dt*g*jnp.abs(phi)**2)*phi 
        """, "LINEAR"],

    "SIMPLE_GL": # relaxation dynamics from Ginzburg Landau in 1D (reaction diffusion eq.)
    ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-.5*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    return jnp.exp(-.5*(2*jnp.abs(phi)**2 - 1)*dt) * phi       
        """,
    "NONLINEAR"],
    
    "COMPLEX_GL": # complex Ginzburg Landau in 1D
    ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-.2*(0.5+2j)*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    return jnp.exp(-.2*((1-1.5j)*2*jnp.abs(phi)**2 - 1)*dt) * phi       
        """,
    "NONLINEAR"],

    "NONLIN_SEQ_NNN": # nonlin SEQ
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*0.5*(1.5-jnp.cos(k)+.5*jnp.cos(2*k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    return jnp.exp(-1j*dt*0.25*jnp.abs(phi)**2) * phi       
        """, "NONLINEAR"],

    "COMPLEX_GL_NNN": # complex Ginzburg Landau in 1D, with instability-inducing term
    ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-.8*(0.5+0.5j)*(.65-jnp.cos(k)+.8*jnp.cos(2*k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    return jnp.exp(-.2*((1-1.5j)*2*jnp.abs(phi)**2 - 1)*dt) * phi       
        """,
    "NONLINEAR"],
    
"SIN_POTENTIAL_RELAXATION": #  reaction diffusion with periodic potential
    ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-.5*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    return jnp.exp(+.1*jnp.sin(15*jnp.abs(phi)**2)*dt) * phi       
        """,
    "NONLINEAR"],

    "NONLIN_SEQ_PHI6": # nonlin SEQ
        ["""
def U_kinetic(phi_k,k,t,dt):
    return jnp.exp(-1j*0.5*(1-jnp.cos(k))*dt) * phi_k
def U_potential(phi,x,t,dt):
    return jnp.exp(-1j*dt*0.25*(jnp.abs(phi)**2 + 2*jnp.abs(phi)**4)) * phi       
        """, "NONLINEAR"],
}

class WaveSimulator(waves.WaveBasics):
    def __init__(self, x_max: float, N_x: int, t_max: float, N_t: int, 
                 rhs_choice: str, rhs_code: str, true_type_linear=None, description_type="WAVE",
                 initials_for_res_check: list = None # list containing initial conditions specified as strings of jax code in the form phi0=f(x)
                 ):
        """
Initialize the wave simulator, with x and ts arrays,
as well as either a rhs_choice denoting the label
of a predefined wave right-hand-side, or rhs_code,
which defines directly the rhs. See 
physics_waves.WaveBasics.set_rhs for explanation
of rhs_code.

x will run from -x_max to +x_max in N_x grid points
t will run from 0 to t_max in N_t grid points

Some choices for rhs_choice:
    nonlinSEQ_1, linSEQ_1, ...
See predefined_wave_systems in this file.
        """
        x_sim = jnp.linspace(-x_max,+x_max,N_x)
        ts_sim = jnp.linspace(0.,t_max,N_t)
        super().__init__(x_sim,ts_sim)

        if rhs_choice is not None:
            self.set_rhs("rhs", predefined_wave_systems[rhs_choice][0])
            self.true_type_linear=predefined_wave_systems[rhs_choice][1]
        else:
            self.set_rhs("rhs", rhs_code)
            self.true_type_linear=true_type_linear
        self.toolboxes={"default":[self.run_wave_experiment], "field":[self.run_field_evolution_experiment]}

        if initials_for_res_check is None:
            gauss_small = 'x_rescaled = x/jnp.max(x)\nphi0=0.1*jnp.exp(-0.5*(x_rescaled**2/0.01))'
            gauss_medium = 'x_rescaled = x/jnp.max(x)\nphi0=jnp.exp(-0.5*(x_rescaled**2/0.01))'
            two_gauss = 'x_rescaled = x/jnp.max(x)\nphi0=0.2*jnp.exp(-0.5*(x_rescaled**2/0.01)) + 0.7*jnp.exp(-0.5*((x_rescaled-0.3)**2/0.01))'
            gauss_mom = 'x_rescaled = x/jnp.max(x)\nphi0=jnp.exp(-0.5*((x_rescaled+0.5)**2/0.01)) * jnp.exp(1j*5*(x_rescaled+0.5))'

            self.initials_for_res_check = [gauss_small, gauss_medium, two_gauss, gauss_mom]
        else:
            self.initials_for_res_check = initials_for_res_check


        if description_type=="WAVE":
            self.description=f"""This is a mystery wave system, namely a complex-valued wave field evolving on a 1D tight-binding lattice of {N_x} lattice points, with periodic boundary conditions. The x grid runs from -{x_max} to +{x_max}. For observational experiments, time runs up to {t_max} with {N_t} time grid points."""
        elif description_type=="FIELD":
            self.description=f"""This is a mystery system, showing the evolution of a complex-valued field evolving on a 1D tight-binding lattice of {N_x} lattice points, with periodic boundary conditions. The x grid runs from -{x_max} to +{x_max}. For observational experiments, time runs up to {t_max} with {N_t} time grid points."""
        else:
            self.description=f"""This is a mystery system, showing the evolution of a complex-valued field evolving on a 1D tight-binding lattice of {N_x} lattice points, with periodic boundary conditions. The x grid runs from -{x_max} to +{x_max}. For observational experiments, time runs up to {t_max} with {N_t} time grid points."""
            
    def run_wave_experiment(self, initial_condition_code:str) -> dict:
        """
        Run one experiment of the mystery wave system,
        where you can choose the initial condition.

        Args:
            initial_condition_code: python code with
                jax syntax that set the variable phi0,
                which represents the complex
                wave field at time 0.
                Use jnp.sin(...) etc, and use the
                array x which represents the position
                coordinate on a 1D grid.

        Returns:
            Solution with entries
            - ts: the time points (1D array)
            - x: the x grid (1D array)
            - phis: the complex wave field solution, 
                a 2D jax.Array of shape [n_ts,n_x]
        """
        return self.run_wave_simulation("rhs", initial_condition_code)

    # the same, but with 'field' instead of 'wave'
    def run_field_evolution_experiment(self, initial_condition_code:str) -> dict:
        """
        Run one experiment of the evolution of the mystery field,
        where you can choose the initial condition.

        Args:
            initial_condition_code: python code with
                jax syntax that set the variable phi0,
                which represents the complex
                field at time 0.
                Use jnp.sin(...) etc, and use the
                array x which represents the position
                coordinate on a 1D grid.

        Returns:
            Solution with entries
            - ts: the time points (1D array)
            - x: the x grid (1D array)
            - phis: the complex field solution, 
                a 2D jax.Array of shape [n_ts,n_x]
        """
        return self.run_wave_simulation("rhs", initial_condition_code)
        
    def save_result_linear(self, predicted_type:str, confidence:str) -> dict:
        """
        Save the result of your analysis. Also state your confidence in the prediction.

        Args:
            predicted_type: The predicted type of the wave field. Should either be 'LINEAR' or 'NONLINEAR'.
            confidence: The confidence in the prediction. Should be 'LOW', 'MEDIUM', or 'HIGH'.
        Returns:
            'save_message':str A message that the prediction has been saved.
        """
        if predicted_type.strip() == self.true_type_linear:
            self.last_result = {'result': True, 'predicted_type': predicted_type, 'true_type': self.true_type_linear, 'confidence': confidence}
        else:
            self.last_result = {'result': False, 'predicted_type': predicted_type, 'true_type': self.true_type_linear, 'confidence': confidence}
        
        return {'save_message': 'Saved predicted type. You can now stop the experiment.'}

    def save_result_find_eom_old(self, code:str) -> dict:
        """
        Save the result of your analysis, providing the code that would define
        the equations of motion of the field.

        You pass a python code that must be of the form:

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
            code: The code.
            
        Returns:
            'save_message' A message that the prediction has been saved.
        """
        self.last_result = {'code': code}
        return {'save_message': "The code has been saved." }
    
    def save_result_find_eom(self, code:str) -> dict:
        """
        Save the result of your analysis, providing the code that would define
        the equations of motion of the field.

        You pass a python code that must be of the form:

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
            code: The code.
            
        Returns:
            'save_message' A message that the prediction has been saved.
        """

        # try to set the rhs with the provided code
        set_rhs_result = self.set_rhs("final_predicted_rhs", code)
        if 'error' in set_rhs_result:
            self.last_result = {'code': code, 'R2_potential': jnp.nan, 'R2_kinetic': jnp.nan, 'R2': jnp.nan,
                            'error': set_rhs_result}
            return {'save_message': "The result has been saved." }

        # simulate for each initial condition
        phis_list = []
        for init in self.initials_for_res_check:
            res = self.run_wave_experiment(init)
            phis_list.append(res['phis'])
        phis_all = jnp.array(phis_list) # shape (n_initials, n_ts, n_x)

        U_pot_pred = self.rhs['final_predicted_rhs']['U_potential']
        U_kin_pred = self.rhs['final_predicted_rhs']['U_kinetic']
        dt = self.ts[1]-self.ts[0]


        def apply_Us_pred(phi, t):
            phi = jnp.fft.ifft( U_kin_pred( jnp.fft.fft(U_pot_pred(phi,self.x,t,dt)), self.k,t,dt ) )
            return phi
        def apply_Us_true(phi, t):
            phi = jnp.fft.ifft( self.rhs['rhs']['U_kinetic']( jnp.fft.fft(self.rhs['rhs']['U_potential'](phi,self.x,t,dt)), self.k,t,dt ) )
            return phi


        try:
            # vmap over initials and time steps
            apply_Us_pred_batched = jax.vmap(jax.vmap(apply_Us_pred, in_axes=(0,0)), in_axes=(0, None)) # batch over first argument
            apply_Us_true_batched = jax.vmap(jax.vmap(apply_Us_true, in_axes=(0,0)), in_axes=(0, None)) # batch over first argument

            phis_all_pred = apply_Us_pred_batched(phis_all, self.ts) # shape (n_initials, n_ts, n_x)
            phis_all_true = apply_Us_true_batched(phis_all, self.ts) # shape (n_initials, n_ts, n_x)

            # calculate R^2 values
            def r2_score(y_true, y_pred):
                ss_res = jnp.sum(jnp.abs(y_true - y_pred)**2)
                ss_tot = jnp.sum(jnp.abs(y_true - jnp.mean(y_true))**2) + 1e-10 # avoid division by zero
                return 1 - ss_res / ss_tot

            vel_true = (phis_all_true - phis_all) / dt
            vel_pred = (phis_all_pred - phis_all) / dt
            r2 = r2_score(vel_true.reshape(-1), vel_pred.reshape(-1))

        except Exception as e:
            self.last_result = {'code': code, 'R2': jnp.nan,
                            'error': f'Error during R2 calculation: {e}'}
            return {'save_message': "The result has been saved." }

        self.last_result = {'code': code, 'R2': float(r2),
                            'error': 'none'}

        return {'save_message': "The result has been saved." }

# Only for backward compatibility: (rather use WaveSimulator directly now)
class WaveSimulator_nonlinSEQ_1(WaveSimulator):
    def __init__(self, x_max: float, N_x: int, t_max: float, N_t: int):
        super().__init__(x_max,N_x,t_max,N_t,"nonlinSEQ_1",None)
        self.true_type_linear="NONLINEAR"
# Only for backward compatibility:
class WaveSimulator_nonlinSEQ_2(WaveSimulator):
    def __init__(self, x_max: float, N_x: int, t_max: float, N_t: int):
        super().__init__(x_max,N_x,t_max,N_t,"nonlinSEQ_2",None)
        self.true_type_linear="NONLINEAR"
        