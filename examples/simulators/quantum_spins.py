import sciexplorer.physics_quantumspins as qspins
import jax
import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt

class QuantumSpinSystem:
    def __init__(self, hamiltonian_code: str = "", N: int = 2, observed=None,
                 default_Bloch_vectors=None, ground_state_experiments=False,
                ground_state_experiments_with_params=False,
                 ground_state_experiments_with_params_variable_N=False,
                 dynamics_with_params=False,
                additional_description="",
                params_dict:dict=None,
                variable_N_result_eval_N:int=10,
                eval_seed:int=0,
                n_random_evals:int=100):
        """

        params_dict: dictionary with parameters that can be set by the agent, in the form 'param_name': [min_value, max_value]
        variable_N_result_eval_N: if the number of spins N can be set by the agent, this is the number
            of spins N that will be used to evaluate the result (e.g. ground state fidelity)

        The hamiltonian_code must access Sx, Sy, Sz which are arrays
        of length N, to construct the Hamiltonian H. Sx,Sy,Sz are
        Pauli matrices.

        If observed is set to a list, the agent can only observe
        those spins of the system whose indices are indicated in the list.

        If default_Bloch_vectors is not None, but a jax.Array of shape (N,3), this means all Bloch vectors
        are by default pointing in that direction and the agent can only
        control those Bloch vectors of the spins given in "observed".

        If ground_state_experiments is True, then instead of offering to observe
        the time-dependent dynamics, we offer to observe ground-state expectation values.

        If ground_state_experiments_with_params is True, then instead of offering to observe
        the time-dependent dynamics, we offer to observe ground-state expectation values,
        with the ability to set system parameters (for phase diagram exploration). 

        additional_description can be used to set any additional text.
        """
        self.N=N
        self.observed=observed
        self.description=f"You do have access to an experimental system of {self.N} spins."
        self.solver=qspins.SEQsolver()
        self.solver.init_spins(N)
        self.hamiltonian_code=hamiltonian_code
        self.n_random_evals=n_random_evals
        self.last_result=None

        self.ground_state_experiments_with_params_variable_N=ground_state_experiments_with_params_variable_N
        self.ground_state_experiments_with_params=ground_state_experiments_with_params
        self.ground_state_experiments=ground_state_experiments
        self.dynamics_with_params=dynamics_with_params

        self.eval_key = jax.random.PRNGKey(eval_seed)

        self.params_dict=params_dict
        self.variable_N_result_eval_N=variable_N_result_eval_N

        if ground_state_experiments:
            self.toolboxes={'default': [self.set_operator_for_ground_state, self.run_experiment_ground_state]}
            self.solver.set_Hamiltonian("theHamiltonian", hamiltonian_code)
        elif ground_state_experiments_with_params:
            self.toolboxes={'default': [self.set_operator_for_ground_state, self.run_experiment_ground_state_with_parameters]}
        elif ground_state_experiments_with_params_variable_N:
            self.toolboxes={'default': [self.set_N,self.set_operator_for_ground_state, self.run_experiment_ground_state_with_parameters]}
            self.description=f"You do have access to an experimental system of N spins, where you can select the number N (size of the system)."
        else: # time dynamics
            if dynamics_with_params:
                self.toolboxes={'default': [self.run_experiment_with_parameters]}
            else:
                self.toolboxes={'default': [self.run_experiment]}
                self.solver.set_Hamiltonian("theHamiltonian", hamiltonian_code)
            self.default_Bloch_vectors=default_Bloch_vectors
            self.description=f"You do have access to an experimental system of {self.N} spins."
            if observed is not None:
                self.description += f"\nYou will only be able to observe the spins {observed} of this system, whose spin expectation values will be returned in this order."
            if default_Bloch_vectors is not None:
                self.description += f"\nYou will only be able to control the N_control={len(observed)} spins that are also observable. All others will be set to some fixed default values every time you start an experiment."
            

        self.description+=additional_description

    def set_N(self, N: int) -> dict:
        """
        Set the number of spins of the experimental system.
        Please do not choose values larger than about N=12, to avoid lengthy numerical computations.
        After setting a new number N, you have to redefine the operators whose expectation values
        you want to evaluate in the ground state, and when running the experiment you have to
        also specify the new N.
        
        Args:
            N: the desired number N of spins.

        Returns:
            Message indicating success.
        """
        if N>16:
            return {"message":"error: N is too large!"}
        
        self.N=N
        self.solver.init_spins(N)
        return {"message":"Successfully set new number N of spins!"}
        
    def run_experiment_ground_state(self, operators: str) -> dict:
        """
        Measure the ground state expectation values for several operators, for the given
        experimental system.

        Args:
            operators: a string with a comma-delimited list of operator labels (previously defined via set_operator)

        Returns:
            dict containing the expectation values for each of the operators (dict key named according to the operator label)
        """
        return self.solver.get_ground_state_expectations("theHamiltonian",operators)

    def run_experiment_ground_state_with_parameters(self, set_params_code: str, operators: str) -> dict:
        """
        Measure the ground state expectation values for several operators, for the given
        experimental system, for given physical parameters. If the experimental system
        has variable size N, you must also set N in the code given here!

        Args:
            set_params_code: python code that sets the numerical values of the system parameters.
            operators: a string with a comma-delimited list of operator labels (previously defined via set_operator)

        Returns:
            dict containing the expectation values for each of the operators (dict key named according to the operator label)
        """
        return_value=self.solver.set_Hamiltonian("theHamiltonian", set_params_code+"\n"+self.hamiltonian_code)
        if 'error' in return_value: # something went wrong even when setting the Hamiltonian!
            return return_value
        return self.solver.get_ground_state_expectations("theHamiltonian",operators)
        
    def set_operator_for_ground_state(self, operator_label:str, operator_code: str) -> dict:
        """
        Define a Hermitian operator, whose expectation value in the ground state of 
        the experimental system can later be evaluated.

        You pass a python code that must produce an operator H, constructing
        it out of provided spin operators. Here Sx is a list of spin operator x-components,
        Sy likewise for the y-components, and Sz for the z-components.
        These are Pauli matrices.

        They can be accessed like Sx[2] etc. Remember to use
        the "@" matrix multiplication operator when taking
        the product of several spin operators.
        
        Otherwise you can use jax.numpy syntax in the form "jnp.sin(...)".
        
        Args:
            operator_label: the label the operator will be stored under.
            operator_code: the python code defining the operator.
    
        Returns:
            Message indicating wether the operator was set successfully.
        """        
        return self.solver.set_operator(operator_label,operator_code)
        
    def run_experiment(self, bloch_vectors:jax.Array, T: float, dt: float) -> dict:
        """
        Run an experiment on the spin system. You can provide the normalized Bloch
        vectors to describe the initial product state. The system will evolve
        according to its time-independent Hamiltonian, and the evolution of the
        spin expectation values will be returned. The spins are described
        by Pauli operators Sx, Sy, and Sz. The number N_obs of observable spin
        operators may be smaller than the total number N of spins. The number N_control
        of controllable spin operators may be smaller than the total number N (see
        description of experimental setup).

        Args:
            bloch_vectors: array (jax.Array) of shape (N_control,3) of Bloch vectors.
            T: total time duration of experiment.
            dt: time step between observations. nsteps will be int(T/dt)+1.

        Returns:
            'ts': jax.Array of shape [nsteps] (with the time steps 'nsteps')
            'Sx_t': jax.Array of shape [N_obs,nsteps] with the expectation values of all observed Sx operators
            'Sy_t': jax.Array of shape [N_obs,nsteps] with the expectation values of all observed Sy operators
            'Sz_t': jax.Array of shape [N_obs,nsteps] with the expectation values of all observed Sz operators
        """
        Hamiltonian=self.solver.Hamiltonian["theHamiltonian"]
        
        if self.default_Bloch_vectors is not None:
            new_bloch_vectors=np.array(self.default_Bloch_vectors)
            for idx,spin_idx in enumerate(self.observed):
                new_bloch_vectors[spin_idx,:]=bloch_vectors[idx,:]
            bloch_vectors=jnp.array(new_bloch_vectors)
        ts,Sx_t,Sy_t,Sz_t=self.solver.do_solve_SEQ(Hamiltonian,bloch_vectors,T,dt)
        if self.observed is None:
            return {"ts":ts, "Sx_t":Sx_t, "Sy_t":Sy_t, "Sz_t":Sz_t}
        else:
            return {"ts":ts, "Sx_t":Sx_t[self.observed,:], "Sy_t":Sy_t[self.observed,:], "Sz_t":Sz_t[self.observed,:]}

    def run_experiment_with_parameters(self, bloch_vectors:jax.Array, T: float, dt: float, parameter_code: str) -> dict:
        """
        Run an experiment on the spin system. You can provide the normalized Bloch
        vectors to describe the initial product state. The system will evolve
        according to its time-independent Hamiltonian, and the evolution of the
        spin expectation values will be returned. The spins are described
        by Pauli operators Sx, Sy, and Sz. The number N_obs of observable spin
        operators may be smaller than the total number N of spins. The number N_control
        of controllable spin operators may be smaller than the total number N (see
        description of experimental setup).

        Args:
            bloch_vectors: array (jax.Array) of shape (N_control,3) of Bloch vectors.
            T: total time duration of experiment.
            dt: time step between observations. nsteps will be int(T/dt)+1.
            parameter_code: python code that sets the parameter numerical values

        Returns:
            'ts': jax.Array of shape [nsteps] (with the time steps 'nsteps')
            'Sx_t': jax.Array of shape [N_obs,nsteps] with the expectation values of all observed Sx operators
            'Sy_t': jax.Array of shape [N_obs,nsteps] with the expectation values of all observed Sy operators
            'Sz_t': jax.Array of shape [N_obs,nsteps] with the expectation values of all observed Sz operators
        """
        self.solver.set_Hamiltonian("theHamiltonian",parameter_code+"\n"+self.hamiltonian_code)
        return self.run_experiment(bloch_vectors, T, dt)

    def fidelity(self, psi1, psi2):
        return jnp.real(jnp.abs( jnp.sum(jnp.conj(psi1) * psi2) )**2 )
        
    def fidelity_for_ground_states(self, H1, H2):
        dim = H1.shape[0]
        N = int(jnp.log2(dim))
        return self.fidelity(self.solver.do_get_ground_state(H1),
                                 self.solver.do_get_ground_state(H2))**(1/N)
    def fidelity_for_Hamiltonians(self, H1, H2):
        dim = H1.shape[0]
        H1_p = H1 - jnp.trace(H1)/dim * jnp.eye(dim)
        H2_p = H2 - jnp.trace(H2)/dim * jnp.eye(dim)
        normH1 = jnp.sqrt(jnp.trace( jnp.conj(jnp.transpose(H1_p)) @ H1_p ))
        normH2 = jnp.sqrt(jnp.trace( jnp.conj(jnp.transpose(H2_p)) @ H2_p ))
        return jnp.real(jnp.trace( jnp.conj(jnp.transpose(H1_p)) @ H2_p ) / jnp.max(jnp.array([normH1, normH2]))**2)

    
    def announce_Hamiltonian(self, Hamiltonian: str) -> dict:
        """
        Announce the correct Hamiltonian, in the form of python code.

        You pass a python code that must produce an operator H, constructing
        it out of provided spin operators. Here Sx is a list of spin operator x-components,
        Sy likewise for the y-components, and Sz for the z-components.
        These are Pauli matrices.

        They can be accessed like Sx[2] etc. Remember to use
        the "@" matrix multiplication operator when taking
        the product of several spin operators.

        If the Hamiltonian contains a tuneable parameter (or several), use the parameter name(s) specified in the problem description and do not substitute numerical values.
        However, if the Hamiltonian contains non-tuneable numerical parameters, specify them as floating point numbers.

        Returns:
            Message that the Hamiltonian has been stored.
        """
            # do this for ground state announce_Hamiltonian result tool:
            #   either for case without params: use 'self.solver.set_Hamiltonian' with the
            #      announced Hamiltonian (from agent) to get that Hamiltonian, and our
            #      correct H. is in self.solver.Hamiltonian["theHamiltonian"]
            #   for case with params: have to know number and names of params, loop over
            #   some random values, construct 'true' Hamiltonian with "A=4\nB=3\nb"+self.hamiltonian_code,
            #   and then also construct agent Hamiltonian from "A=4\nB=3\nb"+announced_Hamiltonian
            #   for case with params and N: set N to some reasonable value, use self.set_N to set this,
            #   proceed further in same manner as above (inserting N=... in addition
            #   to other params)
            # for dynamics: maybe (1/N) log | tr(H1^dagger H2) |
        N = -1
        self.last_result = {'fidelity': jnp.nan, 'N': N, 'error': 'No result yet', 'evals': str(self.n_random_evals), 'Hamiltonian': Hamiltonian}

        try:
            # if (not self.ground_state_experiments_with_params and not self.ground_state_experiments_with_params_variable_N
            #     and not self.dynamics_with_params):

            if (self.ground_state_experiments or 
                self.ground_state_experiments_with_params or
                self.ground_state_experiments_with_params_variable_N):
                fidelity_func = self.fidelity_for_ground_states
            else:
                fidelity_func= self.fidelity_for_Hamiltonians
            fidelities = []
            if self.params_dict is None:
                N = int(self.N)
                self.set_N(N)
                H_true = self.solver.Hamiltonian["theHamiltonian"]
                self.solver.set_Hamiltonian("agentHamiltonian", Hamiltonian)
                H_agent = self.solver.Hamiltonian['agentHamiltonian']
                fidelities = [fidelity_func(H_true, H_agent)]
            else:
                if self.ground_state_experiments_with_params_variable_N:
                    N=int(self.variable_N_result_eval_N)
                    self.set_N(N)
                else:
                    N=int(self.N)
                self.set_N(N)

                for _ in range(self.n_random_evals):
                    init_str = f"N={N}\n"
                    for param, (minp, maxp) in self.params_dict.items():
                        self.eval_key, subkey = jax.random.split(self.eval_key)
                        val = jax.random.uniform(subkey, minval=minp, maxval=maxp)
                        init_str += f"{param}={val}\n"
                    self.solver.set_Hamiltonian("theHamiltonian", init_str + self.hamiltonian_code)
                    H_true = self.solver.Hamiltonian["theHamiltonian"]
                    self.solver.set_Hamiltonian("agentHamiltonian", init_str + Hamiltonian)
                    H_agent = self.solver.Hamiltonian["agentHamiltonian"]
                    fidelities.append(fidelity_func(H_true, H_agent))

            avg_fidelity = float(jnp.mean(jnp.array(fidelities)))
            self.last_result = {'fidelity': avg_fidelity, 'N': N, 'evals': str(self.n_random_evals), 'Hamiltonian': Hamiltonian}
        except Exception as e:
            print(f"Error: {e}")
            self.last_result = {'fidelity': jnp.nan, 'N': N, 'error': str(e), 'evals': str(self.n_random_evals), 'Hamiltonian': Hamiltonian}

        return {"message": "The Hamiltonian has been stored."}
            
    def save_phase_diagram(self, code: str) -> dict:
        """
        Save the approximate phase diagram for this system.

        Args:
            code: Stand-alone python code that will plot a clear and attractive visualization 
            of the approximate phase diagram, with labels for the phases; this code cannot access any previously
            existing variables, but it can use plt (from matplotlib) and np (numpy). Uncertainties
            in the phase boundaries should be indicated.

        Returns:
            Message that the code has been saved.
        """
        self.last_result=code
        locals_dict= {'result':None}
        globals_dict = {'plt':plt, 'np':np}
        try:
            exec(code, globals_dict, locals_dict)
            plt.show()
        except Exception as e:
            print(f"Error: {e}")
        
        return {"message": "The phase diagram has been stored."}
        