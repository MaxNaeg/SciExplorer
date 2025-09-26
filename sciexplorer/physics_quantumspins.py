import functools
from functools import partial
import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np

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

# batched parameters:
parallel_param_runge_kutta_solve = vmap(runge_kutta_solve,in_axes=[None,None,None,0],
                                                   out_axes=0)

# batched initial conditions:
parallel_runge_kutta_solve = vmap(runge_kutta_solve,in_axes=[0,None,None,None],
                                                   out_axes=0)

def seq_rhs(H, psi, t):
    return -1j*(H @ psi)
    
class SEQsolver:
    def __init__(self):        
        self.Hamiltonian={}
        self.Operators={}
        self.description=f"""You have access to a solver for the Schrödinger equation for a spin system, 
where you can set the number of spins using the init_spins tool and then later set the Hamiltonian using
set_Hamiltonian."""
        self.toolboxes = {'SEQsolver': [self.init_spins, self.set_Hamiltonian, self.solve_SEQ],
                         'SEQgroundstate': [self.init_spins, self.set_Hamiltonian, self.set_operator, self.get_ground_state_expectations]}

    def init_spins(self, N: int):
        """
        Set the number of spins in the system. This needs to be called before setting
        the Hamiltonian. It may be changed later if you want to run different simulations, but
        then you need to set the Hamiltonian again.

        Args:
            N: number of spins

        Returns:
            Success message
        """
        self.N=N
        
        self.Xop=jnp.array([[0.,1.],[1.,0.]]) # Pauli X as a sparse matrix
        self.Yop=jnp.array([[0,-1j],[1j,0.]])
        self.Zop=jnp.array([[1.,0.],[0.,-1]])
        self.Sx,self.Sy,self.Sz=self.produce_XYZ(N)
        return {"message": f"Succcessfully set up spin system for {N} spins"}
        
    def operator(self,op,idx,N):
      """
      Produce an operator that has single-qubit operator op acting
      on qubit with index idx, and the rest is identity. N is the total number of qubits.
      idx count starts at 0 and runs up to N-1.
    
      Example: Z_3
      >> operator(Zop,3,10)
    
      Example: X_0 X_2
      >>  operator(Xop,0,10) @ operator(Xop,2,10)
      """
      return jnp.kron( jnp.identity(2**idx), jnp.kron(op, jnp.identity(2**(N-idx-1))) )
    
    def get_identity(self,N):
      """
      Return identity operator for an N-qubit system
      """
      return jnp.identity(2**N)
    
    def expectation(self,psi,op):
      """
      Calculate the expectation value of observable operator op with respect to state psi.
      This takes the real part automatically, assuming op is Hermitean !
      """
      return jnp.real( jnp.sum( jnp.conj(psi) * (op @ psi) ) )
        
    def empty_matrix(self,n_dim):
      """
      Return the empty sparse matrix for a system of Hilbert space dimension n_dim.
      """
      return jnp.zeros((n_dim, n_dim), dtype='complex')
    
    def produce_XYZ(self,N):
      """
      Return X,Y,Z = a list of all Pauli X operators, all Pauli Y, all Pauli Z,
      for given qubit number N
      """
      return ( [self.operator(self.Xop,idx,N) for idx in range(N)] ,
                [self.operator(self.Yop,idx,N) for idx in range(N)] ,
                [self.operator(self.Zop,idx,N) for idx in range(N)] )

    def get_product_state(self, bloch_vectors:jax.Array) -> jax.Array:
        """
        Given the normalized bloch_vectors, of shape [N,3],
        construct a product state and return it.
        
        Returns:
            psi, the product state
        """

        psis=[]
        for idx in range(self.N):
            # extract angles from Bloch vector and construct psi
            nx,ny,nz=bloch_vectors[idx]
            theta=jnp.arccos(nz)
            phi=jnp.arctan2(ny,nx)
            psis.append( jnp.array( [jnp.cos(theta/2), jnp.sin(theta/2)*jnp.exp(1j*phi)]) )

        return functools.reduce(jnp.kron, psis)

    def calculate_Hamiltonian(self, hamiltonian_code: str) -> jax.Array:
        """
        Internal method for actually calculating the Hamiltonian from code.
        """
        #local_dict={"H": None, "Sx": self.Sx, "Sy": self.Sy, "Sz": self.Sz}
        #exec(hamiltonian_code, None, local_dict)
        global_dict={"H": None, "Sx": self.Sx, "Sy": self.Sy, "Sz": self.Sz}
        exec(hamiltonian_code, global_dict, None)
        return global_dict["H"]


    def set_operator(self, operator_label:str, operator_code: str) -> dict:
        """
        Define a Hermitian operator, whose expectation value in the ground state of a system
        can later be evaluated.

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
        try:
            H=self.calculate_Hamiltonian(operator_code)
        except Exception as e:
            return {'error': f"Error in executing provided code: {e}"}
        if H is None:
            return {'error': "Error: Operator H not defined in code."}
        self.Operators[operator_label] = H
        return {'result_str': f"Success: Set operator successfully and saved it under the label: {operator_label}."}
        
    def set_Hamiltonian(self, hamiltonian_label:str, hamiltonian_code: str) -> dict:
        """
        Define the Hamiltonian for a quantum system to be simulated.

        You pass a python code that must produce a Hamiltonian H, constructing
        it out of provided spin operators. Here Sx is a list of spin operator x-components,
        Sy likewise for the y-components, and Sz for the z-components.
        These are Pauli matrices.

        They can be accessed like Sx[2] etc. Remember to use
        the "@" matrix multiplication operator when taking
        the product of several spin operators.
        
        Otherwise you can use jax.numpy syntax in the form "jnp.sin(...)".
        
        Args:
            hamiltonian_label: the label the Hamiltonian will be stored under.
            hamiltonian_code: the python code defining the Hamiltonian.
    
        Returns:
            Message indicating wether the Hamiltonian was set successfully set.
        """
        try:
            H=self.calculate_Hamiltonian(hamiltonian_code)
        except Exception as e:
            print(f"setHamiltonian: Error in executing provided code: {e}")
            return {'error': f"Error in executing provided code: {e}"}
        if H is None:
            print("Error: Hamiltonian H not defined in code.")
            return {'error': "Error: Hamiltonian H not defined in code."}
        self.Hamiltonian[hamiltonian_label] = H
        return {'result_str': f"Success: Set Hamiltonian successfully and saved it under the label: {hamiltonian_label}."}
        
    def do_solve_SEQ(self, Hamiltonian: jax.Array, bloch_vectors: jax.Array, T: float, dt: float) -> dict:
        """
        Internal method for actually solving the SEQ.
        """
        ts=jnp.linspace(0.0, T, int(T/dt)+1)
        psi0 = self.get_product_state(jnp.array(bloch_vectors))
        Psis, ts = runge_kutta_solve( psi0, lambda psi,t,params: seq_rhs(Hamiltonian, psi, t), ts, None )
        nsteps=len(ts)
        # cumbersome way to get all expectation values, doing a loop!
        # should replace by batching
        Sx_t=jnp.vstack([jnp.array([self.expectation(Psis[idx,:],self.Sx[site]) for idx in range(nsteps)]) for site in range(self.N)])
        Sy_t=jnp.vstack([jnp.array([self.expectation(Psis[idx,:],self.Sy[site]) for idx in range(nsteps)]) for site in range(self.N)])
        Sz_t=jnp.vstack([jnp.array([self.expectation(Psis[idx,:],self.Sz[site]) for idx in range(nsteps)]) for site in range(self.N)])
        return ts,Sx_t,Sy_t,Sz_t

    def solve_SEQ(self, hamiltonian_label: str, bloch_vectors: jax.Array,
                  T: float, dt: float) -> dict:
        """
        Solve the time-dependent Schrödinger equation for a given
        fixed Hamiltonian that was defined previously. The initial
        state is given as a product state over the spins making up the
        system. The spins are spin-1/2, described by Pauli matrices
        Sx, Sy, and Sz.

        Args:
            hamiltonian_label: the label for the previously defined Hamiltonian
            bloch_vectors: an array (jax.Array) of shape [N,3], where N is the number
                of spins in the system, and for each spin a unit Bloch vector is prescribed
                that determines the initial direction of the spin.
            T: the time until which the equation should be solved
            dt: the time step size for the solution. nsteps will be int(T/dt)+1.

        Returns:
            'ts': jax.Array of shape [nsteps] (with the time steps 'nsteps')
            'Sx_t': jax.Array of shape [nsteps, N] with the expectation values of all Sx operators
            'Sy_t': jax.Array of shape [nsteps, N] with the expectation values of all Sy operators
            'Sz_t': jax.Array of shape [nsteps, N] with the expectation values of all Sz operators
        """    
        try:
            H = self.Hamiltonian[hamiltonian_label]
        except KeyError:
            return {"error": f"Error: Hamiltonian with label '{hamiltonian_label}' not found."}
        try:
            ts,Sx_t,Sy_t,Sz_t=self.do_solve_SEQ(H, bloch_vectors, T, dt)                
        except Exception as e:
            return {"error": f"Error in solving ODE: {e}"}
        solution={"ts":ts,
                  "Sx_t":Sx_t, "Sy_t":Sy_t, "Sz_t":Sz_t}
        return {'to_save': solution, 'to_model': f"Success: Equation solved."}

    def do_get_ground_state(self, Hamiltonian: jax.Array) -> dict:
        """
        Internal method for actually obtaining the ground state.
        """
        evals,evecs=np.linalg.eigh(Hamiltonian)
        return evecs[:,0] # evals was ordered in ascending order, so this should be ground state

    def get_ground_state_expectations(self, hamiltonian_label: str, operators: str) -> dict:
        """
        Obtain the ground state expectation values of several operators.

        Args:
            hamiltonian_label: the label for the previously defined Hamiltonian
            operators: a string with a comma-delimited list of operator labels (previously defined via set_operator)

        Returns:
            dict containing the expectation values for each of the operators (dict key named according to the operator label)
        """
        try:
            H = self.Hamiltonian[hamiltonian_label]
        except KeyError:
            return {"error": f"Error: Hamiltonian with label '{hamiltonian_label}' not found."}

        psi=self.do_get_ground_state(H)
        
        try:
            results={}
            for oplabel in operators.split(','):
                strippedlabel=oplabel.strip()
                results[strippedlabel]=self.expectation(psi,self.Operators[strippedlabel])
        except Exception as e:
            return {"error": f"Error in obtaining expectation values: {e}"}
        
        return results
        