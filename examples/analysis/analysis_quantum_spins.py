import sciexplorer.general_analysis_tools as analysis
import sciexplorer.physics_quantumspins as qspins
import jax.numpy as jnp

qspin_sim=qspins.SEQsolver()

toolboxes = {'default': [
            qspin_sim.init_spins,
            qspin_sim.set_Hamiltonian,
            qspin_sim.solve_SEQ, 
            analysis.approx_equal, 
            analysis.get_all_previously_saved_results,
            analysis.plot_from_code,
            analysis.execute_code,],
            'groundstate': [
            qspin_sim.init_spins,
            qspin_sim.set_Hamiltonian,
            qspin_sim.set_operator,
            qspin_sim.get_ground_state_expectations,
            analysis.approx_equal,
            analysis.get_all_previously_saved_results,
            analysis.plot_from_code,
            analysis.execute_code,               
            ]}
