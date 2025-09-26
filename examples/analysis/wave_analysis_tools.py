# Analysis tool for wave simulations
# This relies on sciexplorer.physics_waves, 
# which contains the actual simulation code

import sciexplorer.general_analysis_tools as analysis
import sciexplorer.physics_waves as waves
import jax.numpy as jnp

# ...set up x and ts --> can we define this outside this file?
x_max=5
N_x=100
t_max=20.
N_t=200
x_sim = jnp.linspace(-x_max,+x_max,N_x)
ts_sim = jnp.linspace(0.,t_max,N_t)
wave_sim = waves.WaveBasics(x_sim,ts_sim)

# note: there is no real difference here between
# default and field, except the latter everywhere
# only uses the nomenclature 'field', so as not
# to mislead into thinking that we only deal with
# wave equations...

toolboxes = {'default': [wave_sim.set_rhs, wave_sim.run_wave_simulation, 
            analysis.approx_equal, 
            analysis.get_all_previously_saved_results,
            analysis.plot_from_code,
            analysis.execute_code,],
            'field': [wave_sim.set_field_rhs, wave_sim.run_field_simulation, 
            analysis.approx_equal, 
            analysis.get_all_previously_saved_results,
            analysis.plot_from_code,
            analysis.execute_code,]}


