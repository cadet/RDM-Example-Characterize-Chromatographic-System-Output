# ---
# jupyter:
#   jupytext:
#     formats: md:myst,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fit Particle Porosity
#
# Particle porosity is an important consideration in the modeling of column transport. The speed in which a tracer is able to pass through the column is dependent on its interaction with the stationary phase. The porosity is a chemical property of the particles in the stationary phase. The tracer in the mobile phase must be able to penetrate the pores to interact with the stationary phase.
#
# ## Experiment
#
# To fit the particle porosity an experiment is conducted with acetone as a pore penetrating tracer. The tracer is injected into the column and its concentration at the column outlet is measured and compared to the concentration predicted by simulation results.
# - Acetone (pore penetrating tracer './experimental_data/pore_penetrating_tracer.csv')
# - data: time / s and c / mM

# %%
import numpy as np
data = np.loadtxt('experimental_data/pore_penetrating_tracer.csv', delimiter=',')

time_experiment = data[:, 0]
c_experiment = data[:, 1]

from CADETProcess.reference import ReferenceIO
tracer_peak = ReferenceIO(
    'Tracer Peak', time_experiment, c_experiment
)

if __name__ == '__main__':
    _ = tracer_peak.plot(x_axis_in_minutes = False)

# %% [markdown]
# ## Reference Model
#
# Here, initial values for `axial_dispersion` and `bed_porosity` are assumed. The `particle_porosity` will later be optimized, thus an arbitrary value can be set for now. `film_diffusion` is set to a value higher than 0 to allow for the pore penetrating tracer to enter the pores.

# %%
from CADETProcess.processModel import ComponentSystem
component_system = ComponentSystem(['Penetrating Tracer'])

# %%
from CADETProcess.processModel import Inlet, Outlet, LumpedRateModelWithPores

feed = Inlet(component_system, name='feed')
feed.c = [131.75]

eluent = Inlet(component_system, name='eluent')
eluent.c = [0]

column = LumpedRateModelWithPores(component_system, name='column')

column.length = 0.1
column.diameter = 0.0077
column.particle_radius = 34e-6

column.axial_dispersion = 1e-8
column.bed_porosity = 0.3
column.particle_porosity = 0.8

column.film_diffusion = [1]

outlet = Outlet(component_system, name='outlet')

# %%
from CADETProcess.processModel import FlowSheet

flow_sheet = FlowSheet(component_system)

flow_sheet.add_unit(feed)
flow_sheet.add_unit(eluent)
flow_sheet.add_unit(column)
flow_sheet.add_unit(outlet)

flow_sheet.add_connection(feed, column)
flow_sheet.add_connection(eluent, column)
flow_sheet.add_connection(column, outlet)

# %%
from CADETProcess.processModel import Process

Q_ml_min = 0.5  # ml/min
Q_m3_s = Q_ml_min/(60*1e6)
V_tracer = 50e-9  # m3

process = Process(flow_sheet, 'Tracer')
process.cycle_time = 15*60

process.add_event(
    'feed_on',
    'flow_sheet.feed.flow_rate',
    Q_m3_s, 0
)
process.add_event(
    'feed_off',
    'flow_sheet.feed.flow_rate',
    0,
    V_tracer/Q_m3_s
)

process.add_event(
    'feed_water_on',
    'flow_sheet.eluent.flow_rate',
     Q_m3_s,
     V_tracer/Q_m3_s
)

process.add_event(
    'eluent_off',
    'flow_sheet.eluent.flow_rate',
    0,
    process.cycle_time
)

# %% [markdown]
# ## Simulator

# %%
from CADETProcess.simulator import Cadet
simulator = Cadet()

if __name__ == '__main__':
    simulation_results = simulator.simulate(process)
    _ = simulation_results.solution.outlet.inlet.plot()

# %% [markdown]
# ## Comparator

# %%
from CADETProcess.comparison import Comparator

comparator = Comparator()
comparator.add_reference(tracer_peak)
comparator.add_difference_metric(
    'NRMSE', tracer_peak, 'outlet.outlet',
)

if __name__ == '__main__':
    comparator.plot_comparison(simulation_results)

# %% [markdown]
# ## Optimization Problem

# %%
from CADETProcess.optimization import OptimizationProblem

optimization_problem = OptimizationProblem('particle_porosity')

optimization_problem.add_evaluation_object(process)

optimization_problem.add_variable(
    name='particle_porosity',
    parameter_path='flow_sheet.column.particle_porosity',
    lb=0.5, ub=0.99,
    transform='auto'
)

optimization_problem.add_evaluator(simulator)

optimization_problem.add_objective(
    comparator,
    n_objectives=comparator.n_metrics,
    requires=[simulator]
)

def callback(simulation_results, individual, evaluation_object, callbacks_dir='./'):
    comparator.plot_comparison(
        simulation_results,
        file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_comparison.png',
        show=False
    )


optimization_problem.add_callback(callback, requires=[simulator])

# %% [markdown]
# ## Optimizer

# %%
from CADETProcess.optimization import U_NSGA3
optimizer = U_NSGA3()
optimizer.n_max_gen = 3
optimizer.pop_size = 3
optimizer.n_cores = 3

# %% [markdown]
# ## Run Optimization

# %%
optimization_results = optimizer.optimize(
    optimization_problem,
    use_checkpoint=False )

# %% [markdown]
# ### Optimization Progress and Results
#
# The `OptimizationResults` which are returned contain information about the progress of the optimization.
# For example, the attributes `x` and `f` contain the final value(s) of parameters and the objective function.

# %%
print(optimization_results.x)
print(optimization_results.f)

# %% [markdown]
# After optimization, several figures can be plotted to vizualize the results. For example, the convergence plot shows how the function value changes with the number of evaluations.

# %%
optimization_results.plot_convergence()

# %% [markdown]
# The plot_objectives method shows the objective function values of all evaluated individuals. Here, lighter color represent later evaluations. Note that by default the values are plotted on a log scale if they span many orders of magnitude. To disable this, set autoscale=False.

# %%
optimization_results.plot_objectives()

# %% [markdown]
# All figures are saved automatically in the `working_directory`.
# Moreover, results are stored in a `.csv` file.
# - The `results_all.csv` file contains information about all evaluated individuals.
# - The `results_last.csv` file contains information about the last generation of evaluated individuals.
# - The `results_pareto.csv` file contains only the best individual(s).
