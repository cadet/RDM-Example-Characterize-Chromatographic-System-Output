# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fit Pore Transport Parameters
#
# The system we characterize in this example uses a porous stationary phase, such as beads made of cross-linked agarose.
# This means that there exists a stagnant liquid phase within the stationary phase and components can diffuse into and out of this stagnant liquid phase. The tracer in the mobile phase must be able to penetrate the pores to interact with the stationary phase.
#
# In this example we will use the [Lumped Rate Model with Pores](https://cadet.github.io/master/modelling/unit_operations/lumped_rate_model_with_pores.html#lumped-rate-model-with-pores-model) to describe the pore transport. Within this model, two parameters characterize the pore transport:
#
# - The `particle_porosity` describes the ratio of liquid phase volume within the stationary phase to total volume of the stationary phase.
# - The `film_diffusion` desribes the diffusion speed with which components diffuse into and out of the stagnant liquid phase.
#
# One approach to determine these parameters is the inverse method.
# By adjusting the values of the parameters in the simulation model and comparing the resulting behavior to the experimental data, the optimal parameter values that match the observed behavior can be found.
#
# ## Experiment
#
# To fit the pore transport parameters an experiment is conducted with a pore-penetrating tracer.
# The tracer is injected into the column and its concentration at the column outlet is measured.
#
# In this example, acetone is used as a non-binding, pore-penetrating tracer for a hypothetical ion exchange column.
# This is useful if the column is to be used to separate small molecules.
# If the target molecules are large molecules, such as proteins, it can be advisable to perform the pore-penetrating tracer experiment using the target proteins under non-binding conditions, such as with a buffer with a high salt concentration for ion exchange chromatography.
#
# The acetone data is stored in './experimental_data/pore_penetrating_tracer.csv' as time in seconds and concentration in mM.

# %%
import numpy as np

data = np.loadtxt("experimental_data/pore_penetrating_tracer.csv", delimiter=",")

time_experiment = data[:, 0]
c_experiment = data[:, 1]

from CADETProcess.reference import ReferenceIO

tracer_peak = ReferenceIO("Tracer Peak", time_experiment, c_experiment)

if __name__ == "__main__":
    _ = tracer_peak.plot(x_axis_in_minutes=False)

# %% [markdown]
# ## Reference Model
#
# Here, initial values for `film_diffusion` and `particle_porosity` are assumed, as they will later be optimized, thus arbitrary values (within reason) can be set for now. `film_diffusion` is set to a value > 0 m/s to allow for the tracer to enter the pores. `particle_porosity` is set to a value between 0 and 1.

# %%
from CADETProcess.processModel import ComponentSystem

component_system = ComponentSystem(["Penetrating Tracer"])

# %%
from CADETProcess.processModel import Inlet, Outlet, LumpedRateModelWithPores

feed = Inlet(component_system, name="feed")
feed.c = [131.75]

eluent = Inlet(component_system, name="eluent")
eluent.c = [0]

column = LumpedRateModelWithPores(component_system, name="column")

column.length = 0.1
column.diameter = 0.0077
column.particle_radius = 34e-6

column.axial_dispersion = 1e-8
column.bed_porosity = 0.3

column.particle_porosity = 0.5
column.film_diffusion = [1e-5]

outlet = Outlet(component_system, name="outlet")

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
Q_m3_s = Q_ml_min / (60 * 1e6)
V_tracer = 50e-9  # m3

process = Process(flow_sheet, "Tracer")
process.cycle_time = 15 * 60

process.add_event("feed_on", "flow_sheet.feed.flow_rate", Q_m3_s, 0)
process.add_event("feed_off", "flow_sheet.feed.flow_rate", 0, V_tracer / Q_m3_s)

process.add_event(
    "feed_water_on", "flow_sheet.eluent.flow_rate", Q_m3_s, V_tracer / Q_m3_s
)

process.add_event("eluent_off", "flow_sheet.eluent.flow_rate", 0, process.cycle_time)

# %% [markdown]
# ## Simulator

# %%
from CADETProcess.simulator import Cadet

simulator = Cadet()

if __name__ == "__main__":
    simulation_results = simulator.simulate(process)
    _ = simulation_results.solution.outlet.inlet.plot(x_axis_in_minutes=False)

# %% [markdown]
# ## Comparator

# %%
from CADETProcess.comparison import Comparator

comparator = Comparator()
comparator.add_reference(tracer_peak)
comparator.add_difference_metric(
    "NRMSE",
    tracer_peak,
    "outlet.outlet",
)

if __name__ == "__main__":
    comparator.plot_comparison(simulation_results, x_axis_in_minutes=False)

# %% [markdown]
# ## Optimization Problem

# %%
from CADETProcess.optimization import OptimizationProblem

optimization_problem = OptimizationProblem("particle_porosity")

optimization_problem.add_evaluation_object(process)

optimization_problem.add_variable(
    name="particle_porosity",
    parameter_path="flow_sheet.column.particle_porosity",
    lb=0.3,
    ub=0.8,
    transform="auto",
)
optimization_problem.add_variable(
    name="film_diffusion",
    parameter_path="flow_sheet.column.film_diffusion",
    lb=1e-6,
    ub=1e-2,
    transform="auto",
)

optimization_problem.add_evaluator(simulator)

optimization_problem.add_objective(
    comparator, n_objectives=comparator.n_metrics, requires=[simulator]
)


def callback(simulation_results, individual, evaluation_object, callbacks_dir="./"):
    comparator.plot_comparison(
        simulation_results,
        file_name=f"{callbacks_dir}/{individual.id}_{evaluation_object}_comparison.png",
        show=False,
    )


optimization_problem.add_callback(callback, requires=[simulator])

# %% [markdown]
# ## Optimizer

# %%
from CADETProcess.optimization import U_NSGA3

optimizer = U_NSGA3()
optimizer.n_max_gen = 10
optimizer.pop_size = 12
optimizer.n_cores = 12

# %% [markdown]
# ## Run Optimization

# %%
optimization_results = optimizer.optimize(optimization_problem, use_checkpoint=False)

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
