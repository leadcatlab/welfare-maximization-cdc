## Data Processing Overview


Currently, the raw data is defined in terms of components. We have a finite set of component types, and each component type (say 'Other Roofing') has unique dynamics. However, there can be multiple instances of the same component type in a single building and each of these instances can have different replacement and inspection costs. 

## Raw Data
The raw data comes in two files:

`component_costs_raw.csv` - Each row in this file an instance of a particular component type. Each row is associated with a replacement cost and importance index, and a uniformat code that specifies the component type.

`component_dynamics_raw.csv` - Each row in this file is a unique component type associated with a unique uniformat code. Each unique component type is associated with the mean and standard deviation of the shape and the scale parameters that dictate the dynamics of the components belonging to that type. 

## Data Processing
The raw data in combined and processed into a single csv file in the `data_processing.ipynb` notebook. This single file with 37 rows where each row contains scaled replacement cost, inspection cost. shape and scale parameters, and importance score is saved as `components_data_true.csv`. Note that we scale the total maintainance budget value to the range 0 to 1000. Although we expect the scaled replacement costs to be less than 1000, that is not necessarily the case as some of the components have replacement costs greater than the total maintenance budget.

## Component Selection

For now, we select 15 components out of the given 37 components. We select the components based on the following criteria:

1. Avoid components whose replacement cost is greater than the total maintenance cost of the building.
2. Among the remaining components, select the components avoiding both extremes in terms of replacement cost.

Under criteria 1, we lose 2 components which leaves us with 35 components to select the 15 components from. The selected components (components 7-22 when sorted in descending order of replacement cost) are stored in the file `selected_15_components_data.csv`. Note that this file is all we need for generating the POMDP and getting the optimal policies and survival probabilities for each component. 

## Running Simulations
The code to run the simulations is in the `simulations` subfolder. We make a copy of `selected_15_components_data.csv` in this folder. This folder also contains two additional folders:

1. `dynamics`- A folder that stores the dynamics data for different components generated using the `generate_dynamics_data.ipynb` notebook. The dynamics of each component is saved in the file `dynamics_{component_id}.csv`. T

2. `results` - This folder stores the `.jld` files with the state histories and survival probabilities for each component. Simulations for each component will generate 5 files storing state histories, action histories, observation histories, reward histories, and survival probabilities. This folder also contains code for analyzing the results and generating the plots.

## Running the POMDP

1. Make sure that you generate dynamics for all 15 components using the `generate_dynamics_data.ipynb` notebook.
2. Open the `POMDP_parallel.ipynb` notebook and set the indices (in order of rows in the csv file) you want to run in the `indices_to_run` array. 


