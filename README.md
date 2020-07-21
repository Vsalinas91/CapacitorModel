# CapacitorModel For Flash Energy Estimation:

Requirements:
-------------
           -) Seaborn
           -) Scipy
           -) Pandas
           -) Matplotlib


Basic Information and Data Structure:
-------------------------------------
The provided code and data allow for all figures and capacitor energy estimation to be reproduced. The code provided are mainly for plotting routines, with the capacitor model also contained. The data provided are for the initiation locations of the flashes described in Section 2 of the manuscript, and contain the following data (ordered the same as the .csv files provided):

                -)Initiation Time - in seconds
                -)Initiation Time - in minutes
                -)Background Domain Electrostatic Energy - [J | COMMAS Value]
                -)Flash Electrostatic Energy Change - [J | COMMAS Value]
                -)Domain Maximum Vertical Velocity at Init Time - [m/s]
                -)xi,yi,zi - index locations of flash initiations on model grid
                           - multiply by 125. [m] to recover positional coordinates in [m]
                -)group_idx - index for model time step in which each initiation occured.
                -)Flash Area - convex hull plan-view area of each flash initiation.
                -)Separation - separation between charge centers; used for capacitor plate separation 
                -)Updraft Volume > 10m/s - Approximated updraft volumes for w>10m/s
                -)Model Time Step - Model time step, in [s], which align with the updraft volume data.
                -)Capacitor Neutralization Efficiency - eta_c (relative to Flash Electrostatic Energy Change)
                -)COMMAS Neutralization Efficiency - eta_m (relative to domain electrostatic energy)


All data were extracted exactly from the simulation ouput files, with the exception of the flash area, plate separation, updraft volumes, and capacitor neutralization efficiencies which were computed at each inidividual flash location and time (described in Methods Section).

Running Analysis:
-------------------------------------
Running the analysis is done by: python CapacitorModelAnalyses.py. There are three arguments that can be adjusted by the user, if additional analyses are wanted to be made.

                -) Time series interval adjustment - adjusting bin_range will allow for various views of the data at different total and    average time scales. Currently, the default is 60. for 60 second intervals.
                -) uniform_eta - Boolean arguement. If True, user must specify a value for eta by which capacitor energy estimates are adjusted by. Else, if False, the median values of eta_c are used.
                -) eta_u - Uniform eta to be specified by the user if a different value is wished to be used to scale and adjust the capacitor energy estimates. 
    
    
    
Any questions or concerns about issues or errors in these files may be directed to the author at: vicente.salinas@ttu.edu

