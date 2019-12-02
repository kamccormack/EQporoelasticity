"""
Kimmy McCormack

Subduction Zone coupled poroelastic model. Creates a synthetic slip patch along the subduction zone fault plane.
Can either simulate an earthquake or a SSE. Outputs the deformation and pore pressure change as a result of
imposed slip along the boundary

Units are in meters and MPa


########################### USER INPUTS ##################################

ndim = Number of spatial dimensions - [3]
event = type fo event - [EQ = earthquake, SSE = slow slip event, sub = subduction,
						sub_EQ = sub + EQ]
permeability = Is the permeabilty a constant value or mapped variable? - [mapped, constant]


#################### SYNTHETIC EVENT PARAMETERS  ###################

sub_cycle_years = how many years between large earthquakes?
SSE_days = How many days does a slow slip event last?
days = For a SSE, run the model for how many days?
sigma_b = = Std deviation from the center of slip for a gausian slip distribution
xcenter = How far is the center of the slip patch from the trench?
u0_EQ =  For an EQ, what was the max slip [meters]?
u0_sub = subduction rate [m/yr]
u0_SSE = Amount of slip /day in a SSE [meter/day]
SS_migrate =  migration of the SSE along the trench [meters/day]
surface_k =  The  log of the permeability at the surface [m^2]

EQ_type: 'data' - slip vectors input as an excel file
		 'synthetic' - gaussian slip patch
		 'inversion' - imported from inverse model

"""


from firedrake import *
import numpy as np
import time
start = time.time()

""""#################### USER INPUTS ###################"""
"""choose between standard or mixed formulation for groundwater flow equation"""
mixed_form = False

if mixed_form:
	from PE_3field_functions import *
else:
	from PE_2field_functions import *

ndim = 3
mesh_size = 'med' # fine, med, coarse
event = 'EQ'
plot_figs = 'no'
permeability = 'mapped'
data_file = '/fenics/shared/data/Data.xlsx'

origin = np.array([-85.1688, 8.6925]) # origin of top surface of mesh - used as point of rotation
theta = 0.816 # angle of rotation between latitude line and fault trace (42 deg)


""""#################### EVENT PARAMETERS for 3D ###################"""
EQ_type = 'inversion'
sub_cycle_years = 0
SSE_days = 0
sigma_bx = 1.5e4  # Std deviation from the center of slip in trench perpendicular direction
xcenter = 100e3
sigma_by = 2e4  # Std deviation from the center of slip in trench parallel direction
ycenter = 130e3  # How far is the center of the slip patch from south end of the domain?

u0_EQdip = -4.0  # max slip in trench perpendicular direction?
u0_EQstrike = 0.0  # max slip in trench parallel direction?

u0_subdip = .08  # subduction rate [cm/yr] in trench perpendicular direction
u0_substrike = .01  # subduction rate [cm/yr] in trench parallel direction

u0_SSEdip = .01  # meter/day Slow slip rate in trench perpendicular direction
u0_SSEstrike = .01  # meter/day Slow slip rate in trench parallel direction

SS_migratedip = 1e2  # migration of the SSE towards the trench in meters per day
SS_migratestrike = 1e2  # migration of the SSE along the trench in meters per day
surface_k = -10  # The  log of the permeability at the surface
ocean_k = -13

""""#################### Run model code ###################"""
path = "/fenics/shared/firedrake/meshes/"
mesh = Mesh(path+'CR3D_quads'+mesh_size+'.msh')

if EQ_type == 'synthetic':
	Poroelasticity3DSynthetic(data_file, origin, theta, ndim, mesh, plot_figs, event,
	                          permeability, SSE_days, sub_cycle_years, sigma_bx, xcenter, sigma_by, ycenter,
	                          SS_migratedip, SS_migratestrike, u0_EQdip, u0_EQstrike, u0_SSEdip, u0_SSEstrike,
	                          u0_subdip, u0_substrike, surface_k, ocean_k)
else:
	Poroelasticity3D(data_file, origin, theta, ndim, mesh, plot_figs, EQ_type, event, permeability,
	                 sub_cycle_years, u0_subdip, u0_substrike, surface_k, ocean_k)


print("Time elasped", (time.time() - start)/60.0, "minutes")


print("################### THE END ###########################")