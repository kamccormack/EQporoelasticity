"""
Kimmy McCormack

Subduction Zone coupled poroelastic model. Creates a synthetic slip patch along the subduction zone fault plane.
Can either simulate an earthquake or a SSE. Outputs the deformation and pore pressure change as a result of
imposed slip along the boundary

Units are in meters and MPa


########################### USER INPUTS ##################################

ndim = Number of spatial dimensions - [2,3]
event = type of event - [EQ = earthquake, SSE = slow slip event, sub = subduction,
						sub_EQ = sub + EQ ]
loop =  Use last section to loop through a parameter such as surface permeability or event location - [no,yes]
permeability = Is the permeabilty a constant value or mapped variable? - [mapped, constant]


#################### SYNTHETIC EVENT PARAMETERS  ###################

sub_cycle_years = how many years between large earthquakes?
SSE_days = How many days does a slow slip event last?
days = For a SSE, run the model for how many days?
sigma_b = = Std deviation from the center of slip for a Gaussian slip distribution
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

from dolfin import *
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

ndim = 2
mesh_size = 'med'  # fine, med, coarse
event = 'EQ'
plot_figs = 'no'
permeability = 'constant'
data_file = '/fenics/shared/data/Data.xlsx'
loop = 'no'
loop_variable = 'sigma'  # 'sigma' for EQ size and 'kappa' for permeability

origin = np.array([-85.1688, 8.6925]) # origin of top surface of mesh - used as point of rotation
theta = 0.816 # angle of rotation between latitude line and fault trace (42 deg)

""""#################### EVENT PARAMETERS for 2D ###################"""
sub_cycle_years2D = 2
SSE_days2D = 5
days = 10
sigma_b = 1.2e4
xcenter2D = 80e3
u0_EQ = -2.5
u0_sub = .08
u0_SSE = -.006
SS_migrate = 0.0
surface_k2D = -10.5
depth_kappa = -18 #log of permeability at depth
xstick = 110e3 # distance from trench transition from stick to sliding behavior occurs
B = 0.8 # Skempton's coefficient

""""#################### EVENT PARAMETERS for 3D ###################"""
EQ_type = 'inversion'
sub_cycle_years = 2
SSE_days = 10
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

path = "/fenics/shared/meshes"

if ndim == 2:
	mesh = Mesh(path+'/2D_'+mesh_size+'.xml')
	boundaries = MeshFunction("size_t", mesh2D, path+'/2D_'+mesh_size+'_facet_region.xml')
elif ndim == 3:
	mesh = Mesh(path+'/3D_'+mesh_size+'.xml')
	boundaries = MeshFunction("size_t", mesh3D, path+'/3D_'+mesh_size+'_facet_region.xml')

if loop == 'no':
	if ndim == 2:
		Poroelasticity2D(data_file, ndim, mesh, boundaries, plot_figs, event, permeability, days, SSE_days2D,
		                 sub_cycle_years2D, sigma_b, xcenter2D, SS_migrate, u0_EQ, u0_SSE, u0_sub, surface_k2D, ocean_k, B,
		                 loop, loop_variable)

	elif ndim == 3:
		if EQ_type == 'synthetic':
			Poroelasticity3DSynthetic(data_file, origin, theta, ndim, mesh, boundaries, plot_figs, event,
			                          permeability, SSE_days, sub_cycle_years, sigma_bx, xcenter, sigma_by, ycenter,
			                          SS_migratedip, SS_migratestrike, u0_EQdip, u0_EQstrike, u0_SSEdip, u0_SSEstrike,
			                          u0_subdip, u0_substrike, surface_k, ocean_k)

		else:

			Poroelasticity3D(data_file, origin, theta, ndim, mesh, boundaries, plot_figs, EQ_type, event,
			                 permeability,sub_cycle_years, u0_subdip, u0_substrike, surface_k, ocean_k)

	else:
		print "Error - domain must be 2 or 3 dimensions"

	print "Time elasped", (time.time() - start)/60.0, "minutes"

""""#################### LOOP THROUGH VARIABLES ###################"""
"""Only in 2D"""

if loop == 'yes' and ndim == 2:
	if loop_variable == 'kappa':
		loop_through = np.arange(-10, -14, -1)  # make an array of the variable you want to loop through
	elif loop_variable == 'sigma':
		loop_through = np.arange(1e4, 2.6e4, 5e3)
	elif loop_variable == 'B':
		loop_through = np.arange(0.4, 1.0, 0.2)

	for VAR in loop_through:
		if loop_variable == 'kappa':
			Poroelasticity2D(data_file, ndim, mesh, boundaries, plot_figs, event, permeability, days, SSE_days2D,
				                 sub_cycle_years2D, sigma_b, xcenter2D, SS_migrate, u0_EQ, u0_SSE, u0_sub, VAR, VAR,
				                B, loop, loop_variable)
		if loop_variable == 'sigma':
			Poroelasticity2D(data_file, ndim, mesh, boundaries, plot_figs, event, permeability, days,
				                 SSE_days2D, sub_cycle_years2D, VAR, xcenter2D, SS_migrate, u0_EQ, u0_SSE, u0_sub, surface_k2D,
				                 ocean_k, B, loop, loop_variable)
		if loop_variable == 'B':
			Poroelasticity2D(data_file, ndim, mesh, boundaries, plot_figs, event, permeability, days,
				                 SSE_days2D, sub_cycle_years2D, sigma_b, xcenter2D, SS_migrate, u0_EQ, u0_SSE, u0_sub, surface_k2D,
				                 ocean_k, VAR, loop, loop_variable)
	

	print "Time elasped", (time.time() - start)/60.0, "minutes"
print "################### THE END ###########################"