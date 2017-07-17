"""
Kimmy McCormack

Subduction Zone coupled poroelastic model. Creates a synthetic slip patch along the subduction zone fault plane.
Can either simulate an earthquake or a SSE. Outputs the deformation and pore pressure change as a result of
imposed slip along the boundary

Units are in meters and MPa

TO RUN: 1) Open Fenics (this opens a terminal window)
        2) Change the path of the mesh files to the location on your computer (lines 42 and 45)
        3) Go to directory where this file is located
        4) run "python ./CR_2D_poroelastic.py"

########################### USER INPUTS ##################################

ndim = Number of spatial dimensions - [2,3]
event = type fo event - [EQ = earthquake, SSE = slow slip event, sub = subduction,
						sub_EQ = sub + EQ , topo = topography driven flow]
plot_figs = Plot figures with fenics plotter? - [no,yes]
loop =  Use last section to loop through a parameter such as surface permeability or event location - [no,yes]
new_mesh = Creates new indices file to extract solution at surface nodes/GPS
			stations when turned on, otherwise it loads a previously saved file. Must = 'yes' at
			least once to create files. TURN OFF ONLY IF NONE OF THE FOLLOWING HAVE BEEN CHANGED:
			a) mesh file  b) mesh refinement 3) order of basis functions - [no,yes]
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

EQ_type: 'data' - slip vectors input as an excel file,  'synthetic' - gaussian slip patch

"""


from dolfin import *
import numpy as np
import time

from poroelastic_master2 import *
#from poroelastic_master_RT import *
#from poroelastic_master import *
#from poroelastic_master_RT import Poroelasticity2D, Poroelasticity3D, Laplace3D

start = time.time()

#print list_linear_solver_methods()
#print list_lu_solver_methods()
#print list_linear_algebra_backends()
#print list_krylov_solver_methods()
#print list_krylov_solver_preconditioners()


##########################################################################
########################### USER INPUTS ##################################
##########################################################################

ndim = 2
mesh_size = 'coarse'  # fine, med, coarse
event = 'SSE'
plot_figs = 'no'
new_mesh = 'yes'
permeability = 'mapped'
data_file = 'data/Data.xlsx'
loop = 'no'
loop_variable = 'sigma'  # 'sigma' for EQ size and 'kappa' for permeability

origin = np.array([-85.21, 8.64]) # origin of top surface of mesh - used as point of rotation
theta = 0.733 # angle of rotation between latitude line and fault trace

##########################################################################
#################### SYNTHETIC EVENT PARAMETERS for 2D ###################
##########################################################################

sub_cycle_years2D = 2
SSE_days2D = 5
days = 2
sigma_b = 1.5e4
xcenter2D = 75e3
u0_EQ = -4.0
u0_sub = .08
u0_SSE = -.06
SS_migrate = 0.0
surface_k2D = -12
B = 0.7 # Skempton's coefficient
# if event == 'topo':
# 	sub_cycle_years2D *= 2

##########################################################################
#################### SYNTHETIC EVENT PARAMETERS for 3D ###################
##########################################################################

EQ_type = 'data'
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
surface_k = -12  # The  log of the permeability at the surface
ocean_k = -13

##########################################################################
###################  IMPORT MESH FILES   #################################
##########################################################################

path = "/home/fenics/shared/meshes"
if ndim == 2:
	mesh2D = Mesh(path+'/CR2D_'+mesh_size+'.xml')
	boundaries2D = MeshFunction("size_t", mesh2D, path+'/CR2D_'+mesh_size+'_facet_region.xml')
elif ndim == 3:
	mesh3D = Mesh(path+'/CR3D_'+mesh_size+'.xml')
	boundaries3D = MeshFunction("size_t", mesh3D, path+'/CR3D_'+mesh_size+'_facet_region.xml')

#mesh_visco = Mesh("/Users/kam4898/Google_Drive/FEniCS/meshes/CR_split_surf_mantle.xml")
#boundaries_visco = MeshFunction("size_t", mesh_visco,path+"Google_Drive/FEniCS/meshes/CR_split_surf_mantle_facet_region.xml")
#subdomains = MeshFunction("size_t", mesh_visco,path+"Google_Drive/FEniCS/meshes/CR_split_surf_mantle_physical_region.xml")


if loop == 'no':
	if ndim == 2:
		Poroelasticity2D(data_file, ndim, mesh2D, boundaries2D, plot_figs, event, new_mesh, permeability, days, SSE_days2D,
		                 sub_cycle_years2D, sigma_b, xcenter2D, SS_migrate, u0_EQ, u0_SSE, u0_sub, surface_k2D, ocean_k, B, loop, loop_variable)

	elif ndim == 3:

		if EQ_type == 'synthetic':
			Poroelasticity3DSynthetic(data_file, origin, theta, ndim, mesh3D, boundaries3D, plot_figs, event, new_mesh, permeability, SSE_days, sub_cycle_years, sigma_bx, xcenter, sigma_by, ycenter, SS_migratedip, SS_migratestrike, u0_EQdip,
	             u0_EQstrike, u0_SSEdip, u0_SSEstrike, u0_subdip, u0_substrike, surface_k, ocean_k, loop)

		elif EQ_type == 'data':

			Poroelasticity3D(data_file, origin, theta, ndim, mesh3D, boundaries3D, plot_figs, event, new_mesh, permeability,sub_cycle_years, u0_subdip, u0_substrike, surface_k, ocean_k, loop)


	else:
		print "domain must be 2 or 3 dimensions!"

	print "Time elasped", (time.time() - start)/60.0, "minutes"

	if plot_figs == 'yes':
		interactive()  # hold the plot

##########################################################################
################  SO YOU WANT TO RUN A LOOP, EH?  ########################
##########################################################################

if loop == 'yes':

	if loop_variable == 'kappa':
		loop_through = np.arange(-10, -14, -1)  # make an array of the variable you want to loop through
	elif loop_variable == 'sigma':
		loop_through = np.arange(1e4, 2.6e4, 5e3)
	elif loop_variable == 'B':
		loop_through = np.arange(0.4, 1.0, 0.2)

	for i in loop_through:
		if loop_variable == 'kappa':
			if ndim == 2:
				Poroelasticity2D(data_file, ndim, mesh2D, boundaries2D, plot_figs, event, new_mesh, permeability, days, SSE_days2D,
				                 sub_cycle_years2D, sigma_b, xcenter2D, SS_migrate, u0_EQ, u0_SSE, u0_sub, i, B, loop, loop_variable)
		if loop_variable == 'sigma':
			if ndim == 2:
				Poroelasticity2D(data_file, ndim, mesh2D, boundaries2D, plot_figs, event, new_mesh, permeability, days, SSE_days2D,
				                 sub_cycle_years2D, i, xcenter2D, SS_migrate, u0_EQ, u0_SSE, u0_sub, surface_k2D, ocean_k, B, loop, loop_variable)
		if loop_variable == 'B':
			if ndim == 2:
				Poroelasticity2D(data_file, ndim, mesh2D, boundaries2D, plot_figs, event, new_mesh, permeability, days, SSE_days2D,
				                 sub_cycle_years2D, sigma_b , xcenter2D, SS_migrate, u0_EQ, u0_SSE, u0_sub, surface_k2D, i, loop, loop_variable)


	print "Time elasped", (time.time() - start)/60.0, "minutes"


################ Run simulation for Laplace equation ##################

if loop == 'laplace':
	if ndim == 2:
		Laplace2D(data_file, ndim, mesh2D, boundaries2D, plot_figs, event, new_mesh, permeability, days, SSE_days2D,sub_cycle_years2D, sigma_b, xcenter2D, SS_migrate, u0_EQ, u0_SSE, u0_sub, surface_k2D, ocean_k, B, loop, loop_variable)

	elif ndim == 3:

		Laplace3D(data_file, origin, theta, ndim, mesh3D, boundaries3D, plot_figs, event, new_mesh, permeability,
	             sub_cycle_years, surface_k, ocean_k, loop)

if loop == "steady":
	Laplace3Dsteady(data_file, origin, theta, ndim, mesh3D, boundaries3D, plot_figs, event, new_mesh, permeability,
	          sub_cycle_years, surface_k, ocean_k, loop)



print "################### THE END ###########################"