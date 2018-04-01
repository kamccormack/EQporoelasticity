from dolfin import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
from scipy.interpolate import griddata
from tempfile import TemporaryFile
import openpyxl as pyxl
import matplotlib.tri as tri
dolfin.parameters.reorder_dofs_serial = False


event = 'sub'  # syn = synthetic event, data = real EQ, sub = subduction, sub_EQ
loop = 'no'
##########################################################################
############################ LOAD VARIABLES ##############################
##########################################################################
#path = "/Users/kam4898"
#path = "/Users/Kimmy"

results_path = 'results/numpy_results/'
var_path = 'results/numpy_variables/'
x_all = np.load(var_path+"x_all_2D.npy")
y_all = np.load(var_path+"y_all_2D.npy")
size_p = np.load(var_path+"size_p_CR_2D.npy")

meshpath = "/home/fenics/shared/meshes"
mesh_size = 'fine'  # fine, med, coarse
mesh = Mesh(meshpath+'/CR2D_'+mesh_size+'.xml')
boundaries = MeshFunction("size_t", mesh, meshpath+'/CR2D_'+mesh_size+'_facet_region.xml')

surface_dofs = np.load(var_path+"surface_dofs_2D.npy")
ocean_dofs = np.load(var_path+"ocean_dofs_2D.npy")



if event == 'EQ':
	if loop == 'yes':

		# load solution from kappa loop
		Sol_allk10 = np.load("saved_results_poroelastic/Sol_all_2D_EQ_loop_k10.npy")
		Sol_allk11 = np.load("saved_results_poroelastic/Sol_all_2D_EQ_loop_k11.npy")
		Sol_allk12 = np.load("saved_results_poroelastic/Sol_all_2D_EQ_loop_k12.npy")
		Sol_allk13 = np.load("saved_results_poroelastic/Sol_all_2D_EQ_loop_k13.npy")
		dtt = np.load("saved_variables_poroelastic/dtt_comsum_EQ.npy")

		seaflux_k10 = np.load("saved_results_poroelastic/sea_flux_2D_EQ_loop_k10.npy")
		seaflux_k11 = np.load("saved_results_poroelastic/sea_flux_2D_EQ_loop_k11.npy")
		seaflux_k12 = np.load("saved_results_poroelastic/sea_flux_2D_EQ_loop_k12.npy")
		seaflux_k13 = np.load("saved_results_poroelastic/sea_flux_2D_EQ_loop_k13.npy")

		size_ux = (Sol_allk11.shape[0] - size_p) / 2
		size_flux = seaflux_k10.shape[0] / 2
		p_ind = np.argmax(ocean_dofs >= size_p)
		ux_ind = np.argmax(ocean_dofs >= size_p + size_ux)

		ocean_dof_p = ocean_dofs[0:p_ind][np.argsort(x_all[ocean_dofs[0:p_ind]])]
		ocean_dof_ux = ocean_dofs[p_ind:ux_ind][np.argsort(x_all[ocean_dofs[p_ind:ux_ind]])]
		ocean_dof_uy = ocean_dofs[ux_ind::][np.argsort(x_all[ocean_dofs[ux_ind::]])]
		sort_p, sort_u = np.argsort(x_all[ocean_dofs[0:p_ind]]), np.argsort(x_all[ocean_dofs[ux_ind::]])

		Z_pk10 = Sol_allk10[0:size_p, :]
		Z_pk11 = Sol_allk11[0:size_p, :]
		Z_pk12 = Sol_allk12[0:size_p, :]
		Z_pk13 = Sol_allk13[0:size_p, :]

		Z_uxk10 = Sol_allk10[size_p:size_p + size_ux, :]
		Z_uxk11 = Sol_allk11[size_p:size_p + size_ux, :]
		Z_uxk12 = Sol_allk12[size_p:size_p + size_ux, :]
		Z_uxk13 = Sol_allk13[size_p:size_p + size_ux, :]

		Z_uyk10 = Sol_allk10[size_p + size_ux::, :]
		Z_uyk11 = Sol_allk11[size_p + size_ux::, :]
		Z_uyk12 = Sol_allk12[size_p + size_ux::, :]
		Z_uyk13 = Sol_allk13[size_p + size_ux::, :]

		#load solution from sigma loop
		Sol_all10 = np.load("saved_results_poroelastic/Sol_all_2D_EQ_loop_10.npy")
		Sol_all15 = np.load("saved_results_poroelastic/Sol_all_2D_EQ_loop_15.npy")
		Sol_all20 = np.load("saved_results_poroelastic/Sol_all_2D_EQ_loop_20.npy")
		Sol_all25 = np.load("saved_results_poroelastic/Sol_all_2D_EQ_loop_25.npy")
		dtt = np.load("saved_variables_poroelastic/dtt_comsum_EQ.npy")

		seaflux_10 = np.load("saved_results_poroelastic/sea_flux_2D_EQ_loop_10.npy")
		seaflux_15 = np.load("saved_results_poroelastic/sea_flux_2D_EQ_loop_15.npy")
		seaflux_20 = np.load("saved_results_poroelastic/sea_flux_2D_EQ_loop_20.npy")
		seaflux_25 = np.load("saved_results_poroelastic/sea_flux_2D_EQ_loop_25.npy")

		norm_sea = np.array([.06652, 0.997785])

		seaflux_x_k10 = norm_sea[0] * velocity_k10[ocean_dof_uy - size_p]
		seaflux_x_k11 = norm_sea[0] * seaflux_k11[0:size_flux][sort_p]
		seaflux_x_k12 = norm_sea[0] * seaflux_k12[0:size_flux][sort_p]
		seaflux_x_k13 = norm_sea[0] * seaflux_k13[0:size_flux][sort_p]

		seaflux_y_k10 = norm_sea[1] * velocity_k10[ocean_dof_uy - size_p]
		seaflux_y_k11 = norm_sea[1] * seaflux_k11[size_flux::][sort_p]
		seaflux_y_k12 = norm_sea[1] * seaflux_k12[size_flux::][sort_p]
		seaflux_y_k13 = norm_sea[1] * seaflux_k13[size_flux::][sort_p]


		Z_p10 = Sol_all10[0:size_p].reshape(size_p, )
		Z_p15 = Sol_all15[0:size_p].reshape(size_p, )
		Z_p20 = Sol_all20[0:size_p].reshape(size_p, )
		Z_p25 = Sol_all25[0:size_p].reshape(size_p, )

		Z_ux10 = Sol_all10[size_p:size_p + size_ux].reshape(size_ux, )
		Z_ux15 = Sol_all15[size_p:size_p + size_ux].reshape(size_ux, )
		Z_ux20 = Sol_all20[size_p:size_p + size_ux].reshape(size_ux, )
		Z_ux25 = Sol_all25[size_p:size_p + size_ux].reshape(size_ux, )

		Z_uy10 = Sol_all10[size_p + size_ux::].reshape(size_ux, )
		Z_uy15 = Sol_all15[size_p + size_ux::].reshape(size_ux, )
		Z_uy20 = Sol_all20[size_p + size_ux::].reshape(size_ux, )
		Z_uy25 = Sol_all25[size_p + size_ux::].reshape(size_ux, )

		Z_mag10 = np.sqrt(pow(Z_ux10, 2) + pow(Z_uy10, 2))
		Z_mag15 = np.sqrt(pow(Z_ux15, 2) + pow(Z_uy15, 2))
		Z_mag20 = np.sqrt(pow(Z_ux20, 2) + pow(Z_uy20, 2))
		Z_mag25 = np.sqrt(pow(Z_ux25, 2) + pow(Z_uy25, 2))
			
	else:
		sea_flux = np.load("saved_results_poroelastic/sea_flux_2D_EQ.npy")
		Sol_surf = np.load("saved_results_poroelastic/Sol_surf_2D_EQ.npy")
		Sol_all = np.load("saved_results_poroelastic/Sol_all_2D_EQ.npy")
		#sub_cycle = np.load("saved_variables_poroelastic/sub_cycle_EQ.npy")
		dtt = np.load("saved_variables_poroelastic/dtt_comsum_EQ.npy")
		streamfun = np.load("saved_results_poroelastic/Streamfun_2D_EQ.npy")


		velocity = np.load("saved_results_poroelastic/flow_velocity_2D_EQ.npy")

		size_ux = (Sol_all.shape[0] - size_p) / 2
		size_flux = sea_flux.shape[0] / 2

		Z_p = Sol_all[0:size_p, :]
		Z_ux = Sol_all[size_p:size_p + size_ux, :]
		Z_uy = Sol_all[size_p + size_ux::, :]

		norm_sea = np.array([.06652, 0.997785])

		p_ind = np.argmax(ocean_dofs >= size_p)
		ux_ind = np.argmax(ocean_dofs >= size_p + size_ux)

		ocean_dof_p = ocean_dofs[0:p_ind][np.argsort(x_all[ocean_dofs[0:p_ind]])]
		ocean_dof_ux = ocean_dofs[p_ind:ux_ind][np.argsort(x_all[ocean_dofs[p_ind:ux_ind]])]
		ocean_dof_uy = ocean_dofs[ux_ind::][np.argsort(x_all[ocean_dofs[ux_ind::]])]
		sort_p, sort_u = np.argsort(x_all[ocean_dofs[0:p_ind]]), np.argsort(x_all[ocean_dofs[ux_ind::]])

		seaflux_x = norm_sea[0] * velocity[ocean_dof_ux - size_p]
		seaflux_y = norm_sea[1] * velocity[ocean_dof_uy - size_p]

	"""
	dtt = [15, 240, 288e1, 432e2]  # in minutes
	ntime = [96, 120, 15, 122]
	dtt = [i * 60 for i in dtt]  # in seconds
	dtt_repeated = np.repeat(dtt, ntime)
	dtt = np.cumsum(dtt_repeated)
	"""

elif event == 'SSE':
	sea_flux = np.load("saved_results_poroelastic/sea_flux_2D_SSE.npy")
	Sol_surf = np.load("saved_results_poroelastic/Sol_surf_2D_SSE.npy")
	dtt = np.load("saved_variables_poroelastic/dtt_comsum_SSE.npy")

elif event == 'sub':
	sea_flux = np.load(results_path+"sea_flux_2D_sub.npy")
	Sol_surf = np.load(results_path+"Sol_surf_2D_sub.npy")
	Sol_all = np.load(results_path+"Sol_all_2D_sub.npy")
	#sub_cycle = np.load(var_path+"sub_cycle_sub.npy")
	dtt = np.load(var_path+"dtt_comsum_sub.npy")
	streamfun = np.load(results_path+"Streamfun_2D_sub.npy")
	
	velocity = np.load(results_path+"flow_velocity_2D_sub.npy")
	

	
	size_ux = (Sol_all.shape[0] - size_p) / 2
	size_flux = sea_flux.shape[0] / 2
	
	Z_p = Sol_all[0:size_p, :]
	Z_ux = Sol_all[size_p:size_p + size_ux, :]
	Z_uy = Sol_all[size_p + size_ux::, :]
	
	norm_sea = np.array([.06652, 0.997785])
	norm_sea = np.array([0.0, 0.997785])
	
	p_ind = np.argmax(ocean_dofs >= size_p)
	ux_ind = np.argmax(ocean_dofs >= size_p + size_ux)
	
	ocean_dof_p = ocean_dofs[0:p_ind][np.argsort(x_all[ocean_dofs[0:p_ind]])]
	ocean_dof_ux = ocean_dofs[p_ind:ux_ind][np.argsort(x_all[ocean_dofs[p_ind:ux_ind]])]
	ocean_dof_uy = ocean_dofs[ux_ind::][np.argsort(x_all[ocean_dofs[ux_ind::]])]
	sort_p, sort_u = np.argsort(x_all[ocean_dofs[0:p_ind]]), np.argsort(x_all[ocean_dofs[ux_ind::]])
	
	seaflux_x = norm_sea[0] * velocity[ocean_dof_p]
	seaflux_y = norm_sea[1] * velocity[ocean_dof_p + size_p]
	
	print velocity.shape
	print ocean_dof_p.shape
	print size_p
	print seaflux_x.shape
	#quit()
	

elif event == 'sub_EQ':
	sea_flux = np.load("saved_results_poroelastic/sea_flux_2D_sub_EQ.npy")
	Sol_surf = np.load("saved_results_poroelastic/Sol_surf_2D_sub_EQ.npy")
	sub_cycle = np.load("saved_variables_poroelastic/sub_cycle_sub_EQ.npy")
	dtt = np.load("saved_variables_poroelastic/dtt_comsum_sub_EQ.npy")

elif event == 'topo':

	if loop == 'yes':

		# load solution from kappa loop
		Sol_allk10 = np.load("saved_results_poroelastic/Sol_all_2D_topo_loop_k10.npy")
		Sol_allk11 = np.load("saved_results_poroelastic/Sol_all_2D_topo_loop_k11.npy")
		Sol_allk12 = np.load("saved_results_poroelastic/Sol_all_2D_topo_loop_k12.npy")
		Sol_allk13 = np.load("saved_results_poroelastic/Sol_all_2D_topo_loop_k13.npy")
		dtt = np.load("saved_variables_poroelastic/dtt_comsum_topo.npy")

		seaflux_k10 = np.load("saved_results_poroelastic/sea_flux_2D_topo_loop_k10.npy")
		seaflux_k11 = np.load("saved_results_poroelastic/sea_flux_2D_topo_loop_k11.npy")
		seaflux_k12 = np.load("saved_results_poroelastic/sea_flux_2D_topo_loop_k12.npy")
		seaflux_k13 = np.load("saved_results_poroelastic/sea_flux_2D_topo_loop_k13.npy")

		velocity_k10 = np.load("saved_results_poroelastic/flow_velocity_2D_topo_loop_k10.npy")
		velocity_k11 = np.load("saved_results_poroelastic/flow_velocity_2D_topo_loop_k11.npy")
		velocity_k12 = np.load("saved_results_poroelastic/flow_velocity_2D_topo_loop_k12.npy")
		velocity_k13 = np.load("saved_results_poroelastic/flow_velocity_2D_topo_loop_k13.npy")
		
		streamfun_k10 = np.load("saved_results_poroelastic/Streamfun_2D_topo_loop_k10.npy")
		streamfun_k11 = np.load("saved_results_poroelastic/Streamfun_2D_topo_loop_k11.npy")
		streamfun_k12 = np.load("saved_results_poroelastic/Streamfun_2D_topo_loop_k12.npy")
		streamfun_k13 = np.load("saved_results_poroelastic/Streamfun_2D_topo_loop_k13.npy")

		print velocity_k10.shape
		quit()

		size_ux = (Sol_allk11.shape[0] - size_p) / 2
		size_flux = seaflux_k10.shape[0] / 2

		Z_pk10 = Sol_allk10[0:size_p, :]
		Z_pk11 = Sol_allk11[0:size_p, :]
		Z_pk12 = Sol_allk12[0:size_p, :]
		Z_pk13 = Sol_allk13[0:size_p, :]

		Z_uxk10 = Sol_allk10[size_p:size_p + size_ux, :]
		Z_uxk11 = Sol_allk11[size_p:size_p + size_ux, :]
		Z_uxk12 = Sol_allk12[size_p:size_p + size_ux, :]
		Z_uxk13 = Sol_allk13[size_p:size_p + size_ux, :]

		Z_uyk10 = Sol_allk10[size_p + size_ux::, :]
		Z_uyk11 = Sol_allk11[size_p + size_ux::, :]
		Z_uyk12 = Sol_allk12[size_p + size_ux::, :]
		Z_uyk13 = Sol_allk13[size_p + size_ux::, :]

		vel_x_k10 = velocity_k10[0:size_ux, :]
		vel_x_k11 = velocity_k11[0:size_ux, :]
		vel_x_k12 = velocity_k12[0:size_ux, :]
		vel_x_k13 = velocity_k13[0:size_ux, :]

		vel_y_k10 = velocity_k10[size_ux::, :]
		vel_y_k11 = velocity_k11[size_ux::, :]
		vel_y_k12 = velocity_k12[size_ux::, :]
		vel_y_k13 = velocity_k13[size_ux::, :]

		norm_sea = np.array([.06652, 0.997785])

		p_ind = np.argmax(ocean_dofs >= size_p)
		ux_ind = np.argmax(ocean_dofs >= size_p + size_ux)

		ocean_dof_p = ocean_dofs[0:p_ind][np.argsort(x_all[ocean_dofs[0:p_ind]])]
		ocean_dof_ux = ocean_dofs[p_ind:ux_ind][np.argsort(x_all[ocean_dofs[p_ind:ux_ind]])]
		ocean_dof_uy = ocean_dofs[ux_ind::][np.argsort(x_all[ocean_dofs[ux_ind::]])]
		sort_p, sort_u = np.argsort(x_all[ocean_dofs[0:p_ind]]), np.argsort(x_all[ocean_dofs[ux_ind::]])

		seaflux_x_k10 = norm_sea[0] * velocity_k10[ocean_dof_ux - size_p]
		seaflux_x_k11 = norm_sea[0] * seaflux_k11[0:size_flux][sort_p]
		seaflux_x_k12 = norm_sea[0] * seaflux_k12[0:size_flux][sort_p]
		seaflux_x_k13 = norm_sea[0] * seaflux_k13[0:size_flux][sort_p]

		seaflux_y_k10 = norm_sea[1] * velocity_k10[ocean_dof_uy - size_p]
		seaflux_y_k11 = norm_sea[1] * seaflux_k11[size_flux::][sort_p]
		seaflux_y_k12 = norm_sea[1] * seaflux_k12[size_flux::][sort_p]
		seaflux_y_k13 = norm_sea[1] * seaflux_k13[size_flux::][sort_p]


	else:
		Sol_all = np.load("saved_results_poroelastic/Sol_all_2D_topo.npy")
		seaflux = np.load("saved_results_poroelastic/sea_flux_2D_topo.npy")
		velocity = np.load("saved_results_poroelastic/flow_velocity_2D_topo.npy")
		streamfun = np.load("saved_results_poroelastic/Streamfun_2D_topo.npy")
		dtt = np.load("saved_variables_poroelastic/dtt_comsum_topo.npy")
		size_ux = (Sol_all.shape[0] - size_p) / 2
		size_flux = seaflux.shape[0]
		norm_sea = np.array([.06652, 0.997785])


		Z_p = Sol_all[0:size_p, :]
		Z_ux = Sol_all[size_p:size_p + size_ux, :]
		Z_uy = Sol_all[size_p + size_ux::, :]
		vel_x = velocity[0:size_ux, :]
		vel_y = velocity[size_ux::, :]

		p_ind = np.argmax(ocean_dofs >= size_p)
		ux_ind = np.argmax(ocean_dofs >= size_p + size_ux)

		ocean_dof_p = ocean_dofs[0:p_ind][np.argsort(x_all[ocean_dofs[0:p_ind]])]
		ocean_dof_ux = ocean_dofs[p_ind:ux_ind][np.argsort(x_all[ocean_dofs[p_ind:ux_ind]])]
		ocean_dof_uy = ocean_dofs[ux_ind::][np.argsort(x_all[ocean_dofs[ux_ind::]])]
		sort_p, sort_u = np.argsort(x_all[ocean_dofs[0:p_ind]]), np.argsort(x_all[ocean_dofs[ux_ind::]])

		seaflux_x = norm_sea[0] * velocity[ocean_dof_ux - size_p]
		seaflux_y = norm_sea[1] * velocity[ocean_dof_uy - size_p]

		print seaflux.shape
		print seaflux_y.shape







##########################################################################
###################### PARSE AND SORT VARIABLES ##########################
##########################################################################


nsteps = dtt.size



x_p = x_all[0:size_p]
y_p = y_all[0:size_p]
x_u = x_all[size_p:size_p+size_ux]
y_u = y_all[size_p:size_p+size_ux]
x_ux = x_u[0:size_ux]
y_ux = y_u[0:size_ux]
x_ocean_p = x_all[ocean_dof_p]
y_ocean_p = y_all[ocean_dof_p]
x_ocean_u = x_all[ocean_dof_ux]
y_ocean_u = y_all[ocean_dof_ux]


"""
x_stream = np.linspace(x_ux.min(), x_ux.max(), 1000)
y_stream = np.linspace(y_ux.min(), y_ux.max(), 100)
X_stream, Y_stream = np.meshgrid(x_stream, y_stream)
Z_x_stream = griddata((x_ux,y_ux), vel_x_k10,(X_stream, Y_stream))
Z_y_stream = griddata((x_ux,y_ux), vel_y_k10,(X_stream, Y_stream))
speed = -1/(np.log10(np.sqrt(Z_x_stream*Z_x_stream + Z_y_stream*Z_y_stream)))
speed_NaNs = np.isnan(speed)
speed[speed_NaNs] = 0
"""


##########################################################################
######################### CONVERT UNITS ##################################
##########################################################################


# convert dtt from seconds to years
dtt = dtt / (3600 * 24)
# convert sea flux from m/s to m/day
# sea_flux = sea_flux*(3600*24)

# convert MPa to well head change (meters)
# Sol_surf_p9[:] = Sol_surf_p9[:]*1e3/(9.81)
# Sol_surf_p10[:] = Sol_surf_p10[:]*1e3/(9.81)
#Sol_surf_p11[:] = Sol_surf_p11[:] * 1e3 / (9.81)


# Sol_surf_p12[:] = Sol_surf_p12[:]*1e3/(9.81)
# Sol_surf_p13[:] = Sol_surf_p13[:]*1e3/(9.81)



##########################################################################
######################### PLOTTING FUNCTIONS #############################
##########################################################################
def shiftedColorMap(cmap, start = 0, midpoint = 0.5, stop = 1.0, name = 'shiftedcmap'):
	'''
	Function to offset the "center" of a colormap. Useful for
	data with a negative min and positive max and you want the
	middle of the colormap's dynamic range to be at zero

	Input
	-----
	  cmap : The matplotlib colormap to be altered
	  start : Offset from lowest point in the colormap's range.
		  Defaults to 0.0 (no lower ofset). Should be between
		  0.0 and `midpoint`.
	  midpoint : The new center of the colormap. Defaults to
		  0.5 (no shift). Should be between 0.0 and 1.0. In
		  general, this should be  1 - vmax/(vmax + abs(vmin))
		  For example if your data range from -15.0 to +5.0 and
		  you want the center of the colormap at 0.0, `midpoint`
		  should be set to  1 - 5/(5 + 15)) or 0.75
	  stop : Offset from highets point in the colormap's range.
		  Defaults to 1.0 (no upper ofset). Should be between
		  `midpoint` and 1.0.
	'''
	cdict = {
		'red':[],
		'green':[],
		'blue':[],
		'alpha':[]
	}

	# regular index to compute the colors
	reg_index = np.linspace(start, stop, 257)

	# shifted index to match the data
	shift_index = np.hstack([
		np.linspace(0.0, midpoint, 128, endpoint = False),
		np.linspace(midpoint, 1.0, 129, endpoint = True)
	])

	for ri, si in zip(reg_index, shift_index):
		r, g, b, a = cmap(ri)

		cdict['red'].append((si, r, r))
		cdict['green'].append((si, g, g))
		cdict['blue'].append((si, b, b))
		cdict['alpha'].append((si, a, a))

	newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
	plt.register_cmap(cmap = newcmap)

	return newcmap


def plot_pressure_contour():
	fig = plt.figure()
	ax1 = fig.add_subplot(111)


	i = 1 # timestep

	stream = streamfun[:, i]
	Z = Z_p[:, i]
	
	#Zmin, Zmax  = Z.min(), Z.max()
	Zmin, Zmax = -0.002, 0.1

	triang = tri.Triangulation(x_p, y_p, mesh.cells())
	triang_u = tri.Triangulation(x_u, y_u, mesh.cells())
	contour_steps = 80
	num_lines = 20
	orig_cmap = cm.coolwarm
	midpoint = 1 - Zmax / (Zmax + abs(Zmin))
	shifted_cmap = shiftedColorMap(orig_cmap, midpoint = midpoint)

	zero = np.array([0])

	# levels_10 = np.array([Z_10.min(), Z_10.min()/2, Z_10.min()/4, Z_10.min()/8, Z_10.min()/16,
	#                      0, Z_10.max()/16, Z_10.max()/8, Z_10.max()/4, Z_10.max()/2, Z_10.max()])
	levels = np.linspace(Zmin, Zmax, num = contour_steps, endpoint = True)
	lines = np.linspace(Zmin, Zmax, num = num_lines, endpoint = True)
	lines_stream = np.linspace(stream.min(), stream.max(), num = num_lines, endpoint = True)


	skip = (slice(None, None, 4))

	contourf = ax1.tricontourf(triang, Z, levels, cmap = shifted_cmap)
	contour = ax1.tricontour(triang, Z, lines, linewidth = 1, colors = 'black', linestyles = 'solid')
	#quiver = ax1.quiver(x_ocean_u[skip], y_ocean_u[skip], seaflux_x[skip][:, i], seaflux_y[skip][:, i], units = 'inches', scale_units = 'width', width = 1.7e-2, scale = .3/1e5, color = 'darkcyan')
	quiver = ax1.quiver(x_ocean_p[skip], y_ocean_p[skip], seaflux_x[skip][:, i], seaflux_y[skip][:, i],
	                    units = 'inches', scale_units = 'width', width = 1e-2, scale = .00001 / 2e5, color = 'darkcyan')
	
	# quiver = ax1.quiver(x_ux[skip], y_ux[skip], vel_x[skip][:, i], vel_y[skip][:, i], scale_units = 'width', scale = 1.0 / 2e5)
	# streamline10 = ax1.streamplot(x_stream, y_stream, Z_x_stream[:, :, i], Z_y_stream[:, :, i], density = [1.0, 3.0], linewidth = lw)
	streamfun_contour = ax1.tricontour(triang, stream, lines_stream, linewidth =1, colors = 'blue',linestyles = 'solid')



	fig.colorbar(contourf)
	ax1.set_ylim([-20e3, 50e3])
	ax1.set_xlim([0e3, 65e3])

	fig.set_size_inches(8, 2)


	fig.savefig('figures/sub_flownet.png', dpi = 1000)

	# ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	# plt.title('Surf X')
	# plt.xlabel('Meters from trench')
	# plt.ylabel('Meters from trench')

def plot_pressure_contour_sigmaloop():
	fig_10 = plt.figure()
	ax1 = fig_10.add_subplot(111)
	"""
	fig_15 = plt.figure()
	ax2 = fig_15.add_subplot(111)
	fig_20 = plt.figure()
	ax3 = fig_20.add_subplot(111)
	fig_25 = plt.figure()
	ax4 = fig_25.add_subplot(111)
	"""

	triang_p = tri.Triangulation(x_p, y_p, mesh.cells())
	triang_u = tri.Triangulation(x_u, y_u, mesh.cells())


	contour_steps = 200
	orig_cmap = cm.coolwarm
	midpoint10 = 1 - Z_p10.max() / (Z_p10.max() + abs(Z_p10.min()))
	shifted_cmap10 = shiftedColorMap(orig_cmap, midpoint = midpoint10)
	shifted_cmap_u = shiftedColorMap(cm.jet, midpoint = .4)

	zero = np.array([0])

	# levels_10 = np.array([Z_10.min(), Z_10.min()/2, Z_10.min()/4, Z_10.min()/8, Z_10.min()/16,
	#                      0, Z_10.max()/16, Z_10.max()/8, Z_10.max()/4, Z_10.max()/2, Z_10.max()])
	levels_10 = np.linspace(Z_p10.min(), Z_p10.max(), num = contour_steps, endpoint = True)
	levels_15 = np.linspace(Z_p15.min(), Z_p15.max(), num = contour_steps, endpoint = True)
	levels_20 = np.linspace(Z_p20.min(), Z_p20.max(), num = contour_steps, endpoint = True)
	levels_25 = np.linspace(Z_p25.min(), Z_p25.max(), num = contour_steps, endpoint = True)
	
	levels_u = np.linspace(Z_mag10.min(), Z_mag10.max(), num = contour_steps, endpoint = True)


	# ax1.scatter(x_all,y_all)
	#cp10f = ax1.tricontourf(triang_p, Z_p10, levels_10, cmap = shifted_cmap10)
	#cp10f2 = ax1.tricontourf(triang_u, Z_mag10, levels_u, cmap = shifted_cmap_u, alpha = .25)
	cp10 = ax1.tricontour(triang_p, Z_p10, zero, linewidth = 4, colors = 'black', linestyles = 'solid')

	cp15f = ax1.tricontourf(triang_p, Z_p15, levels_15, cmap = shifted_cmap10)
	cp15 = ax1.tricontour(triang_p, Z_p15, zero, linewidth = 4,colors = 'black', linestyles = 'solid')

	# cp20f = ax1.tricontourf(triang_p, Z_p20, levels_10, cmap = shifted_cmap10)
	cp20 = ax1.tricontour(triang_p, Z_p20, zero, linewidth = 4,colors = 'black', linestyles = 'solid')

	# cp25f = ax1.tricontourf(triang_p, Z_p25, levels_10, cmap = shifted_cmap10)
	cp25 = ax1.tricontour(triang_p, Z_p25, zero, linewidth = 4,colors = 'black', linestyles = 'solid')
	# ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	# plt.title('Surf X')
	# plt.xlabel('Meters from trench')
	# plt.ylabel('Meters from trench')

	#fig_10.colorbar(cp10f2)
	# fig_15.colorbar(cp15f)
	# fig_20.colorbar(cp20f)
	# fig_25.colorbar(cp25f)
def plot_pressure_contour_kloop():
	fig_k10 = plt.figure()
	ax1 = fig_k10.add_subplot(111)
	fig_k11 = plt.figure()
	ax2 = fig_k11.add_subplot(111)
	fig_k12 = plt.figure()
	ax3 = fig_k12.add_subplot(111)
	fig_k13 = plt.figure()
	ax4 = fig_k13.add_subplot(111)

	i = 1
	
	#stream_10 = streamfun_k10[:, i]
	#stream_11 = streamfun_k11[:, i]
	#stream_12 = streamfun_k12[:, i]
	#stream_13 = streamfun_k13[:, i]
	
	Z_10 = Z_pk10[:, i]
	Z_11 = Z_pk11[:, i]
	Z_12 = Z_pk12[:, i]
	Z_13 = Z_pk13[:, i]
	
	
	triang = tri.Triangulation(x_p, y_p, mesh.cells())
	triang_u = tri.Triangulation(x_u, y_u, mesh.cells())
	contour_steps = 200
	num_lines = 30
	orig_cmap = cm.coolwarm
	midpoint10 = 1 - Z_10.max() / (Z_10.max() + abs(Z_10.min()))
	midpoint11 = 1 - Z_11.max() / (Z_11.max() + abs(Z_11.min()))
	midpoint12 = 1 - Z_12.max() / (Z_12.max() + abs(Z_12.min()))
	midpoint13 = 1 - Z_13.max() / (Z_13.max() + abs(Z_13.min()))
	shifted_cmap10 = shiftedColorMap(orig_cmap, midpoint = midpoint10)
	shifted_cmap11 = shiftedColorMap(orig_cmap, midpoint = midpoint11)
	shifted_cmap12 = shiftedColorMap(orig_cmap, midpoint = midpoint12)
	shifted_cmap13 = shiftedColorMap(orig_cmap, midpoint = midpoint13)

	zero = np.array([0])

	# levels_10 = np.array([Z_10.min(), Z_10.min()/2, Z_10.min()/4, Z_10.min()/8, Z_10.min()/16,
	#                      0, Z_10.max()/16, Z_10.max()/8, Z_10.max()/4, Z_10.max()/2, Z_10.max()])
	levels_10 = np.linspace(Z_10.min(), Z_10.max(), num = contour_steps, endpoint = True)
	lines_10 = np.linspace(Z_10.min(), Z_10.max(), num = num_lines, endpoint = True)
	#lines_stream_10 = np.linspace(stream_10.min(), stream_10.max(), num = num_lines, endpoint = True)
	levels_11 = np.linspace(Z_11.min(), Z_11.max(), num = contour_steps, endpoint = True)
	lines_11 = np.linspace(Z_11.min(), Z_11.max(), num = num_lines, endpoint = True)
	#lines_stream_11 = np.linspace(stream_11.min(), stream_11.max(), num = num_lines, endpoint = True)
	levels_12 = np.linspace(Z_12.min(), Z_12.max(), num = contour_steps, endpoint = True)
	lines_12 = np.linspace(Z_12.min(), Z_12.max(), num = num_lines, endpoint = True)
	#lines_stream_12 = np.linspace(stream_12.min(), stream_12.max(), num = num_lines, endpoint = True)
	levels_13 = np.linspace(Z_13.min(), Z_13.max(), num = contour_steps, endpoint = True)
	lines_13 = np.linspace(Z_13.min(), Z_13.max(), num = num_lines, endpoint = True)
	#lines_stream_13 = np.linspace(stream_13.min(), stream_13.max(), num = num_lines, endpoint = True)

	skip = (slice(None, None, 1))

	if event =='EQ':

		contour10f = ax1.tricontourf(triang, Z_10, levels_10, cmap = shifted_cmap10)
		contour10 = ax1.tricontour(triang, Z_10, zero, linewidth = 4,colors = 'black', linestyles = 'solid')
		quiver10 = ax1.quiver(x_ocean_u[skip], y_ocean_u[skip], seaflux_x_k10[:,i][skip][:, i], seaflux_y_k10[skip][:, i], color = 'black', scale_units = 'width', width = 1.0/3e2, scale = 1.0 / 2e5)


		contour11f = ax2.tricontourf(triang, Z_11, levels_11, cmap = shifted_cmap11)
		contour11 = ax2.tricontour(triang, Z_11, zero, linewidth = 4,colors = 'black', linestyles = 'solid')
		quiver11 = ax2.quiver(x_ocean_u[skip], y_ocean_u[skip], seaflux_x_k11[skip][:, i], seaflux_y_k11[skip][:, i], units = 'inches', scale_units = 'width', width = 1.0/5e1, scale = 1.0 / 2e5)


		contour12f = ax3.tricontourf(triang, Z_12, levels_12, cmap = shifted_cmap12)
		contour12 = ax3.tricontour(triang, Z_12, zero, linewidth = 4, colors = 'black', linestyles = 'solid')
		quiver12 = ax3.quiver(x_ocean_u[skip], y_ocean_u[skip], seaflux_x_k12[skip][:, i], seaflux_y_k12[skip][:, i], units = 'inches', scale_units = 'width', width = 1.0/5e1, scale = 1.0 / 2e5)

		contour13f = ax4.tricontourf(triang, Z_13, levels_13, cmap = shifted_cmap13)
		contour13 = ax4.tricontour(triang, Z_13, zero, linewidth = 4,colors = 'black', linestyles = 'solid')
		quiver12 = ax4.quiver(x_ocean_u[skip], y_ocean_u[skip], seaflux_x_k13[skip][:, i], seaflux_y_k13[skip][:, i],
						  units = 'inches', scale_units = 'width', width = 1.0/5e1, scale = 1.0 / 2e5)

	if event == 'topo':
		#lw = 15*speed[:,:,i]

		contour10f = ax1.tricontourf(triang, Z_10, levels_10, cmap = shifted_cmap10)
		contour10 = ax1.tricontour(triang, Z_10, lines_10, linewidth = 4,colors = 'black', linestyles = 'solid')
		quiver10 = ax1.quiver(x_ocean_u[skip], y_ocean_u[skip], seaflux_x_k10[skip][:, i], seaflux_y_k10[skip][:, i], units = 'inches', scale_units = 'width', width = 3e-2, scale = 1.0/1e5, color = 'blue')
		#quiver10 = ax1.quiver(x_ux[skip], y_ux[skip], vel_x_k10[skip][:, i], vel_y_k10[skip][:, i], scale_units = 'width', scale = 1.0 / 2e5)
		#streamline10 = ax1.streamplot(x_stream, y_stream, Z_x_stream[:, :, i], Z_y_stream[:, :, i], density = [1.0, 3.0], linewidth = lw)
		#streamfun_contour10 = ax1.tricontour(triang_u, stream_10, lines_stream_10, linewidth=4, colors='blue', linestyles='solid')

		"""
		contour11f = ax2.tricontourf(triang, Z_11, levels_11, cmap = shifted_cmap11)
		contour11 = ax2.tricontour(triang, Z_11, lines_11, linewidth = 4, colors = 'black', linestyles = 'solid')
		quiver11 = ax2.quiver(x_ocean_u[skip], y_ocean_u[skip], seaflux_x_k11[skip][:, i], seaflux_y_k11[skip][:, i],
							  units = 'inches', scale_units = 'width', width = 3e-2, scale = 1.0 / 1e5, color = 'blue')

		contour12f = ax3.tricontourf(triang, Z_12, levels_12, cmap = shifted_cmap12)
		contour12 = ax3.tricontour(triang, Z_12, lines_12, linewidth = 4, colors = 'black', linestyles = 'solid')
		quiver12 = ax3.quiver(x_ocean_u[skip], y_ocean_u[skip], seaflux_x_k12[skip][:, i], seaflux_y_k12[skip][:, i],
							  units = 'inches', scale_units = 'width', width = 3e-2, scale = 1.0 / 1e5, color = 'blue')

		contour13f = ax4.tricontourf(triang, Z_13, levels_13, cmap = shifted_cmap13)
		contour13 = ax4.tricontour(triang, Z_13, lines_13, linewidth = 4, colors = 'black', linestyles = 'solid')
		quiver13 = ax4.quiver(x_ocean_u[skip], y_ocean_u[skip], seaflux_x_k13[skip][:, i], seaflux_y_k13[skip][:, i],
							  units = 'inches', scale_units = 'width', width = 3e-2, scale = 1.0 / 1e5, color = 'blue')


		"""
	# ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	# plt.title('Surf X')
	# plt.xlabel('Meters from trench')
	# plt.ylabel('Meters from trench')
	
	fig_k10.colorbar(contour10f)
	ax1.set_ylim([-4e4, 1e4])
	"""
	ax2.set_ylim([-4e4, 1e4])
	ax3.set_ylim([-4e4, 1e4])
	ax4.set_ylim([-4e4, 1e4])

	fig_k11.colorbar(contour11f)
	fig_k12.colorbar(contour12f)
	fig_k13.colorbar(contour13f)
	"""
	
def plot_def_contour_sigmaloop():
	fig_10 = plt.figure()
	ax1 = fig_10.add_subplot(111)
	"""
	fig_15 = plt.figure()
	ax2 = fig_15.add_subplot(111)
	fig_20 = plt.figure()
	ax3 = fig_20.add_subplot(111)
	fig_25 = plt.figure()
	ax4 = fig_25.add_subplot(111)
	"""
	triang_p = tri.Triangulation(x_p, y_p, mesh.cells())
	triang = tri.Triangulation(x_ux, y_ux, mesh.cells())

	contour_steps = 200
	orig_cmap = cm.jet
	midpoint10 = .25
	shifted_cmap10 = shiftedColorMap(orig_cmap, midpoint = midpoint10)
	zero = np.array([0])

	#levels_10 = np.array([0, Z_mag10.max()/16, Z_mag10.max()/8, Z_mag10.max()/4, Z_mag10.max()/2, Z_mag10.max()])
	levels_10 = np.linspace(Z_mag10.min(), Z_mag10.max(), num = contour_steps, endpoint = True)
	levels_15 = np.linspace(Z_mag15.min(), Z_mag15.max(), num = contour_steps, endpoint = True)
	levels_20 = np.linspace(Z_mag20.min(), Z_mag20.max(), num = contour_steps, endpoint = True)
	levels_25 = np.linspace(Z_mag25.min(), Z_mag25.max(), num = contour_steps, endpoint = True)

	skip = (slice(None, None, 80))

	# ax1.scatter(x_all,y_all)
	cp10f = ax1.tricontourf(triang, Z_mag10, levels_10, cmap = shifted_cmap10, alpha=.5)
	cp10 = ax1.tricontour(triang_p, Z_p10, zero, linewidth = 10 ,colors = 'black', linestyles = 'solid')
	quiver10 = ax1.quiver(x_u[skip],y_u[skip], Z_ux10[skip], Z_uy10[skip])

	# cp10f = ax1.tricontourf(triang, Z_p15, levels_15, cmap = shifted_cmap10)
	# cp10 = ax1.tricontour(triang, Z_p15, zero, linewidth = 4,colors = 'black', linestyles = 'solid')

	# cp10f = ax1.tricontourf(triang, Z_p20, levels_20, cmap = shifted_cmap10)
	# cp10 = ax1.tricontour(triang, Z_p20, zero, linewidth = 4,colors = 'black', linestyles = 'solid')

	# cp25f = ax1.tricontourf(triang, Z_p25, levels_25, cmap = shifted_cmap10)
	# cp25 = ax1.tricontour(triang, Z_p25, zero, linewidth = 4,colors = 'black', linestyles = 'solid')

	# ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	# plt.title('Surf X')
	# plt.xlabel('Meters from trench')
	# plt.ylabel('Meters from trench')

	fig_10.colorbar(cp10f)
	# fig_15.colorbar(cp15f)
	# fig_20.colorbar(cp20f)
	# fig_25.colorbar(cp25f)
	# fig_15.colorbar(cp15f)
	# fig_20.colorbar(cp20f)
	# fig_25.colorbar(cp25f)

def plot_pressure_kappa_movie(Z):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	#i = 3
	Z0 = Z[:,0]



	triang = tri.Triangulation(x_p, y_p, mesh.cells())

	contour_steps = 200
	num_lines = 20
	orig_cmap = cm.coolwarm
	midpoint = 1 - Z0.max() / (Z0.max() + abs(Z0.min()))
	shifted_cmap = shiftedColorMap(orig_cmap, midpoint = midpoint)


	zero = np.array([0])


	levels = np.linspace(Z0.min(), Z0.max(), num = contour_steps, endpoint = True)
	lines = np.linspace(Z0.min(), Z0.max(), num = num_lines, endpoint = True)

	skip = (slice(None, None, 2))


	for i in range(0,30):
		Zi = Z[:, i]
		ax1.clear()
		contour10f = ax1.tricontourf(triang, Zi, levels, cmap = shifted_cmap)
		contour10 = ax1.tricontour(triang, Zi, zero, linewidth = 4,colors = 'black', linestyles = 'solid')
		quiver10 = ax1.quiver(x_ocean_u[skip], y_ocean_u[skip], seaflux_x_k10[skip][:, i], seaflux_y_k10[skip][:, i], color = 'blue', units = 'inches', scale_units = 'inches', width = 3e-2)

		ax1.set_ylim([-4e4, 1e4])







	# ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	# plt.title('Surf X')
	# plt.xlabel('Meters from trench')
	# plt.ylabel('Meters from trench')

	#fig.colorbar(contour10f)

def plot_topo():
	wb_slip = pyxl.load_workbook('water_table.xlsx')
	topo_data = wb_slip['Sheet1']
	X_topo = np.array([[cell.value for cell in col] for col in topo_data['A2':'A21']])
	h_topo = np.array([[cell.value for cell in col] for col in topo_data['B2':'B21']])
	h_topo2 = 0.5 * np.array([[cell.value for cell in col] for col in topo_data['B2':'B21']])
	h_topo3 = 0.25 * np.array([[cell.value for cell in col] for col in topo_data['B2':'B21']])
	h_topo4 = 0.75 * np.array([[cell.value for cell in col] for col in topo_data['B2':'B21']])

	X_topo = X_topo.reshape(X_topo.shape[0], )
	h_topo = h_topo.reshape(h_topo.shape[0], )
	h_topo2 = h_topo2.reshape(h_topo2.shape[0], )
	h_topo3 = h_topo3.reshape(h_topo3.shape[0], )
	h_topo4 = h_topo4.reshape(h_topo4.shape[0], )

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	#ax1.set_title('Nicoya Peninsula Topography')
	ax1.plot(X_topo, h_topo, '-', ms = 3, mew = 2, fillstyle = 'none', color = 'green', linewidth = 3)#, label = 'Land surface')
	ax1.plot(X_topo, h_topo2, '-', ms = 3, mew = 2, fillstyle = 'none', color = 'blue', linewidth = 3)#, label = 'Water table')
	ax1.plot(X_topo, h_topo3, '-', ms = 3, mew = 2, fillstyle = 'none', color = 'blue', linewidth = 1)
	ax1.plot(X_topo, h_topo4, '-', ms = 3, mew = 2, fillstyle = 'none', color = 'blue', linewidth = 1)
	ax1.fill_between(X_topo, h_topo3, h_topo4, alpha = .5, facecolor='blue')


	fig.set_size_inches(8, 1.0)
	plt.xlim((0, 250e3))
	# plt.ylim((5e-8, 10))
	#ax1.legend(loc=2)
	fig.savefig('figures/topo_flow.png', dpi = 400)
	# fig1.savefig('figures/SGD_sea_flux.png', dpi = 400)
	# fig3.savefig('figures/SGD_percent.png', dpi=400)#




##########################################################################
######################### PLOTTING COMMANDS ##############################
##########################################################################


# plot_sea_flux()
plot_pressure_contour()
#plot_pressure_contour_kloop()
#plot_pressure_contour_sigmaloop()
#plot_def_contour_sigmaloop()
#plot_contour_def_sigma()
#plot_pressure_kappa_movie(Z_pk11)
#plot_topo()






plt.show()
