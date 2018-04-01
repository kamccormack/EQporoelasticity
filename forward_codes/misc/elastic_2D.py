"""
FEniCS program for the elastic response to a megathrust earthquake on the overlying continental crust. This is the forward elastic problem given synthetic slip on the bottom boundary.

Kimberly McCormack
08/20/15

3D
"""

from dolfin import *
import numpy as np
import time
import openpyxl as pyxl
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import shutil, os
import scipy.io
import scipy.interpolate
#dolfin.parameters.reorder_dofs_serial = False
import scipy as sp
from scipy.interpolate import griddata
from tempfile import TemporaryFile

"""


"""
start = time.time()

##########################################################################
########################### USER INPUTS ##################################
##########################################################################

plot_figs = 'yes' # Plot figures with fenics plotter?
event = 'syn' # syn = synthetic event, data = real EQ data
new_mesh = 'yes' # Creates new indices file to extract solution at surface nodes/GPS stations when turned on, otherwise it loads a previously saved file. Must = 'yes' at least once to create files. TURN OFF ONLY IF NONE OF THE FOLLOWING HAVE BEEN CHANGED: a) mesh file  b) mesh refinement 3) order of basis functions
drained = 'undrained' #drained or undrained

#################### SYNTHETIC EVENT PARAMETERS #########################

sigma_b = 2e4 # Std deviation from the center of slip(loosely, how big is the area of slip?)
xcenter = 80e3 # How far is the center of the slip patch from the trench?
u0_EQ = -4.0 # For an EQ, what was the max slip?
surface_k = -10 # The  log of the permeability at the surface. Right now, the log(k) reduces linearly with depth (line 74ish). Can change to an expontential decay but need a mesh that is much finer at the top to capture the rapid decay portion smoothly

######################## SOLVER ##########################################

solver = LUSolver('petsc')

##########################################################################
################### DEFINE MESH AND PARAMETERS ###########################
##########################################################################
#Change the path name for the mesh files!

path = "/home/fenics/shared/meshes"

mesh = Mesh(path+"/CR_2D.xml")
boundaries = MeshFunction("size_t", mesh,path+"/CR_2D_facet_region.xml")

V = VectorFunctionSpace(mesh,'Lagrange', 1)
u = TrialFunction(V)
v = TestFunction(V)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
n = FacetNormal(mesh) #normal vector

######################## ELASTIC PARAMETERS #############################

if drained =='drained':
    nu = 0.25  # Poisson's ratio
elif drained == 'undrained':
    nu = .4 # undrained poissons ratio
delta = 1e-6
#E = Constant('5e4') # Young's modulus [MPa]
E = Expression('(5.7972e-10*(-(pow(x[1],3)))) - (7.25283e-5*(-(pow(x[1],2)))) + (3.8486e-6*(-x[1])) + 6.94e4', degree=1) # Young's modulus [MPa]. Fit from Deshon paper
mu   = Expression('E / (2.0*(1.0 + nu))', E=E, nu=nu, degree=1) # shear modulus
lmbda = Expression('E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))', E=E, nu=nu, degree=1)  # Lame's parameter
d = u.geometric_dimension() # number of space dimensions
I = Identity(d)
T = (I - outer(n,n)) # tangent operator for boundary condition


##########################################################################
################### INITIAL AND BOUNDARY CONDITIONS ######################
##########################################################################

#Initial conditions: (('p0','ux0','uy0'))

################### DIRICHLET BOUNDARY CONDITIONS ########################

ub0 = Expression(('0','0'), degree=1)
ub = Expression(('1','0'), degree=1)
bc1 = DirichletBC(V, ub0, boundaries, 1) #top/surface
bc2 = DirichletBC(V, ub, boundaries, 2) #bottom/fault boundary
bc3 = DirichletBC(V, ub0, boundaries, 3) # right/ back side (no slip condition)
bc4 = DirichletBC(V, ub0, boundaries, 4) # mantle (no slip condition)

bcs = [bc4] #These are the BC's that are actually applied
slab = 2 # Boundary number of the subducting slab


############### INITIAL BOUNDARY CONDITION ##############################
u0d = u0_EQ
u0slab = Expression(('u0d*exp(-(pow((x[0] - xcenter),2)/(pow(sigma,2))))', '0'), u0d=u0d, xcenter=xcenter,sigma=sigma_b, degree=1)


##########################################################################
############################ ASSEMBLE SYSTEM #############################
##########################################################################

def strain(w): # strain = 1/2 (grad u + grad u^T)
    return sym(nabla_grad(w))
def sigma(w): # stress = 2 mu strain + lambda tr(strain) I
    return 2.0*mu*strain(w) + lmbda*div(w)*I

a_E = assemble(inner(sigma(u),nabla_grad(v))*dx) #Elasticity
a_Boundary = assemble((1/delta)*dot(u,n)*dot(v,n)*ds(slab) + (1/delta)*inner(T*u ,T*v)*ds(slab)) #Weakly imposed boundary conditions. First term: no displacement normal to the bottom boundary, Second term: tangential displacement  = u0slab

A = a_E + a_Boundary # Left hand side (LHS)

a_Boundary0 = assemble((1/delta)*inner(T*u0slab ,T*v)*ds(slab))


##########################################################################
######## EXTRACT INDICES/COORDINATES OF SURFACE  #########################
##########################################################################

if new_mesh == 'yes': #create surface/GPS indices and save them
    """
    dofmap = V.dofmap()
    dof_xy = dofmap.tabulate_all_coordinates(mesh).reshape((V.dim(), -1))
    x_coords, y_coords  = dof_xy[:, 0], dof_xy[:, 1]
    bmesh = BoundaryMesh(mesh, 'exterior')
    xy_surf = bmesh.coordinates()
    x_surf, y_surf = xy_surf[:,0], xy_surf[:,1]
    bnd_ind = np.where((x_surf < 2.5e5) & (y_surf > -4e3))[0]
    x_surf, y_surf = x_surf[bnd_ind], y_surf[bnd_ind]
    size_surf = x_surf.shape[0]
    size_w_surf = 2*size_surf
    indices_surf = np.empty((size_surf, 2))
    for j in range(0, size_surf): # There has got to be a cleaner way to do this...
        indice_surf = np.where((x_coords[:] == x_surf[j]) & (y_coords[:] == y_surf[j]))[0]
        indices_surf[j, :] = indice_surf
    indices_surf = indices_surf.reshape((size_w_surf), order='F')
    np.save("saved_variables_elastic/x_surf_2D", x_surf)
    np.save("saved_variables_elastic/y_surf_2D", y_surf)
    np.save("saved_variables_elastic/indices_surf_CR_2D", indices_surf)
    np.save("saved_variables_elastic/sizew_surf_CR_2D", size_w_surf)
	"""

elif new_mesh == 'no': # load saved indices/variables
    x_surf = np.load("saved_variables_elastic/x_surf_2D.npy")
    y_surf = np.load("saved_variables_elastic/y_surf_2D.npy")
    indices_surf = np.load("saved_variables_elastic/indices_surf_CR_2D.npy")
    size_w_surf = np.load("saved_variables_elastic/sizew_surf_CR_2D.npy")


##########################################################################
####################### SOLVE THE SYSTEM #################################
##########################################################################

#Sol_surf  = np.empty((size_w_surf, )) # matrix to save surface solution
ufile = File("results_elastic/def.pvd")
#if drained == 'drained':
    #open("saved_results_elastic/Sol_surf_drained_2D.npy", "w")
#elif drained == 'undrained':
    #open("saved_results_elastic/Sol_surf_undrained_2D.npy", "w")

b = Vector()
b = a_Boundary0
[bc.apply(b) for bc in bcs]
[bc.apply(A) for bc in bcs]
u = Function(V)
solver.set_operator(A)
solver.solve(u.vector(), b)

####################### SAVE SOLUTION ###########################
#u.rename('u','u')
#ufile << u

#Sol_surf = u.vector()[indices_surf]

#if drained == 'drained':
#    np.save("saved_results_elastic/Sol_surf_drained_2D", Sol_surf)
#elif drained == 'undrained':
#    np.save("saved_results_elastic/Sol_surf_undrained_2D", Sol_surf)

######################## PLOT SOLUTION ##########################
if event == 'syn' and plot_figs == 'yes':
    #plot(p, key='pressure', mode = 'color')
    #plot(u, key="def", mode = 'displacement',range_min = 0.0, range_max = 4.0)
    plot(u)#, key='def', mode = 'glyphs', range_min = 0.0, range_max = 4.0)
    #plot(p, key="pressure" ,range_min = -.50, range_max = .10)#, mode = 'color')
    #plot(u_post, key = 'post')

elif event == 'data' and plot_figs == 'yes':
    plot(u, key='def', range_min = 0.0, range_max = 4.0, scale=4)
    #plot(p, key='pressure', mode = 'color',range_min = -1.0, range_max = 1.0)


print "Time elasped" , time.time()-start, "seconds"

if plot_figs == 'yes':
    interactive() # hold the plot

#File("u_instant.xml") << u

####################################

