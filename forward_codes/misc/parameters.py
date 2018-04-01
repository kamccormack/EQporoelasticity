import numpy as np
import dolfin as dl

###################### 
expr_label = 'expr_7'
expr_dir = expr_label
#dl.parameters['num_threads'] =1
max_num_threads = 1
dl.set_log_level(20)
dl.set_log_active(active=False)
# write out the param file to the expr dir
###################### Geometry and Mesh ################ 
dim =3
xl = 10000 #Wedth 
yl = 10000 #Length
zl = 500 #Depth
cl_d = 0.0
aq_d = 400
uz_d = 100
offset = 10.0 #screen interval
screen = 237.7
well_center =  dl.Point(xl/2.0, yl/2.0)
nx = 10 #int(xl/1000)
ny = 10 #int(yl/1000)
nz = 3  #int(zl/100)
mesh = dl.BoxMesh(dl.Point(0.0,0.0,0.0), dl.Point(xl,yl,zl), nx, ny, nz)

## initial mesh
#dl.plot(mesh, interactive = False)

#if proc_id == 0:
    #print mesh.num_cells()

########### Forward solve physical parameters #########
mu = 0.001
alpha = 0.8 # Amal: need to change this
rho = 997.97 # density of water: Kg/m^3 
grv = 9.80665 # gravitational acc: m/s^3
beta_w = 4.4e-10 # water compressibility in aquifer: m sec^2 /kg. form aqtesolv tool webpage
properties   = ['G',     'v', 'K',    'E',  'S_sk']
#uz           = [ 3.5e8,  .25, 0.0,    0.0,   0.0  ] 
aq           =[ 3.4e8,  .25, 5.75e-5, .6, 1.9e-5 ] 
#uz           = [ 3.5e8,  .25, 5.78e-8, .6, 9.0e-6 ] # Not the paper values
uz = aq
u_side_0_bc = 'u.n' 

################ functional space parameters ###############
#formulation =  '2_field'
#spaces = [1,1,1] # 2 fields p_k, u_k, kappa_k
#parmeter_space = 'CG'
#oss_list = [[False, True, True, True]] # 2 fields p,u1,u2,u3

formulation =  '3_field_flux'
parmeter_space = 'CG'
spaces = [0,1,1,1] # 3 fields p_k, u_k, q_k, kappa_k
oss_list =[[False, True, True, True, False,False,False]] # 3 fields p, u1, u2, u3, q1, q2, q3
############################# Time #########################
t_init = 0.0
numdays = 3#40.0 
t_final =np.float64( numdays*sec_in_day) # to be compatible with list for the - operator 
ntime = [100]
dtt = [t_final/ntime[0]]


################# fwd/inverse problem settings #################
rel_noise = 0.01
rel_noises = [rel_noise]
noise_inv = [1.0/ rel_noise]
#Omega =  1/(rel_noise)**2

expr_id = 8 # true parameters expression id
rate_days = -9028.0 #m^3/day

orderPrior = 2
gamma = 10.0
delta = 1.0e-5
Theta = dl.Constant( ((1.,0., 0.),(0., 1., 0.),(0., 0., 100.)) ) # only for bilaplacian case

prior_mean_id = 4

# observation time intervals
TM = np.array( ((0.4*t_final,0.9*t_final ),(0.7*t_final,0.95*t_final)) )
# observation location and operator
eps2 = 1e-8;
numx = 10
numy = 10
x_coor_1 = np.linspace(500,9500, numx)
y_coor_1 = np.linspace(500,9500, numy)
y_coor   = np.tile(y_coor_1, numx)
x_coor = x_coor_1.repeat(numy)
Nx_surf = numx*numy
Xpt_surf = np.vstack((x_coor,y_coor, (zl-eps2) * np.ones((1, Nx_surf)) ))
data_type =  'continuous' # 'synthesized'# 

if data_type == 'continuous':
    Omega = 1e-2
else:
    Omega = 1e2

print Xpt_surf

use_prior_mean = True # for the solver init cond
verify_model = True
secOrder = True
verH = True
verify_at =  'prior_mean' #'true_param' #
verf_eps = np.power(.5, np.arange(1,32, 1))
verf_eps = verf_eps[::-1]
#verf_eps = np.logspace(-8,-6, 20)

rel_tolerance = 1e-6
abs_tolerance = 1e-21
max_iter      = 500
inner_rel_tolerance = 1e-21
c_armijo = 1e-4
GN_iter = 1+15
max_backtracking_iter = 50
print_level = 1
cg_coarse_tolerance = 0.5
############### solver choice ####################################
solver = dl.LUSolver("mumps")
solvert =  dl.LUSolver("mumps")


#solver =dl.LUSolver("mumps")# dl.PETScLUSolver()
#solver.parameters["reuse_factorization"] = True 
#solvert =dl.LUSolver("mumps")
#solvert.parameters["reuse_factorization"] = True 


