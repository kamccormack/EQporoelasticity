"""
Jeonghun J. Lee		2017-10-01

Modified by Kimmy McCormack

A numerical solver of elastostatic equation div(Ce(u)) = f with u = 0 on the boundary
and discontinuity of displacement on fault (:= Gamma_I) (rupture)
The lowest order Arnold-Falk-Winther element is used.

Written in mixed form:
A sigma = e(u),
-div(sigma) = f
with weak symmetry elements of AFW (denoted by V) where
gamma = - skw(grad(u))


(stress, deformation, pressure, skew):
(sigma, u, p, gamma) in V : trial functions with sigma n = g on Gamma_N (if exists)
(tau, v, q, eta) in V : test functions with tau n = 0 on Gamma_N

A variational form is:
(A*sigma, tau) + (div(tau), u) +/-(?) (gamma, tau) = <u0 dot(tau, n)>
- (div(sigma), v) + alpha * (grad(p), v) = 0
(Se*p, q)_t - alpha*(u,grad(q))_t + ((k/mu)*grad(p), grad(q)) = 0
(sigma, eta) = 0

The slip boundary condition across the fault is defined as:
<u0 dot(tau, n)> = int_{Gamma_D} (u0, (tau*n))*ds + (f, w) + int_{Gamma_I} (disp, tau*n))*dS

where disp is the jump of displacement at fault.

Last modified date : 2017-09-28
"""

from dolfin import *

tol = 1E-14

solver = LinearSolver('mumps')
prm = solver.parameters
prm['reuse_factorization'] = True  # Saves Factorization of LHS

paraviewpath = 'results/paraview/'
name_append = "chichi_test"
pfilename = paraviewpath + "pressure2D" + name_append + ".pvd"
ufilename = paraviewpath + "def2D" + name_append + ".pvd"
sigfilename = paraviewpath + "stress2D" + name_append + ".pvd"
skwfilename = paraviewpath + "skew2D" + name_append + ".pvd"

pfile = File(pfilename)
ufile = File(ufilename)
sigfile = File(sigfilename)
skwfile = File(skwfilename)


def lin_sys1a(sigma, tau, u, gamma):  # (A*sigma, tau) + (div(tau), u) - (gamma, tau)
	form1 = 0.5 * (1.0 / mu) * (inner(sigma, tau)) * dx - (lam / (4 * mu * (mu + lam))) * (
	sigma[0, 0] + sigma[1, 1]) * (tau[0, 0] + tau[1, 1]) * dx \
	        + (dot(div(tau), u)) * dx \
	        - gamma * (tau[0, 1] - tau[1, 0]) * dx
	return assemble(form1)

def lin_sys2a(sigma, v, p):  # -(div(sigma), v) + alpha * (grad(p), v)
	a_E =  assemble(-(dot(div(sigma), v)) * dx)
	a_Grad = assemble(-p * (nabla_div(v)) * dx)
	return a_E + alpha*a_Grad

def lin_sys3a(p, q):  #  ((k/mu)*grad(p), grad(q))
	a_K = assemble(((kappa) / muf) * inner(nabla_grad(p), nabla_grad(q)) * dx)  # Stiffness matrix
	return dt * a_K

def lin_sys4a(u, p, q):  # (Se*p, q)_t - alpha*(u,grad(q))_t
	a_Mass = assemble(Se * p * q * dx)  # Mass matrix
	a_Div = assemble(nabla_div(u) * q * dx)  # Divergence matrix
	return a_Mass #+ alpha*a_Div

def lin_sys5a(sigma, eta): # (sigma, eta)
	form5 = eta * (sigma[0, 1] - sigma[1, 0]) * dx
	return assemble(form5)

##################################################################

def lin_sys1(sigma, tau, u, gamma):  # (A*sigma, tau) + (div(tau), u) - (gamma, tau)
	form1 = 0.5 * (1.0 / mu) * (inner(sigma, tau)) * dx - (lam / (4 * mu * (mu + lam))) * (
	sigma[0, 0] + sigma[1, 1]) * (tau[0, 0] + tau[1, 1]) * dx \
	        + (dot(div(tau), u)) * dx \
	        - gamma * (tau[0, 1] - tau[1, 0]) * dx
	return form1

def lin_sys2(sigma, v, p):  # -(div(sigma), v) + alpha * (grad(p), v)
	a_E =  (dot(div(sigma), v)) * dx
	a_Grad = -alpha * p * (nabla_div(v)) * dx
	return a_E + a_Grad

def lin_sys3(p, q):  #  ((k/mu)*grad(p), grad(q))
	a_K = ((dt*kappa) / muf) * inner(nabla_grad(p), nabla_grad(q)) * dx  # Stiffness matrix
	return a_K

def lin_sys4(u, p, q):  # (Se*p, q)_t - alpha*(u,grad(q))_t
	a_Mass = Se * p * q * dx  # Mass matrix
	a_Div = alpha*nabla_div(u) * q * dx  # Divergence matrix

	#Interior penalty method? - from poisson example
	# a = dot(grad(v), grad(u)) * dx \
	#     - dot(avg(grad(v)), jump(u, n)) * dS \
	#     - dot(jump(v, n), avg(grad(u))) * dS \
	#     + alpha / h_avg * dot(jump(v, n), jump(u, n)) * dS

	return a_Mass + a_Div

def lin_sys5(sigma, eta): # (sigma, eta)
	form5 = eta * (sigma[0, 1] - sigma[1, 0]) * dx
	return form5

def bound_sys(disp_t, tau):
	Lform = C * dot(0.5 * jump(dot(hor_vec, n)) * avg(disp_t), 0.5 * jump(dot(tau, n))) * dS0(3)
	return Lform

######################### POROELASTIC PARAMETERS #############################
delta = 1e-6  # Dirichlet control regularization on bottom boundary
muf = Constant('1e-9')  # pore fluid viscosity [MPa s]
B = .7  # Skempton's coefficient  ...Should this be constant?
nuu = .4  # undrained poissons ratio
nu = 0.25  # Poisson's ratio
E = Expression('500 * pow(-x[1], 0.516) + 2e4', degree=1)
mu = Expression('E / (2.0*(1.0 + nu))', E=E, nu=nu, degree=1)  # shear modulus
lmbda = Expression('E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))', E=E, nu=nu,
                        degree=1)  # Lame's parameter
alpha = 3 * (nuu - nu) / (B * (1 - 2 * nu) * (1 + nuu))  # Biot's coefficient
Se = Expression('(9*(nuu-nu)*(1-(2*nuu)))/(2*mu*pow(B,2)*(1-2*nu)*pow((1+nuu),2))', nuu=nuu,
                     nu=nu, mu=mu, B=B, degree=1)  # mass specific storage

kappa = Constant('1e-10')
#mu = 1.0;
lam = lmbda  # Lame coefficients
sigma_b = 1e4
xcenter = 25e3
u0_EQ = -4.0

disp = Expression(('u0*exp(-(pow((x[0] - xcenter),2)/(pow(sigma,2))))', '0.',), u0=u0_EQ,
                  xcenter=xcenter, sigma=sigma_b, degree=5)
# disp = Expression(('0.','cos(2*pi*((x[1]/20e3)-0.5))'), degree = 5)
rhs = Expression(('0.', '0.'), degree=5)

path = "/fenics/shared/meshes"
mesh_size = 'coarse'  # fine, med, coarse

mesh = Mesh(path + '/CHI2D_' + mesh_size + '.xml')
boundaries = MeshFunction("size_t", mesh, path + '/CHI2D_' + mesh_size + '_facet_region.xml')
subdomains = MeshFunction("size_t", mesh, path + '/CHI2D_' + mesh_size + '_physical_region.xml')

ds = Measure('ds')(domain=mesh, subdomain_data=boundaries)
# dx = Measure('dx', domain=mesh, subdomain_data=subdomains)
dS0 = Measure('dS')(domain=mesh, subdomain_data=boundaries)


Mh = VectorElement('BDM', mesh.ufl_cell(), 1)
Vh = VectorElement('DG', mesh.ufl_cell(), 0)
Ph = FiniteElement('CG', mesh.ufl_cell(), 1)
Kh = FiniteElement('DG', mesh.ufl_cell(), 0)
ME = FunctionSpace(mesh, MixedElement([Mh, Vh, Ph, Kh]))
Vspace = FunctionSpace(mesh, Vh)
Pspace = FunctionSpace(mesh, Ph)

#initial conditions
pold = Function(Pspace)
uold = Function(Vspace)

# Variational form
n = FacetNormal(mesh)
(tau, v, q, eta) = TestFunctions(ME)
(sigma, u, p, gamma) = TrialFunctions(ME)

sigb0 = Expression((("0", "0"), ("0", "0")), degree=5)
testb = Expression(("0", "1"), degree=5)

bc1 = DirichletBC(ME.sub(0), sigb0, boundaries, 1)  # left side roller
bc2 = DirichletBC(ME.sub(0), sigb0, boundaries, 2)  # left side roller
bc3 = DirichletBC(ME.sub(0), sigb0, boundaries, 5)  # left side roller
bc4 = DirichletBC(ME.sub(0), sigb0, boundaries, 6)  # left side roller

bctest = DirichletBC(ME.sub(1), testb, boundaries, 5)  # left side roller
bctest2 = DirichletBC(ME.sub(1), testb, boundaries, 2)  # left side roller
bctest3 = DirichletBC(ME.sub(1), testb, boundaries, 5)  # left side roller
bctest4 = DirichletBC(ME.sub(1), testb, boundaries, 6)  # left side roller

bcs = [bc1, bc2]
# bcs = []


d = u.geometric_dimension()  # number of space dimensions
I = Identity(d)
T = (I - outer(n, n))
C = Constant(1.)
hor_vec = Constant((-1., 0.))
disp_t = disp - inner(disp, n) * n



dt = 1
t_f = 3
t = 0
count = 0

# Asssembled system
# A = lin_sys1a(sigma, tau, u, gamma) \
#     + lin_sys2a(sigma, v, p) \
#     + lin_sys3a(p, q) \
#     + lin_sys4a(u, p, q) \
#     + lin_sys5a(sigma, eta)
	

#weak form system
a = lin_sys1(sigma, tau, u, gamma) \
     + lin_sys2(sigma, v, p) \
     + lin_sys3(p, q) \
     + lin_sys4(u, p, q) \
     + lin_sys5(sigma, eta)

A = assemble(a)



# need to restrict to tangental plane of fault
#b = assemble(inner(rhs, v)*dx + C * dot(0.5 * jump(dot(hor_vec, n)) * avg(disp_t), 0.5 * jump(dot(tau, n))) * dS0(3))
#b = inner(rhs, v)*dx + C * dot(0.5 * jump(dot(hor_vec, n)) * avg(disp_t), 0.5 * jump(dot(tau, n))) * dS0(3)

b = bound_sys(disp_t, tau) #+ lin_sys4(uold, pold, q)



while t < t_f:

	print "timestep: ", count

	L = assemble(b) #+ lin_sys4a(uold, pold, q)

	[bc.apply(L) for bc in bcs]
	[bc.apply(A) for bc in bcs]

	# Solve problem
	U = Function(ME)

	solve(A, U.vector(), L)
	#solver.set_operator(A)
	#solver.solve(U.vector(), L)

	#solve(a == b, U, bcs)
	(sigma, u, p, gamma) = U.split(deepcopy=True)

	p.rename('pressure', 'pressure')
	pfile << p
	u.rename('u', 'u')
	ufile << u
	sigma.rename('sigma', 'sigma')
	sigfile << sigma
	gamma.rename('skew', 'skew')
	skwfile << gamma


	pold, uold = p, u
	t += dt
	count += 1


# Plot displacement
# plot(u)
# interactive()


