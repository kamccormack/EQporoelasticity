#Calculate charateristic timescales of diffusion


lenth1, length2, length_lower = 20e3, 100e3, 40e3

kappa_upper, kappa_lower = 1e-12, 1e-18

mu = 1e-9

S_upper, S_lower = 4.546e-6, 1.32e-5

D_upper, D_lower = kappa_upper/(mu*S_upper), kappa_lower/(mu*S_lower)

print D_upper, D_lower

t_u_short = lenth1*lenth1/D_upper
t_u_long = length2*length2/D_upper
t_lower = length_lower*length_lower/D_lower

print "characteristic timescale 1: ", t_u_short/(3600*24), " days"
print "characteristic timescale 2: ", t_u_long/(3600*24*365.25), " years"
print "characteristic timescale 3: ", t_lower/(3600*24*365.25), " years"