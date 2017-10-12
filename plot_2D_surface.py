
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


event = 'EQ' # syn = synthetic event, data = real EQ, sub = subduction, sub_EQ
loop = 'yes'
variable = 'sigma' #sigma or kappa
##########################################################################
############################ LOAD VARIABLES ##############################
##########################################################################

resultspath = 'results/numpy_results/'
variablespath = 'results/numpy_variables/'

x_all = np.load(variablespath+"x_all_2D.npy")
y_all = np.load(variablespath+"y_all_2D.npy")
surface_dofs = np.load(variablespath+"surface_dofs_2D.npy")
ocean_dofs = np.load(variablespath+"ocean_dofs_2D.npy")
size_p = np.load(variablespath+"size_p_CR_2D.npy")



if event == 'EQ':
    if loop == 'yes':
        if variable == 'kappa':
            #sea_flux = np.load(resultspath+"sea_flux_total_2D_EQ.npy")
            Sol_surfk10 = np.load(resultspath+"Sol_surf_2D_EQ_loop_k10.npy")
            Sol_surf = np.load(resultspath+"Sol_surf_2D_EQ_loop_k11.npy")
            Sol_surfk12 = np.load(resultspath+"Sol_surf_2D_EQ_loop_k12.npy")
            Sol_surfk13 = np.load(resultspath+"Sol_surf_2D_EQ_loop_k13.npy")
            #dtt = np.load(variablespath+"dtt_comsum_EQ.npy")

            seaflux_k10 = np.load(resultspath+"sea_flux_total_2D_EQ_loop_k10.npy")
            seaflux_k11 = np.load(resultspath+"sea_flux_total_2D_EQ_loop_k11.npy")
            seaflux_k12 = np.load(resultspath+"sea_flux_total_2D_EQ_loop_k12.npy")
            seaflux_k13 = np.load(resultspath+"sea_flux_total_2D_EQ_loop_k13.npy")

        if variable == 'sigma':
            # sea_flux = np.load(resultspath+"sea_flux_total_2D_EQ.npy")
            Sol_surf10 = np.load(resultspath+"Sol_surf_2D_EQ_loop_sig10.npy")
            Sol_surf = np.load(resultspath+"Sol_surf_2D_EQ_loop_sig15.npy")
            Sol_surf20 = np.load(resultspath+"Sol_surf_2D_EQ_loop_sig20.npy")
            Sol_surf25 = np.load(resultspath+"Sol_surf_2D_EQ_loop_sig25.npy")
            #dtt = np.load(variablespath+"dtt_comsum_EQ.npy")

            seaflux_10 = np.load(resultspath+"sea_flux_total_2D_EQ_loop_sig10.npy")
            seaflux_15 = np.load(resultspath+"sea_flux_total_2D_EQ_loop_sig15.npy")
            seaflux_20 = np.load(resultspath+"sea_flux_total_2D_EQ_loop_sig20.npy")
            seaflux_25 = np.load(resultspath+"sea_flux_total_2D_EQ_loop_sig25.npy")

    else:
        sea_flux = np.load(resultspath+"sea_flux_total_2D_EQ.npy")
        Sol_surf = np.load(resultspath+"Sol_surf_2D_EQ.npy")
        sub_cycle = np.load(variablespath+"sub_cycle_EQ.npy")
        dtt = np.load(variablespath+"dtt_comsum_EQ.npy")

    dtt = [15, 240, 288e1, 432e2]  # in minutes
    ntime = [96, 120, 15, 122]
    dtt = [i * 60 for i in dtt]  # in seconds
    dtt_repeated = np.repeat(dtt, ntime)
    dtt = np.cumsum(dtt_repeated)

elif event == 'SSE':
    sea_flux = np.load(resultspath+"sea_flux_2D_SSE.npy")
    Sol_surf = np.load(resultspath+"Sol_surf_2D_SSE.npy")
    dtt = np.load(variablespath+"dtt_comsum_SSE.npy")

elif event == 'sub':
    sea_flux = np.load(resultspath+"sea_flux_2D_sub.npy")
    Sol_surf = np.load(resultspath+"Sol_surf_2D_sub.npy")
    sub_cycle = np.load(variablespath+"sub_cycle_sub.npy")
    dtt = np.load(variablespath+"dtt_comsum_sub.npy")

elif event == 'sub_EQ':
    sea_flux = np.load(resultspath+"sea_flux_2D_sub_EQ.npy")
    Sol_surf = np.load(resultspath+"Sol_surf_2D_sub_EQ.npy")
    sub_cycle = np.load(variablespath+"sub_cycle_sub_EQ.npy")
    dtt = np.load(variablespath+"dtt_comsum_sub_EQ.npy")

elif event == 'topo':
    sea_flux = np.load(resultspath+"sea_flux_2D_sub_EQ.npy")
    sea_flux_topo = np.load(resultspath+"sea_flux_2D_topo.npy")
    Sol_surf = np.load(resultspath+"Sol_surf_2D_topo.npy")
    #sub_cycle = 50
    sub_cycle = np.load(variablespath+"sub_cycle_sub_EQ.npy")
    dtt = np.load(variablespath+"dtt_comsum_topo.npy")
    dtt_subEQ = np.load(variablespath+"dtt_comsum_sub_EQ.npy")



##########################################################################
###################### PARSE AND SORT VARIABLES ##########################
##########################################################################

p_ind = np.argmax(surface_dofs >= size_p)
size_u_surf = x_all[surface_dofs].shape[0]
size_ux = (size_u_surf - p_ind) / 2
ux_ind = p_ind + size_ux


surface_dof_p = surface_dofs[0:p_ind][np.argsort(x_all[surface_dofs[0:p_ind]])]
surface_dof_ux = surface_dofs[p_ind:ux_ind][np.argsort(x_all[surface_dofs[p_ind:ux_ind]])]
surface_dof_uy = surface_dofs[ux_ind::][np.argsort(x_all[surface_dofs[ux_ind::]])]
sort_p, sort_u = np.argsort(x_all[surface_dofs[0:p_ind]]), np.argsort(x_all[surface_dofs[ux_ind::]])


x_surf_p = 1e-3*x_all[surface_dof_p]
x_surf_ux = 1e-3*x_all[surface_dof_ux]
x_surf_uy = 1e-3*x_all[surface_dof_uy]

y_surf_p = 1e-3*y_all[surface_dof_p]
y_surf_ux = 1e-3*y_all[surface_dof_ux]
y_surf_uy = 1e-3*y_all[surface_dof_uy]

if loop == 'yes':
    if variable == 'kappa':
        Sol_surf_pk10 = Sol_surfk10[0:p_ind, :][sort_p, :] * (1e3 / (9.81))
        Sol_surf_pk11 = Sol_surf[0:p_ind, :][sort_p, :] * (1e3 / (9.81))
        Sol_surf_pk12 = Sol_surfk12[0:p_ind, :][sort_p, :] * (1e3 / (9.81))
        Sol_surf_pk13 = Sol_surfk13[0:p_ind, :][sort_p, :] * (1e3 / (9.81))

        Sol_surf_xk10 = Sol_surfk10[p_ind:ux_ind, :][sort_u, :] * 1e2
        Sol_surf_xk11 = Sol_surf[p_ind:ux_ind, :][sort_u, :] * 1e2
        Sol_surf_xk12 = Sol_surfk12[p_ind:ux_ind, :][sort_u, :] * 1e2
        Sol_surf_xk13 = Sol_surfk13[p_ind:ux_ind, :][sort_u, :] * 1e2

        Sol_surf_yk10 = Sol_surfk10[ux_ind::, :][sort_u, :] * 1e2
        Sol_surf_yk11 = Sol_surf[ux_ind::, :][sort_u, :] * 1e2
        Sol_surf_yk12 = Sol_surfk12[ux_ind::, :][sort_u, :] * 1e2
        Sol_surf_yk13 = Sol_surfk13[ux_ind::, :][sort_u, :] * 1e2



    if variable == 'sigma':
        Sol_surf_p10 = Sol_surf10[0:p_ind, :][sort_p, :] * (1e3 / (9.81))
        Sol_surf_p15 = Sol_surf[0:p_ind, :][sort_p, :] * (1e3 / (9.81))
        Sol_surf_p20 = Sol_surf20[0:p_ind, :][sort_p, :] * (1e3 / (9.81))
        Sol_surf_p25 = Sol_surf25[0:p_ind, :][sort_p, :] * (1e3 / (9.81))

        Sol_surf_x10 = Sol_surf10[p_ind:ux_ind, :][sort_u, :] * 1e2
        Sol_surf_x15 = Sol_surf[p_ind:ux_ind, :][sort_u, :] * 1e2
        Sol_surf_x20 = Sol_surf20[p_ind:ux_ind, :][sort_u, :] * 1e2
        Sol_surf_x25 = Sol_surf25[p_ind:ux_ind, :][sort_u, :] * 1e2

        Sol_surf_y10 = Sol_surf10[ux_ind::, :][sort_u, :] * 1e2
        Sol_surf_y15 = Sol_surf[ux_ind::, :][sort_u, :] * 1e2
        Sol_surf_y20 = Sol_surf20[ux_ind::, :][sort_u, :] * 1e2
        Sol_surf_y25 = Sol_surf25[ux_ind::, :][sort_u, :] * 1e2

else:

    Sol_surf_p = Sol_surf[0:p_ind, :][sort_p, :] * (1e3 / (9.81))
    Sol_surf_x = Sol_surf[p_ind:ux_ind, :][sort_u, :]
    Sol_surf_y = Sol_surf[ux_ind::, :][sort_u, :]

nsteps = dtt.size
y_surf_zero = y_surf_ux - y_surf_ux


##########################################################################
######################### CONVERT UNITS ##################################
##########################################################################

if event == 'topo':
    dt_minus = np.concatenate([[0],  dtt_subEQ[0:-1]])
    dt = dtt_subEQ - dt_minus


    m3_h20 = 4e4 * sea_flux * dt
    m3_h20_topo = 4e4 * sea_flux_topo[sub_cycle] * dt

    #Cumulative cubic kilometers SGD
    cum_h20_topo = np.cumsum(m3_h20_topo)*1e-9
    cum_h20_sub = np.cumsum(m3_h20)*1e-9
    cum_h20_both = cum_h20_sub + cum_h20_topo
    
    # month 1 - [216] - 21 days
    # month 2 - [231] - 30 days
    # month 3 - [232]
    # month 4 - [233]
    # month 5 - [234]
    # month 6- [235]

    nummonths = 24
    EQmonthsadd = np.arange(281,281 + (nummonths-1),1)
    EQmonths = np.concatenate([[sub_cycle-1,sub_cycle,266.0], EQmonthsadd]) # indice intervals for months following the EQ
    ratio = np.empty((EQmonths.shape[0]-1))
    #percentage = np.empty((EQmonths.shape[0]-1))
    #dt_percent = np.empty((EQmonths.shape[0]-1))

    for i in range(0,EQmonths.shape[0]-1):
        sub_water = np.sum(m3_h20[EQmonths[i]:EQmonths[i+1]])
        topo_water = np.sum(m3_h20_topo[EQmonths[i]:EQmonths[i+1]])
        #dt_percent[i] = np.sum(dt[EQmonths[i]:EQmonths[i+1]])/(3600*24*30)
        ratio[i] = sub_water/topo_water
        #percentage[i] = 100*sub_water/(sub_water+topo_water)

    percentage = (sea_flux*dt)/(sea_flux*dt+sea_flux_topo[sub_cycle]*dt)

    total_h20_topo = np.sum(m3_h20_topo)
    total_h20_b4 = np.sum(m3_h20[0:sub_cycle])
    total_h20_EQ = np.sum(m3_h20[sub_cycle::1])

    sixmo_h20_topo = np.sum(m3_h20_topo[sub_cycle:sub_cycle+1])/2
    sixmo_h20_EQ = np.sum(m3_h20[sub_cycle:285])

    print 'cubic meters expelled from topographic flow', total_h20_topo
    print 'cubic meters expelled through seafloor before EQ', total_h20_b4
    print 'cubic meters expelled through after EQ', total_h20_EQ
    print '#####################################################'
    print 'cubic meters expelled from topographic flow - 6 months', sixmo_h20_topo
    print 'cubic meters expelled through seafloor in 6 months after EQ', sixmo_h20_EQ
    print 'EQ/topo in 6 months after EQ', sixmo_h20_EQ/sixmo_h20_topo
    

    dtt_subEQ = dtt_subEQ/(3600*24*365) - 50

# convert dtt from seconds to days
dtt = dtt/(3600*24)
if event == 'sub_EQ':
    x_elastic = np.array([dtt[sub_cycle],dtt[-1]]) #timesteps to plot elastic solution
else:
    x_elastic = np.array([dtt[0],dtt[-1]])



##########################################################################
######################### PLOTTING FUNCTIONS #############################
##########################################################################
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
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
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

cmap = mpl.cm.winter

def plot_timeseries(points):

    for i in range(0,points.shape[0]):
        
        idx = np.abs(x_surf[:] - points[i]).argmin()
        fname = points[i]/1e3
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.set_title('Modeled poroelastic timeseries %3d km from trench' % fname)

        ax1.plot(dtt, Sol_surf_p[idx, :], '-k.', label='constant k',ms=10, mew=1.5, fillstyle='none') #pressure
        
        ax2.plot(dtt, Sol_surf_x[idx, :], '-r.', ms=10, mew=1.5, fillstyle='none') # u_x
        #ax2.plot(x_elastic, elastic_results_x[idx, :],'ro', ms = 13, fillstyle='full', label = 'elastic solution') # u_x

        ax3.plot(dtt, Sol_surf_y[idx, :], '-g.', ms=10, mew=1.5, fillstyle='none') # u_y
        #ax3.plot(x_elastic, elastic_results_y[idx, :],'go', ms=13, fillstyle='full', label = 'elastic solution') # u_x
        

        #plt.ylabel('Depth (km)')
        ax1.set_ylabel('meters of head change')
        ax2.set_ylabel('Displacement (cm)')
        ax3.set_ylabel('Displacement (cm)')
        plt.xlabel('Days after EQ')
        """
        legend1 = ax1.legend()
        for label in legend1.get_texts():
            label.set_fontsize('medium')
        legend2 = ax2.legend()
        for label in legend2.get_texts():
            label.set_fontsize('medium')
        legend3 = ax3.legend()
        for label in legend3.get_texts():
            label.set_fontsize('medium')
        """

        fig.savefig('figures/all_k_compare.png', dpi = 400)

def plot_timeseries_head(points):
    
    x_elastic = np.array([0,dtt[endtime-1]])
      
    for i in range(0,points.shape[0]):
        
        idx_p = np.abs(x_surf_p[:] - points[i]).argmin()
        idx_u = np.abs(x_surf_ux[:] - points[i]).argmin()

        print idx_p, idx_u

        fname = points[i]/1e3
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        #ax2 = fig.add_subplot(312)
        #ax3 = fig.add_subplot(313)
        #ax1.set_title('Modeled poroelastic timeseries %3d km from trench' % fname)

        ax1.plot(dtt[0:endtime], Sol_surf_pk10[idx_p, 0:endtime], 'red', label='log(k) = -10', linewidth = 5)
        ax1.plot(dtt[0:endtime], Sol_surf_pk11[idx_p, 0:endtime], 'k', label='log(k) = -11', linewidth = 5)
        ax1.plot(dtt[0:endtime], Sol_surf_pk12[idx_p, 0:endtime], 'lightblue', label='log(k) = -12', linewidth = 5)
        ax1.plot(dtt[0:endtime], Sol_surf_pk13[idx_p, 0:endtime], 'b', label='log(k) = -13', linewidth = 5)

        # ax2.plot(dtt[0:endtime], Sol_surf_xk10[idx_u, 0:endtime], 'red', label = 'log(k) = -10',
        #          linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none', color = 'pink') #pressure
        # ax2.plot(dtt[0:endtime], Sol_surf_xk11[idx_u, 0:endtime], 'k', label = 'log(k) = -11',
        #          linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none') #pressure
        # ax2.plot(dtt[0:endtime], Sol_surf_xk12[idx_u, 0:endtime], 'lightblue', label = 'log(k) = -12',
        #          linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none', color = 'lightblue') #pressure
        # ax2.plot(dtt[0:endtime], Sol_surf_xk13[idx_u, 0:endtime], 'b', label = 'log(k) = -13',
        #          linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none') #pressure
        #
        # ax3.plot(dtt[0:endtime], Sol_surf_yk10[idx_u, 0:endtime], 'red', label = 'log(k) = -10',
        #          linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none', color = 'pink') #pressure
        # ax3.plot(dtt[0:endtime], Sol_surf_yk11[idx_u, 0:endtime], 'k', label = 'log(k) = -11',
        #          linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none') #pressure
        # ax3.plot(dtt[0:endtime], Sol_surf_yk12[idx_u, 0:endtime], 'lightblue', label = 'log(k) = -12',
        #          linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none', color = 'lightblue') #pressure
        # ax3.plot(dtt[0:endtime], Sol_surf_yk13[idx_u, 0:endtime], 'b', label = 'log(k) = -13',
        #          linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none') #pressure

        #plt.ylabel('Depth (km)')
        #ax1.set_ylabel('meters of head change')
        #ax2.set_ylabel('X Displacement (cm)')
        #ax3.set_ylabel('Uplift (cm)')
        #plt.xlabel('Days after EQ')
        #ax1.legend()

        #ax1.set_xlim((0,80))
        #ax2.set_xlim((0, 80))

        #fig.set_size_inches(8, 10)
        plt.xlim((0, 11000))
        #plt.ylim((5e-8, 10))

        # fig1.savefig('figures/SGD_sea_flux.png', dpi = 400)
        # fig3.savefig('figures/SGD_percent.png', dpi=400)#

        #legend1 = ax1.legend()
        #for label in legend1.get_texts():
        #    label.set_fontsize('large')
        # legend2 = ax2.legend()
        # for label in legend2.get_texts():
        #     label.set_fontsize('medium')
        # legend3 = ax3.legend()
        # for label in legend3.get_texts():
        #     label.set_fontsize('medium')

        fig.savefig('figures/all_k_well_head_3yr_%s.png' % str(points[i])[:2], dpi = 400)

def plot_surface_loop():
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    # ax3 = fig.add_subplot(313)
    # ax1.set_title('Modeled poroelastic timeseries %3d km from trench' % fname)

    # ax1.plot(dtt[0:endtime], Sol_surf_p9[idx, 0:endtime], '-r.', label = 'log(k) = -9',ms=10, mew=1.5, fillstyle='none', color = 'red') #pressure
    ax1.plot(x_surf_p, Sol_surf_p10[:, 0], '#F7941D', label = 'sigma = 10',
             linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none', color = 'pink') #pressure
    ax1.plot(x_surf_p, Sol_surf_p15[:, 0], '#BF1E2E', label = 'sigma = 15',
             linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none') #pressure
    ax1.plot(x_surf_p, Sol_surf_p20[:, 0], '#0F76BB', label = 'sigma = 20',
             linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none', color = 'lightblue') #pressure
    ax1.plot(x_surf_p, Sol_surf_p25[:, 0], '#00A54F', label = 'sigma = 25',
             linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none') #pressure


    ax1.set_yticks(ax1.get_yticks()[::2])
    ax1.set_xlim((60.5, 250))


    ax2.plot(x_surf_uy,Sol_surf_y10[:, 0], '#F7941D', label = 'sigma = 10',
             linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none', color = 'pink') #pressure
    ax2.plot(x_surf_uy, Sol_surf_y15[:, 0], '#BF1E2E', label = 'sigma = 15',
             linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none') #pressure
    ax2.plot(x_surf_uy, Sol_surf_y20[:, 0], '#0F76BB', label = 'sigma = 20',
             linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none', color = 'lightblue') #pressure
    ax2.plot(x_surf_uy, Sol_surf_y25[:, 0], '#00A54F', label = 'sigma = 25',
             linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none') #pressure
    ax2.plot(gps_X, gps_Uz, 'ok', label ='GPS Data')

    ax2.set_xlim((60.5, 250))


    ax3.plot(x_surf_ux,Sol_surf_x10[:, 0], '#F7941D', label = 'sig = 10',
             linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none', color = 'pink') #pressure
    ax3.plot(x_surf_ux, Sol_surf_x15[:, 0], '#BF1E2E', label = 'sig = 15',
             linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none') #pressure
    ax3.plot(x_surf_ux, Sol_surf_x20[:, 0], '#0F76BB', label = 'sig = 20',
             linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none', color = 'lightblue') #pressure
    ax3.plot(x_surf_ux, Sol_surf_x25[:, 0], '#00A54F', label = 'sig = 25',
             linewidth = 3)  # ,ms=1, mew=1.5, fillstyle='none') #pressure
    ax3.plot(gps_X, gps_Ux, 'ok', label='GPS Data')


    # plt.ylabel('Depth (km)')
    ax1.set_ylabel('meters of head change')
    # ax2.set_ylabel('Displacement (cm)')
    # ax3.set_ylabel('Displacement (cm)')
    plt.xlabel('Meters from trench')
    # ax1.legend()

    #fig.set_size_inches(8, 5)
    plt.xlim((60.5, 250))
    # plt.ylim((5e-8, 10))


    ax1.set_ylabel('head change (m)')
    ax2.set_ylabel('Uplift (cm)')
    ax3.set_ylabel('X disp. (cm)')
    #ax2.set_xlabel('Meters from trench')
        
    #legend1 = ax1.legend()
    #for label in legend1.get_texts():
    #    label.set_fontsize('medium')
    legend2 = ax2.legend(prop={'size':8})
    for label in legend2.get_texts():
        label.set_fontsize('small')

    fig.savefig('figures/sig_loop_surface.png' , dpi = 400)

def plot_surface_e():
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.set_title('Elastic Solution')
    
    scaleplot = 2.5e3
    
    Ux, Uy  = e_undrained_x, e_undrained_y
    x_surf_0, y_surf_0 = x_surf + scaleplot*Ux, Uy
    
    Ux_post, Uy_post = e_drained_x - e_undrained_x, e_drained_y - e_undrained_y
    x_surf_post, y_surf_post = x_surf_0 + scaleplot*Ux_post,  y_surf_0 + Uy_post
    
    ax1.plot(x_surf_0, y_surf_0,'-r*', ms=2, mew=2, fillstyle='none', color = 'black', label = 'pre-seismic land surface')
    
    ax1.plot(x_surf, y_surf_zero,'-k*', ms=2, mew=2, fillstyle='none', color='red', label = 'co-seismic land surface')
    
    ax1.quiver(x_surf, y_surf_zero, scaleplot*Ux, Uy, scale_units='xy', angles='xy', scale=1, width = 2e-3)
    
    ax2.plot(x_surf_0, y_surf_0,'-r*', ms=2, mew=2, fillstyle='none', color='red', label = 'undrained co-seimic solution')
    
    ax2.plot(x_surf_post, y_surf_post,'-r*', ms=2, mew=2, fillstyle='none', color = 'magenta', label = 'drained co-seimic solution')
    
    ax2.quiver(x_surf_0, y_surf_0, scaleplot*Ux_post, Uy_post, scale_units='xy', angles='xy', scale=1, width = 1.5e-3)
    
    ax1.set_ylabel('Displacement (cm)')
    ax2.set_ylabel('Displacement (cm)')
    ax2.set_xlabel('Meters from trench')
    plt.xlabel('Meters from trench')
    legend1 = ax1.legend()
    for label in legend1.get_texts():
        label.set_fontsize('medium')
    legend2 = ax2.legend()
    for label in legend2.get_texts():
        label.set_fontsize('medium')

def plot_diff_surf(): #Plot the difference between the elastic and poroelastic solutions

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(223)
    #ax1.set_title('Difference between elastic and poroelastic solution')
    ax1.set_title('Co-seismic solution')
    ax2.set_title('Post-seismic solution')
    scaleplot = 2.5e3
    
    Ux_p, Uy_p  = Sol_surf_x[:,0], Sol_surf_y[:,0]
    Ux_e, Uy_e  = e_undrained_x, e_undrained_y
    x_surf_p0, y_surf_p0 = x_surf + Ux_p, Uy_p
    x_surf_e0, y_surf_e0 = x_surf + Ux_e, Uy_e
    Ux_diff0, Uy_diff0 = x_surf + scaleplot*(Ux_p - Ux_e), (Uy_p - Uy_e)
    Ux_surf_diff0, Uy_surf_diff0 = (Ux_p - Ux_e), (Uy_p - Uy_e)
    
    Ux_p_post, Uy_p_post  = Sol_surf_x[:,-1], Sol_surf_y[:,-1]
    Ux_e_post, Uy_e_post  = e_drained_x, e_drained_y
    x_surf_ppost, y_surf_ppost = x_surf_p0 + scaleplot*Ux_p_post, y_surf_p0 + Uy_p_post
    x_surf_epost, y_surf_epost = x_surf_e0 + scaleplot*Ux_e_post, y_surf_e0 + Uy_e_post
    Ux_diff_post, Uy_diff_post = x_surf_e0 + (Ux_p_post - Ux_e_post), (Uy_p_post - Uy_e_post)
    Ux_surf_diff_post, Uy_surf_diff_post = (Ux_p_post - Ux_e_post), (Uy_p_post - Uy_e_post)

    ax1.plot(x_surf_e0, y_surf_zero,'-r*', ms=2, mew=2, fillstyle='none', color='red', label = 'elastic co-seismic')
    ax1.plot(Ux_diff0, Uy_diff0,'-k*', ms=2, mew=2, fillstyle='none', color='blue', label = 'poroelastic co-seismic')
    ax1.quiver(x_surf_e0, y_surf_zero, scaleplot*Ux_surf_diff0, Uy_diff0, scale_units='xy', angles='xy', scale=1, width = 2e-3)

    ax2.plot(x_surf_epost, y_surf_zero,'-r*', ms=2, mew=2, fillstyle='none', color='magenta', label = 'elastic post-seismic')
    ax2.plot(Ux_diff_post, Uy_diff_post,'-k*', ms=2, mew=2, fillstyle='none', color='blueviolet', label = 'poroelastic post-seismic')
    ax2.quiver(x_surf_epost, y_surf_zero, scaleplot*Ux_surf_diff_post, Uy_diff_post, scale_units='xy', angles='xy', scale=1, width = 2e-3)
    
    plt.xlabel('Meters from trench')
    legend1 = ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    for label in legend1.get_texts():
        label.set_fontsize('medium')
    legend2 = ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    for label in legend2.get_texts():
        label.set_fontsize('medium')

def plot_sea_flux():
    
    if event == 'sub_EQ' :
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        #ax2 = fig.add_subplot(312)
        #ax3 = fig.add_subplot(313)
        #ax1.set_title('Flux through seafloor')
        
        # Co-seismic displacement
        ax1.plot(dtt, sea_flux, '-bo', ms=5, mew=1, fillstyle='none', color = 'blue')
        #ax1.plot(dtt[0:50], sea_flux[0:50], '-b*', ms=2, mew=2, fillstyle='none', color = 'blue')
        ax1.plot(dtt[50:-1], sea_flux[50:-1], '-b*', ms=2, mew=2, fillstyle='none', color = 'red')
        #ax2.plot(dtt[0:50], sea_flux[0:50], '-b*', ms=2, mew=2, fillstyle='none', color = 'blue')
        #ax3.plot(dtt[50:-1], sea_flux[50:-1], '-b*', ms=2, mew=2, fillstyle='none', color = 'red')
        #ax1.set_ylabel('Darcy flux (meters/year)')
        #ax2.set_ylabel('Darcy flux (meters/sec)')
        #ax3.set_ylabel('Darcy flux (meters/year)')
        #ax1.set_xlabel('Years')

        ax1.set_yscale('log')
        #ax3.set_yscale('log')
        
    elif event == 'topo':
        ind = np.arange(7)  # the x locations for the groups
        #width = np.concatenate([[1],dt_percent[1::]])
        #months = np.concatenate([[-1],(np.cumsum(dt_percent[0:-1]) - np.sum(dt_percent[0:1]))])
        sixmo = 287
        twoyear = 303

        

        fig1 = plt.figure()
        fig2 = plt.figure()
        fig3 = plt.figure()
        ax1 = fig1.add_subplot(211)
        ax2 = fig1.add_subplot(212)
        ax3 = fig2.add_subplot(111)
        #ax4 = fig2.add_subplot(212)
        ax5 = fig3.add_subplot(111)

   

        ax1.plot(dtt, (3600*24*365)*sea_flux_topo, '-b', label='Topographic flow',
                 ms = 2, mew = 2, fillstyle = 'none', color = 'blue',linewidth=2.0)
        ax1.plot(dtt_subEQ, (3600*24*365)*sea_flux,  '-r', label='Tectonic driven flow',
                 ms=2, mew=2, fillstyle='none', color = 'red',linewidth=2.0)
        ax1.set_xlim([-50, 50])
        ax1.set_yscale('log')
        ax1.legend(fontsize = 'small')

        ax2.plot(dtt[sub_cycle-2:sub_cycle+2], (3600*24*365)*sea_flux_topo[sub_cycle-2:sub_cycle+2],
                 '-b*', label='Topographic flow', ms = 2, mew = 2,
                 fillstyle = 'none', color = 'blue',linewidth=2.0)
        ax2.plot(dtt_subEQ[sub_cycle-2:twoyear], (3600*24*365)*sea_flux[sub_cycle-2:twoyear],
                 '-b*', label='Tectonic driven flow', ms=2, mew=2,
                 fillstyle='none', color = 'red',linewidth=2.0)
        ax2.set_xlim([-.1, 2])
        ax2.set_xlabel('Years after earthquake')
        ax2.set_yscale('log')
        ax2.legend(fontsize = 'small')


        ax3.plot(dtt_subEQ, cum_h20_sub, '-bo', label='Tectonic flow',
                 ms=2, mew=2, fillstyle='none', color = 'red',linewidth=2.0)
        ax3.plot(dtt_subEQ, cum_h20_topo, '-bo', label='Topographic flow',
                 ms=2, mew=2, fillstyle='none', color = 'blue',linewidth=2.0)
        ax3.plot(dtt_subEQ, cum_h20_both, '-bo', label='Total flow',
                 ms=2, mew=2, fillstyle='none', color = 'purple',linewidth=2.0)
        ax3.set_xlim([-50, 50])
        #ax3.set_xlabel('Years after earthquake')
        #ax3.set_ylabel('Cumulative SGD')
        ax3.legend(loc=2)

        print dtt.shape
        print percentage
   
        ax5.plot(dtt_subEQ,100*percentage,'-b.', color = 'blue', linewidth = 3)
        ax5.set_xlim([-.2, 2])
        ax5.set_ylim([0, 100])
        #ax5.set_xlabel('Months after earthquake')
        #ax5.set_ylabel('Percent of SGD driven by tectonic processes')

            
        #percentagetext = np.round(percentage,1)
        #for i, v in enumerate(percentagetext[0:2]):
        #    ax5.text(i-.95, v+1 , str(v), color='darkblue', fontsize = '10')
        #for i, v in enumerate(percentagetext[-1::]):
        #    ax5.text(i + nummonths-2.5, v + 1.4, str(v), color='darkblue', fontsize = '10')
       

        
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title('Flux through seafloor')
        # Co-seismic displacement
        ax1.plot(dtt, sea_flux, '-b*', ms=2, mew=2, fillstyle='none', color = 'blue')

    
    
    #fig3.set_size_inches(7, 4)
    #plt.xlim((0, 100))
    #plt.ylim((5e-8, 10))
    #fig2.savefig('figures/SGD_cumulative.png', dpi = 400)
    #fig1.savefig('figures/SGD_sea_flux.png', dpi = 400)
    fig3.savefig('figures/SGD_percent.png', dpi=400)#

def plot_surf_post():
    
    fig_x = plt.figure()
    ax1 = fig_x.add_subplot(111, projection='3d')
    fig_y = plt.figure()
    ax2 = fig_y.add_subplot(111, projection='3d')
    fig_p = plt.figure()
    ax3 = fig_p.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(dtt, x_surf)
    Z_x = Sol_surf_x_post
    Z_y = Sol_surf_y_post
    Z_p = Sol_surf_p_post



    surfx = ax1.plot_surface(X, Y, Z_x, cmap=cm.coolwarm,linewidth=0, antialiased=False)

    surfy = ax2.plot_surface(X, Y, Z_y, cmap=cm.coolwarm,linewidth=0, antialiased=False)

    surfp = ax3.plot_surface(X, Y, Z_p, cmap=cm.coolwarm,linewidth=0, antialiased=False)


    
    #ax1.zaxis.set_major_locator(LinearLocator(10))
    #ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #plt.title('Surf X')
    
    #plt.xlabel('Meters from trench')
    #plt.ylabel('Meters from trench')

    fig_x.colorbar(surfx)
    fig_y.colorbar(surfy)
    fig_p.colorbar(surfp)

def plot_surf_all():
    
    fig_x = plt.figure()
    ax1 = fig_x.add_subplot(111, projection='3d')
    fig_y = plt.figure()
    ax2 = fig_y.add_subplot(111, projection='3d')
    fig_p = plt.figure()
    ax3 = fig_p.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(dtt, x_surf)
    Z_x = Sol_surf_x
    Z_y = Sol_surf_y
    Z_p = Sol_surf_p
    
    
    
    surfx = ax1.plot_surface(X, Y, Z_x, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    surfy = ax2.plot_surface(X, Y, Z_y, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    surfp = ax3.plot_surface(X, Y, Z_p, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    
    
    #ax1.zaxis.set_major_locator(LinearLocator(10))
    #ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #plt.title('Surf X')
    
    #plt.xlabel('Meters from trench')
    #plt.ylabel('Meters from trench')
    
    fig_x.colorbar(surfx)
    fig_y.colorbar(surfy)
    fig_p.colorbar(surfp)

def plot_surf_early():
    
    fig_x = plt.figure()
    ax1 = fig_x.add_subplot(111, projection='3d')
    fig_y = plt.figure()
    ax2 = fig_y.add_subplot(111, projection='3d')
    fig_p = plt.figure()
    ax3 = fig_p.add_subplot(111, projection='3d')
    
    
    
    X, Y = np.meshgrid(dtt[0:endtime], x_surf)
    Z_x = Sol_surf_x_post[:, 0:endtime]
    Z_y = Sol_surf_y_post[:, 0:endtime]
    Z_p = Sol_surf_p_post[:, 0:endtime]
    
    
    
    surfx = ax1.plot_surface(X, Y, Z_x, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    surfy = ax2.plot_surface(X, Y, Z_y, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    surfp = ax3.plot_surface(X, Y, Z_p, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    
    
    #ax1.zaxis.set_major_locator(LinearLocator(10))
    #ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #plt.title('Surf X')
    
    #plt.xlabel('Meters from trench')
    #plt.ylabel('Meters from trench')
    
    fig_x.colorbar(surfx)
    fig_y.colorbar(surfy)
    fig_p.colorbar(surfp)

def plot_contour_post():
    
    fig_x = plt.figure()
    ax1 = fig_x.add_subplot(111)
    fig_y = plt.figure()
    ax2 = fig_y.add_subplot(111)
    fig_p = plt.figure()
    ax3 = fig_p.add_subplot(111)
    
    X, Y = np.meshgrid(dtt, x_surf)
    Z_x = Sol_surf_x_post
    Z_y = Sol_surf_y_post
    Z_p = Sol_surf_p_post


    contour_steps=20
    levels_x = np.linspace(Z_x.min(), Z_x.max(), num=contour_steps, endpoint=True)
    levels_y = np.linspace(Z_y.min(), Z_y.max(), num=contour_steps, endpoint=True)
    levels_p = np.linspace(Z_p.min(), Z_p.max(), num=contour_steps, endpoint=True)
    
    cpxf = ax1.contourf(X, Y, Z_x, levels_x, cmap=cm.coolwarm)
    #cpx = ax1.contour(X, Y, Z_x, levels_x, linewidth=2, colors = 'black', linestyles = 'solid')
    
    cpyf = ax2.contourf(X, Y, Z_y, levels_y, cmap=cm.coolwarm)
    #cpy = ax2.contour(X, Y, Z_y, levels_y, linewidth=2, colors = 'black', linestyles = 'solid')
    
    cppf = ax3.contourf(X, Y, Z_p, levels_p, cmap=cm.coolwarm)
    #cpp = ax3.contour(X, Y, Z_p, levels_p, linewidth=2, colors = 'black', linestyles = 'solid')
    
    
    
    #ax1.zaxis.set_major_locator(LinearLocator(10))
    #ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #plt.title('Surf X')
    
    #plt.xlabel('Meters from trench')
    #plt.ylabel('Meters from trench')
    
    fig_x.colorbar(cpxf)
    fig_y.colorbar(cpyf)
    fig_p.colorbar(cppf)

def plot_topo():
    wb_slip = pyxl.load_workbook('water_table.xlsx')
    topo_data = wb_slip['Sheet1']
    X_topo = np.array([[cell.value for cell in col] for col in topo_data['A2':'A21']])
    h_topo = np.array([[cell.value for cell in col] for col in topo_data['B2':'B21']])
    h_topo2 = 0.75 * np.array([[cell.value for cell in col] for col in topo_data['B2':'B21']])

    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Nicoya Peninsula Topography')
    ax1.plot(X_topo,h_topo,'-b*', ms=3, mew=2, fillstyle='none', color = 'green',linewidth = 3)
    ax1.plot(X_topo,h_topo2,'-b*', ms=3, mew=2, fillstyle='none', color = 'blue',linewidth = 3)

    fig.set_size_inches(7, 4)
    #plt.xlim((0, 100))
    #plt.ylim((5e-8, 10))
    fig.savefig('figures/topo_flow.png', dpi = 400)
    #fig1.savefig('figures/SGD_sea_flux.png', dpi = 400)
    #fig3.savefig('figures/SGD_percent.png', dpi=400)#
    
    
##########################################################################
######################### PLOTTING COMMANDS ##############################
##########################################################################


#Array points along the surface to be plotted as timeseries
#points = np.arange(60e3, 150e3, 5e5)
#points = 1e3*np.array([60,65,70,75,80,85,90,95,100,105,110,120,130,140,150 ])
#points = 1e3*np.array([63, 65, 80, 95, 110, 130])
points = 1e3*np.array([65, 95, 130])
endtime = 353

#plot_timeseries(points)
#plot_timeseries_head(points)
plot_surface_loop()
#plot_surface_p()
#plot_surface_e()
#plot_diff_surf()
#plot_sea_flux()
#plot_topo()


#plot_surf_post()
#plot_surf_all()
#plot_surf_early()
#plot_contour_post()











