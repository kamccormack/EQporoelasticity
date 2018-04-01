import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.interpolate import griddata
from tempfile import TemporaryFile
import openpyxl as pyxl



wb = pyxl.load_workbook('well_data.xlsx')
perm = wb['Sheet4']
k = np.array([[cell.value for cell in col] for col in perm['A2':'G34']])



for i in range(0,6):
    plt.figure(1)
    plt.plot( k[:,i+1],k[:,0],linewidth=2) #pressure
    #sol_plotgps[0].plot(Sol_gps_k[idx_gps, :], '--k', linewidth=3, fillstyle='none') #pressure

plt.gca().invert_yaxis()
plt.ylabel('Depth (km)')
plt.xlabel('log(k)')

plt.savefig('k_plot', dpi=500)
plt.show()



"""
    wb_slip = pyxl.load_workbook('magnitude_grid.xlsx')
    slip_grid = wb_slip['Sheet1']
    X_slip = 1e3*np.array([[cell.value for cell in col] for col in slip_grid['A2':'A256']])
    Y_slip = 1e3*np.array([[cell.value for cell in col] for col in slip_grid['B2':'B256']])
    U_X_slip = np.array([[cell.value for cell in col] for col in slip_grid['C2':'C256']])
    U_Y_slip = np.array([[cell.value for cell in col] for col in slip_grid['D2':'D256']])
    
    X_slip = X_slip.reshape(X_slip.shape[0],)
    Y_slip = Y_slip.reshape(Y_slip.shape[0],)
    U_X_slip = U_X_slip.reshape(U_X_slip.shape[0],)
    U_Y_slip = U_Y_slip.reshape(U_Y_slip.shape[0],)
    
    X_slip_fine = np.linspace(0.0, 2.5e5, 250, endpoint=True)
    Y_slip_fine = np.linspace(0.0, 2.5e5, 250, endpoint=True)
    x_grid, y_grid = np.meshgrid(X_slip_fine, Y_slip_fine)
    #slip_data_UX_fine = griddata((X_slip, Y_slip), U_X_slip, (x_bottom, y_bottom), method='linear')
    
    slip_UX = griddata((X_slip, Y_slip), U_X_slip, (x_grid, y_grid), method='cubic')
    slip_UY = griddata((X_slip, Y_slip), U_Y_slip, (x_grid, y_grid), method='cubic')
    slip_mag = griddata((X_slip, Y_slip), np.sqrt(pow(U_X_slip,2) + pow(U_Y_slip,2)), (x_grid, y_grid), method='linear')
    
    
    print slip_UX.shape
    print slip_mag.shape
    
    slip_mag = slip_mag.reshape(X_slip_fine.shape[0], Y_slip_fine.shape[0])
    
    
    levels = np.arange(0, 3.5, .25)
    v = np.arange(0, 3.5, .25)
    v2 = np.arange(.5, 3.5, .5)
    norm = plt.cm.colors.Normalize(0, 3.5)
    cmap = plt.cm.rainbow
    
    fig = plt.figure()
    #cmap=plt.cm.get_cmap(cmap, len(levels) - 1)
    plt.contour(X_slip_fine, Y_slip_fine, slip_mag, v2, linewidths=0.5, colors='k')
    plt.contourf(X_slip_fine, Y_slip_fine, slip_mag,v)
    #plt.contourf(X_slip_fine, Y_slip_fine, slip_mag)
    #plt.clim(8,16)
    CB=plt.colorbar(ticks=v2)
    #fig.set_size_inches(6, 6)
    fname = 'Magnitude.png'
    #plt.savefig(fname, dpi=200)
    
    plt.show()
    
    quit()
    """