# coding: utf-8

import numpy as np
from tqdm import tqdm

#print('Imported SlantStack_v2')

def slant(p,tau,data1,dx,dz,z_ini,x_ini):
    """
    Computes the slant stack value for a given tau-p.

    Parameters:
    -----------
    p : float
        Slope of the line (angular coefficient).
    tau : float
        Linear coefficient of the line.
    data1 : numpy.ndarray
        A 2D array of shape [nz,nx] representing the reflectivity model
        of the subsurface.
    dx : float
        Sampling interval in the x direction.
    dz : float
        Sampling interval in the z direction.
    z_ini : int
        Index of the first sample of the gather in the z direction.
    x_ini : int
        Index of the first sample of the gather in the x direction.

    Returns:
    --------
    s : float
        Coherence value (semblance).
    x_grid : numpy.ndarray 
        Indices of the samples in the x direction.
    z_grid : numpy.ndarray 
        Indices of the samples in the z direction.
        
    Notes:
    ------
    - This function calculates the coherence of a line parameterized by (tau,p) 
    in relation with the reflectivity model data1. 
    
    - It is possible to use (x_grid,y_grid) to plot the defined line over the 
    model and visually check the funcionality of the function.
    
    References:
    -----------
    
    [1] Neidell, N. S. (1997). Perceptions in seismic imaging Part 2: Reflective and 
    diffractive contributions to seismic imaging. The Leading Edge 16:8, 1121-1123
    
    """
    
    [nz1,ntr1]=data1.shape
    dataC = data1.copy()
    x = np.arange(x_ini*dx,(x_ini+ntr1)*dx,dx)
    z = tau + p*x
    z_grid = np.int64(np.round(z/dz)-z_ini)
    x_grid = np.int64((np.round(x/dx))-x_ini)
    s_n = 0
    s_d = 0
    
    plot = "no" #Only change for testing and debugging
    
    for k in range(ntr1):
        
        if (z_grid[k]<nz1) & (z_grid[k]>=0):            
            s_n += dataC[z_grid[k],x_grid[k]]
            s_d += dataC[z_grid[k],x_grid[k]]**2
            
            if plot == "yes":
                dataCC=data1.copy()
                dataCC[z_grid[k],x_grid[k]] = .5
                plt.imshow(dataC, extent=[0, ntr1*dx,nz1*dt, 0])
                plt.imshow(dataCC, extent=[0, ntr1*dx,nz1*dt, 0])
                plt.axis('auto')
                plt.plot(x,z,'r',label="t (x) = tau + p*x \nt (x) = %s + %s*x" % (tau,p))
                plt.legend()
                plt.colorbar()
                plt.show()
    
    if s_d == 0:
        s=1e-16 #avoid nan's
    
    else:
        s = s_n**2/ntr1/s_d

        
    return s, x_grid, z_grid

def slant_stack(pmin,pmax,dp,taumin,taumax,dtau,data1,dx,dt,t_ini,x_ini):
    """
    Computes the slant stack of a seismic dataset for a range of slownesses and
    intercept times.

    Parameters:
    -----------
    pmin, pmax, dp : float
        Minimum, maximum, and step values for angular coefficient (slope of the line).
    taumin, taumax, dtau : float
        Minimum, maximum, and step intercept time values (linear coefficient).
    data1 : numpy.ndarray of shape [nz,nx]
        A 2D array of shape [nz,nx] representing the reflectivity model
        of the subsurface.
    dx, dt : float
        Spatial and temporal sampling intervals.
    t_ini, x_ini : float
        Origin coordinates.
    
    Returns:
    --------
    S : numpy.ndarray of shape (len(tau), len(p))
        Coherence matrix, each cell represents the coherence value for a tau-p pair.
    pmax, taumax : float
        Slope and intercept time values with the maximum slant stack power (highest coherence).
    Smax : float
        Maximum coherence value.
    ip, itau : int
        Indices of the maximum coherence value in the p and tau arrays.
    """
    
    p = np.arange(pmin,pmax,dp)
    tau = np.arange(taumin,taumax,dtau)
    
    # S é o painel que retorna a semblance para cada par (tau,p)
    S = np.zeros([len(tau),len(p)])
    Smax = 0
    
    for ip in (range(len(p))):
        for itau in range(len(tau)):
            S[itau,ip],x,t=slant(p[ip],tau[itau],data1,dx,dt,t_ini,x_ini)
            if S[itau,ip]>Smax:
                Smax = S[itau,ip]
                pmax=p[ip]
                taumax=tau[itau]
                i_pmax=ip
                i_taumax=itau
                
    return S, pmax, taumax, Smax, ip, itau


### NEW FUNCTIONS ####

def slant_local(data1,x_ini,z_ini,dx,dz,x0,z0,p):
    """
    Calculates the slant stack measure at a local location specified by the coordinates x0 and z0
    (image point), for a given slope value p.

    Parameters:
    -----------
    data1 : numpy.ndarray of shape (nz1, ntr1)
        A 2D array of shape [nz,nx] representing the reflectivity model
        of the subsurface.
    x_ini : float
        The x coordinate of the first trace in the seismic data.
    z_ini : float
        The z coordinate of the first sample in the seismic data.
    dx : float
        The x sampling interval in the seismic data.
    dz : float
        The z sampling interval in the seismic data.
    x0 : float
        The x coordinate of the central point of the slant stack line. 
    z0 : float
        The z coordinate of the central point of the slant stack line.
    p : float
        The slowness value used to calculate the slant stack measure.

    Returns:
    --------
    s : float
        Coherence value (semblance).
    x_grid : numpy.ndarray of shape (ntr1,)
        Indices of the samples in the x direction.
    z_grid : numpy.ndarray of shape (ntr1,)
        Indices of the samples in the z direction.
        
    Notes:
    ------
    - This function calculates the coherence of a line of slope defined by `p` 
    in relation with the reflectivity model data1 for a given image point (x0,z0).
    
    - It differs from the `slant` function only in the way each one defines the line.
    In this case, it is not necessary to input the intercept time, only the image point 
    (x0,z0) and the slope (p).
    
    - It is possible to use (x_grid,y_grid) to plot the defined line over the 
    model and visually check the funcionality of the function.
    
    References:
    -----------
    [1] Neidell, N. S. (1997). Perceptions in seismic imaging Part 2: Reflective and 
    diffractive contributions to seismic imaging. The Leading Edge 16:8, 1121-1123
    
    """
    
    [nz1,ntr1]=data1.shape
    dataC = data1.copy()
    
    x = np.arange(x_ini,(x_ini+ntr1),1)
    z = p*(x-x0) + z0
    x_grid = np.int64((np.round(x))-x_ini)
    z_grid = np.int64(np.round(z)-z_ini)
    
    s_n = 0
    s_d = 0
    
    plot = "no" #WARNING! Change only for debugging!
    
    if plot=="yes":
        plt.imshow(dataC, aspect="auto")
        print(f"shape da matriz = {data1.shape}")
        print(f"ponto imagem = [{z0},{x0}]")
        print(f"p = {p}")
        print(f"x_grid = {x_grid}")
        print(f"z_grid = {z_grid}")
    
    for k in range(ntr1):
        if (x_grid[k]<ntr1) and (z_grid[k]>=0 and z_grid[k]<nz1):
            s_n += dataC[z_grid[k],x_grid[k]]
            s_d += dataC[z_grid[k],x_grid[k]]**2
            
            if plot=="yes":
                plt.plot(x_grid[k],z_grid[k],"r.")
    
    if s_d == 0:
        s=1e-16 #avoid nan's
    else:
        s = s_n**2/ntr1/s_d
    
    if plot=="yes":
        plt.show()
        print(f"s = {s}")
    
    return s, x_grid, z_grid


def slant_local_p(data1,x_ini,z_ini,dx,dz,x0,z0,pmin,pmax,dp):
    """
    Calculates the maximum coherence value and its corresponding slope for a given
    data and a range of slopes. Uses the `slant_local` function to calculate the 
    coherence values over a range of slopes.

    Parameters:
    -----------        
    data1 : numpy.ndarray of shape (nz1, ntr1)
        A 2D array of shape [nz,nx] representing the reflectivity model
        of the subsurface.
    x_ini : float
        The x coordinate of the first trace in the seismic data.
    z_ini : float
        The z coordinate of the first sample in the seismic data.
    dx : float
        The x sampling interval in the seismic data.
    dz : float
        The z sampling interval in the seismic data.
    x0 : float
        The x coordinate of the central point of the slant stack line. 
    z0 : float
        The z coordinate of the central point of the slant stack line.
    pmin, pmax : float
        Minimum and maximum values of the slopes to consider.
    dp : float
        Step size for the slope range.

    Returns:
    --------
    Smax : float
        Maximum coherence value obtained for the given slope range.
    pmax : float
        Slope corresponding to the maximum slant stack value.
    """
    
    ps = np.arange(pmin,pmax,dp)
        
    Smax = 0
    pmax = 0
    
    for ip in range(len(ps)):
        S,x,z = slant_local(data1,x_ini,z_ini,dx,dz,x0,z0,ps[ip])
        
        if S>Smax:
            Smax = S
            pmax = ps[ip]
            
    return Smax,pmax


def local_windowS(data1,xwin,zwin,x_ini,z_ini,dx,dz,x0,z0,pmin,pmax,dp):
    
    """
    Computes the slant stack maximum value and corresponding p value for 
    a local window centered at (x_ini, z_ini). 
    
    The window is defined as data1[z_ini:(z_ini+zwin),x_ini:(x_ini+xwin)].
    
    Auxiliary function used by `local_window`.
    
    Parameters:
    -----------
    data1 : numpy.ndarray of shape (nz1, ntr1)
        A 2D array of shape [nz,nx] representing the reflectivity model
        of the subsurface.
    xwin : int
        The width of the window in number of samples along the x-axis.
    zwin : int
        The height of the window in number of samples along the z-axis.
    x_ini : float
        The x coordinate of the first trace in the seismic data.
    z_ini : float
        The z coordinate of the first sample in the seismic data.
    dx : float
        The x sampling interval in the seismic data.
    dz : float
        The z sampling interval in the seismic data.
    x0 : float
        The x coordinate of the central point of the slant stack line. 
    z0 : float
        The z coordinate of the central point of the slant stack line.
    pmin, pmax : float
        Minimum and maximum values of the slopes to consider.
    dp : float
        Step size for the slope range.
            
    Returns:
    --------
    s : float
        The maximum cohernece value.
    pm : float
        Slope corresponding to the maximum slant stack value.
    """
    
    W=data1[z_ini:(z_ini+zwin),x_ini:(x_ini+xwin)] 
    s, pm = slant_local_p(W,x_ini,z_ini,dx,dz,x0,z0,pmin,pmax,dp)

    return s, pm


def local_window(data1,xwin,zwin,x_ini,z_ini,dx,dz,pmin,pmax,dp):
    """
    Compute local slant stack for a given model.
    
    Parameters:
    -----------
    data1 : numpy.ndarray of shape (nz1, ntr1)
        A 2D array of shape [nz,nx] representing the reflectivity model
        of the subsurface.
    xwin : int
        The width of the window in number of samples along the x-axis.
    zwin : int
        The height of the window in number of samples along the z-axis.
    x_ini : float
        The x coordinate of the first trace in the seismic data.
    z_ini : float
        The z coordinate of the first sample in the seismic data.
    dx : float
        The x sampling interval in the seismic data.
    dz : float
        The z sampling interval in the seismic data.
    x0 : float
        The x coordinate of the central point of the slant stack line. 
    z0 : float
        The z coordinate of the central point of the slant stack line.
    pmin, pmax : float
        Minimum and maximum values of the slopes to consider.
    dp : float
        Step size for the slope range.
        
    Returns:
    --------
    p_max : ndarray of shape (ntr1, nz1); matrix
        Maximum slope values computed from local windowed slant stacks.
    s_max : ndarray of shape (ntr1, nz1); matrix
        Maximum correlation coefficients computed from local windowed slant stacks.
        
    Notes:
    ------
    - The `pmax` matrix is used for the calculation of the horizontal and vertical components
    of reflector normal, necessary for the anti-stationary phase filter w(s,x,r) construction. 
    
    """
    
    p = np.arange(pmin,pmax,dp)
    
    [ntr1,nz1] = data1.T.shape
    
    p_max = np.zeros(data1.shape)
    s_max = np.zeros(data1.shape)

    
    for i in tqdm(range(0,nz1)): #eixo t
        for j in range(0,ntr1): #eixo x
            
            #Caso A
            if (i-np.int64(zwin/2))<=0:
                z_ini=0 #t_ini em grid
                zwinA=np.int64(zwin/2)+i

                #Caso A1
                if (j-np.int64(xwin/2))<=0:
                    #print("1")
                    x_ini=0
                    xwinA=np.int64(xwin/2)+j
                    smaxS,pmaxS = local_windowS(data1,xwinA,zwinA,x_ini,z_ini,dx,dz,j,i,pmin,pmax,dp)
                    p_max[i,j]=pmaxS
                    s_max[i,j]=smaxS
                    
                #Caso A2
                elif (j-np.int64(xwin/2))>0 and (j+np.int64(xwin/2))<ntr1:
                    #print("2")
                    x_ini=j-np.int64(xwin/2)
                    xwinA=xwin
                    smaxS,pmaxS = local_windowS(data1,xwinA,zwinA,x_ini,z_ini,dx,dz,j,i,pmin,pmax,dp)
                    p_max[i,j]=pmaxS
                    s_max[i,j]=smaxS
                
                # Caso A3
                elif (j+np.int64(xwin/2))>=ntr1:
                    #print("3")
                    x_ini=j-np.int64(xwin/2)
                    xwinA= xwin - (j + np.int64(xwin/2) - ntr1)
                    smaxS,pmaxS = local_windowS(data1,xwinA,zwinA,x_ini,z_ini,dx,dz,j,i,pmin,pmax,dp)
                    p_max[i,j]=pmaxS
                    s_max[i,j]=smaxS
            
            #Caso B
            elif (i+np.int64(zwin/2))>=nz1:
                z_ini=(i-np.int64(zwin/2))
                zwinA=zwin - (i + np.int64(zwin/2) - nz1)
                tau = np.arange(z_ini*dz, (z_ini+zwinA)*dz,dz)
                #Caso B1
                if (j-np.int64(xwin/2))<=0:
                    #print("1")
                    x_ini=0
                    xwinA=np.int64(xwin/2)+j
                    smaxS,pmaxS = local_windowS(data1,xwinA,zwinA,x_ini,z_ini,dx,dz,j,i,pmin,pmax,dp)
                    p_max[i,j]=pmaxS
                    s_max[i,j]=smaxS
                    
                #Caso B2
                elif (j-np.int64(xwin/2))>0 and (j+np.int64(xwin/2))<ntr1:
                    #print("2")
                    x_ini=j-np.int64(xwin/2)
                    xwinA= xwin
                    smaxS,pmaxS = local_windowS(data1,xwinA,zwinA,x_ini,z_ini,dx,dz,j,i,pmin,pmax,dp)
                    p_max[i,j]=pmaxS
                    s_max[i,j]=smaxS
                
                # Caso B3
                elif (j+np.int64(xwin/2))>=ntr1:
                    #print("3")
                    x_ini=j-np.int64(xwin/2)
                    xwinA= xwin - (j + np.int64(xwin/2) - ntr1)
                    smaxS,pmaxS = local_windowS(data1,xwinA,zwinA,x_ini,z_ini,dx,dz,j,i,pmin,pmax,dp)
                    p_max[i,j]=pmaxS
                    s_max[i,j]=smaxS
                    
            # Caso C 
            elif (i-np.int64(zwin/2))>0 and (i+np.int64(zwin/2))<nz1:
                z_ini=(i-np.int64(zwin/2))
                zwinA = zwin
                tau = np.arange(z_ini*dz, (z_ini+zwinA)*dz,dz)
                #Caso C1
                if (j-np.int64(xwin/2))<=0:
                    #print("1")
                    x_ini=0
                    xwinA=np.int64(xwin/2)+j
                    smaxS,pmaxS = local_windowS(data1,xwinA,zwinA,x_ini,z_ini,dx,dz,j,i,pmin,pmax,dp)
                    p_max[i,j]=pmaxS
                    s_max[i,j]=smaxS
                    
                #Caso C2
                elif (j-np.int64(xwin/2))>0 and (j+np.int64(xwin/2))<ntr1:
                    #print("2")
                    x_ini=j-np.int64(xwin/2)
                    xwinA= xwin
                    smaxS,pmaxS = local_windowS(data1,xwinA,zwinA,x_ini,z_ini,dx,dz,j,i,pmin,pmax,dp)
                    p_max[i,j]=pmaxS
                    s_max[i,j]=smaxS
               
                # Caso C3
                elif (j+np.int64(xwin/2))>=ntr1:
                    #print("3")
                    x_ini=j-np.int64(xwin/2)
                    xwinA= xwin - (j + np.int64(xwin/2) - ntr1)
                    smaxS,pmaxS = local_windowS(data1,xwinA,zwinA,x_ini,z_ini,dx,dz,j,i,pmin,pmax,dp)
                    p_max[i,j]=pmaxS
                    s_max[i,j]=smaxS
    
    return p_max,s_max