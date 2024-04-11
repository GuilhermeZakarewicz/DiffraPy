# coding: utf-8

import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from tqdm import tqdm
from scipy.fft import rfft, rfftfreq, irfft
import cmath



def ricker(nps,fr,dt):
    """
    Generate a Ricker wavelet signal (Mexican Hat).

    Parameters:
    -----------
    nps : int
        The number of samples in the output signal.
    fr : float
        The central frequency of the wavelet in Hz.
    dt : float
        The time step of the output signal in seconds.

    Returns:
    --------
    numpy.ndarray
        The Ricker wavelet signal as a 1-D numpy array.
        
    """
    npt = nps * dt
    t = np.arange(-npt/2,npt/2,dt)
    rick1=(1-t *t * fr**2 *np.pi**2  ) * np.exp(- t**2 * np.pi**2 * fr**2 )
    rick=rick1[int(np.round(nps/2))-(int(np.round(1/fr/dt)))+1:nps]
    l = len(rick)
    if l<nps:
        rick2=np.append(rick,np.zeros([1,nps-1]))
    l=nps
    rick=rick2
    return np.array(rick)


def sub2ind(array_shape, rows, cols):
    """
    Convert subscripts to linear indices.

    Parameters:
    -----------
    array_shape : tuple of int
        The shape of the array to which the indices correspond.
    rows : array-like
        The row indices.
    cols : array-like
        The column indices.

    Returns:
    --------
    array-like
        The linear indices corresponding to the input subscripts.

    Notes:
    ------
    - The input arrays `rows` and `cols` must have the same shape.
    - The function assumes 0-based indexing.
    - The function uses broadcasting to compute the linear indices.

    Examples:
    ---------
    >>> sub2ind((3, 4), [0, 1, 2], [0, 1, 2])
    array([0, 5, 10])
    >>> sub2ind((3, 4), [2, 1], [3, 2])
    array([11, 6])
    
    """
    return cols*array_shape[0] + rows

def buildL2(L,Z,X,ind,z0,x0,z1,x1):
    """
    Calculates the lengths of the line segments defined by the endpoints 
    (z0, x0) and (z1, x1), and updates the values in the specified indices 
    of the 2D array L.
    
    Given a 2D array `L` representing a grid of line segment lengths,
    and indices `Z` and `X` indicating the size of the grid,
    this function updates the `L` array with the length of a line segment
    defined by two points, specified by `z0`, `x0`, `z1`, and `x1`.

    Parameters:
    -----------
    L : numpy.ndarray of shape (Z*X, Z*X)
        2D array containing the pairwise distances between all points in the grid.
    Z : int
        Number of rows in the grid.
    X : int
        Number of columns in the grid.
    ind : int
        Index in the flattened 2D array L where the distance values should be updated.
    z0, x0 : float
        Coordinates of the starting point of the line segment.
    z1, x1 : float
        Coordinates of the ending point of the line segment.

    Returns:
    --------
    L : numpy.ndarray of shape (Z*X, Z*X)
        The input array L with the updated distance values.
        
    """
    [pz,px,j]=lineseg2(z0,x0,z1,x1)
    for i in range(0,j-1):
        l = np.linalg.norm([pz[i+1]-pz[i],px[i+1]-px[i]])
        #a = np.floor((pz[i+1]+pz[i])/2)-1
        #teste***
        a = int(np.floor((pz[i+1]+pz[i])/2)-1)
        if a == Z:
            a = Z-1
        elif a==-1:
            a = 0
        #b = np.floor((px[i+1]+px[i])/2)-1
        #teste***
        b = int(np.floor((px[i+1]+px[i])/2)-1)
        if b == X:
            b = X-1
        elif b == -1:
            b = 0
        L[ind,sub2ind([Z,X],a,b)]=l
    
    return L


def subs2(sZ,sX):
    """
    Construct a sparse matrix D of size (2sZ-1)*(2sX-1) x (2sZ-2)*(2sX-2), 
    where each row of D corresponds to a horizontal or vertical line 
    segment in a grid of size sZ x sX, and each column of D corresponds 
    to the length of the line segment.

    Parameters:
    -----------
    sZ : int
        The number of rows in the grid.
    sX : int
        The number of columns in the grid.

    Returns:
    --------
    scipy.sparse.lil_matrix: A sparse matrix of size 
    (2sZ-1)*(2sX-1) x (2sZ-2)*(2sX-2), 
    where each row corresponds to a horizontal or vertical 
    line segment in the grid, and each column corresponds to 
    the length of the line segment.
    
    """
    z = 2*sZ-1
    x = 2*sX-1
    z1 = z-1
    x1 = x-1
    #dA = csr_matrix(np.zeros([z*x,z1*x1]))
    dA = lil_matrix((z*x,z1*x1))
    for j in range(0,z):
        for i in range(0,x):
            dA = buildL2(dA,z1,x1,sub2ind([z,x],j,i),sZ,sX,j,i)
    
    return dA

def lineseg2(z0,x0,z1,x1):
    """
    Compute a line segment between two points in 2D space using Bresenham's algorithm.

    Parameters:
    -----------
    z0 : float
        The z-coordinate of the starting point.
    x0 : float
        The x-coordinate of the starting point.
    z1 : float
        The z-coordinate of the end point.
    x1 : float
        The x-coordinate of the end point.

    Returns:
    --------
    [pz,px,j] : list
        A list containing three items:
            - `pz`: A sorted array of z-coordinates along the line segment.
            - `px`: A sorted array of x-coordinates along the line segment.
            - `j`: The number of coordinates in the line segment.

    Notes:
    ------
    - This function uses Bresenham's algorithm to compute a line segment between two points in 2D space.
    - The line segment is returned as two arrays, `pz` and `px`, representing the z- and x-coordinates along the segment, respectively.
    - The length of the line segment is returned as `j`.
    - The function assumes that the input coordinates are in increasing order (i.e., z0 <= z1 and x0 <= x1).

    Examples:
    ---------
    >>> lineseg2(0, 0, 3, 3)
    [[0., 1., 1., 2., 2., 3., 3., 4.],[0., 1., 1., 2., 2., 3., 3., 4.]),8]
    
    """
    z1=z1+1
    x1=x1+1
    dz = (z1-z0)
    dx = (x1-x0)
    sgnz = np.sign(dz)
    sgnx = np.sign(dx)
    pz=[]
    px=[]
    pz.append(z0)
    px.append(x0)
    j = 2
    if sgnz!=0:
        zrange = np.arange(z0+sgnz,z1,sgnz)
        for z in zrange:
            pz.append(z)
            px.append(x0 + (z-z0)*dx/dz)
            j = j+1
    if sgnx!=0:
        xrange = np.arange(x0+sgnx,x1,sgnx)
        for x in xrange:
            px.append(x)
            pz.append(z0+(x-x0)*dz/dx)
            j = j+1
            
            
    pz.append(z1)
    px.append(x1)
    px = np.sort(px)
    pz = np.sort(pz)
    if  sgnx==-sgnz:
        px=np.flip(px)
    
    return [pz,px,j]

def Mray(SW,SP,DX):
    """
    Compute the travel time of a seismic ray with respect to a source position.

    Given a slowness model `SW`, a starting point `SP` and a step size `DX`,
    this function traces a seismic ray from `SP` through `SW` and computes
    the travel time of the ray at each point. It is based on the shortest path method.
    The output is a table `Ttable` of travel times that correspond to the traced
    ray path.

    Parameters:
    -----------
    SW : numpy.ndarray
        2D array of slowness values representing the slowness model (1/velocity model).
    SP : tuple or list
        Tuple or list of two integers representing the starting point of the ray [0,src].
    DX : float
        Step size for tracing the ray through the slowness model (model's discretization).
        Grid spacing along the x-axis.

    Returns:
    --------
    Ttable : numpy.ndarray
        2D array of travel times corresponding to the traced ray path through the
        slowness model, taking into account the travel time of the ray with respect
        to the surface normal at each point. It has the same dimensions [ntr,nz] of
        the SW grid.

    Notes:
    ------
    The algorithm works by iteratively propagating the ray through the slowness model 
    in small steps of size `DX` and computing the travel time of the ray at each point
    (grid node). All the visited nodes record the travel time value. Among all the time 
    values, the function decide which is the minimum. Then, it uses the selected node as
    a secondary source (following the Huygen's principle) and repeat the process until all
    interested nodes are visited. 

    References:
    -----------
    [1] Moser, T.J. (1991). Shortest path calculation of seismic rays: Geophysics, 
        56, 59–67.
    
    [2] Shearer, P.M. (2009). Introduction to Seismology, Second Edition. Cambridge
        University Press.
        
    """
    [Z,X]=SW.shape
    ddef = 10000
    delt = np.max(SW.flatten())
    sZ = 7; sX = 7;
    dA = subs2(sZ,sX)
    
    ZZ = Z+2*sZ-1
    XX = X+2*sX-1
    T = np.ones([ZZ,XX])*ddef
    mark = np.ones([ZZ,XX])*ddef
    
    Z2 = Z + 2*sZ - 2
    X2 = X+2*sX - 2
    S = np.ones([Z2,X2])
    
    Z1 = np.arange(sZ-1,Z+sZ)
    X1 = np.arange(sX-1,X+sX)
    mark[np.ix_(Z1.flatten(),X1.flatten())] = 0
    
    Z2 = np.arange(sZ-1,Z+sZ-1)
    X2 = np.arange(sX-1,X+sX-1)
    S[np.ix_(Z2.flatten(),X2.flatten())] = SW
    S[np.ix_([Z+sZ-1],X2.flatten())] = 2*S[np.ix_([Z+sZ-2],X2.flatten())] - S[np.ix_([Z+sZ-3],X2.flatten())]
    S[np.ix_(Z2.flatten(),[X+sX-1])] = 2*S[np.ix_(Z2.flatten(),[X+sX-2])] - S[np.ix_(Z2.flatten(),[X+sX-3])]
    S[np.ix_([Z+sZ-1],[X+sX-1])] = 2*S[np.ix_([Z+sZ-2],[X+sX-2])] - S[np.ix_([Z+sZ-3],[X+sX-3])]
    
    dz = -sZ+1
    dx = -sX+1
    SP = np.array(SP)
    
    
    z = SP[0]
    x = SP[1]
    z = z+sZ-1
    x = x+sX-1
    T[z,x] = 0
    mark[z,x] = ddef
    
    a = 2*sZ-1
    b = 2*sX-1
    aa = np.arange(-sZ+1,sZ)
    bb = np.arange(-sX+1,sX)
    aas = np.arange(-sZ+1,sZ-1)
    bs = np.arange(-sX+1,sX-1)
    
    AS = S[np.ix_((aas+z).flatten(),(bs+x).flatten())]
    aaa = aa + z
    bbb = bb + x
    TT = T[np.ix_(aaa.flatten(),bbb.flatten())]
    
    K = dA*AS.flatten()+T[z,x]
    KK=np.reshape(K,[13,13])
    BB=np.minimum(KK,TT)
    T[np.ix_(aaa.flatten(),bbb.flatten())] = np.minimum(np.reshape(dA*AS.flatten('F')+T[z,x],[a,b]),TT)
    
    
    maxt = np.max(np.max(T[z-1:z+1,x-1:x+1]))
    while 1:
        H = np.argwhere(T + mark <= maxt + delt)
        hz = H[:,0]
        hx = H[:,1]
        hsz = len(hz)
        for ii in range(0,hsz):
            z = hz[ii]
            x = hx[ii]
            maxt = np.max([maxt,T[z,x]])
            mark[z,x] = ddef
            AS = S[np.ix_((aas + z).flatten(), (bs + x).flatten())]
            aaa = aa + z
            bbb = bb + x
            TT = T[np.ix_(aaa.flatten(),bbb.flatten())]
            T[np.ix_(aaa.flatten(),bbb.flatten())] = np.minimum(np.reshape(dA*AS.flatten('F')+T[z,x],[a,b]),TT)
        if mark[np.ix_(Z2.flatten(),X2.flatten())].all():
            break
        Ttable = T[np.ix_(Z2.flatten(),X2.flatten())]*DX
    
    return Ttable


def raymodel3(SW,dx,nx,filename):
    """
    Computes travel times of seismic waves through a 2D subsurface model
    using the MRay function for ray tracing. Performs seismic ray tracing 
    by calling the Mray function to compute travel times for each source-
    receiver pair and then saves the resulting travel time data to a binary
    file.

    Parameters:
    -----------
    SW : numpy.ndarray
        A 2D array representing the subsurface model.
    dx : float
        Grid spacing along both x and y axes.
    nx : int
        Number of grid points along the x axis.
    filename : str
        Name of the file to save the computed travel times.

    Returns:
    --------
    traveltimesrc : list
        A list containing the travel times of all the seismic waves
        from the source point to all the grid points in the model.
        Each element in the list is a 2D array of shape [nx,nz],
        representing the travel times from a single source point.
        It is saved in the binary file `filename`. 

    """
    DX = dx
    traveltimesrc=[]
    sx=np.arange(0,nx)*DX
    
    for ixsrc in tqdm(range(0,nx)):
        SP = [0,ixsrc]
        Ttable = Mray(SW,SP,DX)
        traveltimesrc.append(Ttable[:,:])
        
    with open(filename, 'wb') as f:
        np.save(f, traveltimesrc)
    
    return traveltimesrc


def kirchhoffModeling(nsx,ngx,dsx,nx,nt,dt,TTh,R,W,filename):
    """
    Computes synthetic seismic common-shot gathers by Kirchhoff modeling.
    
    The method consists of computing the travel times of seismic waves from 
    a source point to all the grid points in the model using a ray tracing 
    algorithm, and then calculating the seismic amplitudes at each receiver 
    location by summing the contributions of all the rays that reach that point.
    
    The amplitudes of the seismic waves are calculated using the Kirchhoff 
    integral, which is an integral equation that relates the incident wave 
    field to the scattered wave field at a given receiver location.

    Parameters:
    -----------
    nsx : int
        Number of sources (shots).
    ngx : int
        Number of receivers (traces).
    dsx : int
        Step size for sources along the x-axis.
    nx : int
        Number of grid points along the x-axis.
    nt : int
        Number of time samples.
    dt : float
        Time step.
    TTh : numpy.ndarray
        A 3D array of shape [nsx,nz,nx] containing the travel times of
        all the seismic waves from all the source points to all the grid points
        in the model. It is calculated before by the `raymodel3` function.
    R : numpy.ndarray
        A 2D array of shape [nz,nx] representing the reflectivity coefficients
        of the subsurface model. 
    W : numpy.ndarray
        A 1D array of length nt representing the temporal source waveform. 
        It is a Ricker wavelet calculated by the `ricker` function.
    filename : str
        Base name of the files to save the computed common-shot gathers. One
        file will be saved for each source point, with a suffix indicating
        the source index.

    Returns:
    --------
    files : list of numpy.ndarray
        A list of length nsx containing the common-shot gathers computed for
        each source point. Each element in the list is a 2D array of shape
        [nt-2,ngx], representing the seismic trace data for a given source
        (sx).
        
    """

    nsx=nx 
    ngx=nx 

    #Loop over shots
    for isx in tqdm(range(0,nsx,dsx)):
        D = np.zeros([nt,ngx])
        TSX = (TTh[isx,:,:]/dt+1).astype(int)  # Traveltime  (indexes)

        # Loop Over Traces
        for gx in range(0,ngx):
            TXG=(TTh[gx,:,:]/dt+1).astype(int)  # Traveltime in heterogeneous medium (indexes)

            #Loop over time sample in a trace
            for t in range(0,nt):
                M=W[t-(TSX+TXG)+nt+1]*R
                #D[t,gx]=np.sum(M.flatten('F'));
                D[t,gx]=np.sum(M.flatten())
            gather1=np.diff(D[:,:],n=2,axis=0)
        
        file = str(filename)+"_{}".format(isx)
        with open(file, "wb") as f:
            np.save(f, gather1)
            
    files = []

    for i in range(0,nsx,dsx):
        file = str(filename)+"_{}".format(isx)
        with open(file, 'rb') as file:
            (gather) = np.load(file)
            files.append(gather)
            
    return files


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
    
    #[nz1,ntr1]=data1.shape
    #dataC = data1.copy()
    #x = np.arange(x_ini,(x_ini+ntr1),1)
    #z = p*(x-x0) + z0
    #x_grid = np.int64((np.round(x))-x_ini)
    #z_grid = np.int64(np.round(z)-z_ini)
    
    [nz1,ntr1]=data1.shape
    dataC = data1.copy()
    x = np.arange(x_ini*dx,(x_ini+ntr1)*dx,dx) #now considering the dx,dz information
    z = p*(x-(x0*dx))+ z0*dz
    z_grid = np.int64(np.round(z/dz)-z_ini)
    x_grid = np.int64((np.round(x/dx))-x_ini)
    
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
        The maximum coherence value.
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




def taper(ntr,nz,app,isx,igx):
    """
    Applies a Hanning taper to the seismic data to reduce edge effects.
    
    This function computes a taper window for a seismic trace. The taper 
    window is used to gradually reduce the amplitude of the trace towards 
    its beginning and end, in order to avoid edge effects when processing 
    the data.
    
    The function uses the np.hanning() function to compute a Hanning window 
    of length 2*app. The window is centered around the midpoint of the 
    source and receiver traces (cmp), and is then applied to the appropriate 
    section of the ar array based on the position of the midpoint and the 
    length of the trace.

    Parameters:
    -----------
    ntr : int
        Number of traces in the seismic data.
    nz : int
        Number of grid points in the z-axis.
    app : int
        Number of samples to apply the taper at each edge of the trace.
    isx : int
        Index of the current source in the survey.
    igx : int
        Index of the current trace in the seismic data.

    Returns:
    --------
    ar : numpy.ndarray
        A 2D array of shape [nz,ntr] representing the tapered seismic data.
    
    """
    ar = np.zeros([nz,ntr])
    cmp = int((isx+igx)/2)
    window = np.hanning(2*app)
    
    if (cmp-app)<0: 
        if (cmp+app)>ntr:
            ntr_2 = int(ntr/2)
            lw_2 = int(len(window)/2)            
            ar[:,:] = window[(lw_2 - ntr_2):(lw_2 + ntr_2)]  
        else:
            ar[:,0:(cmp+app)] = window[abs(cmp-app):]   
    elif (cmp+app)>ntr:
        ar[:,(abs(cmp-app)):] = window[0:(ntr - abs(cmp-app))]
    else:
        ar[:,(cmp-app):(cmp+app)] = window 
    
    return ar


def peso(TTh,dt,X,Y,igx,isx):
    """
    Computes the Kirchhoff weighting function w(s,r,x) for a given source-
    receiver pair in a 2D subsurface model, also called imaging condition 
    or anti-stationary phase filter. 

    Parameters:
    -----------
    TTh : numpy.ndarray
        A 3D array of shape [nsx,nz,nx] containing the travel times of
        all the seismic waves from all the source points to all the grid points
        in the model. It is calculated by the `raymodel3` function.
    dt : float
        The time sampling interval in seconds.
    X : numpy.ndarray
        A 2D array of shape [nz,ntr] representing the x component of the dip
        flocal_windowield obtained by a local slant-stack approach with the function 
        ``. 
    Y : numpy.ndarray
        A 2D array of shape [nz,ntr] representing the y component of the dip
        field obtained by a local slant-stack approach with the function 
        `local_window`. 
    igx : int
        The index of the receiver point where the Kirchhoff integral is being
        computed.
    isx : int
        The index of the source point where the Kirchhoff integral is being
        computed.

    Returns:
    --------
    w : numpy.ndarray
        A 2D array of shape [nz,ntr] representing the Kirchhoff weighting
        function for the given source-receiver pair. 
    
    References:
    -----------
    [1] Moser, T.j. and Howard, C.B. (2008). Diffraction imaging in depth.
    Geophysical Prospecting.
    
    """
    
    gH = np.gradient(TTh, axis=2) #gradiente horizontal  #diferença entre colunas (do modelo de velocidade)
    gV = np.gradient(TTh, axis=1) #gradiente vertical    #diferença entre linhas

    prV = gV[igx,:,:] 
    prH = gH[igx,:,:] 

    psV = gV[isx,:,:] 
    psH = gH[isx,:,:] 

    pH = psH + prH
    pV = psV + prV
    
    norma = np.sqrt(pH**2 + pV**2)

    for idx, x in np.ndenumerate(norma): 
        if x==0:
            norma[idx]=1e-16 #avoid nan's

    w = (pH/norma * X) + (pV/norma * Y)
    
    return w


def phase_shift(gather):
    """
    Applies a phase shift of pi/4 radians or 45 degrees to each trace of a seismic 
    gather. The phase shift is implemented in the frequency domain by multiplying 
    the Fourier transform of each trace by a complex exponential. The 
    phase-shifted traces are then reconstructed by applying the inverse 
    Fourier transform.
    
    Parameters:
    -----------
    gather : numpy.ndarray
        A 2D array representing the seismic gather to be phase-shifted.

    Returns:
    --------
    phase_gather : numpy.ndarray
        A 2D array representing the phase-shifted seismic gather. Each trace
        in the output array has been phase-shifted by pi/4 radians relative
        to the corresponding trace in the input array.
    
    """
    phase_gather = gather.copy()
    n_traces = gather.shape[-1]
    
    for i in range(n_traces):
        trace = gather[:,i]
        signalFFT = rfft(trace) 
        newSignalFFT = signalFFT * cmath.rect( 1., np.pi/4 ) 
        newSignal = irfft(newSignalFFT, n=gather.shape[0]) 
        phase_gather[:,i] = newSignal
        
    return phase_gather


def ssf(nz,nx,dx,R,SW):
    """
    Calculate the spherical spreading factor.

    Parameters
    ----------
    igx : int
        Index of the source location in the x-direction.
    nz : int
        Number of grid points in the z-direction.
    nx : int
        Number of grid points in the x-direction.
    dx : int
        The spacing between traces in the x direction.
    R : numpy.ndarray
        distance matrix
    SW : numpy.ndarray
        Slowness model (1/velocity model)

    Returns
    -------
    numpy.ndarray
        The calculated ssf.

    """
    r_mask_ssf = (R==dx/1000)
    R[r_mask_ssf]=dx
    ssf = np.sqrt((1/R)*SW)
    return ssf


def kirchhoffMigration(gather,isx,dx,dz,dt,win,dwin,app_ref,app_dif,TTh,X,Y,sm):
    """
    Perform Kirchhoff Migration on seismic data to create conventional 
    reflection and diffraction migration images. It includes the 
    obliquity factor, phase shift, aperture, window, and weighting
    function w(s,x,r).

    Parameters:
    -----------
    gather : array-like
        A single gather or an array of gathers to be migrated.
    isx : int
        The index of the source trace in the gather, considered for 
        the single gather case.
    dx : float
        The spacing between traces in the x direction.
    dz : float
        The spacing between samples in the z direction.
    dt : float
        The sampling interval in seconds.
    win : int
        The half-length of the migration window in samples.
    dwin : float
        The step size between migration windows in seconds.
    app_ref : int
        The aperture size in number of traces for the conventional migration.
    app_diff : int
        The aperture size in number of traces for the diffraction-oriented migration
    TTh : array-like
        The traveltime table calculated by the raymodel3 function.
    X : array-like
        The X component of the dip field. Should be calculated with np.sin(m_theta).
        It is obtained with a local slant-stack approach by the function 
        `local_window`. 
    Y : array-like
        The Y component of the dip field. Should be calculated with np.cos(m_theta).
        It is obtained with a local slant-stack approach by the function 
        `local_window`
    sm : array-like
        Maximum correlation coefficients computed from local windowed slant stacks.

    Returns:
    --------
    refl_mig : array-like
        The conventional reflection migration image where w = w(s,x,r).
    diff_mig : array-like
        The diffraction migration image where w = (1 - w(s,x,r)).
        
    Notes:
    ------
    - The function first checks whether the input data is for a single shot or multiple 
    shots, then processes the seismic data by phase shifting and applying the obliquity 
    factor. It then applies a window to the data, tapers the traces, and calculates the 
    migration for each window position. Finally, if the input data is composed by multiple
    shots, it stacks the migration images for each window position to produce the final migration 
    image. Both conventional and diffraction migration images are generated simultaneously.
    
    - An important step of the migration process is finding the optimal parameters of window
    and aperture. A small window may overlook the event's energy, while a large one may include 
    noise and unrelated energy. Similarly, a small aperture may disregard diffraction energy, 
    while a large one may consider unrelated noise and energy. Generating an migrated image without
    considering the effects of these parameters may lead to spurious results. 
    """
    
    gather = np.array(gather)
    
    #single shot
    if gather.ndim == 2:
    
        gather = phase_shift(gather)

        timer=np.round(TTh/dt)+1
        #timer=TTh

        window = np.arange(-win,win,dwin)
        [nt,ntr]=gather.shape
        [ntr2,nz,nx]=timer.shape
        if ntr!=ntr2:
            print('Gather and traveltime table have different trace numbers')

        #mig=np.zeros([nz,nx])
        refl_mig = np.zeros([nz,nx])
        diff_mig = np.zeros([nz,nx])

        IX = np.arange(0,nx*dx,dx)
        IZ = np.arange(0,nz*dz,dz)
        [IIX,IIZ] = np.meshgrid(IX,IZ)
        # Loop over each trace of the shot gather at src isx
        
        for igx in tqdm(range(0,ntr)):
            w = peso(TTh,dt,X,Y,igx,isx)
            ###
            w_reff = (w**4)*(sm)
            w_diff = (1-(w**4))
            ###
            trace_reflwin = np.zeros([nz,nx])
            trace_diffwin = np.zeros([nz,nx])
            R = np.sqrt(IIZ**2 + (IIX-(igx+isx)/2*dx)**2)
            r_mask = (R==0)
            R[r_mask]= dx/1000
            obli = IIZ/R
            trace_appref = taper(ntr,nz,app_ref,isx,igx)
            trace_appdif = taper(ntr,nz,app_dif,isx,igx) 

            for j in range(len(window)): #somar amplitudes da curva de difração com uma janela 
                t = timer[isx,0:nz,0:nx] + timer[igx,0:nz,0:nx] #t_{d}
                twin = t + window[j]
                t2 = (twin<nt)*twin 
                trace1=gather.T[np.ix_([igx],t2.flatten().astype(np.int32))] 
                #trace1 = trace1.reshape([nz,nx])*(w)
                trace_refl1 = trace1.reshape([nz,nx])*(w_reff)
                trace_diff1 = trace1.reshape([nz,nx])*(w_diff)
                #trace1 = trace1*trace_app
                trace_refl = trace_refl1*trace_appref
                trace_diff = trace_diff1*trace_appdif
                #trace_win = trace_win+trace1
                trace_reflwin = trace_reflwin + trace_refl
                trace_diffwin = trace_diffwin + trace_diff

            #mig[0:nz,0:nx]=mig[0:nz,0:nx] + trace_reflwin*obli
            refl_mig[0:nz,0:nx] = refl_mig[0:nz,0:nx] + trace_reflwin*obli
            diff_mig[0:nz,0:nx] = diff_mig[0:nz,0:nx] + trace_diffwin*obli
    
    #multiple shots, stack
    elif gather.ndim == 3:

        gathers_shifted = []

        for i in gather:
            gather_shifted = phase_shift(i)
            gathers_shifted.append(gather_shifted)

        files = gathers_shifted
        timer=np.round(TTh/dt)+1
        
        #If the number of shots != ntr, we need to create an  
        #array with the isx indexes. This snippet assures it. 
        n_isx = np.shape(files[0])[1] #ntr
        d_isx = len(files) #number of shots         
        isxs = np.linspace(0,n_isx,d_isx,endpoint=False) #array with the isx indexes
        
        refl_migs = []
        diff_migs = []

        for count,gather in tqdm(enumerate(files)):
            isx = int(np.round(isxs[count])) #acessing the correct index for source position isx
            window = np.arange(-win,win,dwin)
            [nt,ntr] = gather.shape
            [ntr2,nz,nx] = timer.shape
            if ntr!=ntr2:
                print('Gather and traveltime table have different trace numbers')

            #mig=np.zeros([nz,nx])
            #mig_final=np.zeros([nz,nx])
            
            refl_mig_isx = np.zeros([nz,nx])
            diff_mig_isx = np.zeros([nz,nx])

            IX = np.arange(0,nx*dx,dx)
            IZ = np.arange(0,nz*dz,dz)
            [IIX,IIZ] = np.meshgrid(IX,IZ)

            for igx in range(0,ntr):
                
                w = peso(TTh,dt,X,Y,igx,isx)
                w_reff = (w**4)*(sm)
                w_diff = (1-(w**4))
                
                #trace_win = np.zeros([nz,nx])
                trace_reflwin = np.zeros([nz,nx])
                trace_diffwin = np.zeros([nz,nx])
                R = np.sqrt(IIZ**2 + (IIX-(igx+isx)/2*dx)**2)
                r_mask = (R==0)
                R[r_mask]= dx/1000
                obli = IIZ/R
                trace_appref = taper(ntr,nz,app_ref,isx,igx)
                trace_appdif = taper(ntr,nz,app_dif,isx,igx) 

                for j in range(len(window)):
                    t = timer[isx,0:nz,0:nx] + timer[igx,0:nz,0:nx] #t_{d}
                    twin = t + window[j]
                    t2 = (twin<nt)*twin 
                    trace1 = gather.T[np.ix_([igx],t2.flatten().astype(np.int32))]
                    #trace1 = trace1.reshape([nz,nx])*w 
                    trace_refl1 = trace1.reshape([nz,nx])*(w_reff)
                    trace_diff1 = trace1.reshape([nz,nx])*(w_diff)
                    #trace1 = trace1*trace_app
                    trace_refl = trace_refl1*trace_appref
                    trace_diff = trace_diff1*trace_appdif
                    #trace_win = trace_win+trace1
                    trace_reflwin = trace_reflwin + trace_refl
                    trace_diffwin = trace_diffwin + trace_diff

                #mig[0:nz,0:nx]=mig[0:nz,0:nx] + trace_win*obli
                refl_mig_isx[0:nz,0:nx] = refl_mig_isx[0:nz,0:nx] + trace_reflwin*obli
                diff_mig_isx[0:nz,0:nx] = diff_mig_isx[0:nz,0:nx] + trace_diffwin*obli

            #migs.append(mig)
            refl_migs.append(refl_mig_isx)
            diff_migs.append(diff_mig_isx)

        #mig_final = np.add.reduce(migs)
        refl_mig = np.add.reduce(refl_migs)
        diff_mig = np.add.reduce(diff_migs)
        
    return refl_mig,diff_mig



def plot(grid,filename,nx,dx,nz,dz,xlabel="",ylabel="",cmp="viridis",it=None,color_bar=False):
    """
    Plots a 2D grid using matplotlib. 

    Parameters
    ----------
    grid : numpy.ndarray
        2D array of data to plot.
    filename : str
        Name of the output file to save the plot.
    nx : int
        Number of grid points in the x-direction.
    dx : int
        Grid spacing in the x-direction.
    nz : int
        Number of grid points in the z-direction.
    dz : int
        Grid spacing in the z-direction.
    xlabel : str, optional
        Label for the x-axis, by default "".
    ylabel : str, optional
        Label for the y-axis, by default "".
    cmp : str, optional
        Matplotlib colormap to use, by default "viridis". Try other colormaps: "Greys",
        "hot", "seismic", "plasma"...
        https://matplotlib.org/stable/tutorials/colors/colormaps.html
    it : str, optional
        Interpolation method to use, by default None. Try others: "bicubic", "gaussian",
        "nearest", "sinc"...
        https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
    color_bar : boolean, optional
        Set color_bar = True for visualizing the colorbar.

    Returns
    -------
    None

    Notes
    -----
    Saves the plot as a PNG image in the "Data" directory (300 dpi).
    """
    
    csfont = {'fontname':'Times New Roman'}

    fig, (ax1) = plt.subplots(1,1,figsize=(8,6))

    im1 = ax1.imshow(grid, extent=[0, nx*dx, nz*dz, 0], aspect='auto', cmap=cmp, interpolation=it)

    ax1.set_xlabel(xlabel, **csfont, fontsize=17)
    ax1.set_ylabel(ylabel, **csfont, fontsize=17)
    
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top')
    ax1.tick_params(axis='x', which='major', pad=0, labelsize=15)
    ax1.tick_params(axis='y', which='major', pad=1, labelsize=15)

    for tick in ax1.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax1.get_yticklabels():
        tick.set_fontname("Times New Roman")

    if color_bar:
        divider1 = make_axes_locatable(ax1)

        cax1 = divider1.append_axes("right", size="5%", pad=0.3)
        cbar1 = plt.colorbar(im1, cax=cax1)
        cbar1.ax.tick_params(labelsize=12)
        for l in cbar1.ax.yaxis.get_ticklabels():
            l.set_family("Times New Roman")

    plt.savefig("Data/{}.png".format(str(filename)),dpi=300)
    plt.show()




###########################################################################################################
###########################################################################################################

                #####################################
                ##Section for testing new functions##
                #####################################

###########################################################################################################
###########################################################################################################          

###########TESTE TESTE TESTE TESTE TESTE TESTE



def migration_teste(gather,isx,dx,dz,dt,win,dwin,app_ref,app_dif,TTh,X,Y,sm,SW):
    """
    Function for testing modifications on the migration.
    
    """
    
    gather = np.array(gather)
    
    #############
    #single shot#
    #############
    if gather.ndim == 2:
    
        gather = phase_shift(gather) #Phase Shift

        timer=np.round(TTh/dt)+1

        window = np.arange(-win,win,dwin)
        [nt,ntr]=gather.shape
        [ntr2,nz,nx]=timer.shape
        if ntr!=ntr2:
            print('Gather and traveltime table have different trace numbers')

        refl_mig = np.zeros([nz,nx])
        diff_mig = np.zeros([nz,nx])

        IX = np.arange(0,nx*dx,dx)
        IZ = np.arange(0,nz*dz,dz)
        [IIX,IIZ] = np.meshgrid(IX,IZ)
       
        # Loop over each trace of the shot gather at src isx
        for igx in tqdm(range(0,ntr)):
            
            w = peso(TTh,dt,X,Y,igx,isx)
            w = np.abs(w)
            
            ######### Single Shot
            #########
            #########
            w_reff = (w**4)*(np.sqrt(sm))
            w_diff = (1-w**4)
            #########
            #########
            #########
                      
            trace_reflwin = np.zeros([nz,nx])
            trace_diffwin = np.zeros([nz,nx])
            R = np.sqrt(IIZ**2 + (IIX-(igx+isx)/2*dx)**2)
            r_mask = (R==0)
            R[r_mask]= dx/1000
            obli = IIZ/R #Obliquity Factor
            trace_appref = taper(ntr,nz,app_ref,isx,igx) #Tapering
            trace_appdif = taper(ntr,nz,app_dif,isx,igx) #Tapering
            sf = ssf(nz,nx,dx,R,SW) #Spherical Spreading Factor

            for j in range(len(window)): #somar amplitudes da curva de difração com uma janela 
                t = timer[isx,0:nz,0:nx] + timer[igx,0:nz,0:nx] #t_{d}
                twin = t + window[j]
                t2 = (twin<nt)*twin 
                trace1=gather.T[np.ix_([igx],t2.flatten().astype(np.int32))] 
                #trace1 = trace1.reshape([nz,nx])*(w)
                trace_refl1 = trace1.reshape([nz,nx])*(w_reff)
                trace_diff1 = trace1.reshape([nz,nx])*(w_diff)
                #trace1 = trace1*trace_app
                trace_refl = trace_refl1*trace_appref
                trace_diff = trace_diff1*trace_appdif
                #trace_win = trace_win+trace1
                trace_reflwin = trace_reflwin + trace_refl
                trace_diffwin = trace_diffwin + trace_diff

            #mig[0:nz,0:nx]=mig[0:nz,0:nx] + trace_reflwin*obli
            refl_mig[0:nz,0:nx] = refl_mig[0:nz,0:nx] + trace_reflwin*obli#*sf
            diff_mig[0:nz,0:nx] = diff_mig[0:nz,0:nx] + trace_diffwin*obli#*sf
    
    
    #######################
    #multiple shots, stack#
    #######################
    elif gather.ndim == 3:

        gathers_shifted = []

        for i in gather:
            gather_shifted = phase_shift(i)
            gathers_shifted.append(gather_shifted)

        files = gathers_shifted
        timer=np.round(TTh/dt)+1
        
        n_isx = np.shape(files[0])[1] #ntr
        d_isx = len(files) #number of shots         
        isxs = np.linspace(0,n_isx,d_isx,endpoint=False) #array with the isx indexes
        
        refl_migs = []
        diff_migs = []

        for count,gather in tqdm(enumerate(files)):
            isx = int(np.round(isxs[count])) #acessing the correct index for source position isx
            window = np.arange(-win,win,dwin)
            [nt,ntr] = gather.shape
            [ntr2,nz,nx] = timer.shape
            if ntr!=ntr2:
                print('Gather and traveltime table have different trace numbers')
            
            refl_mig_isx = np.zeros([nz,nx])
            diff_mig_isx = np.zeros([nz,nx])

            IX = np.arange(0,nx*dx,dx)
            IZ = np.arange(0,nz*dz,dz)
            [IIX,IIZ] = np.meshgrid(IX,IZ)

            for igx in range(0,ntr):
                w = peso(TTh,dt,X,Y,igx,isx)
                
                ######### #Stack
                #########
                #########
                #w_reff = (sm/2)*(w**4)
                w_reff = (np.abs(w)**4)*(np.sqrt(sm))
                w_diff = (1-np.abs(w)**4)
                #########
                #########
                #########
                
                trace_reflwin = np.zeros([nz,nx])
                trace_diffwin = np.zeros([nz,nx])
                R = np.sqrt(IIZ**2 + (IIX-(igx+isx)/2*dx)**2)
                r_mask = (R==0)
                R[r_mask]= dx/1000
                obli = IIZ/R
                trace_appref = taper(ntr,nz,app_ref,isx,igx)
                trace_appdif = taper(ntr,nz,app_dif,isx,igx) 
                sf = ssf(nz,nx,dx,R,SW) #Spherical Spreading Factor

                for j in range(len(window)):
                    t = timer[isx,0:nz,0:nx] + timer[igx,0:nz,0:nx] #t_{d}
                    twin = t + window[j]
                    t2 = (twin<nt)*twin 
                    trace1 = gather.T[np.ix_([igx],t2.flatten().astype(np.int32))]
                    #trace1 = trace1.reshape([nz,nx])*w 
                    trace_refl1 = trace1.reshape([nz,nx])*(w_reff)
                    trace_diff1 = trace1.reshape([nz,nx])*(w_diff)
                    #trace1 = trace1*trace_app
                    trace_refl = trace_refl1*trace_appref
                    trace_diff = trace_diff1*trace_appdif
                    #trace_win = trace_win+trace1
                    trace_reflwin = trace_reflwin + trace_refl
                    trace_diffwin = trace_diffwin + trace_diff

                #mig[0:nz,0:nx]=mig[0:nz,0:nx] + trace_win*obli
                refl_mig_isx[0:nz,0:nx] = refl_mig_isx[0:nz,0:nx] + trace_reflwin*obli#*sf
                diff_mig_isx[0:nz,0:nx] = diff_mig_isx[0:nz,0:nx] + trace_diffwin*obli#*sf

            #migs.append(mig)
            refl_migs.append(refl_mig_isx)
            diff_migs.append(diff_mig_isx)

        #mig_final = np.add.reduce(migs)
        refl_mig = np.add.reduce(refl_migs)
        diff_mig = np.add.reduce(diff_migs)
        
    return refl_mig,diff_mig
   