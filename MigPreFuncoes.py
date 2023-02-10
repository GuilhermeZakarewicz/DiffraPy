# coding: utf-8

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from tqdm import tqdm
from scipy.fft import rfft, rfftfreq, irfft
import cmath

print('Imported MigPreFuncoes now')

def ricker(nps,fr,dt):
    npt = nps*dt
    t = np.arange(-npt/2,npt/2,dt)
    #t = np.linspace(-npt/2,npt/2,nps)
    rick1=(1-t *t * fr**2 *np.pi**2  ) * np.exp(- t**2 * np.pi**2 * fr**2 )
    rick=rick1[int(np.round(nps/2))-(int(np.round(1/fr/dt)))+1:nps]
    l = len(rick)
    if l<nps:
        rick2=np.append(rick,np.zeros([1,nps-1]))
    l=nps
    rick=rick2
    return np.array(rick)

# Equivalente ao sub2ind do Matlab
def sub2ind(array_shape, rows, cols):
    #return rows*array_shape[1] + cols
    return cols*array_shape[0] + rows

def buildL2(L,Z,X,ind,z0,x0,z1,x1):
    [pz,px,j]=lineseg2(z0,x0,z1,x1)
    for i in range(0,j-1):
        l = np.linalg.norm([pz[i+1]-pz[i],px[i+1]-px[i]])
        a = np.floor((pz[i+1]+pz[i])/2)-1
        if a == Z:
            a = Z-1
        elif a==-1:
            a = 0
        b = np.floor((px[i+1]+px[i])/2)-1
        if b == X:
            b = X-1
        elif b == -1:
            b = 0
        L[ind,sub2ind([Z,X],a,b)]=l
    return L


def subs2(sZ,sX):
    z = 2*sZ-1
    x = 2*sX-1
    z1 = z-1
    x1 = x-1
    #dA = csr_matrix(np.zeros([z*x,z1*x1]))
    dA = lil_matrix((z*x,z1*x1))
    for j in range(0,z):
        for i in range(0,x):
            #print(i)
            dA = buildL2(dA,z1,x1,sub2ind([z,x],j,i),sZ,sX,j,i)
    return dA

def lineseg2(z0,x0,z1,x1):
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


def kirchoffModeling(nsx,ngx,dsx,nx,nt,dt,TTh,R,W,filename):
    """
    Kirchhoff Modeling of synthetic seismic common-shot gathers. 
    Calculate the common-shot gathers for the input model. 
    
    Parameters
    ----------
    nsx : number of shots. Minimum = 1, maximum = nx
    ngx : number of receivers. Minimum = 1, maximum = nx
    dsx : shots spacing (grid value)
    nx : grid size in X
    nt : number of time samples
    dt : 
    TTh : travel-time table
    R : reflectivity model
    W :
    path : [str] path to the directory to store the files
    
    
    Returns
    -------
    files : np.array [(nsx/dsx),nt,nx] with the shot gathers 
    The funtion saves the files in directory [path] 
    """
    nsx=nx #105  #numero de tiros (pode ser no máximo o valor de TT axis=0)
    ngx=nx #105   #numero de receptores (pode ser no máximo o valor de TT axis=0)

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


def taper(ntr,ns,app,isx,igx):
    
    ar = np.zeros([ns,ntr])
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
    Calcula a função peso w(s,x,r)
    Entrada:
    TTh - tabela do tempo de trânsito calculada com a função raymodel3
    dt - discretização do tempo (s)
    X - componente X do modelo; X = np.sin(m_theta); X.shape = [nz,nx]
    Y - componente Y do modelo; Y = np.cos(m_theta); Y.shape = [nz,nx]
    igx - posição do receptor 
    isx - posição da fonte
    Saída:
    w - função peso (w.shape=[nz,nx])
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
    Realiza a meia-derivada (phase-shift de 45 graus) para os traços sísmicos de um gather.
    Entrada:
    gather - dado sísmico (nt,ntr)
    Saída:
    phase_gather - dado sísmico após phase-shift em todos os traços (nt,ntr)
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



def migvsp_winapp(gather,isx,dx,dz,dt,win,dwin,app,TTh,X,Y,epsilon):
    """
    Calcula a migração para 1 arquivo (1 tiro) com janela (window) e abertura (aperture)
    Considera a função peso w = w(s,r,t)
    
    Entrada:
    gather - dado sísmico (nt,ntr)
    isx - posição do tiro
    dx - discretização no eixo x (m)
    dz - discretização no eixo z (m)
    dt - discretização do tempo (s)
    win - (tamanho da janela)/2
    dwin - passo da janela. Preferencialmente, dwin=dt
    app - tamanho da abertura
    TTh - tabela do tempo de trânsito calculada com a função raymodel3
    X - componente X da normal do modelo; X = np.sin(m_theta); X.shape = [nz,nx]
    Y - componente Y da normal do modelo; Y = np.cos(m_theta); Y.shape = [nz,nx]
    epsilon - entre 0 e 1- relacionado à frequência (zona de Fresnel)
    Saída:
    mig - imagem migrada com janela e abertura. Formato: matriz [nt,ntr]
    """
    gather = phase_shift(gather)
    
    timer=np.round(TTh/dt)+1
    #timer=TTh
    
    window = np.arange(-win,win,dwin)
    [nt,ntr]=gather.shape
    [ntr2,nz,nx]=timer.shape
    if ntr!=ntr2:
        print('Gather e traveltime table tem numero diferente de traços')
        
    mig=np.zeros([nz,nx])
    
    IX = np.arange(0,nx*dx,dx)
    IZ = np.arange(0,nz*dz,dz)
    [IIX,IIZ] = np.meshgrid(IX,IZ)
    # Loop over each trace of the shot gather at src isx
    for igx in tqdm(range(0,ntr)):
        w = peso(TTh,dt,X,Y,igx,isx)
        #mask = w>(1-epsilon)
        trace_win = np.zeros([nz,nx])
        R = np.sqrt(IIZ**2 + (IIX-(igx+isx)/2*dx)**2)
        r_mask = (R==0)
        R[r_mask]= dx/1000 #trocando 0 por algo pequeno
        obli = IIZ/R
        trace_app = taper(ntr,nz,app,isx,igx) 
        
        for j in range(len(window)): #somar amplitudes da curva de difração com uma janela 
            t = timer[isx,0:nz,0:nx] + timer[igx,0:nz,0:nx] #t_{d}
            twin = t + window[j]
            t2 = (twin<nt)*twin 
            trace1=gather.T[np.ix_([igx],t2.flatten().astype(np.int32))] 
            trace1 = trace1.reshape([nz,nx])*(w) 
            trace1 = trace1*trace_app
            trace_win = trace_win+trace1
        
        mig[0:nz,0:nx]=mig[0:nz,0:nx] + trace_win*obli
        
    return mig





def migstack_winapp(files,isx,dx,dz,dt,win,dwin,app,TTh,X,Y,epsilon):
        
    """
    Calcula a migração para vários arquivos (todos os tiros ao longo de uma linha sísmica) com janela (window) e abertura (aperture)
    Stack das imagens migradas de cada tiro
    Considera a função peso w = w(s,r,t)
    
    Entrada:
    files - lista (array) com os dados sísmicos. 
    isx - posição do tiro
    dx - discretização no eixo x (m)
    dz - discretização no eixo z (m)
    dt - discretização do tempo (s)
    win - (tamanho da janela)/2
    dwin - passo da janela. Preferencialmente, dwin=dt
    app - tamanho da abertura
    TTh - tabela do tempo de trânsito calculada com a função raymodel3
    X - componente X do modelo; X = np.sin(m_theta); X.shape = [nz,nx]
    Y - componente Y do modelo; Y = np.cos(m_theta); Y.shape = [nz,nx]
    
    Saída:
    mig - imagem migrada com janela e abertura. Formato: matriz [nt,ntr]
    """

    gathers_shifted = []

    for i in files:
        gather_shifted = phase_shift(i)
        gathers_shifted.append(gather_shifted)

    files = gathers_shifted
    
    timer=np.round(TTh/dt)+1
    #timer=TTh
    migs = []
        
    for count,gather in tqdm(enumerate(files)):
        isx = count
        #print(f"shot {isx}")
    
        window = np.arange(-win,win,dwin)
        [nt,ntr]=gather.shape
        [ntr2,nz,nx]=timer.shape
        if ntr!=ntr2:
            print('Gather e traveltime table tem numero diferente de traços')

        mig=np.zeros([nz,nx])
        mig_final=np.zeros([nz,nx])

        IX = np.arange(0,nx*dx,dx)
        IZ = np.arange(0,nz*dz,dz)
        [IIX,IIZ] = np.meshgrid(IX,IZ)
        # Loop over each trace of the shot gather at src isx
        for igx in range(0,ntr):
            w = peso(TTh,dt,X,Y,igx,isx)
            #mask = w>(1-epsilon)
            trace_win = np.zeros([nz,nx])
            R = np.sqrt(IIZ**2 + (IIX-(igx+isx)/2*dx)**2)
            r_mask = (R==0)
            R[r_mask]= dx/1000
            obli = IIZ/R
            trace_app = taper(ntr,nz,app,isx,igx) 

            for j in range(len(window)):
                t = timer[isx,0:nz,0:nx] + timer[igx,0:nz,0:nx] #t_{d}
                twin = t + window[j]
                t2 = (twin<nt)*twin 
                trace1=gather.T[np.ix_([igx],t2.flatten().astype(np.int32))] 
                trace1 = trace1.reshape([nz,nx])*w #seção? imshow em um separado
                trace1 = trace1*trace_app
                trace_win = trace_win+trace1

            mig[0:nz,0:nx]=mig[0:nz,0:nx] + trace_win*obli
        
        migs.append(mig)
    
    mig_final = np.add.reduce(migs)
        
    return mig_final






def migvsp_winapp_diff(gather,isx,dx,dz,dt,win,dwin,app,TTh,X,Y,epsilon):
        
    """
    Calcula a imagem de difração para 1 arquivo (1 tiro) com janela (window) e abertura (aperture)
    Considera a função peso w = 1 - w(s,r,t)
    
    Entrada:
    gather - dado sísmico (nt,ntr)
    isx - posição do tiro
    dx - discretização no eixo x (m)
    dz - discretização no eixo z (m)
    dt - discretização do tempo (s)
    win - (tamanho da janela (s))/2
    dwin - passo da janela. Preferencialmente, dwin=dt
    app - tamanho da abertura
    TTh - tabela do tempo de trânsito calculada com a função raymodel3
    X - componente X do modelo; X = np.sin(m_theta); X.shape = [nz,nx]
    Y - componente Y do modelo; Y = np.cos(m_theta); Y.shape = [nz,nx]
    
    Saída:
    mig - imagem de difrações a partir de migração com janela e abertura. Formato: matriz [nt,ntr]
    """
    gather = phase_shift(gather)    
    timer=np.round(TTh/dt)+1
    #timer=TTh
    window = np.arange(-win,win,dwin)
    [nt,ntr]=gather.shape
    [ntr2,nz,nx]=timer.shape
    if ntr!=ntr2:
        print('Gather e traveltime table tem numero diferente de traços')
        
    mig=np.zeros([nz,nx])
    
    IX = np.arange(0,nx*dx,dx)
    IZ = np.arange(0,nz*dz,dz)
    [IIX,IIZ] = np.meshgrid(IX,IZ)
    # Loop over each trace of the shot gather at src isx
    for igx in tqdm(range(0,ntr)):
        w = peso(TTh,dt,X,Y,igx,isx)
        #mask = w<(1-epsilon)
        trace_win = np.zeros([nz,nx])
        R = np.sqrt(IIZ**2 + (IIX-(igx+isx)/2*dx)**2)
        r_mask = (R==0)
        R[r_mask]= dx/1000
        obli = IIZ/R
        trace_app = taper(ntr,nz,app,isx,igx) 
        
        for j in range(len(window)):
            t = timer[isx,0:nz,0:nx] + timer[igx,0:nz,0:nx] #t_{d}
            twin = t + window[j]
            t2 = (twin<nt)*twin 
            trace1=gather.T[np.ix_([igx],t2.flatten().astype(np.int32))] 
            trace1 = trace1.reshape([nz,nx])*(1-w) 
            trace1 = trace1*trace_app
            trace_win = trace_win+trace1
        
        mig[0:nz,0:nx]=mig[0:nz,0:nx] + trace_win*obli
        
    return mig







def migstack_winapp_diff(files,isx,dx,dz,dt,win,dwin,app,TTh,X,Y,epsilon):
            
    """
    Calcula a migração para vários arquivos (todos os tiros ao longo de uma linha sísmica) com janela (window) e abertura (aperture)
    Stack das imagens migradas de cada tiro
    Considera a função peso w = 1 - w(s,r,t)
    
    Entrada:
    files - lista (array) com os dados sísmicos. 
    isx - posição do tiro
    dx - discretização no eixo x (m)
    dz - discretização no eixo z (m)
    dt - discretização do tempo (s)
    win - (tamanho da janela)/2
    dwin - passo da janela. Preferencialmente, dwin=dt
    app - tamanho da abertura
    TTh - tabela do tempo de trânsito calculada com a função raymodel3
    X - componente X do modelo; X = np.sin(m_theta); X.shape = [nz,nx]
    Y - componente Y do modelo; Y = np.cos(m_theta); Y.shape = [nz,nx]
    
    Saída:
    mig - imagem de difrações a partir de migrações com janela e abertura. Formato: matriz [nt,ntr]
    """

    gathers_shifted = []

    for i in files:
        gather_shifted = phase_shift(i)
        gathers_shifted.append(gather_shifted)

    files = gathers_shifted
    
    timer=np.round(TTh/dt)+1
    #timer=TTh
    migs = []
        
    for count,gather in tqdm(enumerate(files)):
        isx = count
        #print(f"shot {isx}")
    
        window = np.arange(-win,win,dwin)
        [nt,ntr]=gather.shape
        [ntr2,nz,nx]=timer.shape
        if ntr!=ntr2:
            print('Gather e traveltime table tem numero diferente de traços')

        mig=np.zeros([nz,nx])
        mig_final=np.zeros([nz,nx])

        IX = np.arange(0,nx*dx,dx)
        IZ = np.arange(0,nz*dz,dz)
        [IIX,IIZ] = np.meshgrid(IX,IZ)
        # Loop over each trace of the shot gather at src isx
        for igx in range(0,ntr):
            w = peso(TTh,dt,X,Y,igx,isx)
            #mask = w<(1-epsilon)
            trace_win = np.zeros([nz,nx])
            R = np.sqrt(IIZ**2 + (IIX-(igx+isx)/2*dx)**2)
            r_mask = (R==0)
            R[r_mask]= dx/1000
            obli = IIZ/R
            trace_app = taper(ntr,nz,app,isx,igx) 

            for j in range(len(window)):
                t = timer[isx,0:nz,0:nx] + timer[igx,0:nz,0:nx] #t_{d}
                twin = t + window[j]
                t2 = (twin<nt)*twin 
                trace1=gather.T[np.ix_([igx],t2.flatten().astype(np.int32))] 
                trace1 = trace1.reshape([nz,nx])*(1-w) #seção? imshow em um separado
                trace1 = trace1*trace_app
                trace_win = trace_win+trace1

            mig[0:nz,0:nx]=mig[0:nz,0:nx] + trace_win*obli
        
        migs.append(mig)
    
    mig_final = np.add.reduce(migs)
        
    return mig_final



def kirchhoffMigration(gather,isx,dx,dz,dt,win,dwin,app,TTh,X,Y):
    """
    Calculate the Kirchhoff Migration for a single gather
    or the stacked migration for several gathers  in a array.
    Includes the obliquity factor, phase shift, aperture, window
    and weight factor w. 
    
    Parameters
    ----------
    files : lista (array) com os dados sísmicos. 
    isx : posição do tiro
    dx : discretização no eixo x (m)
    dz : discretização no eixo z (m)
    dt : discretização do tempo (s)
    win : (tamanho da janela)/2
    dwin : passo da janela. Preferencialmente, dwin=dt
    app : tamanho da abertura
    TTh : tabela do tempo de trânsito calculada com a função raymodel3
    X : componente X do modelo; X = np.sin(m_theta); X.shape = [nz,nx]
    Y : componente Y do modelo; Y = np.cos(m_theta); Y.shape = [nz,nx]
    
    Returns
    -------
    refl_mig : conventional reflection migration image (w = w(s,r,t))
    diff_mig : diffraction migration image (w = 1)
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
            print('Gather e traveltime table tem numero diferente de traços')

        #mig=np.zeros([nz,nx])
        refl_mig = np.zeros([nz,nx])
        diff_mig = np.zeros([nz,nx])

        IX = np.arange(0,nx*dx,dx)
        IZ = np.arange(0,nz*dz,dz)
        [IIX,IIZ] = np.meshgrid(IX,IZ)
        # Loop over each trace of the shot gather at src isx
        for igx in tqdm(range(0,ntr)):
            w = peso(TTh,dt,X,Y,igx,isx)            
            trace_reflwin = np.zeros([nz,nx])
            trace_diffwin = np.zeros([nz,nx])
            R = np.sqrt(IIZ**2 + (IIX-(igx+isx)/2*dx)**2)
            r_mask = (R==0)
            R[r_mask]= dx/1000
            obli = IIZ/R
            trace_app = taper(ntr,nz,app,isx,igx) 

            for j in range(len(window)): #somar amplitudes da curva de difração com uma janela 
                t = timer[isx,0:nz,0:nx] + timer[igx,0:nz,0:nx] #t_{d}
                twin = t + window[j]
                t2 = (twin<nt)*twin 
                trace1=gather.T[np.ix_([igx],t2.flatten().astype(np.int32))] 
                #trace1 = trace1.reshape([nz,nx])*(w)
                trace_refl1 = trace1.reshape([nz,nx])*(w)
                trace_diff1 = trace1.reshape([nz,nx])*(1-w)
                #trace1 = trace1*trace_app
                trace_refl = trace_refl1*trace_app
                trace_diff = trace_diff1*trace_app
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
        
        #migs = []
        refl_migs = []
        diff_migs = []

        for count,gather in tqdm(enumerate(files)):
            isx = count
            window = np.arange(-win,win,dwin)
            [nt,ntr] = gather.shape
            [ntr2,nz,nx] = timer.shape
            if ntr!=ntr2:
                print('Gather e traveltime table tem numero diferente de traços')

            #mig=np.zeros([nz,nx])
            #mig_final=np.zeros([nz,nx])
            
            refl_mig_isx = np.zeros([nz,nx])
            diff_mig_isx = np.zeros([nz,nx])

            IX = np.arange(0,nx*dx,dx)
            IZ = np.arange(0,nz*dz,dz)
            [IIX,IIZ] = np.meshgrid(IX,IZ)

            for igx in range(0,ntr):
                w = peso(TTh,dt,X,Y,igx,isx)
                #trace_win = np.zeros([nz,nx])
                trace_reflwin = np.zeros([nz,nx])
                trace_diffwin = np.zeros([nz,nx])
                R = np.sqrt(IIZ**2 + (IIX-(igx+isx)/2*dx)**2)
                r_mask = (R==0)
                R[r_mask]= dx/1000
                obli = IIZ/R
                trace_app = taper(ntr,nz,app,isx,igx) 

                for j in range(len(window)):
                    t = timer[isx,0:nz,0:nx] + timer[igx,0:nz,0:nx] #t_{d}
                    twin = t + window[j]
                    t2 = (twin<nt)*twin 
                    trace1 = gather.T[np.ix_([igx],t2.flatten().astype(np.int32))]
                    #trace1 = trace1.reshape([nz,nx])*w 
                    trace_refl1 = trace1.reshape([nz,nx])*(w)
                    trace_diff1 = trace1.reshape([nz,nx])*(1-w)
                    #trace1 = trace1*trace_app
                    trace_refl = trace_refl1*trace_app
                    trace_diff = trace_diff1*trace_app
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







###########TESTE TESTE TESTE TESTE TESTE TESTE

def migration_teste(gather,isx,dx,dz,dt,win,dwin,app,TTh,X,Y,sm):
    """
    Calculate the Kirchhoff Migration for a single gather
    or the stacked migration for several gathers  in a array.
    Includes the obliquity factor, phase shift, aperture, window
    and weight factor w. 
    
    Parameters
    ----------
    files : lista (array) com os dados sísmicos. 
    isx : posição do tiro
    dx : discretização no eixo x (m)
    dz : discretização no eixo z (m)
    dt : discretização do tempo (s)
    win : (tamanho da janela)/2
    dwin : passo da janela. Preferencialmente, dwin=dt
    app : tamanho da abertura
    TTh : tabela do tempo de trânsito calculada com a função raymodel3
    X : componente X do modelo; X = np.sin(m_theta); X.shape = [nz,nx]
    Y : componente Y do modelo; Y = np.cos(m_theta); Y.shape = [nz,nx]
    
    Returns
    -------
    refl_mig : conventional reflection migration image (w = w(s,r,t))
    diff_mig : diffraction migration image (w = 1)
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
            print('Gather e traveltime table tem numero diferente de traços')

        #mig=np.zeros([nz,nx])
        refl_mig = np.zeros([nz,nx])
        diff_mig = np.zeros([nz,nx])

        IX = np.arange(0,nx*dx,dx)
        IZ = np.arange(0,nz*dz,dz)
        [IIX,IIZ] = np.meshgrid(IX,IZ)
        # Loop over each trace of the shot gather at src isx
        for igx in tqdm(range(0,ntr)):
            
            w = peso(TTh,dt,X,Y,igx,isx)
            w = w*sm #semblance como peso na função peso
            w = w**2
            
            trace_reflwin = np.zeros([nz,nx])
            trace_diffwin = np.zeros([nz,nx])
            R = np.sqrt(IIZ**2 + (IIX-(igx+isx)/2*dx)**2)
            r_mask = (R==0)
            R[r_mask]= dx/1000
            obli = IIZ/R
            trace_app = taper(ntr,nz,app,isx,igx) 

            for j in range(len(window)): #somar amplitudes da curva de difração com uma janela 
                t = timer[isx,0:nz,0:nx] + timer[igx,0:nz,0:nx] #t_{d}
                twin = t + window[j]
                t2 = (twin<nt)*twin 
                trace1=gather.T[np.ix_([igx],t2.flatten().astype(np.int32))] 
                #trace1 = trace1.reshape([nz,nx])*(w)
                trace_refl1 = trace1.reshape([nz,nx])*(w)
                trace_diff1 = trace1.reshape([nz,nx])*(1-w)
                #trace1 = trace1*trace_app
                trace_refl = trace_refl1*trace_app
                trace_diff = trace_diff1*trace_app
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
        
        #migs = []
        refl_migs = []
        diff_migs = []

        for count,gather in tqdm(enumerate(files)):
            isx = count
            window = np.arange(-win,win,dwin)
            [nt,ntr] = gather.shape
            [ntr2,nz,nx] = timer.shape
            if ntr!=ntr2:
                print('Gather e traveltime table tem numero diferente de traços')

            #mig=np.zeros([nz,nx])
            #mig_final=np.zeros([nz,nx])
            
            refl_mig_isx = np.zeros([nz,nx])
            diff_mig_isx = np.zeros([nz,nx])

            IX = np.arange(0,nx*dx,dx)
            IZ = np.arange(0,nz*dz,dz)
            [IIX,IIZ] = np.meshgrid(IX,IZ)

            for igx in range(0,ntr):
                w = peso(TTh,dt,X,Y,igx,isx)
                w = w*sm #semblance como peso na função peso
                w = w**2
                trace_reflwin = np.zeros([nz,nx])
                trace_diffwin = np.zeros([nz,nx])
                R = np.sqrt(IIZ**2 + (IIX-(igx+isx)/2*dx)**2)
                r_mask = (R==0)
                R[r_mask]= dx/1000
                obli = IIZ/R
                trace_app = taper(ntr,nz,app,isx,igx) 

                for j in range(len(window)):
                    t = timer[isx,0:nz,0:nx] + timer[igx,0:nz,0:nx] #t_{d}
                    twin = t + window[j]
                    t2 = (twin<nt)*twin 
                    trace1 = gather.T[np.ix_([igx],t2.flatten().astype(np.int32))]
                    #trace1 = trace1.reshape([nz,nx])*w 
                    trace_refl1 = trace1.reshape([nz,nx])*(w)
                    trace_diff1 = trace1.reshape([nz,nx])*(1-w)
                    #trace1 = trace1*trace_app
                    trace_refl = trace_refl1*trace_app
                    trace_diff = trace_diff1*trace_app
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