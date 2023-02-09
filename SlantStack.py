# coding: utf-8

import numpy as np
from tqdm import tqdm

print('Imported SlantStack_v2')

def slant(p,tau,data1,dx,dz,z_ini,x_ini):
    """
    Calcula a semblance da reta dada definida por (tau + p*x)
    p - tangente de theta (prof./dist.)
    tau - profunidade (m)
    data1 - dado [s,m]
    dz -
    x_ini - [coordenada do grid]
    z_ini -
    Saída:
    s -
    x_grid -
    z_grid -
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
    Retorna um painel de semblane
    e a semblance máxima para um conjunto de retas
    pmin, pmax (angulos minimo e maximo)
    taumin taumax (tempos -- x=0 -- minimo e maximo (em segundos))
    data1, dx, e dt referem-se ao dado
    x_ini é a primeira amostra em x [grid coord]
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
    Calculates the semblance for each image point of the input data.
    
    Input:
    data1 - reflectivity model 
    x_ini - (grid coord) initial coordinate of x-axis
    z_ini - (grid coord) initial coordinate of z-axis
    dx,dz -
    x0,z0 -
    p -
    
    Output:
    s -
    x_grid -
    z_grid -
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
    Calcula a semblance para o ponto imagem para diferentes inclinações (pmin,pmax,dp)
    
    Entrada:
    data1 -
    x_ini -
    z_ini -
    dx -
    x0 -
    y0 -
    pmin -
    pmax -
    dp -
    
    Saída:
    Smax - 
    pmax - inclinação que gera a semblance máxima
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
    Define a nova matriz a partir da janela definida para cálculo do slant_local_p()
    
    Entrada:
    data1: dado, conferir se precisa ser data1.T
    xwin: tamanho da janela em x (grid coord.)
    zwin: tamanho da janela em t (grid coord.)
    dx,dz: discretização do modelo data1
    dp: referente ao angulo (cos(theta)*dt/dx)
    tau:
    z_ini, x_ini: (grid coord.)
    
    Saída: 
    m_taumax e m_pmax, em que cada célula armazena os valores máximos de tau e p para aquele ponto no modelo
    """
    
    W=data1[z_ini:(z_ini+zwin),x_ini:(x_ini+xwin)] 
    s, pm = slant_local_p(W,x_ini,z_ini,dx,dz,x0,z0,pmin,pmax,dp)

    return s, pm


def local_window(data1,xwin,zwin,x_ini,z_ini,dx,dz,pmin,pmax,dp):
    """
    data1: dado, conferir se precisa ser data1.T
    xwin: tamanho da janela em x (grid coord.)
    zwin: tamanho da janela em t (grid coord.) 
    pmin,pmax,dp: referente ao angulo (cos(theta)*dt/dx)
    dx,dz: discretização do modelo data1
    
    Retorna duas matrizes com dimensões de data1: 
    m_taumax e m_pmax, em que cada célula armazena os valores máximos de tau e p para aquele ponto no modelo
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