# coding: utf-8

import numpy as np
from tqdm import tqdm

print('Imported SlantStack now')

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


def local_windowS(data1,xwin,zwin,dx,dz,p,tau,z_ini,x_ini):
    """
    Entrada:
    data1: dado, conferir se precisa ser data1.T
    xwin: tamanho da janela em x (grid coord.)
    zwin: tamanho da janela em t (grid coord.)
    dx,dz: discretização do modelo data1
    dp: referente ao angulo (cos(theta)*dt/dx)
    tau:
    z_ini, x_ini (grid coord.)
    
    Saída: 
    m_taumax e m_pmax, em que cada célula armazena os valores máximos de tau e p para aquele ponto no modelo
    """
    
    [ntr1,nz1] = data1.T.shape

    W=data1[z_ini:(z_ini+zwin),x_ini:(x_ini+xwin)] #atenção para não errar aqui

    s, x_grid, z_grid = slant(p,tau,W,dx,dz,z_ini,x_ini)

    return s

def local_window(data1,xwin,zwin,dx,dz,pmin,pmax,dp):
    """
    data1: dado, conferir se precisa ser data1.T
    xwin: tamanho da janela em x (grid coord.)
    zwin: tamanho da janela em t (grid coord.) 
    pmin,pmax,dp: referente ao angulo (cos(theta)*dt/dx)
    dx,dz: discretização do modelo data1
    
    Retorna duas matrizes com dimensões de data1: 
    m_taumax e m_pmax, em que cada célula armazena os valores máximos de tau e p para aquele ponto no modelo
    """
    
    #dp=0.1
    #pmin=-3
    #pmax=3
    p = np.arange(pmin,pmax,dp)
    
    [ntr1,nz1] = data1.T.shape
    
    m_taumax = np.zeros(data1.shape)
    m_pmax = np.zeros(data1.shape)
    s_max = np.zeros(data1.shape)

    
    for i in tqdm(range(0,nz1)): #eixo t
        for j in range(0,ntr1): #eixo x
            
            #Caso A
            if (i-np.int64(zwin/2))<=0:
                z_ini=0 #t_ini em grid
                zwinA=np.int64(zwin/2)+i
                tau = np.arange(z_ini*dz, (z_ini+zwinA)*dz,dz)
                #Caso A1
                if (j-np.int64(xwin/2))<=0:
                    #print("1")
                    x_ini=0
                    xwinA=np.int64(xwin/2)+j
                    smax = 0
                    pmaxS = 0
                    taumaxS = 0
                    for ip in p:
                        for itau in tau:
                            s = local_windowS(data1,xwinA,zwinA,dx,dz,ip,itau,z_ini,x_ini)
                            if s>smax:
                                smax = s
                                pmaxS = ip
                                taumaxS = itau
                    
                    m_taumax[i,j]=taumaxS
                    m_pmax[i,j]=pmaxS
                    s_max[i,j]=smax
                    
                #Caso A2
                elif (j-np.int64(xwin/2))>0 and (j+np.int64(xwin/2))<ntr1:
                    #print("2")
                    x_ini=j-np.int64(xwin/2)
                    xwinA= xwin
                    smax = 0
                    pmaxS = 0
                    taumaxS = 0
                    for ip in p:
                        for itau in tau:
                            s = local_windowS(data1,xwinA,zwinA,dx,dz,ip,itau,z_ini,x_ini)
                            if s>smax:
                                smax = s
                                pmaxS = ip
                                taumaxS = itau
                    
                    m_taumax[i,j]=taumaxS
                    m_pmax[i,j]=pmaxS
                    s_max[i,j]=smax
                
                # Caso A3
                elif (j+np.int64(xwin/2))>=ntr1:
                    #print("3")
                    x_ini=j-np.int64(xwin/2)
                    xwinA= xwin - (j + np.int64(xwin/2) - ntr1)
                    smax = 0
                    pmaxS = 0
                    taumaxS = 0
                    for ip in p:
                        for itau in tau:
                            s = local_windowS(data1,xwinA,zwinA,dx,dz,ip,itau,z_ini,x_ini)
                            if s>smax:
                                smax = s
                                pmaxS = ip
                                taumaxS = itau
                    
                    m_taumax[i,j]=taumaxS
                    m_pmax[i,j]=pmaxS
                    s_max[i,j]=smax
            
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
                    smax = 0
                    pmaxS = 0
                    taumaxS = 0
                    for ip in p:
                        for itau in tau:
                            s = local_windowS(data1,xwinA,zwinA,dx,dz,ip,itau,z_ini,x_ini)
                            if s>smax:
                                smax = s
                                pmaxS = ip
                                taumaxS = itau
                    
                    m_taumax[i,j]=taumaxS
                    m_pmax[i,j]=pmaxS
                    s_max[i,j]=smax
                    
                #Caso B2
                elif (j-np.int64(xwin/2))>0 and (j+np.int64(xwin/2))<ntr1:
                    #print("2")
                    x_ini=j-np.int64(xwin/2)
                    xwinA= xwin
                    smax = 0
                    pmaxS = 0
                    taumaxS = 0
                    for ip in p:
                        for itau in tau:
                            s = local_windowS(data1,xwinA,zwinA,dx,dz,ip,itau,z_ini,x_ini)
                            if s>smax:
                                smax = s
                                pmaxS = ip
                                taumaxS = itau
                    
                    m_taumax[i,j]=taumaxS
                    m_pmax[i,j]=pmaxS
                    s_max[i,j]=smax
                
                # Caso B3
                elif (j+np.int64(xwin/2))>=ntr1:
                    #print("3")
                    x_ini=j-np.int64(xwin/2)
                    xwinA= xwin - (j + np.int64(xwin/2) - ntr1)
                    smax = 0
                    pmaxS = 0
                    taumaxS = 0
                    for ip in p:
                        for itau in tau:
                            s = local_windowS(data1,xwinA,zwinA,dx,dz,ip,itau,z_ini,x_ini)
                            if s>smax:
                                smax = s
                                pmaxS = ip
                                taumaxS = itau
                    
                    m_taumax[i,j]=taumaxS
                    m_pmax[i,j]=pmaxS
                    s_max[i,j]=smax
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
                    smax = 0
                    pmaxS = 0
                    taumaxS = 0
                    for ip in p:
                        for itau in tau:
                            s = local_windowS(data1,xwinA,zwinA,dx,dz,ip,itau,z_ini,x_ini)
                            if s>smax:
                                smax = s
                                pmaxS = ip
                                taumaxS = itau
                    
                    m_taumax[i,j]=taumaxS
                    m_pmax[i,j]=pmaxS
                    s_max[i,j]=smax
                    
                #Caso C2
                elif (j-np.int64(xwin/2))>0 and (j+np.int64(xwin/2))<ntr1:
                    #print("2")
                    x_ini=j-np.int64(xwin/2)
                    xwinA= xwin
                    smax = 0
                    pmaxS = 0
                    taumaxS = 0
                    for ip in p:
                        for itau in tau:
                            s = local_windowS(data1,xwinA,zwinA,dx,dz,ip,itau,z_ini,x_ini)
                            if s>smax:
                                smax = s
                                pmaxS = ip
                                taumaxS = itau
                    
                    m_taumax[i,j]=taumaxS
                    m_pmax[i,j]=pmaxS
                    s_max[i,j]=smax
               
                # Caso C3
                elif (j+np.int64(xwin/2))>=ntr1:
                    #print("3")
                    x_ini=j-np.int64(xwin/2)
                    xwinA= xwin - (j + np.int64(xwin/2) - ntr1)
                    smax = 0
                    pmaxS = 0
                    taumaxS = 0
                    for ip in p:
                        for itau in tau:
                            s = local_windowS(data1,xwinA,zwinA,dx,dz,ip,itau,z_ini,x_ini)
                            if s>smax:
                                smax = s
                                pmaxS = ip
                                taumaxS = itau
                    
                    m_taumax[i,j]=taumaxS
                    m_pmax[i,j]=pmaxS
                    s_max[i,j]=smax
    
    return m_taumax,m_pmax,s_max
