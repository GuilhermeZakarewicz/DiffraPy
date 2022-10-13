# coding: utf-8

import numpy as np

print('Imported SlantStack now')



def slant(p,tau,data1,dx,dt,t_ini,x_ini):
    # Retorna a semblance da reta dada por p, tau
    #p = angulo (s/m)
    #tau = tempo (s)
    # data1 = dado [s,m]
    # dt [s]
    # x_ini [coordenada do grid]
    
    [nt1,ntr1]=data1.shape
    dataC = data1.copy()
    #ct2= nt1/2
    #cx2= (ntr1)*dx/2
    x = np.arange(x_ini*dx,(x_ini+ntr1)*dx,dx)
    t = tau + p*x
    #x = np.arange(0,ntr1*dx,dx)
    #t = tau + p*x
    
    t_grid = np.int64(np.round(t/dt)-t_ini)
    x_grid = np.int64((np.round(x/dx))-x_ini)
    s_n = 0
    s_d = 0
    
    plot = "no"  
    
    for k in range(ntr1):
        
        if (t_grid[k]<nt1) & (t_grid[k]>=0):
            #print(f"dataC[t_grid[k],x_grid[k]]={dataC[t_grid[k],x_grid[k]]}")
            #print(f"k={k}")
            #print(f"t_grid[k]={t_grid[k]}")
            #print(f"x_grid[k]={x_grid[k]}")
            
            s_n += dataC[t_grid[k],x_grid[k]]
            s_d += dataC[t_grid[k],x_grid[k]]**2
            
            if plot == "yes":
                dataCC=data1.copy()
                dataCC[t_grid[k],x_grid[k]] = .5
                plt.imshow(dataC, extent=[0, ntr1,nt1*dt, 0])
                plt.imshow(dataCC, extent=[0, ntr1,nt1*dt, 0])
                plt.axis('auto')
                plt.plot(x,t,'r',label="t (x) = tau + p*x \nt (x) = %s + %s*x" % (tau,p))
                plt.legend()
                #plt.ylim([nt*dt,0])
                #plt.xlim([0,ntr])
                plt.colorbar()
                plt.show()
                
        #else:
        #    print(f"t_grid[k]={t_grid[k]}")
    
    if s_d == 0:
        #print('sd = 0, sn =',s_n)
        #print(f"p={p}")
        #print(f"tau={tau}")
        #print(f"dataC[t_grid[k],x_grid[k]]={dataC[t_grid[k],x_grid[k]]}")
        s=0 #posso fazer isso?
    
    else:
        s = s_n**2/ntr1/s_d

        
    return s, x_grid, t_grid
    #return s, x, t

def slant_stack(pmin,pmax,dp,taumin, taumax, dtau,data1,dx,dt,t_ini,x_ini):
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


def local_windowS(data1,xwin,twin,dx,dt,p,tau,t_ini,x_ini):
    """
    data1: dado, conferir se precisa ser data1.T
    xwin: tamanho da janela em x (grid coord.)
    twin: tamanho da janela em t (grid coord.) 
    pmin,pmax,dp: referente ao angulo (cos(theta)*dt/dx)
    dx,dt: discretização do modelo data1
    
    t_ini, x_ini (grid coord.)
    Retorna duas matrizes com dimensões de data1: 
    m_taumax e m_pmax, em que cada célula armazena os valores máximos de tau e p para aquele ponto no modelo
    """
    
    
    #[ntr1,nt1]=data1.shape
    [ntr1,nt1] = data1.T.shape
    
    #m_taumax=np.zeros([ntr1,nt1])
    #m_taumax = np.zeros(data1.shape)
    #m_pmax = np.zeros(data1.shape)
    
    #for i in tqdm(range(0,nt1)): #0 a 61, eixo t
     #   for j in range(0,ntr1): #0 a 32, eixo x
            
    
    #x_ini=0
    #x_fim=(j+xwin)*dx
    #W=data1[0:(j+xwin),0:(i+twin)]#aqui tem chance de eu sequelar...
    W=data1[t_ini:(t_ini+twin),x_ini:(x_ini+xwin)]#aqui tem chance de eu sequelar...
    #print('W.shape =',W.shape)

    s, x_grid, t_grid = slant(p,tau,W,dx,dt,t_ini,x_ini)
    
    #plt.imshow(W, extent=[x_ini*dx,(x_ini+xwin)*dx,(t_ini+twin)*dt,t_ini*dt],aspect='auto')
    #plt.plot((x_ini+ x_grid)*dx,(t_ini+t_grid)*dt,'r')
    return s

def local_window(data1,xwin,twin,dx,dt):
    """
    data1: dado, conferir se precisa ser data1.T
    xwin: tamanho da janela em x (grid coord.)
    twin: tamanho da janela em t (grid coord.) 
    pmin,pmax,dp: referente ao angulo (cos(theta)*dt/dx)
    dx,dt: discretização do modelo data1
    
    Retorna duas matrizes com dimensões de data1: 
    m_taumax e m_pmax, em que cada célula armazena os valores máximos de tau e p para aquele ponto no modelo
    """
    
    dp=0.0001
    pmin=-0.0032
    pmax=0.0032
    p = np.arange(pmin,pmax,dp)
    
    #[ntr1,nt1]=data1.shape
    [ntr1,nt1] = data1.T.shape
    
    #m_taumax=np.zeros([ntr1,nt1])
    m_taumax = np.zeros(data1.shape)
    m_pmax = np.zeros(data1.shape)
    s_max = np.zeros(data1.shape)

    
    for i in tqdm(range(0,nt1)): #0 a 61, eixo t
        for j in range(0,ntr1): #0 a 32, eixo x
            #Caso A
            if (i-np.int64(twin/2))<=0:
                t_ini=0 #t_ini em grid
                twinA=np.int64(twin/2)+i
                tau = np.arange(t_ini*dt, (t_ini+twinA)*dt,dt)
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
                            s = local_windowS(data1,xwinA,twinA,dx,dt,ip,itau,t_ini,x_ini)
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
                            s = local_windowS(data1,xwinA,twinA,dx,dt,ip,itau,t_ini,x_ini)
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
                            s = local_windowS(data1,xwinA,twinA,dx,dt,ip,itau,t_ini,x_ini)
                            if s>smax:
                                smax = s
                                pmaxS = ip
                                taumaxS = itau
                    
                    m_taumax[i,j]=taumaxS
                    m_pmax[i,j]=pmaxS
                    s_max[i,j]=smax
            
            #Caso B
            elif (i+np.int64(twin/2))>=nt1:
                t_ini=(i-np.int64(twin/2))
                twinA=twin - (i + np.int64(twin/2) - nt1)
                tau = np.arange(t_ini*dt, (t_ini+twinA)*dt,dt)
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
                            s = local_windowS(data1,xwinA,twinA,dx,dt,ip,itau,t_ini,x_ini)
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
                            s = local_windowS(data1,xwinA,twinA,dx,dt,ip,itau,t_ini,x_ini)
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
                            s = local_windowS(data1,xwinA,twinA,dx,dt,ip,itau,t_ini,x_ini)
                            if s>smax:
                                smax = s
                                pmaxS = ip
                                taumaxS = itau
                    
                    m_taumax[i,j]=taumaxS
                    m_pmax[i,j]=pmaxS
                    s_max[i,j]=smax
            # Caso C 
            elif (i-np.int64(twin/2))>0 and (i+np.int64(twin/2))<nt1:
                t_ini=(i-np.int64(twin/2))
                twinA = twin
                tau = np.arange(t_ini*dt, (t_ini+twinA)*dt,dt)
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
                            s = local_windowS(data1,xwinA,twinA,dx,dt,ip,itau,t_ini,x_ini)
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
                            s = local_windowS(data1,xwinA,twinA,dx,dt,ip,itau,t_ini,x_ini)
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
                            s = local_windowS(data1,xwinA,twinA,dx,dt,ip,itau,t_ini,x_ini)
                            if s>smax:
                                smax = s
                                pmaxS = ip
                                taumaxS = itau
                    
                    m_taumax[i,j]=taumaxS
                    m_pmax[i,j]=pmaxS
                    s_max[i,j]=smax
    
    return m_taumax,m_pmax,s_max

