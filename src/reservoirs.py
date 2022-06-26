from brian2 import *
from smallworld import get_smallworld_graph



def reservoir(self):

    neuron_spacing = 50*umetre
    width = self.N/4.0*neuron_spacing
    ref=float(self.refractory)*ms

    eqs = '''
    dv/dt = (-v)/(30*ms) : 1
    ref : second
    x : metre
    y : metre
    z : metre
    '''
  
    G = NeuronGroup(self.N, eqs, threshold='v>15', reset='v = 13.5', refractory='ref', method='exact', dt=100*us)
    G.v = 'rand()*1.5+13.5'

    if self.learning=="Maass":
        S = Synapses(G, G, model='w : 1', on_pre='v_post += w', method='linear', dt=100*us)

    elif self.learning=="STSP":
        S = Synapses(G, G,
                    '''
                    w : 1
                    du/dt = ((U - u)/(150*ms)) : 1 (clock-driven)
                    dr/dt = ((1 - r)/(20*ms)) : 1 (clock-driven)
                    ''',
                    on_pre='''
                    v_post += w*(r*u)/U
                    r += -u*r
                    u += U*(1-u)
                    ''',

                    namespace={'U': self.STSP_U},method='linear', dt=100*us
                    )

    elif self.learning=="STDP":
        S = Synapses(G, G,
                    '''
                    w : 1
                    dapre/dt = -apre/(20*ms) : 1 (clock-driven)
                    dapost/dt = -apost/(20*ms) : 1 (clock-driven)
                    ''',
                    on_pre='''
                    v_post += w
                    apre += 0.01
                    w = clip(w+apost, 0, 0.01)
                    ''',
                    on_post='''
                    apost += -0.01*(20*ms)/(20*ms)*1.05
                    w = clip(w+apre, 0, 0.01)
                    ''', method='linear', dt=100*us)


    elif self.learning=="LSTP":
        S = Synapses(G, G,
                    '''
                    w : 1
                    dapre/dt = -apre/(20*ms) : 1 (clock-driven)
                    dapost/dt = -apost/(20*ms) : 1 (clock-driven)
                    du/dt = ((U - u)/(150*ms)) : 1 (clock-driven)
                    dr/dt = ((1 - r)/(20*ms)) : 1 (clock-driven)
                    ''',
                    on_pre='''
                    v_post += w
                    apre += 0.01
                    w = clip(w+apost, 0, 0.01)
                    v_post += w*(r*u)/U
                    r += -u*r
                    u += U*(1-u)
                    ''',
                    on_post='''
                    apost += -0.01*(20*ms)/(20*ms)*1.05
                    w = clip(w+apre, 0, 0.01)
                    ''', namespace={'U': self.STSP_U}, method='linear', dt=100*us)

    else:
        print("Error: please select learning type.")

    # Sparsity !!
    if self.topology=="rnd":
        S.connect(condition='i!=j',p=self.res_sparsity)
        print(f"Random topology generated with probability of connection = {self.res_sparsity}")
    
    # Sparsity !!
    elif self.topology=="geo":
        G,S = gen_geometric(G,S,self.N,self.dims,lam=None,dist_coeff=self.res_sparsity)

    # Sparsity !!   
    elif self.topology=="smw":
        S = gen_small_world(S,self.N,self.beta,self.res_sparsity)

    else:
        print("Error: please select topology type.")
        
    if self.x_atory == True:
        n_type = np.random.randint(0,100,self.N)
        w_scale = 1
        EE = w_scale * .5 
        EI = w_scale * 2.5
        IE = w_scale * -2 
        II = w_scale * -2 

        EI_split = int(.2*self.N)
        EE_EI_IE_II = np.zeros((4,1))
        for pre in range(self.N):
            for post in range(self.N):
                if n_type[pre] <= EI_split and n_type[post] <= EI_split:
                    S.w[pre,post] = II
                    S.delay[pre,post] = .8*ms
                    G[pre].ref=2*ms

                elif n_type[pre] <= EI_split and n_type[post] > EI_split:
                    S.w[pre,post] = IE
                    S.delay[pre,post] = .8*ms

                elif n_type[pre] > EI_split and n_type[post] <= EI_split:
                    S.w[pre,post] = EI
                    S.delay[pre,post] = .8*ms

                elif n_type[pre] > EI_split and n_type[post] > EI_split:
                    S.w[pre,post] = EE
                    S.delay[pre,post] = 1.5*ms
                    G[pre].ref=3*ms

    else:
        for pre in range(self.N):
            for post in range(self.N):
                S.w[pre,post] = rand()
                G.ref[pre] = self.refractory*ms
        S.delay = self.delay*ms

    # W = np.zeros((self.N,self.N))
    # for i in range(self.N):
    #     for j in range(self.N):
    #         if (len(S.w[i,j])) > 0:
    #             W[i,j] = S.w[i,j]
    # print(W.shape)
    W = 0
    return G, S, W



def gen_geometric(G,S,N,dims,lam,dist_coeff):
    """
    Parameters to vary:
    - dimensions
    - lambda
    - distance coefficient
    """
    neuron_spacing = 50*umetre
    width = N/4.0*neuron_spacing
    for z_i in range(dims[2]):
        for y_i in range(dims[1]):
            for x_i in range(dims[0]):
                idx = z_i*dims[1]*dims[0] + y_i*dims[0] + x_i*1
                G.x[idx] = x_i*neuron_spacing
                G.y[idx] = y_i*neuron_spacing
                G.z[idx] = z_i*neuron_spacing

    # Sweep?
    lam = 2*width
    # Sparsity !!
    S.connect(condition='i!=j',p='dist_coeff*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/(lam**2))')
    print(f"Geomtric topology generated with dimensions {dims}")
    return G, S

def gen_small_world(S,N,beta,res_sparsity):
    """
    Parameters to vary:
    - Beta
    """
    k_over_2 = 2
    sm = get_smallworld_graph(N, k_over_2, beta)
    sm_i = []
    sm_j = []

    for n_i in range(len(sm)):
        for n_j in sm[n_i]:
            sm_i.append(n_i)
            sm_j.append(n_j)
    # Sparsity !!
    for ind in range(len(sm_i)):
        eye = sm_i[ind]
        jay = sm_j[ind]
        S.connect(condition='i==eye and j==jay',p=res_sparsity)
    print(f"Small-world topology generated with beta={beta}")
    return S