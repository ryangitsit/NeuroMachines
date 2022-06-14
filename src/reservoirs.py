from brian2 import *
from smallworld import get_smallworld_graph



def reservoir(self):

    neuron_spacing = 50*umetre
    width = self.N/4.0*neuron_spacing
    ref=float(self.refractory)*ms

    eqs = '''
    dv/dt = (-v)/(30*ms) : 1
    x : metre
    y : metre
    z : metre
    '''

    if self.learning=="Maass":
        G = NeuronGroup(self.N, eqs, threshold='v>1', reset='v = 0', refractory=ref, method='exact')
        G.v = 'rand()'
        S = Synapses(G, G, model='w : 1', on_pre='v_post += 0.2*w')

    elif self.learning=="STSP":
        G = NeuronGroup(self.N, eqs, threshold='v>1', reset='v = 0', refractory=ref, method='euler')
        G.v = 'rand()'
        S = Synapses(G, G,
                    '''
                    w : 1
                    du/dt = ((0.5 - u)/(150*ms)) : 1 (clock-driven)
                    dr/dt = ((1 - r)/(20*ms)) : 1 (clock-driven)
                    ''',
                    on_pre='''
                    v_post += w*(r*u)/0.5
                    r += -u*r
                    u += 0.5*(1-u)
                    ''',

                    method='linear'
                    )

    elif self.learning=="STDP":
        G = NeuronGroup(self.N, eqs, threshold='v>1', reset='v = 0', refractory=ref, method='euler')
        G.v = 'rand()'
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
                    ''', method='linear')


    elif self.learning=="LSTP":
        G = NeuronGroup(self.N, eqs, threshold='v>1', reset='v = 0', refractory=ref, method='euler')
        G.v = 'rand()'
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
                    ''', namespace={'U': self.STSP_U}, method='linear')


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
        
    for pre in range(self.N):
        for post in range(self.N):
            S.w[pre,post]=rand()
    
    S.delay = self.delay*ms

    return G, S



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