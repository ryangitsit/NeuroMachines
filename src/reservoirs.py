from brian2 import *
from smallworld import get_smallworld_graph



def reservoir(config):

    neuron_spacing = 50*umetre
    width = config.neurons/4.0*neuron_spacing
    ref=float(config.refractory)*ms

    # define leaky differential equation
    # spacial dimensions for geometric topologies
    eqs = '''
    dv/dt = (-v)/(30*ms) : 1
    ref : second
    x : metre
    y : metre
    z : metre
    '''
    
    # Generate neuron population with typical LSM baseline/thresholds
    G = NeuronGroup(config.neurons, eqs, threshold='v>15', reset='v = 13.5', refractory='ref', method='exact', dt=config.DT*us)
    
    # Initialize voltages radomly within baseline/thresh range
    G.v = 'rand()*1.5+13.5'

    # Simple LIF model, v increases by w for all pre-synaptic spikes
    if config.learning=="Maass":
        S = Synapses(G, G, model='w : 1', on_pre='v_post += w', method='linear', dt=config.DT*us)

    # Short-Term Synaptic Plasticity
    # Parameters from Mongillo2008
    # With pre-firing, u increases and r depletes
    # higher u results in facilitation through great v increases
    # smaller r means depression through the same v update
    # therefore frequent firing results in facilliation then depression
    elif config.learning=="STSP":
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

                    namespace={'U': config.STSP_U},method='linear', dt=config.DT*us
                    )

    # Spike Timing Dependent Plasticity
    # firing traces associated with pre and post synaptic firing
    # weight updates according to association between these firings
    elif config.learning=="STDP":
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
                    ''', method='linear', dt=config.DT*us)

    # A straightforward superposition of STDP and STSP
    elif config.learning=="LSTP":
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
                    ''', namespace={'U': config.STSP_U}, method='linear', dt=config.DT*us)

    else:
        print("Error: please select learning type.")

    # note here reservoir sparsity defines probability of any two neurons
    # being connected
    if config.topology=="rnd":
        S.connect(condition='i!=j',p=config.res_sparsity)
        print(f"Random topology generated with probability of connection = {config.res_sparsity}")
    
    # here res_sparsity defines the lambda parameter
    elif config.topology=="geo":
        #G,S = gen_geometric(G,S,config.neurons,config.dims,lam=None,dist_coeff=config.res_sparsity)
        G,S = gen_geometric(G,S,config.neurons,config.dims,lam=config.res_sparsity,dist_coeff=None)
  
    # here res_sparsity defines the k/2 parameter
    elif config.topology=="smw":
        S = gen_small_world(S,config.neurons,config.beta,config.res_sparsity)

    else:
        print("Error: please select topology type.")
        
    if config.x_atory == True:
        n_type = np.random.randint(0,100,config.neurons)
        w_scale = 1
        EE = w_scale * .5 
        EI = w_scale * 2.5
        IE = w_scale * -2 
        II = w_scale * -2 

        EI_split = int(.2*config.neurons)
        EE_EI_IE_II = np.zeros((4,1))
        for pre in range(config.neurons):
            for post in range(config.neurons):
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
        W  = S.w

    else:
        S.w = W = np.random.rand(len(S.w))
        G.ref = config.refractory*ms
        S.delay = config.delay*ms

    return G, S, W



def gen_geometric(G,S,neurons,dims,lam,dist_coeff):
    """
    Parameters to vary:
    - dimensions
    - lambda
    - distance coefficient
    """
    neuron_spacing = 50*umetre
    width = neurons/4.0*neuron_spacing
    for z_i in range(dims[2]):
        for y_i in range(dims[1]):
            for x_i in range(dims[0]):
                idx = z_i*dims[1]*dims[0] + y_i*dims[0] + x_i*1
                G.x[idx] = x_i*neuron_spacing
                G.y[idx] = y_i*neuron_spacing
                G.z[idx] = z_i*neuron_spacing

    # Sweep?
    #lam = 2*width
    lam = lam * 10 * width
    dist_coeff = 0.3
    S.connect(p='dist_coeff*exp(-((x_pre-x_post)**2+(y_pre-y_post)**2+(z_pre-z_post)**2)/(lam**2))')
    print(f"Geomtric topology generated with dimensions {dims}")
    return G, S

def gen_small_world(S,neurons,beta,res_sparsity):
    """
    Parameters to vary:
    - Beta
    """
    # k_over_2 = 2
    k_over_2 = int(res_sparsity*10)
    sm = get_smallworld_graph(neurons, k_over_2, beta)
    sm_i = []
    sm_j = []

    for n_i in range(len(sm)):
        for n_j in sm[n_i]:
            sm_i.append(n_i)
            sm_j.append(n_j)

    for ind in range(len(sm_i)):
        eye = sm_i[ind]
        jay = sm_j[ind]
        S.connect(condition='i==eye and j==jay',p=res_sparsity)
    print(f"Small-world topology generated with beta={beta}")
    return S