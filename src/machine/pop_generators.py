from brian2 import *
from topologies import *


def general_generator(configs):

    dic = configs
    for key, value in dic.items():
        str = key
        globals()[str] = value

    print(beta)

    N = neurons
    neuron_spacing = 50*umetre
    width = N/4.0*neuron_spacing
    ref=float(refractory)*ms

    eqs = '''
    dv/dt = (-v)/(30*ms) : 1
    x : metre
    y : metre
    z : metre
    '''

    if learning=="Maass":
        G = NeuronGroup(N, eqs, threshold='v>1', reset='v = 0', refractory=ref, method='exact')
        G.v = 'rand()'
        S = Synapses(G, G, model='w : 1', on_pre='v_post += 0.2*w')

    elif learning=="STSP":
        G = NeuronGroup(N, eqs, threshold='v>1', reset='v = 0', refractory=ref, method='euler')
        G.v = 'rand()'
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

                    method='linear',
                    namespace={'U': STSP_U}
                    )

    elif learning=="STDP":
        G = NeuronGroup(N, eqs, threshold='v>1', reset='v = 0', refractory=ref, method='euler')
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
    else:
        return print("Error: please select learning type.")

    # Sparsity !!
    if topology=="rnd":
        S.connect(condition='i!=j',p=res_sparsity)
        print(f"Random topology generated with probability of connection = {res_sparsity}")
    
    # Sparsity !!
    elif topology=="geo":
        G,S = gen_geometric(G,S,N,dims,lam=None,dist_coeff=res_sparsity)

    # Sparsity !!   
    elif topology=="smw":
        S = gen_small_world(S,N,beta,res_sparsity)

    else:
        print("Error: please select topology type.")
        
    for pre in range(neurons):
        for post in range(neurons):
            S.w[pre,post]=rand()
    
    S.delay = delay*ms

    """"
    parameters still to vary:
        tau = '30*ms'
        neuron_spacing = 50*umetre
        width = N/4.0*neuron_spacing
        tau_d = 20*ms
        tau_f = 150*ms
        U = .7
        Vepsp = 1
        delay_time = 1.5*ms

        tau = '30*ms'
        neuron_spacing = 50*umetre
        width = N/4.0*neuron_spacing

        tau_d = 20*ms
        tau_f = 150*ms
        U = .7
        Vepsp = 1
        delay_time = 1.5*ms

        N = 135
        neuron_spacing = 50*umetre
        width = N/4.0*neuron_spacing
        tau = 30*ms
        c = 0.3
        delay_time = 1.5*ms
    """

    return G, S


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





































### Legacy Methods


def gen_Maass(N,ref,delays,top):

    N = 135
    neuron_spacing = 50*umetre
    width = N/4.0*neuron_spacing
    tau = 30*ms
    c = 0.3
    delay_time = 1.5*ms

    if ref==True:
        refract=3*ms
    else:
        refract=0*ms


    eqs = '''
    dv/dt = (13.5-v)/(30*ms) : 1 (unless refractory)
    x : metre
    '''

    # Initialize Population (parameters can be made less complex)
    G = NeuronGroup(N, eqs, threshold='v>15', reset='v = 13.5', refractory=refract, method='exact')
    G.x = 'i*neuron_spacing'
    G.v = 'rand()*1.5+13.5'
    S = Synapses(G, G, model='w : 1', on_pre='v_post += 0.2*w')
    S.connect(condition='i!=j',p='c*exp(-(x_pre-x_post)**2/(2*width**2))')


    if top == "geo":
        S.connect(condition='i!=j',p='.3*exp(-(x_pre-x_post)**2/(2*width**2))')
    elif top == "rand":
        S.connect(condition='i!=j',p='.1')
    else:
        return print("Error: please select topology type.")

    # Random weights
    for pre in range(N):
        for post in range(N):
            S.w[pre,post]=rand()

    if delays==True:
        S.delay = 1.5*ms

    # M = StateMonitor(G, 'v', record=True)
    # spikemon = SpikeMonitor(G)

    # net = Network(G,S)

    return G, S #, M, spikemon




def gen_p_delta_pop(N):
    """
    The p-delta rule is taken from Auer2001 as per Maass2002.  It employs
    a population of parallel perceptrons with linear "squashing" function
    along with the additional constraint that 
    """
    pass


def gen_STSP(N,ref,delays,top):
    tau = '30*ms'
    neuron_spacing = 50*umetre
    width = N/4.0*neuron_spacing

    tau_d = 20*ms
    tau_f = 150*ms
    U = .7
    Vepsp = 1
    delay_time = 1.5*ms

    if ref==True:
        refract=3*ms
    else:
        refract=0*ms

    eqs = '''
    dv/dt = (13.5-v)/(30*ms) : 1
    x : metre
    '''

    # Neuron has one variable x, its position
    G = NeuronGroup(N, eqs, threshold='v>15', reset='v = 13.5', refractory=refract, method='euler')
    G.x = 'i*neuron_spacing'
    G.v = 'rand()*1.5+13.5'

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


    if top == "geo":
        S.connect(condition='i!=j',p='.3*exp(-(x_pre-x_post)**2/(2*width**2))')
        
    elif top == "rand":
        S.connect(condition='i!=j',p='.1')
    else:
        return print("Error: please select topology type.")
        
    for pre in range(N):
        for post in range(N):
            S.w[pre,post]=rand()

    if delays==True:
        S.delay = 1.5*ms

    return G, S


    
def gen_STDP(N,ref,delays,top):
    tau = '30*ms'
    neuron_spacing = 50*umetre
    width = N/4.0*neuron_spacing

    tau_d = 20*ms
    tau_f = 150*ms
    U = .7
    Vepsp = 1
    delay_time = 1.5*ms

    if ref==True:
        refract=3*ms
    else:
        refract=0*ms

    eqs = '''
    dv/dt = (13.5-v)/(30*ms) : 1
    x : metre
    '''

    # Neuron has one variable x, its position
    G = NeuronGroup(N, eqs, threshold='v>15', reset='v = 13.5', refractory=refract, method='euler')
    G.x = 'i*neuron_spacing'
    G.v = 'rand()*1.5+13.5'

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
    

    if top == "geo":
        S.connect(condition='i!=j',p='.3*exp(-(x_pre-x_post)**2/(2*width**2))')
    elif top == "rand":
        S.connect(condition='i!=j',p='.1')
    else:
        return print("Error: please select topology type.")
        
    for pre in range(N):
        for post in range(N):
            S.w[pre,post]=rand()
        
    if delays==True:
        S.delay = 1.5*ms

    M = StateMonitor(S, ['w', 'apre', 'apost'], record=True)

    return G, S



def gen_readout(N,classes):

    eqs = '''
    dv/dt = (13.5-v)/(30*ms) : 1
    '''

    G = NeuronGroup(classes, eqs, threshold='v>15', reset='v = 13.5', method='euler')

    return G

def gen_pot_current():  

    eqs = '''
    dv/dt = (16-v)/(3*ms) : 1
    '''

    G = NeuronGroup(1, eqs, threshold='v>15', reset='v = 13.5', method='euler')

    return G    

def gen_dep_current():

    eqs = '''
    dv/dt = (-v)/(3*ms) : 1
    '''

    G = NeuronGroup(1, eqs, threshold='v>1', reset='v = 0', method='euler')

    return G   