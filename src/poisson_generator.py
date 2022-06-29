from brian2 import *
import numpy as np


def gen_poisson_pattern(channels, rate_low, rate_high, sim):
    """
    Poisson Pattern Generator
    - Creates a random vector of length 'channels'
    - Uses each value of this vector a neurons poisson spiking rate
    - Uses Brians built in poisson spike generator function
        - Would be faster to just use numpy
    - Does one neuron at a time and returns spiking times and indices
    """
    spikes_i = []
    spikes_t = []

    #rate_low = randint(1,50)
    # rate_high=rate_low+1
    
    rand_rates = np.random.randint(rate_low,rate_high, channels)
    print(rand_rates)

    for input in range(channels):
        input_rate=(rand_rates[input])*Hz
        P = PoissonGroup(1, rates=input_rate)
        MP = SpikeMonitor(P)
        net = Network(P, MP)
        net.run(sim*ms)
        spikes_i = np.append(spikes_i, MP.i[:]+input)
        spikes_t = np.append(spikes_t, MP.t[:])
    indices = array(spikes_i)
    times = array(spikes_t)*1000
    return rand_rates, indices, times

def create_jitter(sigma,times):
    """
    Time Jitter
    - Creates a Gaussian time jitter with standard dev = sigma
    - Acts on each spike in a time array 
    - Returns new 'jittered' spike timing values
    """
    mu = 0
    # sigma = 5
    for t in range(len(times)):
        spike_move = np.random.normal(mu, sigma)
        if spike_move > 0 and spike_move <= 500:
            times[t] += spike_move
    return times




### Legacy - Ignore

def generate_poisson(N,n,rate_low,rate_high):
    P = []
    for i in range(n):
        rand_rate = np.random.randint(rate_low,rate_high)
        P.append(PoissonGroup(N, np.arange(1)*Hz + (rand_rate)*Hz))
    # SP = Synapses(Po, G, on_pre='v+=.5')
    # SP.connect(j='i')
    print(len(P))
    return(P)

def gen_poisson_basic(G,N,rate,spike):
    spike=0.5
    P = PoissonGroup(N, np.arange(rate)*Hz + 10*Hz)
    SP = Synapses(P, G, on_pre='v+=1')
    SP.connect(j='i')
    return P,SP
