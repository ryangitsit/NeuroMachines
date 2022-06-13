from brian2 import *
from smallworld import get_smallworld_graph


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