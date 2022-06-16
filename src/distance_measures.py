#%%
# from liquid_object_test import LiquidState
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.linalg import norm
from processing import read_in_ranks
import itertools
from sklearn.cluster import KMeans
from processing import read_in_ranks, txt_to_spks, one_hot
#%%

def get_input(sweep,classes,replicas,N,T):

    encoded_inputs = {}
    for pattern in classes:
        for rep in range(replicas):

            location = f"results/{sweep}/inputs/spikes/pat{pattern}_rep{rep}.txt"
            dat,indices,times = txt_to_spks(location)
            hot_matrix = one_hot(N,T,indices,times)
            encoded_inputs[f"{pattern}-{rep}"] = hot_matrix

    return encoded_inputs

sweep = 'new_sweep'
classes = ["A","B","C"]
replicas = 3
N = 100
T = 100
inputs = get_input(sweep, classes, replicas, N, T)

#%%

input_vectors=[]
for pattern in classes:
    for rep in range(replicas):
        input_vectors.append(np.concatenate(inputs[f"{pattern}-{rep}"]))

km = KMeans(3)
clusts = km.fit_predict(input_vectors)
centers = km.cluster_centers_
print(clusts)
# if len(set(clusts[:3]))==1 and len(set(clusts[3:6]))==1 and len(set(clusts[6:9]))==1:

#%%

def get_states(sweep,exp,classes,replicas,N,T):

    encoded_states = {}
    for pattern in classes:
        for rep in range(replicas):
            location = f"results/{sweep}/liquid/spikes/{exp}_pat{pattern}_rep{rep}.txt"
            dat,indices,times = txt_to_spks(location)
            hot_matrix = one_hot(N,T,indices,times)
            encoded_states[f"{pattern}-{rep}"] = hot_matrix

    return encoded_states

##highest performance
#exp = "Maass_geo=(randNone_geo[8, 8, 1]_smNone)__N=64_IS=0.2_RS=0.45_ref=0.0_delay=0.0"

##lowest performance
# exp = "Maass_rnd=(rand0.45_geoNone_smNone)__N=64_IS=0.2_RS=0.45_ref=1.5_delay=0.0"
# states = get_states(sweep,exp,classes,replicas,N,T)


#%%
def complete_distance_measure(inputs,classes,replicas,time):
    euc_dict = {}
    fro_dict = {}
    c = 0
    all_input_patterns=[]
    for i,let1 in enumerate(classes):
        for j in range(replicas):
            all_input_patterns.append((let1,j))
    unique_pairs = list(itertools.combinations(all_input_patterns,2))

    all_euclidean = {}
    all_frobenius = {}

    for pair in unique_pairs:
        euclidean = []
        frobenius = []

        l1=pair[0][0]
        r1=pair[0][1]
        l2=pair[1][0]
        r2=pair[1][1]

        for t in range(time):
            s1 = inputs[f"{l1}-{r1}"][:,t]
            s2 = inputs[f"{l2}-{r2}"][:,t]

            # for log difference
            s1 = np.log(s1+.01)
            s2 = np.log(s2+.01)

            euc = distance.euclidean(s1,s2)
            frob = norm((s1,s2), 'fro')
            euclidean.append(euc)
            frobenius.append(frob)

        all_euclidean[f"{l1}{r1}-{l2}{r2}"] = euclidean
        all_frobenius[f"{l1}{r1}-{l2}{r2}"] = frobenius

    return all_euclidean, all_frobenius

euc_in, fro_in = complete_distance_measure(inputs,classes,3,100)

# %%

def total_separation(dist_dict):
    all_sep = []
    for v in dist_dict.values():
        all_sep.append(np.sum(v))
    total_separation = np.sum(all_sep)
    return total_separation

total_euclidean = total_separation(euc_in)
total_frobenius = total_separation(fro_in)

print(total_euclidean)
print(total_frobenius)

#%%

def grouped_distance_measure(inputs,classes,replicas,time):
    grouped_euclidean = {}
    grouped_frobenius = {}

    between_groups = []

    groups = {}
    for i,let in enumerate(classes):
        groups[let] = []
        for j in range(replicas):
            groups[let].append((let,j))

    for v in groups.values():
        unique_pairs = list(itertools.combinations(v,2))

        group = []
        for pair in unique_pairs:
            euclidean = []
            frobenius = []

            l1=pair[0][0]
            r1=pair[0][1]
            l2=pair[1][0]
            r2=pair[1][1]

            for t in range(time):
                s1 = inputs[f"{l1}-{r1}"][:,t]
                s2 = inputs[f"{l2}-{r2}"][:,t]
                
                # for log difference
                s1 = np.log(s1+.01)
                s2 = np.log(s2+.01)

                euc = distance.euclidean(s1,s2)
                frob = norm((s1,s2), 'fro')
                euclidean.append(euc)
                frobenius.append(frob)

            grouped_euclidean[f"{l1}{r1}-{l2}{r2}"] = euclidean
            grouped_frobenius[f"{l1}{r1}-{l2}{r2}"] = frobenius

            group.append(euclidean)

        between_groups.append(np.mean(group,axis=0))

    between = []
    B = list(itertools.combinations(between_groups,2))
    for b in B:
        between.append(distance.euclidean(b[0],b[1]))
    between = np.sum(between)

    return grouped_euclidean, grouped_euclidean, between

g_e,g_f,between = grouped_distance_measure(inputs,classes,3,100)



#%%


def total_distance_loop(sweep,classes,replicas,N,T):

    tot_EUC = []
    grp_EUC = []
    BET = []

    perf_rankings = read_in_ranks(sweep,f'{sweep}-rankings')
    # perf_rankings = read_in_ranks(sweep,'full_sweep-separation_rankings')

    for k in perf_rankings.keys():
        print(k)
        name = k
        states = get_states(sweep,name,classes,replicas,N,T)
        
        t_euc, t_fro = complete_distance_measure(states,classes,replicas,T)
        g_euc, g_fro, between = grouped_distance_measure(states,classes,replicas,T)
        
        tot_euc = total_separation(t_euc)
        grp_euc = total_separation(g_euc)

        tot_EUC.append(tot_euc)
        grp_EUC.append(grp_euc)

        BET.append(between)

    return tot_EUC, grp_EUC, BET


# ham_in, euc_in, fro_in = distance_measure(inputs,classes)
tot_EUC, grp_EUC, BET = total_distance_loop(sweep,classes,replicas,N,T)



#%%
with open(f'results/{sweep}/analysis/frobenius_total_dist.txt', 'w') as fp:
    for item in FROS:
        fp.write("%s\n" % item)
    print('Done')

#%%
plt.figure(figsize=(16,8))

#plt.plot(tot_EUC,label='total euclidean')
#plt.plot(grp_EUC,label='grouped euclidean')
#plt.plot(BET,label='between euclidean')
#plt.plot((np.array(tot_EUC)/np.array(grp_EUC))[:390],label='total/grouped')
#plt.plot(np.array(tot_EUC)/np.array(BET),label='total/between')

plt.plot(np.array(grp_EUC)/np.array(BET),label='group/between')

plt.legend()
#print(EUCS)

"""
Try comparing variance of groups against total variance
Try comparing ordered pair wise distances of with opw of states 

"""


#%%

rankings = read_in_ranks(sweep,f'{sweep}-rankings')

separation_ratio = {}
for i,(k,v) in enumerate(rankings.items()):
    separation_ratio[k] = tot_EUC[i]/BET[i]

sep_rank = dict(sorted(separation_ratio.items(), key=lambda item: item[1],reverse=True))

for k,v in sep_rank.items():
    print(f"{v} - {k}")

#%%

from performance import ranking_analysis, hist_ranked

features = {
    'Maass':0,
    'STDP':0,
    'STSP':0,
    'LSTP':0,
    'rnd=':0,
    'geo=':0,
    'smw':0,
    'RS=0.01':0,
    'RS=0.05':0,
    'RS=0.1':0,
    'delay=0.0':0,
    'delay=1.5':0,
    'delay=3.0':0,

    'Maass_rnd=':0,
    'Maass_geo=':0,
    'Maass_smw=':0,

    'STDP_rnd=':0,
    'STDP_geo=':0,
    'STDP_smw=':0,

    'STSP_rnd=':0,
    'STSP_geo=':0,
    'STSP_smw=':0,

    'LSTP_rnd=':0,
    'LSTP_geo=':0,
    'LSTP_smw=':0,
    
    'sm0.0':0,
    'sm0.25':0,
    'sm0.5':0,
    'sm0.75':0,

    'ref=0.0':0,
    'ref=1.5':0,

}

lim = 50
feat = ranking_analysis(sep_rank,features,lim)

keys_list = list(feat)
keys = keys_list[13:25]

hist_ranked(keys,feat)


"""
 - Compare both orderings with clusterability
 - Ranking for clusterability?
 - With performance
 - What are the significance of STSP and STDP-smw here?
"""



# %%
# Legacy below this line, ignore
# -------------------------------------------------

def distance_measure(inputs,classes):
    euc_dict = {}
    ham_dict = {}
    fro_dict = {}

    for i,let1 in enumerate(classes[:2]):
        for j,let2 in enumerate(classes[1:]):
            if let1 != let2:
                dists = []
                hams = []
                fros = []
                for t in range(T):
                    s1 = inputs[f"{let1}-0"][:,t]
                    s2 = inputs[f"{let2}-0"][:,t]
                    dist = distance.euclidean(s1,s2)
                    dists.append(dist)
                    ham = distance.hamming(s1,s2)
                    hams.append(ham)
                    fro = norm((s1,s2), 'fro')
                    fros.append(fro)
                euc_dict[f"{let1}-{let2}"] = dists
                ham_dict[f"{let1}-{let2}"] = hams
                fro_dict[f"{let1}-{let2}"] = fros

    #for k,v in ham_dict.items():
        # print(f"Average distance of {k}:")
        # print(f"Hamming: {np.mean(v)}")
        # print(f"Euclidean: {np.mean(euc_dict[k])}")
        # print(f"Frobenius: {np.mean(fro_dict[k])}\n")

    return ham_dict, euc_dict, fro_dict

ham_in, euc_in, fro_in = distance_measure(inputs,classes)
# ham_state, euc_state, fro_state = distance_measure(states,classes)


def distance_comparison(ham_in, euc_in, fro_in,ham_state, euc_state, fro_state):
    all_dicts = [ham_in, euc_in, fro_in,ham_state, euc_state, fro_state]
    for i,(k,v) in enumerate(ham_in.items()):
        h_in = np.mean(ham_in[k])
        e_in = np.mean(euc_in[k])
        f_in = np.mean(fro_in[k])
        h_st = np.mean(ham_state[k])
        e_st = np.mean(euc_state[k])
        f_st = np.mean(fro_state[k])

        # print(f"Input - State Difference for {k}")
        # print(f"Hamming: {h_in - h_st}")
        # print(f"Euclidean: {e_in - e_st}")
        # print(f"Frobenius: {f_in - f_st}\n")

        print(f"Expanse factor for {k}")
        print(f"Hamming: {h_st/h_in}")
        print(f"Euclidean: {e_st/e_in}")
        print(f"Frobenius: {f_st/f_in}\n")


    # for x in all_dicts:
    #     ranked = dict(reversed(sorted(x.items(), key=lambda item: item[1])))
    #     print(f"Ranking Comparison near-to-far:")
    #     for key,value in ranked.items():
    #         print(key)




#distance_comparison(ham_in, euc_in, fro_in,ham_state, euc_state, fro_state)
# %%

def distance_loop(sweep,classes,replicas,N,T):
    HAMS = []
    EUCS = []
    FROS = []
    pats = len(classes)
    directory = f'results/{sweep}/liquid/spikes'

    filename = os.listdir(directory)[1]
    file = os.path.join(directory, filename)
    # dat,indices,times = txt_to_spks(file)
    # length = int(np.ceil(np.max(times)))
    # neurons = np.max(indices) + 1

    # experiments = int(len(os.listdir(directory))/(len(classes)*replicas))

    #steps = np.arange(0,len(os.listdir(directory)),replicas*pats)[:3]

    perf_rankings = read_in_ranks(sweep,'full_sweep-rankings')
    # perf_rankings = read_in_ranks(sweep,'full_sweep-separation_rankings')
    for k in perf_rankings.keys():
        print(k)
        name = k
        states = get_states(sweep,name,classes,replicas,N,T)
        ham_state, euc_state, fro_state = distance_measure(states,classes)
        HAMS.append(ham_state)
        EUCS.append(euc_state)
        FROS.append(fro_state)

    # for exp in steps:            
    #     name = os.listdir(directory)[exp][:-len('_patA_rep0.txt')]
    #     print(name)
    #     states = get_states(sweep,name,classes,replicas,N,T)
    #     ham_state, euc_state, fro_state = distance_measure(states,classes)
    #     HAMS.append(ham_state)
    #     EUCS.append(euc_state)
    #     FROS.append(fro_state)

    return HAMS, EUCS, FROS

sweep = 'STSP_sweep'
N = 100
T = 100
classes = ["A","B","C"]

# ham_in, euc_in, fro_in = distance_measure(inputs,classes)
SHAMS, SEUCS, SFROS = distance_loop(sweep,classes,replicas,N,T)


#%%

def distance_plot(LIST,label,list):
    dists_AB=[]
    dists_AC=[]
    dists_BC=[]
    for H in LIST:
        for k,v in H.items():
            #print(np.mean(fro_in[k]))
            # if k == "A-B":
            #     dists_AB.append(np.mean(v)/np.mean(list[k]))
            # if k == "A-C":
            #     dists_AC.append(np.mean(v)/np.mean(list[k]))
            # if k == "B-C":
            #     dists_BC.append(np.mean(v)/np.mean(list[k]))
            if k == "A-B":
                dists_AB.append(np.mean(v))
            if k == "A-C":
                dists_AC.append(np.mean(v))
            if k == "B-C":
                dists_BC.append(np.mean(v))
                
    LIST_mean = np.mean([dists_AB,dists_AC,dists_BC], axis=0)
    plt.plot(LIST_mean, label=label)

plt.figure(figsize=(16,10))
plt.title("Distance Growth Coefficient State/Input")
#distance_plot(HAMS,'Hamming',ham_in)
distance_plot(SEUCS,'Euclidean',euc_in)
distance_plot(SFROS,'Frobenius',fro_in)
plt.legend()
# %%
