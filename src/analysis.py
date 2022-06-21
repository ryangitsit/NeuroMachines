#%%
from processing import txt_to_spks
import numpy as np
import os
import matplotlib.pyplot as plt
from plotting import *

from processing import txt_to_spks
from plotting import raster_plot
import json


""""
Simply set the location of the accuracies folder for a sweep on line 36
and run this file.  The output will be a ranked list of configurations
ordered by average accuracy for all classes at the final moment of each
simulation.

ToDo:
 - Auto multiplot
 - Ranking at different timesteps (area under the curve?)
 - PCA
 - 1/f measures
 - Synchonicity measures
"""
#%%
 
sweep  = 'hei_X'

# def weight_pull(sweep):

#     directory = f'results/{sweep}/weights'
#     all_accs = {}

#     for filename in os.listdir(directory):
#         file = os.path.join(directory, filename)
#         print(file[:-4])
#         with open(file, 'rb') as f:
#             accs = np.load(f, allow_pickle=True)
#         all_accs[file[len(directory)+1:-4]] = accs

#     return all_accs

# all_Ws = weight_pull(sweep)
#print(all_Ws['Maass_geo=(randNone_geo[4, 4, 4]_smNone)_N=64_IS=0.3_RS=0.2_ref=0.0_delay=0.0_U=1.0'][0,63])

#%%
# x = []
# y = []
# for i in range(63):
#     for j in range(63):
#         x.append(i)
#         y.append(j)
# for k,W in all_Ws.items():
#     plt.figure(figsize=(16, 8))
#     for i in range(len(x)):
#         plt.plot(x[i], y[i], '.k', ms = W[x[i],y[i]]*15)
    # plt.show()
    # title(f'Synaptic Connections and Weights {k}')
    # xlabel('Source neuron position (um)')
    # ylabel('Target neuron position (um)');

#%%
# sweep  = 'instant_sweep'

def performance_pull(sweep):
    """
    Pull .npy Files for Analysis
    - Iterate over all accuracy files for a given sweep
    - Load them into a dictionary with file name as key
    - Note there is an accuracy array of all time steps for each pattern
    
    Example:
        dict[key] = [acc["A"][t=0],acc["A"][t=1],...][acc["B"][t=0]...],...]
    """
    directory = f'results/{sweep}/performance/accuracies'
    all_accs = {}

    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)

        with open(file, 'rb') as f:
            accs = np.load(f, allow_pickle=True)

        sub = {filename:accs}
        all_accs.update(sub)
        #all_accs[file] = accs

    return all_accs

all_accs = performance_pull(sweep)

#%%
def average_performance(all_accs):
    """
    Average Accuracies Across Patterns
    - Return a dictionary with file (config) names for keys
    - Average accs over time accross patterns for values
    """
    all_avgs = {}
    for file, accs in all_accs.items():
        avg = 0
        accs_array = []
        for acc in accs:
            accs_array.append(acc)
        accs_array = np.array(accs_array)
        avg = np.mean(accs_array, axis=0 )
        sub = {file:avg}
        all_avgs.update(sub)
    return all_avgs

all_avgs = average_performance(all_accs)
print(len(all_avgs['STSP_rnd=(rand0.3_geoNone_smNone)_N=135_IS=0.4_RS=0.3_ref=3.0_delay=1.5_U=0.6.npy']))


#%%

def accs_plots(all_avgs):
    """
    Plot All Average Accuracies

    QUESTION: Why are some avg accs identical for different configs on small experiments?
    """
    plt.figure(figsize=(16, 10))
    for file, avg in all_avgs.items():
        plt.title(f"Prediction Certainty Over Time",fontsize=24)
        plt.plot(avg,label=file[:8])
        plt.ylim(0,1)
        #plt.xlim(20,100)
        plt.xlabel("Time (ms), dt=1ms",fontsize=22)
        plt.ylabel("Ratio of Correctness",fontsize=22)
    #plt.legend()
    plt.show()

accs_plots(all_avgs)

#%%

def best_performance(all_avgs,sweep,write):
    """
    Rank Best Average Performance at Final Time Step
    - Just one of many possible ways to rank performance
    - Take average accuracy at final time step
    - Store in dict with config filename as key and final avg acc as val
    - Sort by top performance
    - Write to .json in performance directory if desired (write==True)
    """
    finals = []
    for file, avgs in all_avgs.items():
        finals.append(avgs[-1])
    ordered = np.argsort(finals)
    key_list = list(all_avgs)
    ordered_name = []
    for i in ordered:
        ordered_name.append(key_list[i])
    ranking = ordered_name[::-1]
    ranked = {}
    for rank in ranking:
        sub = {rank[:-4]:all_avgs[rank][-1]}
        ranked.update(sub)

    if write==True:
        js = json.dumps(ranked)
        path = f'results/{sweep}/performance/{sweep}-rankings.json'
        f = open(path,"w")
        f.write(js)
        f.close()
    return ranked

write  = True
ranked = best_performance(all_avgs,sweep,write)

def best_overall(all_avgs,sweep,write):
    overall = {}
    for file, avgs in all_avgs.items():
        overall[file] = np.mean(avgs)
    overall = dict(sorted(overall.items(), key=lambda item: item[1], reverse=True))

    if write==True:
        js = json.dumps(overall)
        path = f'results/{sweep}/performance/overall-rankings.json'
        f = open(path,"w")
        f.write(js)
        f.close()
    return overall

write  = True
over_ranked = best_overall(all_avgs,sweep,write)

from processing import print_rankings
print_rankings(over_ranked,"Overall",9)
print_rankings(ranked,"Performance",9)

#%%

names = []
print("\n\n     ### Performance Rankings ###\n")
count = 0
for key,value in ranked.items():
    count += 1
    if count < 100:
        print(f"{key}: {value}")
    names.append(key)
print("\n\n")
print(len(names))


#%%
def top_plot(names,patterns,sweep,save):

    """
    Plotting Top 5 Performers and Their Replicas
    - Create a subplot grid
    - For each pattern replica
        - For each Top5 performer
            - Convert top performing names to paths
            - Pull the appropriate spikea
            - Raster plot them into the subplot grid
    """

    fig, axs = plt.subplots(5, 3,figsize=(24,14))
    plt.title("Title")

    top_5 = names[:5]
    print(top_5)
    for i,pattern in enumerate(patterns):
        suffix = "_pat"+pattern+"_rep0.txt"
        prefix = f'results/{sweep}/liquid/spikes/'
        for j,name in enumerate(top_5):
            #print(name+suffix)
            dat, indices, times = txt_to_spks(prefix+name+suffix)
            axs[j, i].plot(times, indices, '.k', ms=.7)
            axs[j, i].set_title(name, size=8)

    for ax in axs.flat:
        ax.set(xlabel='time (ms)', ylabel='neuron index')

    for ax in axs.flat:
        ax.label_outer()

    if save==True:
        path = f'results/{sweep}/performance/bottom_performers.png'
        plt.savefig(path)
    plt.show()

save=True
#patterns=["A","B","C"]
patterns=["ZERO","ONE","TWO"]
top_plot(names,patterns,sweep,save)



# %%
from performance import ranking_analysis, hist_ranked


features = {
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

    'Maass':0,
    'STDP':0,
    'STSP':0,
    'LSTP':0,
    'rnd=':0,
    'geo=':0,
    'smw':0,
    # 'RS=0.01':0,
    # 'RS=0.05':0,
    # 'RS=0.1':0,
    # 'RS=0.1':0,
    # 'RS=0.2':0,
    # 'RS=0.3':0,
    # 'IS=0.1':0,
    # 'IS=0.2':0,
    # 'IS=0.3':0,

    'delay=0.0':0,
    'delay=1.5':0,
    'delay=3.0':0,
    # 'IS=0.1_RS=0.1':0,
    # 'IS=0.2_RS=0.1':0,
    # 'IS=0.3_RS=0.1':0,
    # 'IS=0.1_RS=0.2':0,
    # 'IS=0.2_RS=0.2':0,
    # 'IS=0.3_RS=0.2':0,
    # 'IS=0.1_RS=0.3':0,
    # 'IS=0.2_RS=0.3':0,
    # 'IS=0.3_RS=0.3':0,

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

    'IS=0.12':0,
    'IS=0.14':0,
    'IS=0.15':0,
    'IS=0.16':0,
    'IS=0.18':0,
    'IS=0.2':0,
    'IS=0.3':0,
    'IS=0.4':0,
    'IS=0.5':0,

    # 'geo[16, 2, 2]':0,
    # 'geo[4, 4, 4]':0,
    # 'geo[8, 4, 2]':0,
    
    # 'sm0.0':0,
    # 'sm0.25':0,
    # 'sm0.5':0,
    # 'sm0.75':0,

    # 'ref=0.0':0,
    # 'ref=1.5':0,
    '111':0

}

lim = 20

feat = ranking_analysis(ranked,features,lim)


#%%
keys_list = list(feat)
keys = keys_list[0:12]
print(keys)


hist_ranked(keys,feat)

# %%

# %%
