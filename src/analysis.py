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
Simply set the location of the accufracies folder for a sweep on line 36
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

sweep  = 'dense_sweep'
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

print(ranked)

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
def top_plot(names,sweep,save):

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
    patterns=["A","B","C"]

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

            # legacy from old naming format, ignore
            '''
            # txt_file = prefix+name[:-4]+suffix
            # if name[0]=="M":
            #     under = txt_file[:5+len(prefix)]+"_"+txt_file[5+len(prefix):]
            # else:
            #     under = txt_file[:4+len(prefix)]+"_"+txt_file[4+len(prefix):]
            # try:
            #     dat, indices, times = txt_to_spks(under)
            # except:
            #     if under[61] == "_":
            #         dunder = under[:61]+under[62:]
            #     else:
            #         dunder = under[:62]+under[63:]
            #     dat, indices, times = txt_to_spks(dunder)
            '''

    for ax in axs.flat:
        ax.set(xlabel='time (ms)', ylabel='neuron index')

    for ax in axs.flat:
        ax.label_outer()

    if save==True:
        path = f'results/{sweep}/performance/bottom_performers.png'
        plt.savefig(path)
    plt.show()

save=True
top_plot(names,sweep,save)



# %%
from performance import ranking_analysis

features = {
    'Maass':0,
    'STDP':0,
    'STSP':0,
    'rnd=':0,
    'geo=':0,
    'smw':0,

    'RS=0.15':0,
    'RS=0.3':0,
    'RS=0.45':0,
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
    
    'sm0.0':0,
    'sm0.25':0,
    'sm0.5':0,
    'sm0.75':0,

    'ref=0.0':0,
    'ref=1.5':0,

}


lim = 93
feat = ranking_analysis(names,features,lim)

#%%
def hist_ranked(keys,feat):
    labels = ['Maass', 'STDP','STSP']

    RND=[]
    GEO=[]
    SMW=[]
    for i in range(3):
        RND.append(feat[keys[3*i]]/100)
        GEO.append(feat[keys[3*i+1]]/100)
        SMW.append(feat[keys[3*i+2]]*.75/100)


    #%%
    x = np.arange(len(labels))  # the label locations
    print(x)
    x= np.array([0.5,1,1.5])
    width = 0.1  # the width of the bars

    #fig, plt = plt.subplots()
    plt.figure(figsize=(7,7))
    plt.style.use('seaborn-muted')

    rects1 = plt.bar(x - width-.01, RND, width, label='random')
    rects2 = plt.bar(x, GEO, width, label='geometric')
    rects3 = plt.bar(x + width+.01, SMW, width, label='small-world')


    plt.ylabel('Percent Present',fontsize=18)
    plt.title('Clusterable Configurations',fontsize=22)
    plt.xticks(x, labels,fontsize=18)
    plt.legend(fontsize=20) # using a size in points
    plt.legend(fontsize="x-large") # using a named size
    # plt.bar_label(rects1, padding=3)
    # plt.bar_label(rects2, padding=3)

    plt.tight_layout()

    plt.show()


keys_list = list(feat)
keys = keys_list[12:21]

hist_ranked(keys,feat)