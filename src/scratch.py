#%%
from processing import read_json, write_dict
import matplotlib.pyplot as plt
import numpy as np
from itertools import repeat

sweep = "SuperSweep4"
#sweep = "SuperSweep_MNIST_asymm"

dirName = f'results/{sweep}/analysis/distance_measures/'
item = 'inter'
dict = read_json(dirName,item)

dirName = f'results/{sweep}/analysis/'
item = 'finals'
finals = read_json(dirName,item)
#%%

def dist_plot(params,dists,over):
    dists = list(repeat([], len(params)))
    colors = ['r','b','g','k']
    param_dict = {}
    for p in params:
        param_dict[p] = []
    count=0
    for x,k in enumerate(over.keys()):
        for i,param in enumerate(params):
            if param in k:
                
                col = i
                if len(dict[k]) == 1:
                    # print("single")
                    dist = dict[k]
                else:
                    # print("sum")
                    dist = sum(dict[k])
                dists[i].append([x,dist])
                param_dict[param].append([x,dist])
                count+=1

    dists = np.array(dists)
    plt.figure(figsize=(15,15))
    for i,(k,v) in enumerate(param_dict.items()):
        param_dict[k] = np.array(param_dict[k])
        plt.plot(param_dict[k][:,0],np.log(param_dict[k][:,1]),'.',color=colors[i],label=params[i])

    plt.legend()
    plt.show()

RULES = [["Maass","STSP","STDP","LSTP"],
["_rnd=","_geo=","_smw="],
["IS=0.1","IS=0.2","IS=0.3"],
["RS=0.1","RS=0.2","RS=0.3"],
["reset","continuous"],
["XTrue","XFalse"]]

for rules in RULES:
    dist_plot(rules,dict,finals)

#%%
