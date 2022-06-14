
#%%
from typing import KeysView
import numpy as np
import matplotlib.pyplot as plt
from processing import read_in_ranks, print_rankings


# sweep = 'full_sweep'
# ranked = read_in_ranks(sweep,'full_sweep-rankings')

sweep = 'poisson'
ranked = read_in_ranks(sweep,f'{sweep}-rankings')
#print_rankings(ranked,"Performance",100)


def ranking_analysis(rankings,dict,lim):
    for i,(key,value) in enumerate(rankings.items()):
        if i < lim:
            for k,v in dict.items():
                if k in key:
                    dict[k] += 1
    for i,(k,v )in enumerate(dict.items()):
        if (i)%3==0:
            print("-")
        print(f'{k}: {v}')
    return dict

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
feat = ranking_analysis(ranked,features,lim)


#%%
keys_list = list(feat)
keys = keys_list[13:25]
print(keys)
#%%
def hist_ranked(keys,feat):
    labels = ['Maass', 'STDP','STSP','LSTP']

    RND=[]
    GEO=[]
    SMW=[]
    for i in range(4):
        RND.append(feat[keys[3*i]]/100)
        GEO.append(feat[keys[3*i+1]]/100)
        SMW.append(feat[keys[3*i+2]]/100)

    print(GEO)
    #%%
    x = np.arange(len(labels)) 
    # print(x)
    # x= np.array([0.5,1,1.5])

    width = 0.18  

    #fig, plt = plt.subplots()
    plt.figure(figsize=(7,7))
    plt.style.use('seaborn-muted')

    rects1 = plt.bar(x - width-.01, RND, width, label='random')
    rects2 = plt.bar(x, GEO, width, label='geometric')
    rects3 = plt.bar(x + width+.01, SMW, width, label='small-world')


    plt.ylabel('Percent Present',fontsize=18)
    plt.title('Configurations in top 100 Performers',fontsize=22)
    plt.xticks(x, labels,fontsize=18)
    plt.legend(fontsize=20) # using a size in points
    plt.legend(fontsize="x-large") # using a named size
    # plt.bar_label(rects1, padding=3)
    # plt.bar_label(rects2, padding=3)

    plt.tight_layout()

    plt.show()

hist_ranked(keys,feat)

#%%
