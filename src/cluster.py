#%%
import json
from processing import read_json,write_json
from sklearn.cluster import KMeans
import numpy as np


## PCs
# type = 'PCs'
# dirName = 'results/full_sweep/performance'
# item='full_sweep_99_stacks'

## Full Dimensionality
type = 'groups'
dirName = 'results/sparse_sweep/analysis'
#dirName = 'results/full_sweep/analysis'

item = "m=99_groups"

j_stacks = read_json(dirName,item)


sweep = 'sparse_sweep'
write=False

def cluster_check(dict,dirName,sweep,type,write):
    clusters = {}
    count=0
    success=0
    for key,value in j_stacks.items():
        #print(key)

        if type == 'groups':
            value = np.concatenate(value)

        km = KMeans(3)
        clusts = km.fit_predict(value)
        centers = km.cluster_centers_
        #print(clusts)

        if len(set(clusts[:3]))==1 and len(set(clusts[3:6]))==1 and len(set(clusts[6:9]))==1:
            #print('success')
            clusters[key] = 1
            count+=1
            success+=1
            #print(key)
        
        else:
            clusters[key] = 0
            count+=1

    if write==True:
        dirName=f'results/{sweep}/performance/clustering/'
        exp='-'
        name='clusters'
        write_json(clusters,dirName,exp,name)

    print(f'Total success = {success}/{count}')

    return clusters

clusters = cluster_check(j_stacks,dirName,sweep,type,write)

mcount = 0
stdp = 0
stsp = 0
smw = 0
rnd=0
geo=0

clusterable = {}
for k,v in clusters.items():
    if v == 1: # or v ==0:
        #print(k)
        clusterable[k] = 1
        if k[:5] == 'Maass':
            mcount+=1
        elif k[:4] == 'STDP':
            stdp+=1
        elif k[:4] == 'STSP':
            stsp+=1
        if k[5:8] == "smw" or k[6:9] == "smw":
            smw+=1
        if k[5:8] == "rnd" or k[6:9] == "rnd":
            rnd+=1
        if k[5:8] == "geo" or k[6:9] == "geo":
            geo+=1

print(f'Maass = {mcount}\nSTDP = {stdp}\nSTSP = {stsp}\nSmall-world = {smw}\nRandom = {rnd}\nGeo = {geo}')

#%%

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
feat = ranking_analysis(clusterable,features,lim)

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


#%%

# import matplotlib.pyplot as plt
# from processing import unit_dict

# def cluster_plotting(pcs,clusts,centers):

#     classes=["A","B","C"]
#     replicas = 3

#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     fig.set_figheight(15)
#     fig.set_figwidth(15)

#     markers = ["$A$","$B$","$C$"]
#     colors = ['r','g','b']
    
#     count = 0
#     #print(pcs)
#     for key,val in pcs.items():
#         count = 0
#         for i,c in enumerate(classes):
#             for j in range(replicas):
#                 # print(count, i)
#                 pc = val[count]
#                 ax.scatter(pc[0],pc[1],pc[2], marker=markers[i],color=colors[i],s=200,label=classes[i])
#                 count+=1

#         means = {}
#         for i,let in enumerate(classes):
#             #print(replicas*i+replicas)
#             mean = np.mean((pcs[key][replicas*i:replicas*i+replicas]),axis=0)
#             means[let] = mean
#             ax.scatter(mean[0],mean[1],mean[2], marker='o',color=colors[i],s=200,label=classes[i])
        

#     # ccolors = ['orange','purple','yellow']
#     # for i in range(3):
#     #     count = 0
#     #     #print(i)
#     #     ax.scatter(centers[i, 0],
#     #         centers[i, 1],
#     #         centers[i, 2]+2,
#     #         s = 250,
#     #         marker='o',
#     #         c=ccolors[clusts[i+i*3]],
#     #         label='centroids')
#     #     print(clusts)
#     #     print(clusts[i+i*3])
#     #     for j in range(3):
#     #         pc = pcs[list(pcs.keys())[0]][count]
#     #         ax.scatter(pc[0],pc[1],pc[2], marker='*',color=ccolors[count],s=500,label=classes[i])
#     #         count +=1 
#     ax.scatter(km.cluster_centers_[:, 0],
#                 km.cluster_centers_[:, 1],
#                 km.cluster_centers_[:, 2],
#                 s = 50,
#                 marker='*',
#                 c='w',
#                 label='centroids')

#     plt.xlim(-5,5)
#     plt.ylim(-5,5)
#     # plt.legend()
#     ax.set_zlim(-5,5)

#     ax.set_xlabel('Replica 0 Component')
#     ax.set_ylabel('Replica 1 Component')
#     ax.set_zlabel('Replica 2 Component')

#     plt.show()
#     plt.close()

# single = unit_dict(j_stacks,'Maass_geo=(randNone_geo[8, 8, 1]_smNone)__N=64_IS=0.2_RS=0.3_ref=1.5_delay=0.0')
# km = KMeans(3)
# clusts = km.fit_predict(single)
# centers = km.cluster_centers_
# cluster_plotting(single,clusts,centers)