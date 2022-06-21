#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from processing import *
import itertools
""""
File description
"""
#%%

sweep = 'hei_X'
#classes=["A","B","C"]
classes=["ZERO","ONE","TWO"]
replicas = 3
components = 3
moment=490
write = True

#%%

# Legacy
def pcs_all(groups):
    """
    PCA through groups
    - iterate through each group
    - stack their group state at given time for each class
    - generate principle components for each stack
    - store in a dicitonary, associating pcs with appropriate label
    """
    pcs = {}
    for key,value in groups.items():
        mat = np.stack(groups[key], axis=0 )
        mat = StandardScaler().fit_transform(mat)
        pca = PCA(n_components=3)
        pca.fit(mat)
        X = pca.transform(mat)
        pcs[key]=X
    return pcs

# Updated version
def pc_stack(groups):
    """
    PCA through groups
    - iterate through each group
    - stack their group state at given time for each class
    - generate principle components for entire stack
    - store in a dicitonary
    """
    all_norms = []
    for k,v in groups.items():
        for i, slice in enumerate(v):
            norm = (np.array(slice)-np.mean(slice))
            all_norms.append(norm)
    all_norms = np.stack(all_norms)
    pc_obj = PCA(n_components=3)
    pc_slice = pc_obj.fit_transform(all_norms)
    return pc_slice

#%%

def analysis_loop(sweep,classes,replicas,moment,components,write):
    """
    Generate and Store First 3 PCs for all Experiments in a Directory
    - Group experiments by configuration
    - Store grouped spikes in dict per config (extra)
    - Create matrix of PCs across replicas for each class of all configs
        - Use `pc_all` function
    - Store in dictionary by config file name
    - Write dictionary to json file for subsequent analysis

    ToDo:
    - Import congigurations cleanly
    - Check the dimensional math on PCA setup and be sure it is correct
    """
    np.seterr(divide='ignore', invalid='ignore')
    pats = len(classes)

    directory = f'results/{sweep}/liquid/spikes'

    filename = os.listdir(directory)[1]
    file = os.path.join(directory, filename)
    dat,indices,times = txt_to_spks(file)
    length = int(np.ceil(np.max(times)))
    neurons = np.max(indices) + 1

    experiments = int(len(os.listdir(directory))/(len(classes)*replicas))

    exp_pcs = {}
    stacks = {}
    spikes={}
    grouped={}
    # iterate through all experiments and group one hot encoded spikes
    # by class and replica, generate 3 pcs for each and store in dictionary
    for exp in range(experiments):
        groups={}
        spike={}
        I=[]
        T=[]
        IT = []
        for pat,label in enumerate(classes):
            groups[label]=[]
            spikes[label]=[]
            for r in range(replicas):
                i = exp*pats*replicas + pat*replicas + r
            
                filename = os.listdir(directory)[i]
                file = os.path.join(directory, filename)

                if (i) % 9 == 0:
                    exp_name=file[len(directory)+1:-14]
                    exp_pcs[exp_name] = []
                    print(f"{exp}-{i} experiment: {exp_name}")

                dat,indices,times = txt_to_spks(file)
                # spike[label] = zip(indices,times)
                # I.append(indices)
                # T.append(times)
                IT.append(indices)
                IT.append(times)
                spike[label] = IT

                mat=one_hot(neurons,length,indices,times)
                slice = []
                for n in range(neurons):
                    slice.append(mat[n][moment])
                groups[label].append(slice)

        # generate 3 pcs for each replica and class
        pcs = pcs_all(groups) # Legacy
        stack = pc_stack(groups)

        # store in nice tensor format and add to dictionary of experiments
        pc_mat = []
        for ind,label in pcs.items():
            pc_mat.append((pcs[ind]))
        pc_mat=np.array(pc_mat)
        pc_mat.reshape(pats,replicas,components)
        exp_pcs[exp_name] = pc_mat.tolist()

        stacks[exp_name] = np.array(stack).tolist()

        # SPIKES
        # spikes[exp_name] = zip(np.concatenate(I),np.concatenate(T))
        spikes[exp_name] = spike


        grouped[exp_name] =[]
        for k,v in groups.items():
            grouped[exp_name].append(np.array(v).tolist())

    # store dictionary in json format
    if write==True:
        print("write")
        js = json.dumps(exp_pcs)
        sweep = directory[len('results/'):-len('/liquid/spikes')]
        path = directory[:-len('/liquid/spikes')] + f'/performance/{sweep}_pcs.json'
        f = open(path,"w")
        f.write(js)
        f.close()

        js2 = json.dumps(stacks)
        path2 = f'results/{sweep}/performance/{sweep}_{moment}_stacks.json'
        f2 = open(path2,"w")
        f2.write(js2)

        js3 = json.dumps(grouped)
        path3 = f'results/{sweep}/analysis/m={moment}_groups.json'
        dirName= f'results/{sweep}/analysis'
        try:
            os.makedirs(dirName)    
        except FileExistsError:
            pass
        f3 = open(path3,"w")
        f3.write(js3)
        f3.close()

    return spikes, groups, exp_pcs, stacks


spikes, groups, exp_pcs, stacks = analysis_loop(sweep,classes,replicas,moment,components,write)
#%%
print(len(list(stacks)))
print()
#%%
def path_stacks(sweep,classes,replicas,m1,m2,name):

    np.seterr(divide='ignore', invalid='ignore')
    pats = len(classes)
    directory = f'results\{sweep}\liquid\spikes'

    path_stacks = []

    groups={}
    m_range = np.arange(m1,m2,1)[:10]
    for m in range(m1,m2):
        print(m)
        for pat,label in enumerate(classes):
            groups[label]=[]

            for r in range(replicas):
                i = 1*pats*replicas + pat*replicas + r
            
                filename = directory + name + f"_pat{label}_rep{r}.txt"
                # file = os.path.join(directory, filename)
                file = filename

                if (i) % 9 == 0:
                    exp_name=file[len(directory)+1:-14]

                dat,indices,times = txt_to_spks(file)
                length = 700 #int(np.ceil(np.max(times)))
                neurons = 700 # np.max(indices) + 1
                # print("n=",neurons)

                mat=one_hot(neurons,length,indices,times)
                slice = []
                for n in range(neurons):
                    slice.append(mat[n][m])
                groups[label].append(slice)

        # generate 3 pcs for each replica and class
        stack = pc_stack(groups)

        path_stacks.append(np.array(stack).tolist())

    return path_stacks
        

replicas=3
components=3
m1 = 0
m2 = 699
name = '\Maass_geo=(randNone_geo[4, 4, 4]_smNone)_N=1000_IS=0.2_RS=0.3_ref=3.0_delay=1.5_U=0.6'


path_sep_minus = path_stacks(sweep,classes,replicas,m1,m2,name)


#%%

dirName = f'results/{sweep}/performance'
item = f'{sweep}_{moment}_stacks'
j_stacks = read_json(dirName,item)

#%%

def mean_separation(stacks,classes,replicas,write=True):
    """
    Calculate Mean Distance Between Mean Pattern Replica PCs for All Configs
    - Return sorted dictionary
    """

    separation = {}

    for key,val in stacks.items():
        #print(key)
        means = []
        metric = []
        reps = []
        for i,let in enumerate(classes):
            reps_pos = stacks[key][replicas*i:replicas*i+replicas]
            #print(reps_pos)
            mean_position = np.mean((reps_pos),axis=0)
            stand = np.std((reps_pos),axis=0)
            means.append(mean_position)
            metric.append(stand)

            reps_dists=[]
            combos = list(itertools.combinations(list(reps_pos),2))
            #print(combos)
            ####
            for c in combos:
                reps_dists.append(distance.euclidean(c[0],c[1]))

            # for a,vr1 in enumerate(reps_pos):
            #     for b,vr2 in enumerate(reps_pos):
            #         if a!=b:
            #             reps_dists.append(distance.euclidean(reps_pos[a],reps_pos[b]))
            reps.append(np.mean(reps_dists))

            # print((stacks[key][replicas*i:replicas*i+replicas]))
            # print(mean_position)
            # print(stand)
            # print(np.mean(reps_dists))
            # print("\n")
            
        dists=[]
        #print(list(means))
        dist_combos = list(itertools.combinations(list(means),2))
        print(len(dist_combos))
        for c in dist_combos:
            dists.append(distance.euclidean(c[0],c[1]))

        # for i,v1 in enumerate(means):
        #     for j,v2 in enumerate(means):
        #         if i!=j:
        #             dists.append(distance.euclidean(means[i],means[j]))
        #print(len(reps))
        #separation[key] = np.mean(dists) - np.mean(reps)
        separation[key] = np.log(np.mean(dists))/np.log(np.mean(metric))
        #separation[key] = np.log(np.mean(dists))/np.log(np.mean(rep))
        #separation[key] = np.mean(dists) - np.mean(metric)
        #separation[key] = np.mean(dists)/np.mean(metric)
        #separation[key] = np.mean(dists)
        

    ranked_separation = dict(reversed(sorted(separation.items(), key=lambda item: item[1])))

    if write==True:
        js = json.dumps(ranked_separation)
        path = f'results/{sweep}/performance/{sweep}-separation_rankings.json'
        f = open(path,"w")
        f.write(js)
        f.close()
    
    return ranked_separation

replicas = 3
ranked_separation = mean_separation(j_stacks,classes,replicas)

#%%

item = f"{sweep}-rankings"
ranked_performance = read_in_ranks(sweep,item)

print_rankings(ranked_performance,"Performace",9)
print_rankings(ranked_separation,"Separation",9)

#%%

def ranking_comparison(dict1,dict2):
    I = []
    J = []
    for i, (k1,v1) in enumerate(dict1.items()):
        for j, (k2,v2) in enumerate(dict2.items()):
            if k1[:-2] == k2:
                I.append(i)
                J.append(j)
            #print(k1,k2)
    return I,J

I,J = ranking_comparison(ranked_separation, ranked_performance)

#print(I,J)
#%%

# from matplotlib import cm
# from colorspacious import cspace_converter

def ranking_comparison_plot(I,J):
    N = len(I) + 1
    I = np.array(I) + 1
    J = np.array(J) + 1
    plt.figure(figsize=(16, 10))
    plt.plot(np.zeros(N-1), np.arange(1,N,1, dtype=int), 'ok', ms=100/N, color='orange')
    plt.plot(np.ones(N-1), np.arange(1,N,1, dtype=int), 'ok', ms=100/N, color='blue')
    # print(np.arange(1,N,1, dtype=int))
    for i, j in zip(I, J):
        plt.plot([0, 1], [i, j], '-k', linewidth=5/(np.abs(i-j)*2+1))

    plt.title("Ranking Comparison Between PCA Separation and Performance", fontsize=24)
    plt.xticks([0, 1], ['PCA', 'Performance'], fontsize=22)
    plt.ylabel('Experiment Configuration Ranking', fontsize=22)
    plt.xlim(-0.1, 1.1)
    plt.ylim(np.max(I)+np.int(N/10), np.min(I)-np.int(N/10))
    plt.yticks(np.arange(1,N,16, dtype=int))

    type = 'means_only'

    #plt.savefig(f"results/{sweep}/performance/rank_compare_{sweep}_{type}.png")
    plt.show()

ranking_comparison_plot(I,J)


#%%


def pc_plotting(pcs,classes,replicas):


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    fig.set_figheight(15)
    fig.set_figwidth(15)

    markers = ["$A$","$B$","$C$"]
    colors = ['r','g','b']
    
    count = 0
    #print(pcs)
    for key,val in pcs.items():
        count = 0

        for i,c in enumerate(classes):
            for j in range(replicas):
                #print(count, i)
                pc = val[count]
                ax.scatter(pc[0],pc[1],pc[2], marker=markers[i],color=colors[i],s=250,label=classes[i])
                count+=1
                print(f"{c}-{j}",pc[0],pc[1],pc[2])
        means = {}
        for i,let in enumerate(classes):
            #print(replicas*i+replicas)
            mean = np.mean((pcs[key][replicas*i:replicas*i+replicas]),axis=0)
            means[let] = mean
            ax.scatter(mean[0],mean[1],mean[2], marker='o',color=colors[i],s=450,label=classes[i])
            #print(f"{c}-{j}",mean[0],mean[1],mean[2])
    plt.xlim(-2.5,2.5)
    plt.ylim(-2.5,2.5)
    # plt.legend()
    ax.set_zlim(-5,5)

    ax.set_xlabel('Replica 0 Component')
    ax.set_ylabel('Replica 1 Component')
    ax.set_zlabel('Replica 2 Component')
    plt.title("Positions in PCA Space",fontsize=24)
    plt.show()


def unit_dict(dict,key):
    unit_dict = {}
    for i, (k,v) in enumerate(dict.items()):
        if k == key:
            unit_dict[k] = v
    return unit_dict

# high sep
# sing = 'Maass_rnd=(rand0.3_geoNone_smNone)_N=135_IS=0.14_RS=0.3_ref=3.0_delay=1.5_U=0.6_p'
sing = 'Maass_smw=(rand111.0_geoNone_sm0.25)_N=135_IS=0.18_RS=0.3_ref=3.0_delay=1.5_U=0.6_p'
# low sep


# high perf
# sing = 'STSP_rnd=(rand0.3_geoNone_smNone)_N=135_IS=0.2_RS=0.3_ref=3.0_delay=1.5_U=0.6_p'
# low perf

# high sep

single = unit_dict(j_stacks,sing)
pc_plotting(single,classes,replicas)



#%%

def mean_separation(stacks,classes,replicas,write=True):
    """
    Calculate Mean Distance Between Mean Pattern Replica PCs for All Configs
    - Return sorted dictionary
    """

    separation = {}

    for key,val in stacks.items():
        #print(key,val)
        means = []
        metric = []
        reps = []
        for i,let in enumerate(classes):
            #print(replicas*i,replicas*i+replicas)
            reps_pos = stacks[key][replicas*i:replicas*i+replicas]
            print(reps_pos)
            mean_position = np.mean((reps_pos),axis=0)
            print(mean_position)

            stand = np.std((reps_pos),axis=0)
            means.append(mean_position)
            metric.append(stand)

            reps_dists=[]
            combos = list(itertools.combinations(list(reps_pos),2))
            for c in combos:
                reps_dists.append(distance.euclidean(c[0],c[1]))
            reps.append(np.mean(reps_dists))

        dists=[]
        dist_combos = list(itertools.combinations(list(means),2))
        for c in dist_combos:
            dists.append(distance.euclidean(c[0],c[1]))

        print(len(reps))
        #separation[key] = np.mean(dists) - np.mean(reps)
        ##separation[key] = np.mean(dists)/np.mean(reps)
        #separation[key] = np.mean(dists) - np.mean(metric)
        #separation[key] = np.mean(dists)/np.mean(metric)
        #separation[key] = np.mean(dists)
        

    ranked_separation = dict(reversed(sorted(separation.items(), key=lambda item: item[1])))

    return ranked_separation

replicas = 3
separation = mean_separation(single,classes,replicas)

#%%
a = [3,8,12]
plt.hlines(1,1,20, 'k')  # Draw a horizontal line
plt.xlim(0,21)
plt.ylim(0.5,1.5)
c = ['r','b','g']
y = np.ones(np.shape(a))  
for i in range(3):
    plt.plot(a[i],y[i],'.k',ms = 20, color = c[i])  # Plot a line at each location specified in a
plt.axis('off')
plt.show()

#%%

def path_plotting(path_stacks):

    # classes=["A","B","C"]
    classes=["A","B","C"]
    replicas = 3

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    fig.set_figheight(15)
    fig.set_figwidth(15)

    markers = ["$A$","$B$","$C$"]
    colors = ['r','g','b']
    
    count = 0
    #print(pcs)
    xline = []
    yline = []
    zline = []
    lines = {'A':[],
    'B':[],
    'C':[]}
    for time, exp in enumerate(path_stacks):
        count = 0
        for i,c in enumerate(classes):
            for j in range(replicas):
                #print(count, i)
                pc = exp[count]
                #ax.scatter(pc[0],pc[1],pc[2], marker=markers[i],color=colors[i],s=200,label=classes[i])
                count+=1

        means = {}
        for i,let in enumerate(classes):
            #print(replicas*i+replicas)
            mean = np.mean((exp[replicas*i:replicas*i+replicas]),axis=0)
            means[let] = mean
            lines[let].append([mean[0],mean[1],mean[2]])
            #ax.scatter(mean[0],mean[1],mean[2], marker='o',s=(time)*3, color=colors[i],label=classes[i])
            #plt.plot(xline,yline,zline, color=colors[i],label=classes[i])


    start = 50
    until = 100
    for i,(k,v) in enumerate(lines.items()):
        xs = []
        ys = []
        zs = []
        # for c in v:
        #     print("----------")
        #     print(c)
        #     print("----------")
        for c in v:
            xs.append(c[0])
            ys.append(c[1])
            zs.append(c[2])
        # print(zs)
        plt.plot(xs[start:until],ys[start:until],zs[start:until], color=colors[i],label=classes[i])


    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.legend()
    ax.set_zlim(-1,1)

    ax.set_xlabel('Replica 0 Component')
    ax.set_ylabel('Replica 1 Component')
    ax.set_zlabel('Replica 2 Component')

    plt.show()

path_plotting(path_sep_minus)

#print(path_sep_minus)


#%%
# LEGACY - Ignore
##################################################################################
##################################################################################
##################################################################################

def read_in_pcs(sweep):
    """
    Read in Stored PC Json
    - Extract only first PC across replicas for each pattern
    - Return a dictionary of coordinates per pattern
    - In form:
    dic = {
        filename: [[A-xyz-coordinates],[B-xyz-coordinates],[C-xyz-coordinates]]
    }
    """
    place = f'results/{sweep}/performance/{sweep}_pcs.json'
    f = open(place)
    data = json.load(f)
    f.close()
    pc_coordinates = {}
    for i,name in data.items():
        pc_coordinates[i] = []
        for j in range(len(data[i][0])):
            pc_coordinates[i].append([])
            for k in range(len(data[i])):
                pc_coordinates[i][j].append(data[i][j][k][0])
    return pc_coordinates

# pc_coordinates = read_in_pcs(sweep)
# print(pc_coordinates)
# for k,v in pc_coordinates.items():
#     print(k,"\n",v,"\n")

############

import warnings

def mean_distances(coordinates):
    """
    Calculate Mean Distance Between Pattern PCs for All Configs
    - Return sorted dictionary

    ToDo:
    - Deal with Nan values better
    - Check actual order against sorted
    - Compare with performance rankings
    """
    mean_dists = {}
    for key,value in coordinates.items():
        list = coordinates[key]
        dists = []
        for i in list:
            for j in list:
                if i!=j:
                    dists.append(distance.euclidean(i,j))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.mean(dists)
            

        # try:
        #     mean = np.mean(dists)
        # except RuntimeWarning:
        #     print("Nan")

        #print(f"{mean} - {key}")
        mean_dists[key] = mean

    ranked_mean_dists = dict(reversed(sorted(mean_dists.items(), key=lambda item: item[1])))
    
    return ranked_mean_dists

# ranked_pca = mean_distances(pc_coordinates)

# print_rankings(ranked_pca,"PCA")

###########


def get_liquids():

    # best performing config
    configs = {
        'patterns': 3, 
        'length': 100, 
        'channels': 40, 
        'replicas': 3,
        'location': 'multi_sweep',
        'learning': 'STDP', 
        'topology': 'small-world', 
        'rand_p': None,
        'dims': None, 
        'beta': 0.5, 
        'neurons': 100, 
        'refractory': 0.0,
        'delay': 1.5, 
        'flow':False,
        'plotting':False
        }

    classes=["A","B","C"]

    multi_indices, multi_times, mats, labels = one_hot_sets(configs,classes,liquids=None)
    return configs, classes, multi_indices, multi_times, mats, labels

# configs, classes, multi_indices, multi_times, mats, labels = get_liquids()

def pc_analysis(configs,classes, multi_indices, multi_times, mats, labels):

    groups = group(mats,labels,95)
    print(groups)
    
    pcs = pcs_all(groups)
    print(pcs)
    
    return pcs

# pc_analysis(configs, classes, multi_indices, multi_times, mats, labels)