#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from processing import *

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

def path_stacks(sweep,classes,replicas,m1,m2,name,write):

    np.seterr(divide='ignore', invalid='ignore')
    pats = len(classes)
    directory = f'results\{sweep}\liquid\spikes'

    path_stacks = []

    groups={}

    for m in range(m1,m2):
        for pat,label in enumerate(classes):
            groups[label]=[]

            for r in range(replicas):
                i = 1*pats*replicas + pat*replicas + r
            
                #filename = directory + name + f"_pat{label}_rep{r}.txt"
                file = os.path.join(directory, name) + f"_pat{label}_rep{r}.txt"
                

                if (i) % 9 == 0:
                    exp_name=file[len(directory)+1:-14]

                dat,indices,times = txt_to_spks(file)
                length = m2
                neurons = np.max(indices) + 1

                mat=one_hot(neurons,length,indices,times)
                slice = []
                for n in range(neurons):
                    slice.append(mat[n][m])
                groups[label].append(slice)

        # generate 3 pcs for each replica and class
        stack = pc_stack(groups)
        path_stacks.append(np.array(stack).tolist())

    if write==True:
        js2 = json.dumps(path_stacks)
        dir = f'results/{sweep}/analysis/paths/coords/'
        try:
            os.makedirs(dir)    
        except FileExistsError:
            pass
        path2=f'results/{sweep}/analysis/paths/coords/{name}.json'
        f2 = open(path2,"w")
        f2.write(js2)
        f2.close()

    return path_stacks

def path_plotting(paths,name,save):

    classes=["A","B","C"]
    replicas = 3

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    fig.set_figheight(15)
    fig.set_figwidth(15)

    markers = ["$A$","$B$","$C$"]
    colors = ['r','g','b']
    
    count = 0
    lines = {'A':[],
    'B':[],
    'C':[]}
    for time, exp in enumerate(paths):
        means = {}
        for i,let in enumerate(classes):
            mean = np.mean((exp[replicas*i:replicas*i+replicas]),axis=0)
            means[let] = mean
            lines[let].append([mean[0],mean[1],mean[2]])

    for i,(k,v) in enumerate(lines.items()):
        xs = []
        ys = []
        zs = []
        for c in v:
            xs.append(c[0])
            ys.append(c[1])
            zs.append(c[2])
        plt.plot(xs,ys,zs, color=colors[i],label=classes[i])

    plt.xlim(-2,2)
    plt.ylim(-2,2)
    ax.set_zlim(-2,2)
    plt.legend()
    plt.title(f'Pattern Path In PCA for Mean Replica PCs\n{name}')
    ax.set_xlabel('Replica 0 Component')
    ax.set_ylabel('Replica 1 Component')
    ax.set_zlabel('Replica 2 Component')

    if save==True:
        dir = f'results/{sweep}/analysis/paths/plots/'
        try:
            os.makedirs(dir)    
        except FileExistsError:
            pass
        plt.savefig(f'results/{sweep}/analysis/paths/plots/{name}.png')
        plt.close()
    else:
        plt.show()

#%%

def experiment_loop(sweep,classes,replicas,m1,m2,write):
    pats = len(classes)
    directory = f'results/{sweep}/liquid/spikes'

    filename = os.listdir(directory)[1]
    file = os.path.join(directory, filename)
    dat,indices,times = txt_to_spks(file)
    length = int(np.ceil(np.max(times)))
    neurons = np.max(indices) + 1

    experiments = int(len(os.listdir(directory))/(len(classes)*replicas))

    steps = np.arange(0,len(os.listdir(directory)),replicas*pats)

    for exp in steps:            
        name = os.listdir(directory)[exp][:-len('_patA_rep0.txt')]
        print(name)
        file = os.path.join(directory, filename)
        paths = path_stacks(sweep,classes,replicas,m1,m2,name,write)
        path_plotting(paths,name,write)


#%%
sweep = 'full_sweep'
classes = ['A','B','C']
replicas = 3
m1 = 0
m2 = 100
write = True

experiment_loop(sweep,classes,replicas,m1,m2,write)

#%%

