import numpy as np
import os
import json
from plotting import raster_run_input, raster_save




##################### Spike Processing Functions #######################

def spks_to_txt(N,indices,times,prec,name):
    """
    Convert Brain spikes to txt file
     - Each line is a neuron indexe
     - Firing times are recorded at at their appropriate neuron row
    """
    with open(f'{name}.txt', 'w') as f:
        for row in range(N):
            for i in range(len(indices)):
                if row == indices[i]:
                    if row == 0:
                        f.write(str(np.round(times[i],prec)))
                        f.write(" ")
                    else:
                        f.write(str(np.round(times[i],prec)))
                        f.write(" ")
            f.write('\n')


def txt_to_spks(file):
    """"
    Convert txt file back to Brian style spikes
     - Two parallel arrays of spike times and associated neuron indices
    """
    mat = []
    with open(file) as f:
        for line in f:
            arr = line.split(' ')
            mat.append(np.array(arr))
    dat = []
    indi = []
    timi = []
    for i in range(len(mat)):
        row = []
        for j in range(len(mat[i])):
            if mat[i][j] != '\n':
                indi.append(i)
                row.append(float(mat[i][j]))
                timi.append(float(mat[i][j]))
        dat.append(row)
    indi = np.array(indi)
    timi = np.array(timi)
    return dat,indi,timi



def one_hot(N,length,indices,times):
    """"
    One hot encode spiking data into NEURONS x TIME 
     - Time columns (rows?) are divided into 1ms steps
        - If spike falls into this range for a given neuron, that neuron
          index at that time index is incremented by 1
     - Not actually one hot, but rather binned, with multiple spikes
       allowed per index
    """
    hot_matrix = np.zeros((N,length))
    for t in range(len(times)):
        if times[t] < length:
            hot_matrix[indices[t],int(np.floor(times[t]))] +=1
    return hot_matrix




####################### Bureaucratic Functions #########################

def read_in_ranks(sweep,item):
    """
    Read in performance rankings from stored dictionary
    """
    place = f'results/{sweep}/performance/{item}.json'
    f = open(place)
    ranks = json.load(f)
    f.close()
    return ranks


def read_json(dirName,item):
    #place = f'results/{sweep}/performance/{sweep}_{item}.json'
    place = f'{dirName}/{item}.json'
    f = open(place)
    data = json.load(f)
    f.close()
    for k,v in data.items():
        data[k] = np.array(v)
    return data


def write_json(dict,dirName,exp,name):
    for k,v in dict.items():
        dict[k] = np.array(v).tolist()
    try:
        os.makedirs(dirName)    
    except FileExistsError:
        pass
    js = json.dumps(dict)
    path = f'{dirName}{name}_{exp}.json'
    f = open(path,"w")
    f.write(js)
    f.close()


def write_dict(dict,path,name):
    for k,v in dict.items():
        dict[k] = np.array(v).tolist()
    try:
        os.makedirs(path)    
    except FileExistsError:
        pass
    js = json.dumps(dict)
    path = f'{path}/{name}.json'
    f = open(path,"w")
    f.write(js)
    f.close()

def save_spikes(N,T,times,indices,location,item,show):
    #spiking
    dirName = f"results/{location}/spikes"
    try:
        os.makedirs(dirName)    
    except FileExistsError:
        pass
    spks_to_txt(N,indices,times,prec=8,name=f"results/{location}/spikes/{item}")

    #plotting
    dirName2 = f"results/{location}/plots"
    try:
        os.makedirs(dirName2)    
    except FileExistsError:
        pass
    # if N < 100 and "inputs" in location:
    #     raster_run_input(times,indices,dirName2,item)
    # else:
    # if item[-1] == '0' or len(item) < 20:
    #raster_save(times,indices,dirName2,item,show)


def billboard(word):
    print("\n\n")
    for i in range(3):
        print(f"######### {word} #########   ######### {word} #########   ######### {word} #########   ")
    print("\n")



def unit_dict(dict,key):
    unit_dict = {}
    for i, (k,v) in enumerate(dict.items()):
        if k == key:
            unit_dict[k] = v
    return unit_dict






###

# Legacy
# def one_hot_sets(configs,classes,liquids):
#     """
#     - Takes spikes either directly from liquids or from saved directories
#     - Returns a matrix of one-hot-encoded spikes for 1ms times steps
#     - Note that all patterns and replicas are concatenated in the form:
#         - A0, B0, C0, A1, B1, C1, A2, B2, C2,...
#     """

#     dic = configs
#     for key, value in dic.items():
#         str = key
#         globals()[str] = value

#     IND = []
#     TIM = []
#     mats = []
#     labels=[]

#     if flow == False:
#         for rep in range(replicas):
#             for pat in classes:

#                 dat,indices,times = txt_to_spks(f"results/{location}/liquid/spikes/"+full_loc+f"_pat{pat}_rep{rep}.txt")

#                 IND.append(np.array(indices)[:])
#                 TIM.append(times[:]/1000)
#                 # if plotting:
#                 #     print("plotting")
#                 #     raster_plot(times[:]/1000,np.array(indices)[:])
#                 mats.append(one_hot(neurons,length,np.array(indices)[:],times[:]))
#                 #print(one_hot(135,array(indices)[:],times[:]).shape)
#                 labels.append(pat)

#     elif flow == True:
#         for rep in range(replicas):
#             for pattern in classes:

#                 indices_ = np.array(liquids[pattern][rep][:,0])
#                 times = np.array(liquids[pattern][rep][:,1])

#                 indices = indices_.astype(int)

#                 IND.append(np.array(indices)[:])
#                 TIM.append(times[:])
#                 # if plotting:
#                 #     raster_plot(times[:]*ms,array(indices)[:])
#                 mats.append(one_hot(neurons,length,np.array(indices)[:],times[:]))
#                 #print(one_hot(135,array(indices)[:],times[:]).shape)
#                 labels.append(pattern)

#     for t in range(len(TIM)):
#         TIM[t] += length*t

#     multi_indices = np.concatenate(IND)
#     multi_times = np.concatenate(TIM)
#     print(f'''
#     Spike times and indices imported
#     Folder: {location}
#     Patterns: {classes}
#     Replicas: {replicas}
#     Length: {len(multi_times)}
#     ''')

#     return multi_indices, multi_times, mats, labels


# def group(mats,labels,time):
#     """
#     Grouping
#     - group one-hot encoded sets for each replica by class label
#     - store in a dictionay
#     """
#     classes = set(labels)
#     groups = {}
#     for c in classes:
#         groups[c] = []

#     for i, lab in enumerate(labels):
#         for group in classes:
#             if lab == group:
#                 groups[group].append(mats[i][time])
#     return groups