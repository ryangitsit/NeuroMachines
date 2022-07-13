#%%

import os
from os.path import dirname
import numpy as np
import matplotlib.pyplot as plt
import itertools
import string
import pickle
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial import distance
import warnings
from os.path import exists

from processing import txt_to_spks,write_dict,read_json

#%%
"""
Plan:

- Generate one-hot encoded sets of liquid states and store them
- Create time-slice groups across all liquids
- Generate PCs for each slice and store
- Analyze average distance over time of PCs and full dimensions
    - Centroid separations
- Plot separation measures over performance
"""

class PerformanceAnalysis():
    def __init__(self,config,save,show):
        self.sweep = config.dir
        self.type = config.input_name
        self.patterns = config.patterns
        self.replicas = config.replicas
        self.classes = config.classes
        self.save = save
        self.show = show

    def __str__(self):
        return f"Dataset: \n{self.__dict__}"

    def performance_pull(self):
        """
        Pull .npy Files for Analysis
        - Iterate over all accuracy files for a given sweep
        - Load them into a dictionary with file name as key
        - Note there is an accuracy array of all time steps for each pattern
        
        Example:
            dict[key] = [acc["A"][t=0],acc["A"][t=1],...][acc["B"][t=0]...],...]
        """
        self.directory = f'results/{self.sweep}/performance/accuracies'
        self.all_accs = {}
        self.all_avgs = {}
        self.all_totals = {}
        self.all_finals = {}
        for filename in os.listdir(self.directory):
            file = os.path.join(self.directory, filename)
            with open(file, 'rb') as f:
                accs = np.load(f, allow_pickle=True)
            key = file[len(self.directory)+1:-4]
            self.all_accs[key] = accs
            self.all_avgs[key] = np.mean(accs, axis=0 )
            self.all_totals[key] = np.sum(np.mean(accs, axis=0 ))
            self.all_finals[key] = np.mean(accs, axis=0 )[-1:][0]
        return self.all_accs, self.all_avgs, self.all_totals, self.all_finals

    def perfromance_t(self,t):
        perf_at_t = []
        for k,v in self.all_avgs.items():
            perf_at_t.append(v[t])
        print(f"Average performance and time {t} is {np.mean(perf_at_t)} for {len(self.all_avgs)} experiments.")


    def accs_plots(self,tops=None):
        """
        Plot certainty trajectorys for all configurations in sweep
        """
        dict = self.all_avgs.items()
        plt.figure(figsize=(16, 10))
        for i,(file, avg) in enumerate(dict):
            if tops==None:
                plt.plot(avg)
            else:
                if file in tops:
                    plt.plot(avg)
            plt.title(f"Prediction Certainty Over Time",fontsize=24)
            plt.ylim(0,1.01)
            plt.xlabel("Time (ms), dt=1ms",fontsize=22)
            plt.ylabel("Certainty for Correct Classification",fontsize=22)
        if self.save==True:
            if tops==None:
                path = f'results/{self.sweep}/analysis/perfplot.png'
            else:
                path = f'results/{self.sweep}/analysis/top{len(tops)}_perfplot.png'
            dirName = f'results/{self.sweep}/analysis/'
            try:
                os.makedirs(dirName)    
            except FileExistsError:
                pass

            plt.savefig(path)
        if self.show==True:
            plt.show()
        else:
            plt.close()

    def rankings(self):
        self.final_perf_ranking = dict(sorted(self.all_finals.items(), key=lambda item: item[1], reverse=True))
        self.total_perf_ranking = dict(sorted(self.all_totals.items(), key=lambda item: item[1], reverse=True))
        return self.final_perf_ranking, self.total_perf_ranking

    def print_rankings(self,dict,name,vals):
        print(f"\n      ### {name} Rankings ###\n")
        for i, (key,value) in enumerate(dict.items()):
            if i < vals:
                print(np.round(value,4), " - ", key)
        print("\n")
        print(f"Mean of {name} value: {np.array(list(dict.values())).mean()}")
        print(f"Standard Deviation of {name} value: {np.array(list(dict.values())).std()}")

    def performance_statistics(self,config,dict,lim):
        """
        Determine number of occurnces for all hyperparameters within
        top performing limit.  Also find parameter pairings.
        """
        self.lim = lim
        hyper_parameters = {
            "learning":[],
            "topology":[],
            "input_sparsity":[],
            "res_sparsity":[],
            "rndp":[],
            "dims":[],
            "beta":[],
            "refractory":[],
            "delay":[],
            "x_atory":[],
            "feed":[]
            }
        for k in list(dict)[:lim]:
            directory = f'results/{config.dir}/configs'
            filename = f'config_{k}.pickle'
            file = os.path.join(directory, filename)
            file_to_read = open(file, "rb")
            experiment = pickle.load(file_to_read)
            file_to_read.close()
            for param in hyper_parameters.keys():
                hyper_parameters[param].append(experiment.__dict__[param])
        self.hyperparams = hyper_parameters
        print(f"Hyper Parameter Occurences in Top {lim} Performers")
        for k,v in hyper_parameters.items():
            if k == 'dims':
                for i in range(len(v)):
                    v[i] = str(v[i])
            uniques = set(v)
            occ = {}
            for u in uniques:
                occ[u] = v.count(u)
            print(f"{k} => {occ}")
        
        hyper_combos = list(itertools.combinations(list(hyper_parameters),3))
        self.combos = {}
        for param_combo in hyper_combos:
            sets = []
            for i in range(lim):
                sets.append((hyper_parameters[param_combo[0]][i],hyper_parameters[param_combo[1]][i],hyper_parameters[param_combo[2]][i]))
            unique_combos = set(sets)
            occ_combos = {}
            for u in unique_combos:
                occ_combos[u] = sets.count(u)
            # print(param_combo)
            # for k,v in occ_combos.items():
            #     if str(k[0]) != 'None' and str(k[1]) != 'None':
            #         print("   ",k," - ",v)
            self.combos[param_combo] = occ_combos

    def hist_ranked(self):
        """
        Plot histogram for occurnces of learing-topology pairs in 
        defined top performers.
        """
        labels = list(set(self.hyperparams["learning"]))
        dict = self.combos[list(self.combos)[0]]
        RND=[0,0,0,0]
        GEO=[0,0,0,0]
        SMW=[0,0,0,0]
        for i,l in enumerate(labels):
            for k,v in dict.items():
                if k[0] == l:
                    if k[1] == 'geo':
                        GEO[i]+=v*100/self.lim
                    if k[1] == 'rnd':
                        RND[i]+=v*300/self.lim
                    if k[1] == 'smw':
                        SMW[i]+=v*100/self.lim

        x = np.arange(len(labels)) 
        width = 0.18  
        plt.figure(figsize=(7,7))
        plt.style.use('seaborn-muted')

        rects1 = plt.bar(x - width-.01, RND, width, label='random')
        rects2 = plt.bar(x, GEO, width, label='geometric')
        rects3 = plt.bar(x + width+.01, SMW, width, label='small-world')

        plt.ylabel('Percent Present',fontsize=18)
        plt.title(f'Configurations in top {self.lim} Performers',fontsize=22)
        plt.xticks(x, labels,fontsize=18)
        plt.legend(fontsize=20) # using a size in points
        plt.legend(fontsize="x-large") # using a named size
        plt.tight_layout()

        if self.save==True:
            path = f'results/{self.sweep}/analysis/stats_{self.lim}.png'
            plt.savefig(path)
        if self.show==True:
            plt.show()
        else:
            plt.close()

    def hist_alt(self):
        """
        Plot histogram for occurnces of learing-topology pairs in 
        defined top performers.
        """
        labels = list(set(self.hyperparams["learning"]))


        dict = self.combos[('learning', 'x_atory', 'feed')]

        plain=[0,0,0,0]
        xt=[0,0,0,0]
        cont=[0,0,0,0]
        xcont=[0,0,0,0]
        for i,l in enumerate(labels):
            for k,v in dict.items():
                if k[0] == l:
                    if k == (l,'False','reset'):
                        plain[i]+=v*100/self.lim
                    if k == (l,'True','reset'):
                        xt[i]+=v*100/self.lim
                    if k == (l,'False','continuous'):
                        cont[i]+=v*100/self.lim
                    if k == (l,'True','continuous'):
                        xcont[i]+=v*100/self.lim

                    # if k == (l,False,'reset'):
                    #     plain[i]+=v*100/self.lim
                    # if k == (l,True,'reset'):
                    #     xt[i]+=v*100/self.lim
                    # if k == (l,False,'continuous'):
                    #     cont[i]+=v*100/self.lim
                    # if k == (l,True,'continuous'):
                    #     xcont[i]+=v*100/self.lim

        x = np.arange(len(labels)) 
        width = 0.18  
        plt.figure(figsize=(7,7))
        plt.style.use('seaborn-muted')

        rects1 = plt.bar(x - 1.5*width-.03, plain, width, label='plain')
        rects2 = plt.bar(x - .5*width-.01, xt, width, label='x_atory')
        rects3 = plt.bar(x + .5*width+.01, cont, width, label='continous')
        rects3 = plt.bar(x + 1.5*width+.03, xcont, width, label='x and continuous')

        plt.ylabel('Percent Present',fontsize=18)
        plt.title(f'Configurations in top {self.lim} Performers',fontsize=22)
        plt.xticks(x, labels,fontsize=18)
        plt.legend(fontsize=20) # using a size in points
        plt.legend(fontsize="x-large") # using a named size
        plt.tight_layout()

        if self.save==True:
            path = f'results/{self.sweep}/analysis/stats_xcon_{self.lim}.png'
            plt.savefig(path)
        if self.show==True:
            plt.show()
        else:
            plt.close()

    def hist_alt_top(self):
        """
        Plot histogram for occurnces of learing-topology pairs in 
        defined top performers.
        """
        labels = list(set(self.hyperparams["topology"]))


        dict = self.combos[('topology', 'x_atory', 'feed')]

        plain=[0,0,0]
        xt=[0,0,0]
        cont=[0,0,0]
        xcont=[0,0,0]
        for i,l in enumerate(labels):
            for k,v in dict.items():
                if k[0] == l:
                    if l == 'rnd':
                        if k == (l,'False','reset'):
                            plain[i]+=v*300/self.lim
                        if k == (l,'True','reset'):
                            xt[i]+=v*300/self.lim
                        if k == (l,'False','continuous'):
                            cont[i]+=v*300/self.lim
                        if k == (l,'True','continuous'):
                            xcont[i]+=v*300/self.lim
                    else:
                        if k == (l,'False','reset'):
                            plain[i]+=v*100/self.lim
                        if k == (l,'True','reset'):
                            xt[i]+=v*100/self.lim
                        if k == (l,'False','continuous'):
                            cont[i]+=v*100/self.lim
                        if k == (l,'True','continuous'):
                            xcont[i]+=v*100/self.lim
                    # if l == 'rnd':
                    #     if k == (l,False,'reset'):
                    #         plain[i]+=v*300/self.lim
                    #     if k == (l,True,'reset'):
                    #         xt[i]+=v*300/self.lim
                    #     if k == (l,False,'continuous'):
                    #         cont[i]+=v*300/self.lim
                    #     if k == (l,True,'continuous'):
                    #         xcont[i]+=v*300/self.lim
                    # else:
                    #     if k == (l,False,'reset'):
                    #         plain[i]+=v*100/self.lim
                    #     if k == (l,True,'reset'):
                    #         xt[i]+=v*100/self.lim
                    #     if k == (l,False,'continuous'):
                    #         cont[i]+=v*100/self.lim
                    #     if k == (l,True,'continuous'):
                    #         xcont[i]+=v*100/self.lim
        
        x = np.arange(len(labels)) 
        width = 0.18  
        plt.figure(figsize=(7,7))
        plt.style.use('seaborn-muted')

        rects1 = plt.bar(x - 1.5*width-.03, plain, width, label='plain')
        rects2 = plt.bar(x - .5*width-.01, xt, width, label='x_atory')
        rects3 = plt.bar(x + .5*width+.01, cont, width, label='continous')
        rects3 = plt.bar(x + 1.5*width+.03, xcont, width, label='x and continuous')

        plt.ylabel('Percent Present',fontsize=18)
        plt.title(f'Configurations in top {self.lim} Performers',fontsize=22)
        plt.xticks(x, labels,fontsize=18)
        plt.legend(fontsize=20) # using a size in points
        plt.legend(fontsize="x-large") # using a named size
        plt.tight_layout()

        if self.save==True:
            path = f'results/{self.sweep}/analysis/stats_xcontop_{self.lim}.png'
            plt.savefig(path)
        if self.show==True:
            plt.show()
        else:
            plt.close()


    def top_plot(self,size,tb,lst):
        """
        Plotting Top 5 Performers and Their Replicas
        - Create a subplot grid
        - For each pattern replica
            - For each Top5 performer
                - Convert top performing names to paths
                - Pull the appropriate spikes
                - Raster plot them into the subplot grid
        """
        plt.figure(figsize=(24,14))
        
        fig, axs = plt.subplots(size, 3,figsize=(24,14))
        if tb == 'top':
            top_5 = list(self.final_perf_ranking)[:size]
        elif tb == 'bottom':
            top_5 = list(self.final_perf_ranking)[-size:]
        elif tb == "list":
            top_5 = lst
            size = len(lst)

        for i,pattern in enumerate(self.classes[:self.patterns]):
            suffix = "_pat"+pattern+"_rep0.txt"
            prefix = f'results/{self.sweep}/liquid/spikes/'
            for j,name in enumerate(top_5):
                if not exists(prefix+name+suffix):
                    break
                #print(name+suffix)
                dat, indices, times = txt_to_spks(prefix+name+suffix)
                axs[j, i].plot(times, indices, '.k', ms=.7)#/size)
                axs[j, i].set_xlim([0, 700])
                if i == 0:
                    axs[j, i].set_title(name, size=6)
            axs[j, i].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
        # for ax in axs.flat:
        #     ax.set(xlabel='time (ms)', ylabel='neuron index')
        # for ax in axs.flat:
        #     ax.label_outer()

        if self.save==True:
            path = f'results/{self.sweep}/analysis/{tb}{size}_performers.png'
            plt.savefig(path)
        if self.show==True:
            plt.show()
        else:
            plt.close()


# sweep = "SuperSweep"

# directory = f'results/{sweep}/configs'
# filename = os.listdir(directory)[1]
# file = os.path.join(directory, filename)
# file_to_read = open(file, "rb")
# config = pickle.load(file_to_read)
# file_to_read.close()
# save = False
# show = True

# full_analysis = PerformanceAnalysis(config,save,show)
# full_analysis.performance_pull()
# finals, totals = full_analysis.rankings()
# # full_analysis.top_plot(30)
# # full_analysis.performance_statistics(config,totals,100)
# # full_analysis.hist_alt()
# # full_analysis.hist_alt_top()
# dirName = f'results/{config.dir}/analysis/'
# item = 'all_pcs'
# PCs = read_json(dirName,item)


#%%
class StateAnalysis():
    def __init__(self,config,save,show):
        self.save = save
        self.show = show
        self.directory = f'results/{config.dir}/liquid/spikes'

    def print_config(self,config):
        print(config.__dict__)

    def analysis_loop(self,config,new_pcs):
        """
        Gather States and PCs
         - Iterate over all saved one-hot-encoded matrices
         - Store them in a dictionary with experiment names as keys
         - Meanwhile, generate first 3 PCs across states of the same time
            - For all samples of an experiemnt
            - Store in dictionary
        """
        np.seterr(divide='ignore', invalid='ignore')
        
        if config.old_encoded == True:
            mat_dir = f'results/{config.dir}/performance/liquids/encoded/'
        else:
            mat_dir = f'results/{config.dir}/liquid/encoded/'
        experiments = len(os.listdir(mat_dir))

        count = 0
        self.MATs = {}
        self.PCs = {}
        for exp in range(experiments):
            filename = os.listdir(mat_dir)[exp]
            file = os.path.join(mat_dir, filename)
            exp_name = file[len(mat_dir)+4:-4]
            print(f"{exp} - experiment: {exp_name}")

            mat = np.load(file, allow_pickle=True)
            self.MATs[exp_name] = mat
            # Across all
            pcs_times = []
            for t in range(config.length):
                step = 0
                pc_pats = []
                norms = []
                for p,pattern in enumerate(config.classes):
                    # norms = []
                    for r in range(config.replicas):
                        slice = mat[step][:,t]
                        norm = np.array(slice) - np.mean(slice)
                        norms.append(norm)
                        step+=1
                norms = np.array(norms)
                pc_obj = PCA(n_components=3)
                pc_slice = pc_obj.fit_transform(norms)
                #print(pc_slice)
                pcs_times.append(np.array(pc_slice))
            pcs_times = np.array(pcs_times)
            self.PCs[exp_name] = pcs_times

        dict = self.PCs
        path = f'results/{config.dir}/analysis/'
        name = 'all_pcs'
        write_dict(dict,path,name)


        # dict = self.MATs
        # name = 'all_mats'
        # write_dict(dict,path,name)

        return self.MATs, self.PCs

    def full_pc_plot(self,config,key,moment):
        """
        Plot PC coordinates for all samples in an experiment
         - Include centroids of each class
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        fig.set_figheight(15)
        fig.set_figwidth(15)

        #markers = ["$A$","$B$","$C$"]
        markers = ["$ZERO$","$ONE$","$TWO$"]
        colors = ['r','g','b']

        # for i,position in enumerate(self.PCs[key][moment]):
        count=0
        for i,pat in enumerate(config.classes):
            pat_pos = []
            for j in range(config.replicas):
                position = self.PCs[key][moment][count]
                # ax.scatter(position[0],position[1],position[2],marker=markers[i],color=colors[i],s=750,label=config.classes[i])
                pat_pos.append(position)
                count+=1
            mean_pat = np.mean(np.array(pat_pos),axis=0)
            ax.scatter(mean_pat[0],mean_pat[1],mean_pat[2],marker='.',color=colors[i],s=500,label=config.classes[i])

        plt.xlim(-2.5,2.5)
        plt.ylim(-2.5,2.5)
        ax.set_zlim(-2.5,2.5)
        # plt.legend()
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        plt.title("Positions in PCA Space",fontsize=24)
        if self.save == True:
            dirName = f"results/{config.dir}/analysis/full_PCs/{key}"
            try:
                os.makedirs(dirName)    
            except FileExistsError:
                pass
            plt.savefig(f'results/{config.dir}/analysis/full_PCs/{key}/PC_t={moment}.png')
        if self.show == True:
            plt.show()
        plt.close()

    def full_path_plot(self,config,key,range_low,range_high):
        """
        Plot path of a PC centroids for each class in an experiment
            - range arguemnts define time frame of path
            - if none, range of full experiment by default
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        fig.set_figheight(25)
        fig.set_figwidth(25)

        #markers = ["$A$","$B$","$C$"]
        markers = ["$ZERO$","$ONE$","$TWO$"]
        labels=["ZERO","ONE","TWO"]
        colors = ['r','g','b']

        x =  [[] for _ in range(config.patterns)]
        y = [[] for _ in range(config.patterns)]
        z = [[] for _ in range(config.patterns)]

        if range_low == None:
            range_low = 0
            range_high = config.length
        
        for t in range(range_low,range_high):
            count=0
            for i,pat in enumerate(config.classes):
                pat_pos = []
                for j in range(config.replicas):
                    position = self.PCs[key][t][count]
                    pat_pos.append(position)
                    count+=1
                mean_pat = np.mean(np.array(pat_pos),axis=0)
                x[i].append(mean_pat[0])
                y[i].append(mean_pat[1])
                z[i].append(mean_pat[2])

        for i in range(config.patterns):
            xs = x[i]
            ys = y[i]
            zs = z[i]
            plt.plot(xs,ys,zs, linewidth = .5,color=colors[i],label=labels[i])

        plt.xlim(-2.5,2.5)
        plt.ylim(-2.5,2.5)
        ax.set_zlim(-2.5,2.5)
        plt.legend()
        if range_low == None:
            plt.title(f"Average Replica Positions in PCA Space Traced Across Time\n{key}",fontsize=24)
            dirName = f"results/{config.dir}/analysis/full_paths/"
        else:
            plt.title(f"Average Replica Positions in PCA Space Traced Across Time from {range_low} to {range_high}\n{key}",fontsize=24)
            dirName = f'results/{config.dir}/analysis/full_paths[{range_low}{range_high}]/'
        if self.save == True:
            try:
                os.makedirs(dirName)    
            except FileExistsError:
                pass
            if range_low == None:
                plt.savefig(f'results/{config.dir}/analysis/full_paths/paths_{key}.png')
            else:
                plt.savefig(f'results/{config.dir}/analysis/full_paths[{range_low}{range_high}]/paths_{key}.png')
        if self.show == True:
            plt.show()
        plt.close()

    def pc_polygons(self,config,key,moment):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
        """
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        fig.set_figheight(15)
        fig.set_figwidth(15)

        #markers = ["$A$","$B$","$C$"]
        markers = ["$ZERO$","$ONE$","$TWO$"]
        colors = ['r','g','b']

        # for i,position in enumerate(self.PCs[key][moment]):
        count=0
        for i,pat in enumerate(config.classes):
            pat_pos = []
            for j in range(config.replicas):
                position = self.PCs[key][moment][count]
                ax.scatter(position[0],position[1],position[2],marker='.',color=colors[i],s=750,label=config.classes[i])
                pat_pos.append(position)
                count+=1
            verts = [np.array(pat_pos)]
            # print(verts.shape)
            ax.add_collection3d(Poly3DCollection(verts, 
            facecolors=colors[i], linewidths=1, edgecolors=colors[i], alpha=.25))

        plt.xlim(-2.5,2.5)
        plt.ylim(-2.5,2.5)
        ax.set_zlim(-2.5,2.5)
        # plt.legend()
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        plt.title("Positions in PCA Space",fontsize=24)
        if self.save == True:
            dirName = f"results/{config.dir}/analysis/pc_polygons/{key}"
            try:
                os.makedirs(dirName)    
            except FileExistsError:
                pass
            plt.savefig(f'results/{config.dir}/analysis/pc_polygons/{key}/PC_t={moment}.png')
        if self.show == True:
            plt.show()
        plt.close()
# save=True
# show=False
# state_analysis = StateAnalysis(config,save,show)
# state_analysis.PCs = PCs
# for i in range(700):
#     state_analysis.pc_polygons(config,list(totals)[6],i)


#%%

class DistanceAnalysis():
    def __init__(self,config,save,show):
        self.save = save
        self.show = show
        self.sweep = config.dir
        # self.directory = f'results/{config.dir}/'
        self.dirName = dirName = f'results/{config.dir}/analysis/distance_measures/'
        if exists(self.dirName):
            self.intra = read_json(dirName,"intra")
            self.intra_mean_dist = read_json(dirName,"intra_mean")
            self.inter= read_json(dirName,"inter")
            self.clust = read_json(dirName,"clust")
            self.diff = read_json(dirName,"diff")
            self.diff_sum = read_json(dirName,"dif_sum")

    def distance_measure(self,config,single):
        """
        Determine and store the following metrics for states at each time
            - Intra class mean position
            - Intra class total separation
            - Inter class separation distnaces (between centroids)
                - Note, should try paiwise
            - K-means clusterabilty
            - Difference between total inter and intra(normalized)
        """
        reps = config.replicas
        intra_ranges = [[x*reps,x*reps+reps] for x in range(config.patterns)]
        intra_t = []
        inter_t = []
        clust_t = []
        intra_mean_dist_t = []
        for t in range(config.length):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(3)
                clusts = km.fit_predict(single[:,:,t])
            eucs = []
            intra_means = []
            clusting=[]
            for intra in intra_ranges:
                combinations = list(itertools.combinations(list(range(intra[0],intra[1])),2))
                euc = [(distance.euclidean(single[comb[0],:,t],single[comb[1],:,t])) for comb in combinations]
                pos = [single[y,:,t] for y in range(intra[0],intra[1])]
                eucs.append(euc)
                intra_means.append(np.mean(pos,axis=0))
                if len(set(clusts[intra[0]:intra[1]]))==1 and len(set(clusts))==3:
                    clusting.append(1)
                # std?
            intra_t.append([eucs])
            intra_mean_dist_t.append(np.mean(np.concatenate(eucs)))
            inter_combs = list(itertools.combinations(list(range(config.patterns)),2))
            inter_dists = [distance.euclidean(intra_means[comb[0]],intra_means[comb[1]]) for comb in inter_combs]
            inter_t.append(np.mean(inter_dists))
            if np.sum(clusting)==3:
                clust_t.append(1)
            else:
                clust_t.append(0)
        # print(len(intra_t), len(intra_mean_dist_t), len(inter_t),len(clust_t))
        diff_t = np.array(inter_t)-np.array(intra_mean_dist_t)/config.replicas
        #print(diff_t)
        return intra_t, intra_mean_dist_t, inter_t,clust_t,diff_t

    def all_dists(self,config,MATs):
        """
        Perform and store distance measures for all experiments
        """
        self.intra = {}
        self.intra_mean_dist = {}
        self.inter= {}
        self.clust = {}
        self.diff = {}
        self.diff_sum = {}
        for i,(k,v) in enumerate(MATs.items()):
            print(i," - ",k)
            self.intra[k], self.intra_mean_dist[k], self.inter[k], self.clust[k], self.diff[k] = self.distance_measure(config,v)
            if np.sum(self.clust[k])>0:
                print(f"         {np.sum(self.clust[k])} successful moments of clustering!")
            self.diff_sum[k] =  np.sum(self.diff[k])
        
        dist_ranked = dict(reversed(sorted(self.diff_sum.items(), key=lambda item: item[1])))
        for k,v in dist_ranked.items():
            print(np.round(v,4)," - ",k)


        dirName = f'results/{config.dir}/analysis/distance_measures/'
        try:
            os.makedirs(dirName)    
        except FileExistsError:
            pass

        write_dict(self.intra,dirName,"intra")
        write_dict(self.intra_mean_dist,dirName,"intra_mean")
        write_dict(self.inter,dirName,"inter")
        write_dict(self.clust,dirName,"clust")
        write_dict(self.diff,dirName,"diff")
        write_dict(self.diff_sum,dirName,"dif_sum")
            
    def dist_plot(self,dict):
        """
        Plot distances over performance rankings
        """
        ordered_distance = []
        plt.figure(figsize=(10,10))
        for k in dict.keys():
            ordered_distance.append(self.diff_sum[k])
        plt.plot(ordered_distance,'.k')
        if self.save==True:
            path = f'results/{self.sweep}/analysis/distance_plot.png'
            plt.savefig(path)
        if self.show==True:
            plt.show()
        else:
            plt.close()


class MetaAnalysis():
    def __init__(self,config,save,show):
        self.save = save
        self.show = show
        self.directory = f'results/{config.dir}/'

    def show_all(self,config,key):
        """"
        Show relevant plots for specific experiment
        """
        for i in range(config.patterns):
            plot = Image.open(f'{self.directory}/liquid/plots/{key}_pat{config.classes[i]}_rep0.png')
            plot.show()
        plot = Image.open(f'{self.directory}/performance/plots/{key}_performance.png')
        plot.show()
        plot = Image.open(f'{self.directory}/analysis/full_paths[150699]/paths_{key}.png') #[150699]
        plot.show()

    def dict_compare(self,config,dict1,dict2):
        """
        Compare relationship between two ranked dictionaries
         - For each exerpiment, store position from each dict
        """
        I = []
        J = []
        same = 0
        for i, (k1,v1) in enumerate(dict1.items()):
            for j, (k2,v2) in enumerate(dict2.items()):
                if k1 == k2:
                    I.append(i)
                    J.append(j)
        
        total_difference=0
        for i in range(len(I)):
            total_difference+= np.abs(I[i] - J[i])
            if I[i] - J[i] ==0:
                same+=1
        print("Total difference in rankings: ",total_difference)
        print("Amound of exact same rankings: ",same,"/",len(dict1))
        return I,J

    def ranking_comparison_plot(self,I,J):
        """"
        Plot lines between rankings (thinkness for closer orderings)
        """
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

    def dist_plot(self,rank,dict):
        ordered_distance = []
        plt.figure(figsize=(10,10))
        for k in rank.keys():
            ordered_distance.append(dict[k])
        plt.plot(ordered_distance,'.k')
        plt.show()

    # def control_test(self,second_sweep):





# %%

