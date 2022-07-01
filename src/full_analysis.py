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
            plt.ylim(0,1.1)
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
            "x_atory":[]
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
        
        hyper_combos = list(itertools.combinations(list(hyper_parameters),2))
        self.combos = {}
        for param_combo in hyper_combos:
            sets = []
            for i in range(lim):
                sets.append((hyper_parameters[param_combo[0]][i],hyper_parameters[param_combo[1]][i]))
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
        print(dict)
        RND=[0,0,0,0]
        GEO=[0,0,0,0]
        SMW=[0,0,0,0]
        for i,l in enumerate(labels):
            for k,v in dict.items():
                if k[0] == l:
                    if k[1] == 'geo':
                        GEO[i]+=v*100/self.lim
                    if k[1] == 'rnd':
                        RND[i]+=v*100/self.lim
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


    def top_plot(self):
        """
        Plotting Top 5 Performers and Their Replicas
        - Create a subplot grid
        - For each pattern replica
            - For each Top5 performer
                - Convert top performing names to paths
                - Pull the appropriate spikes
                - Raster plot them into the subplot grid
        """
        fig, axs = plt.subplots(5, 3,figsize=(24,14))
        plt.title("Title")
        top_5 = list(self.final_perf_ranking)[:5]
        print(self.classes[:self.patterns])
        for i,pattern in enumerate(self.classes[:self.patterns]):
            suffix = "_pat"+pattern+"_rep0.txt"
            prefix = f'results/{self.sweep}/liquid/spikes/'
            for j,name in enumerate(top_5):
                if not exists(prefix+name+suffix):
                    break
                #print(name+suffix)
                dat, indices, times = txt_to_spks(prefix+name+suffix)
                axs[j, i].plot(times, indices, '.k', ms=.7)
                axs[j, i].set_title(name, size=6)
        for ax in axs.flat:
            ax.set(xlabel='time (ms)', ylabel='neuron index')
        for ax in axs.flat:
            ax.label_outer()

        if self.save==True:
            path = f'results/{self.sweep}/analysis/top_performers.png'
            plt.savefig(path)
        if self.show==True:
            plt.show()
        else:
            plt.close()


class StateAnalysis():
    def __init__(self,config,save,show):
        self.save = save
        self.show = show
        self.directory = f'results/{config.dir}/liquid/spikes'

    def print_config(self,config):
        print(config.__dict__)

    def analysis_loop(self,config):
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
            self.MATs[exp_name] = mat
            self.PCs[exp_name] = pcs_times

        
        dict = self.PCs
        path = f'results/{config.dir}/analysis/'
        name = 'all_pcs'
        write_dict(dict,path,name)

        # dict = self.MATs
        # name = 'all_mats'
        # write_dict(dict,path,name)

        return self.MATs, self.PCs

    # def analysis_loop(self,config):
    #     np.seterr(divide='ignore', invalid='ignore')
    #     experiments = int(len(os.listdir(self.directory))/(config.patterns*config.replicas))
    #     count = 0
    #     self.MATs = {}
    #     self.PCs = {}
    #     for exp in range(experiments):
    #         for pat,label in enumerate(config.classes):
    #             for r in range(config.replicas):
    #                 filename = os.listdir(self.directory)[count]
    #                 file = os.path.join(self.directory, filename)
    #                 if (count) % 9 == 0:
    #                     a = file
    #                     b = '_pat'
    #                     pat_loc = [(i, i+len(b)) for i in range(len(a)) if a[i:i+len(b)] == b]
    #                     exp_name=file[len(self.directory)+1:pat_loc[0][0]]
    #                     print(f"{exp}-{count} experiment: {exp_name}")

    #                     mat_path = f'results/{config.dir}/performance/liquids/encoded/mat_{exp_name}.npy'
    #                     mat = np.load(mat_path, allow_pickle=True)

    #                     # # Across each replica within a pattern
    #                     # pcs_times = []
    #                     # for t in range(config.length):
    #                     #     step = 0
    #                     #     pc_pats = []
    #                     #     for p,pattern in enumerate(config.classes):
    #                     #         norms = []
    #                     #         for r in range(config.replicas):
    #                     #             slice = mat[step][:,t]
    #                     #             norm = np.array(slice) - np.mean(slice)
    #                     #             norms.append(norm)
    #                     #             step+=1
    #                     #         norms = np.array(norms)
    #                     #         pc_obj = PCA(n_components=3)
    #                     #         pc_slice = pc_obj.fit_transform(norms)
    #                     #         pc_pat = pc_slice[:,0]
    #                     #         pc_pats.append(pc_pat)
    #                     #     pcs_times.append(np.array(pc_pats))
    #                     # pcs_times = np.array(pcs_times)

    #                     # Across all
    #                     pcs_times = []
    #                     for t in range(config.length):
    #                         step = 0
    #                         pc_pats = []
    #                         norms = []
    #                         for p,pattern in enumerate(config.classes):
    #                             # norms = []
    #                             for r in range(config.replicas):
    #                                 slice = mat[step][:,t]
    #                                 norm = np.array(slice) - np.mean(slice)
    #                                 norms.append(norm)
    #                                 step+=1
    #                         norms = np.array(norms)
    #                         pc_obj = PCA(n_components=3)
    #                         pc_slice = pc_obj.fit_transform(norms)
    #                         #print(pc_slice)
    #                         pcs_times.append(np.array(pc_slice))
    #                     pcs_times = np.array(pcs_times)
    #                     self.MATs[exp_name] = mat
    #                     self.PCs[exp_name] = pcs_times

    #                 #dat,indices,times = txt_to_spks(file)
    #                 count+=1
        
    #     dict = self.PCs
    #     path = f'results/{config.dir}/analysis/'
    #     name = 'all_pcs'
    #     write_dict(dict,path,name)

    #     return self.MATs, self.PCs


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
                ax.scatter(position[0],position[1],position[2],marker=markers[i],color=colors[i],s=750,label=config.classes[i])
                pat_pos.append(position)
                count+=1
            mean_pat = np.mean(np.array(pat_pos),axis=0)
            ax.scatter(mean_pat[0],mean_pat[1],mean_pat[2],marker='.',color=colors[i],s=350,label=config.classes[i])

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
    

class DistanceAnalysis():
    def __init__(self,config,save,show):
        self.save = save
        self.show = show
        # self.directory = f'results/{config.dir}/'

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
        plot = Image.open(f'{self.directory}/liquid/plots/{key}_pat{config.classes[0]}_rep0.png')
        plot.show()
        plot = Image.open(f'{self.directory}/performance/plots/{key}_performance.png')
        plot.show()
        # plot = Image.open(f'{self.directory}/analysis/full_paths/paths_{key}.png')
        # plot.show()

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

    # def control_test(self,second_sweep):


# sweep = "hei_large2"

# directory = f'results/{sweep}/configs'
# filename = os.listdir(directory)[1]
# file = os.path.join(directory, filename)
# file_to_read = open(file, "rb")
# config = pickle.load(file_to_read)
# file_to_read.close()
# save = False
# show = True

# dirName = f'results/{sweep}/analysis/distance_measures/'
# item = 'dif_sum'
# diff_sum = read_json(dirName,item)
# item = 'clust'
# clust = read_json(dirName,item)
# clusts = {}

# clustered = 0
# for k,v in clust.items():
#     clusts[k] = np.sum(v)
#     if np.sum(v) >0:
#         clustered+=1
# print(clustered)
# clusts = dict(sorted(clusts.items(), key=lambda item: item[1],reverse=True))


# full_analysis = PerformanceAnalysis(config,save,show)
# full_analysis.performance_pull()
# finals, totals = full_analysis.rankings()
# full_analysis.print_rankings(finals,"Final Performance",20)
# full_analysis.print_rankings(totals,"Total Performance",20)

# meta = MetaAnalysis(config,save,show)
# I,J =  meta.dict_compare(config,totals,clusts)
# meta.ranking_comparison_plot(I,J)

# # meta.show_all(list(totals)[0])

# # for k in list(finals)[:10]:
# #     meta.show_all(k)

# # full_analysis.print_rankings(finals,"Final Performance",336)

# # full_analysis.print_rankings(totals,"Total Performance",336)
# # # full_analysis.performance_statistics(finals,20)
# # # # full_analysis.hist_ranked()

# # # top_finals=dict(itertools.islice(finals.items(),8))
# # # top_totals=dict(itertools.islice(totals.items(),8))

# # # full_analysis.print_rankings(totals,"Final Performance",10)

# # # config.dir = sweep + "_rerun"
# # # re_analysis = PerformanceAnalysis(config,save,show)
# # # re_analysis.performance_pull()
# # # refinals, retotals = re_analysis.rankings()
# # # # full_analysis.performance_statistics(finals,20)
# # # # full_analysis.hist_ranked()

# # # # top_finals=dict(itertools.islice(finals.items(),8))
# # # # top_totals=dict(itertools.islice(totals.items(),8))

# # # re_analysis.print_rankings(retotals,"Final Performance",10)

# # # for i,(k,v) in enumerate(full_analysis.all_finals.items()):
# # #     print(np.abs(v - re_analysis.all_finals[k]))

# # ### STATES ###
# # state_analysis = StateAnalysis(config,save,show)
# # MATs, PCs = state_analysis.analysis_loop()

# # # # key = 'Maass_geo=(randNone_geo[45, 3, 1]_smNone)_N=135_IS=0.17_RS=0.3_ref=3.0_delay=1.5_U=0.6'
# # # key = 'LSTP_smw=(randNone_geoNone_sm0.0)_N=135_IS=0.17_RS=0.1_ref=3.0_delay=0.0_U=0.6'

# # ## Plot all full paths ##
# # # for i in range(len(PCs)):
# # #     state_analysis.full_path_plot(list(PCs)[i])

# # ## Plot all PCs for one config ## 
# # # for t in range(config.length):
# # #     state_analysis.full_pc_plot(list(totals)[-1],t)

# # # for t in range(config.length):
# # #     state_analysis.pc_plot(list(PCs)[23],t)
# #     # state_analysis.pc_plot('Maass_geo=(randNone_geo[9, 5, 3]_smNone)_N=135_IS=0.17_RS=0.1_ref=3.0_delay=1.5_U=0.6',t)



# # #%%



# # # from sklearn.cluster import KMeans
# # # from scipy.spatial import distance
# # # import warnings
# # # dist = DistsanceAnalysis(config,MATs,PCs,save,show)
# # # # single = MATs[list(MATs)[0]]
# # # # dist.distance_measure(single)v
# # # dist.all_dists()
# # # #%%
# # # ordered_distance = []
# # # plt.figure(figsize=(10,10))
# # # for k in totals.keys():
# # #     ordered_distance.append(dist.diff_sum[k])
# # # plt.plot(ordered_distance,'.k')
# # #%%


# # ### META ###

# # meta = MetaAnalysis(config,save,show)
# # # meta.show_all(list(totals)[0])

# # # for k in top_finals.keys():
# #     # meta.show_all(k)

# # # for k in list(totals)[-30:-10]:
# # #     meta.show_all(k)




# %%

