#%%

import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import string
import pickle
from sklearn.decomposition import PCA
from PIL import Image

from processing import txt_to_spks,one_hot

"""
Plan:

- Generate one-hot encoded sets of liquid states and store them
- Create time-slice groups across all liquids
- Generate PCs for each slice and store
- Analyze average distance over time of PCs and full dimensions
    - Centroid separations
- Plot separation measures over performance
"""

# Performance (final accuracy)
# Full Plot
# Top Plot

# PCs and Groups of time slices (store)

# Average separation over time in both spaces (log comparison)

# Determine coherent expansion coefficient of input

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

    def accs_plots(self,tops=None):
        dict = self.all_avgs.items()
        plt.figure(figsize=(16, 10))
        for i,(file, avg) in enumerate(dict):
            if tops==None:
                plt.plot(avg)
            else:
                if file in tops:
                    plt.plot(avg)
            plt.title(f"Prediction Certainty Over Time",fontsize=24)
            plt.ylim(0,1)
            plt.xlabel("Time (ms), dt=1ms",fontsize=22)
            plt.ylabel("Certainty for Correct Classification",fontsize=22)
        if self.save==True:
            if tops==None:
                path = f'results/{self.sweep}/analysis/perfplot.png'
            else:
                path = f'results/{self.sweep}/analysis/top{len(tops)}_perfplot.png'
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

    def performance_statistics(self,dict,lim):
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
            "delay":[]
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
            print(param_combo)
            for k,v in occ_combos.items():
                if str(k[0]) != 'None' and str(k[1]) != 'None':
                    print("   ",k," - ",v)
            self.combos[param_combo] = occ_combos

    def hist_ranked(self):
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
        plt.show()


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
        for i,pattern in enumerate(self.classes):
            suffix = "_pat"+pattern+"_rep0.txt"
            prefix = f'results/{self.sweep}/liquid/spikes/'
            for j,name in enumerate(top_5):
                #print(name+suffix)
                dat, indices, times = txt_to_spks(prefix+name+suffix)
                axs[j, i].plot(times, indices, '.k', ms=.1)
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

    def print_config(self):
        print(config.__dict__)

    def analysis_loop(self):
        np.seterr(divide='ignore', invalid='ignore')
        experiments = int(len(os.listdir(self.directory))/(config.patterns*config.replicas))
        count = 0
        self.MATs = {}
        self.PCs = {}
        for exp in range(experiments):
            for pat,label in enumerate(config.classes):
                for r in range(config.replicas):
                    filename = os.listdir(self.directory)[count]
                    file = os.path.join(self.directory, filename)
                    if (count) % 9 == 0:
                        a = file
                        b = '_pat'
                        pat_loc = [(i, i+len(b)) for i in range(len(a)) if a[i:i+len(b)] == b]
                        exp_name=file[len(self.directory)+1:pat_loc[0][0]]
                        print(f"{exp}-{count} experiment: {exp_name}")

                        mat_path = f'results/{config.dir}/performance/liquids/encoded/mat_{exp_name}.npy'
                        mat = np.load(mat_path, allow_pickle=True)

                        # # Across each replica within a pattern
                        # pcs_times = []
                        # for t in range(config.length):
                        #     step = 0
                        #     pc_pats = []
                        #     for p,pattern in enumerate(config.classes):
                        #         norms = []
                        #         for r in range(config.replicas):
                        #             slice = mat[step][:,t]
                        #             norm = np.array(slice) - np.mean(slice)
                        #             norms.append(norm)
                        #             step+=1
                        #         norms = np.array(norms)
                        #         pc_obj = PCA(n_components=3)
                        #         pc_slice = pc_obj.fit_transform(norms)
                        #         pc_pat = pc_slice[:,0]
                        #         pc_pats.append(pc_pat)
                        #     pcs_times.append(np.array(pc_pats))
                        # pcs_times = np.array(pcs_times)

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

                    #dat,indices,times = txt_to_spks(file)
                    count+=1

        return self.MATs, self.PCs

    def pc_plot(self,key,moment):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        fig.set_figheight(15)
        fig.set_figwidth(15)

        #markers = ["$A$","$B$","$C$"]
        markers = ["$ZERO$","$ONE$","$TWO$"]
        colors = ['r','g','b']

        for i,position in enumerate(self.PCs[key][moment]):
            ax.scatter(position[0],position[1],position[2],marker=markers[i],color=colors[i],s=750,label=config.classes[i])

        plt.xlim(-5,5)
        plt.ylim(-5,5)
        ax.set_zlim(-5,5)
        plt.legend()

        # ax.set_xlabel('Replica 0 Component')
        # ax.set_ylabel('Replica 1 Component')
        # ax.set_zlabel('Replica 2 Component')
        plt.title("Positions in PCA Space",fontsize=24)
        if save == True:
            dirName = f"results/{config.dir}/analysis/PCs/{key}/"
            try:
                os.makedirs(dirName)    
            except FileExistsError:
                pass
            plt.savefig(f'results/{config.dir}/analysis/PCs/{key}PC_t={moment}.png')
        if show == True:
            plt.show()
        plt.close()

    def full_pc_plot(self,key,moment):
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

        plt.xlim(-5,5)
        plt.ylim(-5,5)
        ax.set_zlim(-5,5)
        # plt.legend()
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        plt.title("Positions in PCA Space",fontsize=24)
        if save == True:
            dirName = f"results/{config.dir}/analysis/full_PCs/{key}"
            try:
                os.makedirs(dirName)    
            except FileExistsError:
                pass
            plt.savefig(f'results/{config.dir}/analysis/full_PCs/{key}/PC_t={moment}.png')
        if show == True:
            plt.show()
        plt.close()

    def path_plot(self,key):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        fig.set_figheight(25)
        fig.set_figwidth(25)

        #markers = ["$A$","$B$","$C$"]
        markers = ["$ZERO$","$ONE$","$TWO$"]
        labels=["ZERO","ONE","TWO"]
        colors = ['r','g','b']
        x = []
        y = []
        z = []

        for i in range(config.patterns):
            xs = self.PCs[key][:,i][:,0]
            ys = self.PCs[key][:,i][:,1]
            zs = self.PCs[key][:,i][:,2]
            plt.plot(xs,ys,zs, linewidth = .5,color=colors[i],label=labels[i])

        plt.xlim(-5,5)
        plt.ylim(-5,5)
        ax.set_zlim(-5,5)
        plt.legend()
        plt.title(f"Positions in PCA Space Traced Across Time\n{key}",fontsize=24)
        dirName = f"results/{config.dir}/analysis/paths/"
        try:
            os.makedirs(dirName)    
        except FileExistsError:
            pass
        plt.savefig(f'results/{config.dir}/analysis/paths/paths_{key}.png')
        plt.close()

    def full_path_plot(self,key):
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

        for t in range(config.length):
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

        plt.xlim(-5,5)
        plt.ylim(-5,5)
        ax.set_zlim(-5,5)
        plt.legend()
        plt.title(f"Average Replica Positions in PCA Space Traced Across Time\n{key}",fontsize=24)
        if save == True:
            dirName = f"results/{config.dir}/analysis/full_paths/"
            try:
                os.makedirs(dirName)    
            except FileExistsError:
                pass
            plt.savefig(f'results/{config.dir}/analysis/full_paths/paths_{key}.png')
        if show == True:
            plt.show()
        plt.close()
    

class MetaAnalysis():
    def __init__(self,config,save,show):
        self.save = save
        self.show = show
        self.directory = f'results/{config.dir}/'

    def show_all(self,key):
        # import cv2
        # for pat in config.classes:
        #     plot = Image.open(f'{self.directory}/liquid/plots/{key}_pat{pat}_rep0.png')
        #     plot.show()
        # plot = Image.open(f'{self.directory}/performance/plots/{key}_performance.png')
        # plot.show()
        plot = Image.open(f'{self.directory}/analysis/full_paths/paths_{key}.png')
        plot.show()

        # img1 = cv2.imread(f'{self.directory}/analysis/full_paths/paths_{key}.png')
        # img2 = cv2.imread(f'{self.directory}/performance/plots/{key}_performance.png')
        # print(img1.shape, img2.shape)
        # movex=1000
        # movey=1000
        # img1 = img1[movex:movex+img2.shape[0],movey:movey+img2.shape[1]]
        # print(img1.shape)
        # horz = np.concatenate((img1, img2), axis=1)
        # cv2.imshow('Title', horz)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # def control_test(self,second_sweep):



sweep = "hei_phei"

directory = f'results/{sweep}/configs'
filename = os.listdir(directory)[1]
file = os.path.join(directory, filename)
file_to_read = open(file, "rb")
config = pickle.load(file_to_read)
file_to_read.close()
save = True
show = False

full_analysis = PerformanceAnalysis(config,save,show)
full_analysis.performance_pull()
finals, totals = full_analysis.rankings()
full_analysis.print_rankings(finals,"Final Performance",336)

full_analysis.print_rankings(totals,"Total Performance",336)
# full_analysis.performance_statistics(finals,20)
# # full_analysis.hist_ranked()

# top_finals=dict(itertools.islice(finals.items(),8))
# top_totals=dict(itertools.islice(totals.items(),8))

# full_analysis.print_rankings(totals,"Final Performance",10)

# config.dir = sweep + "_rerun"
# re_analysis = PerformanceAnalysis(config,save,show)
# re_analysis.performance_pull()
# refinals, retotals = re_analysis.rankings()
# # full_analysis.performance_statistics(finals,20)
# # full_analysis.hist_ranked()

# # top_finals=dict(itertools.islice(finals.items(),8))
# # top_totals=dict(itertools.islice(totals.items(),8))

# re_analysis.print_rankings(retotals,"Final Performance",10)

# for i,(k,v) in enumerate(full_analysis.all_finals.items()):
#     print(np.abs(v - re_analysis.all_finals[k]))

### STATES ###
state_analysis = StateAnalysis(config,save,show)
MATs, PCs = state_analysis.analysis_loop()

# # key = 'Maass_geo=(randNone_geo[45, 3, 1]_smNone)_N=135_IS=0.17_RS=0.3_ref=3.0_delay=1.5_U=0.6'
# key = 'LSTP_smw=(randNone_geoNone_sm0.0)_N=135_IS=0.17_RS=0.1_ref=3.0_delay=0.0_U=0.6'

## Plot all full paths ##
# for i in range(len(PCs)):
#     state_analysis.full_path_plot(list(PCs)[i])

## Plot all PCs for one config ## 
# for t in range(config.length):
#     state_analysis.full_pc_plot(list(totals)[-1],t)

# for t in range(config.length):
#     state_analysis.pc_plot(list(PCs)[23],t)
    # state_analysis.pc_plot('Maass_geo=(randNone_geo[9, 5, 3]_smNone)_N=135_IS=0.17_RS=0.1_ref=3.0_delay=1.5_U=0.6',t)



#%%
class DistsanceAnalysis():
    def __init__(self,MATs,PCs,config,save,show):
        self.save = save
        self.show = show
        # self.directory = f'results/{config.dir}/'


    def distance_measure(self,single):
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

    def all_dists(self):
        self.intra = {}
        self.intra_mean_dist = {}
        self.inter= {}
        self.clust = {}
        self.diff = {}
        self.diff_sum = {}
        for i,(k,v) in enumerate(MATs.items()):
            print(i," - ",k)
            self.intra[k], self.intra_mean_dist[k], self.inter[k], self.clust[k], self.diff[k] = self.distance_measure(v)
            if np.sum(self.clust[k])>0:
                print(f"Clustering with {np.sum(self.clust[k])} success")
            self.diff_sum[k] =  np.sum(self.diff[k])
        
        dist_ranked = dict(reversed(sorted(self.diff_sum.items(), key=lambda item: item[1])))
        for k,v in dist_ranked.items():
            print(np.round(v,4)," - ",k)
            

from sklearn.cluster import KMeans
from scipy.spatial import distance
import warnings
dist = DistsanceAnalysis(config,MATs,PCs,save,show)
# single = MATs[list(MATs)[0]]
# dist.distance_measure(single)v
dist.all_dists()
#%%
ordered_distance = []
plt.figure(figsize=(10,10))
for k in totals.keys():
    ordered_distance.append(dist.diff_sum[k])
plt.plot(ordered_distance,'.k')
#%%


### META ###

meta = MetaAnalysis(config,save,show)
# meta.show_all(list(totals)[0])

# for k in top_finals.keys():
    # meta.show_all(k)

for k in list(totals)[-30:-10]:
    meta.show_all(k)



