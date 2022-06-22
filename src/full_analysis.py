import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import string
import pickle
from sklearn.decomposition import PCA

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

    def top_plot(self):

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
        top_5 = list(self.final_perf_ranking)[:5]
        for i,pattern in enumerate(self.classes):
            suffix = "_pat"+pattern+"_rep0.txt"
            prefix = f'results/{self.sweep}/liquid/spikes/'
            for j,name in enumerate(top_5):
                #print(name+suffix)
                dat, indices, times = txt_to_spks(prefix+name+suffix)
                axs[j, i].plot(times, indices, '.k', ms=.1)
                axs[j, i].set_title(name, size=8)
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
        experiments = 1 #int(len(os.listdir(self.directory))/(config.patterns*config.replicas))
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

                        pcs_times = []
                        for t in range(config.length):
                            step = 0
                            pc_pats = []
                            for p,pattern in enumerate(config.classes):
                                norms = []
                                for r in range(config.replicas):
                                    slice = mat[step][:,t]
                                    norm = np.array(slice) - np.mean(slice)
                                    norms.append(norm)
                                    step+=1
                                norms = np.array(norms)
                                pc_obj = PCA(n_components=3)
                                pc_slice = pc_obj.fit_transform(norms)
                                pc_pat = pc_slice[:,0]
                                pc_pats.append(pc_pat)
                            pcs_times.append(np.array(pc_pats))
                        pcs_times = np.array(pcs_times)

                        # pcs_times = []
                        # for t in range(200,201): #range(config.length):
                        #     step = 0
                        #     pc_pats = []
                        #     norms = []
                        #     for p,pattern in enumerate(config.classes):
                        #         print(p)
                        #         # norms = []
                        #         for r in range(config.replicas):
                        #             print(r)
                        #             slice = mat[step][:,t]
                        #             norm = np.array(slice) - np.mean(slice)
                        #             norms.append(norm)
                        #             step+=1
                        #     norms = np.array(norms)
                        #     pc_obj = PCA(n_components=3)
                        #     pc_slice = pc_obj.fit_transform(norms)
                        #     print(pc_slice)
                        #     pc_pat = pc_slice[:,0]
                        #     pc_pats.append(pc_pat)

                        #     pcs_times.append(np.array(pc_pats))
                        # pcs_times = np.array(pcs_times)

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
    

sweep = "hei_phei"

directory = f'results/{sweep}/configs'
filename = os.listdir(directory)[1]
file = os.path.join(directory, filename)
file_to_read = open(file, "rb")
config = pickle.load(file_to_read)
file_to_read.close()

#print(config.__dict__)


# type = "Heidelberg"
# patterns = 3
# replicas = 3
save = False
show = False
# full_analysis = PerformanceAnalysis(sweep,type,patterns,replicas,save,show)
# classes = full_analysis.classes

state_analysis = StateAnalysis(config,save,show)

MATs, PCs = state_analysis.analysis_loop()

# for t in range(config.length):
#     state_analysis.pc_plot(list(PCs)[23],t)
    # state_analysis.pc_plot('Maass_geo=(randNone_geo[9, 5, 3]_smNone)_N=135_IS=0.17_RS=0.1_ref=3.0_delay=1.5_U=0.6',t)

# for i in range(len(PCs)):
#     state_analysis.path_plot(list(PCs)[i])

# full_analysis.performance_pull()
# full_analysis.accs_plots()
# finals, totals = full_analysis.rankings()
# full_analysis.print_rankings(finals,"Final Performance",10)
# full_analysis.print_rankings(totals,"Total Performance",10)

# top_finals=dict(itertools.islice(finals.items(),20))
# top_totals=dict(itertools.islice(totals.items(),20))
# full_analysis.accs_plots(top_totals)
# full_analysis.accs_plots(top_finals)
# full_analysis.top_plot()





