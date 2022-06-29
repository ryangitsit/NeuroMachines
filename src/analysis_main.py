from full_analysis import PerformanceAnalysis, StateAnalysis, DistanceAnalysis, MetaAnalysis
import os
import pickle
from os.path import exists

from processing import read_json

""""
Execution file for full_analysis.py
"""


def main():
    sweep = "hei_test"
    save = True
    show = False

    # import configuration settings for a given sweep
    directory = f'results/{sweep}/configs'
    filename = os.listdir(directory)[1]
    file = os.path.join(directory, filename)
    file_to_read = open(file, "rb")
    config = pickle.load(file_to_read)
    file_to_read.close()

    config.patterns = 3
    config.replicas = 3
    config.classes = config.classes[:config.patterns]

    ### Performance Analysis ###
    """
     - Ranks performance on total and final certainties
     - Super plot of all certainty trajectories
     - Plot parameter correlation statistics
     - Multi-plot best performers
    """
    full_analysis = PerformanceAnalysis(config,save,show)
    full_analysis.performance_pull()
    full_analysis.accs_plots()
    finals, totals = full_analysis.rankings()
    full_analysis.print_rankings(finals,"Final Performance",50)
    full_analysis.print_rankings(totals,"Total Performance",50)
    full_analysis.performance_statistics(config,totals,25)
    full_analysis.hist_ranked()
    # top_finals=dict(itertools.islice(finals.items(),20))
    # top_totals=dict(itertools.islice(totals.items(),20))
    # full_analysis.accs_plots(top_totals)
    # full_analysis.accs_plots(top_finals)
    full_analysis.top_plot()


    ### State Analysis ###
    # config.old_encoded = False
    # state_analysis = StateAnalysis(config,save,show)

    # # If analysis has already been run once, use saved results
    # if exists(f'results/{sweep}/analysis/all_pcs.json'):
    #     dirName = f'results/{sweep}/analysis/'
    #     item = 'all_pcs'
    #     PCs = read_json(dirName,item)
    #     state_analysis.PCs = PCs
    #     # item2 = 'all_mats'
    #     # MATs = read_json(dirName,item2)
    #     # state_analysis.MATs = MATs
    # else:
    #     MATs, PCs = state_analysis.analysis_loop(config)

    # # Plot all full paths 
    # print(f"Plotting all PC paths...")
    # for i in range(len(PCs)):
    #     state_analysis.full_path_plot(config,list(PCs)[i],150,650)

    # # Plot PCs at every moment for top total performer
    # # print(f"Plotting all PCs of top total performing experiment...\n {list(totals)[0]}")
    # # for t in range(config.length):
    # #     state_analysis.full_pc_plot(config,list(totals)[0],t)

    # ### Distance Analysis ###
    # #Determine distance metrics across states for different samples
    # dist = DistanceAnalysis(config,save,show)
    # dist.all_dists(config,MATs)

if __name__ == "__main__":
    main()
