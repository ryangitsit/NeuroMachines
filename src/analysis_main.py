from full_analysis import PerformanceAnalysis, StateAnalysis, DistanceAnalysis, MetaAnalysis
import os
import pickle
from os.path import exists

from processing import read_json

""""
Execution file for full_analysis.py
"""


def main():
    #sweep = "rerun_LSTP"
    # sweep = "hei_large2"
    sweep = "SuperSweep"
    save = True
    show = False

    # import configuration settings for a given sweep
    directory = f'results/{sweep}/configs'
    filename = os.listdir(directory)[1]
    file = os.path.join(directory, filename)
    file_to_read = open(file, "rb")
    config = pickle.load(file_to_read)
    file_to_read.close()


    ### Performance Analysis ###
    """
     - Ranks performance on total and final certainties
     - Super plot of all certainty trajectories
     - Plot parameter correlation statistics
     - Multi-plot best performers
    """
    full_analysis = PerformanceAnalysis(config,save,show)
    full_analysis.performance_pull()
    for i in range(20,35):
        full_analysis.perfromance_t(i)

    full_analysis.accs_plots()
    finals, totals = full_analysis.rankings()
    full_analysis.print_rankings(finals,"Final Performance",100)
    full_analysis.print_rankings(totals,"Total Performance",100)
    full_analysis.performance_statistics(config,totals,100) # must not exceed experiment total
    full_analysis.hist_ranked()
    full_analysis.top_plot(20)

    # top_finals=dict(itertools.islice(finals.items(),20))
    # top_totals=dict(itertools.islice(totals.items(),20))
    # full_analysis.accs_plots(top_totals)
    # full_analysis.accs_plots(top_finals)

    


    ### State Analysis ###
    config.old_encoded = False
    state_analysis = StateAnalysis(config,save,show)

    # If analysis has already been run once, use saved results
    if exists(f'results/{sweep}/analysis/all_pcs.json'):
        # dirName = f'results/{sweep}/analysis/'
        # item = 'all_pcs'
        # PCs = read_json(dirName,item)
        # state_analysis.PCs = PCs
        # # item2 = 'all_mats'
        # # MATs = read_json(dirName,item2)
        # # state_analysis.MATs = MATs
        MATs, PCs = state_analysis.analysis_loop(config,True)
    else:
        MATs, PCs = state_analysis.analysis_loop(config,False)

    # # Plot all full paths 
    # print(f"Plotting all PC paths...")
    # for i in range(50,200):
    #     print(list(totals)[i])
    #     state_analysis.full_path_plot(config,list(totals)[i],150,650)

    # # Plot PCs at every moment for top total performer
    # # print(f"Plotting all PCs of top total performing experiment...\n {list(totals)[0]}")
    # # for t in range(config.length):
    # #     state_analysis.full_pc_plot(config,list(totals)[0],t)

    # ### Distance Analysis ###
    #Determine distance metrics across states for different samples
    dist = DistanceAnalysis(config,save,show)
    dist.all_dists(config,MATs)

    ### Meta Analysis ###
    # meta = MetaAnalysis(config,save,show)
    # meta.show_all(config,list(totals)[0])


if __name__ == "__main__":
    main()
