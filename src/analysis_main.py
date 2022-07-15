from re import L
from full_analysis import PerformanceAnalysis, StateAnalysis, DistanceAnalysis, MetaAnalysis
import os
import pickle
from os.path import exists
import numpy as np

from processing import read_json, write_dict

""""
Execution file for full_analysis.py
"""


def main():
    # sweep = "rerun_LSTP"
    # sweep = "hei_large2"
    sweep = "SuperSweep_MNIST_asymm"
    # sweep = "LightSweep_4r"

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
    # for i in range(20,35):
    #     full_analysis.perfromance_t(i)

    full_analysis.accs_plots()
    finals, totals = full_analysis.rankings()

    full_analysis.print_rankings(finals,"Final Performance",520)
    full_analysis.print_rankings(totals,"Total Performance",200)



    # dict = finals
    # path = f'results/{config.dir}/analysis/'
    # name = "finals"
    # write_dict(dict,path,name)
    

    full_analysis.performance_statistics(config,totals,182) # must not exceed experiment total

    full_analysis.hist_ranked()
    full_analysis.hist_alt()
    full_analysis.hist_alt_top()
    # k1 = 'Maass_smw=(randNone_geoNone_sm0.0)_N=135_IS=0.2_RS=0.1_ref=0.0_delay=1.5_U=0.6_XTrue_feedcontinuous_IDNone'
    # k2 = 'LSTP_geo=(randNone_geo[27, 5, 1]_smNone)_N=135_IS=0.1_RS=0.2_ref=0.0_delay=1.5_U=0.6_XFalse_feedcontinuous_IDNone'
    # k3 = 'STDP_smw=(randNone_geoNone_sm0.0)_N=135_IS=0.1_RS=0.1_ref=0.0_delay=1.5_U=0.6_XFalse_feedcontinuous_IDNone'
    # k4 = 'STSP_rnd=(rand0.2_geoNone_smNone)_N=135_IS=0.3_RS=0.2_ref=0.0_delay=1.5_U=0.6_XTrue_feedreset_IDNone'
    # lst = [k2,k1,k3,k4]
    lst = list(totals)[:5]
    full_analysis.top_plot(5,"list",lst)

    # top_finals=dict(itertools.islice(finals.items(),10))
    # top_totals=dict(itertools.islice(totals.items(),10))
    # full_analysis.accs_plots(top_totals)
    # full_analysis.accs_plots(top_finals)

    


    # # ### State Analysis ###
    config.old_encoded = False
    state_analysis = StateAnalysis(config,save,show)

    # # If analysis has already been run once, use saved results
    if exists(f'results/{sweep}/analysis/all_pcs.json'):
        MATs, PCs = state_analysis.analysis_loop(config,False)
    else:
        MATs, PCs = state_analysis.analysis_loop(config,True)

    # dirName = f'results/{sweep}/analysis/'
    # item = 'all_pcs'
    # PCs = read_json(dirName,item)
    # state_analysis.PCs = PCs
    # k = 'STSP_rnd=(rand0.2_geoNone_smNone)_N=135_IS=0.3_RS=0.2_ref=0.0_delay=1.5_U=0.6_XTrue_feedreset_IDNone'


    # list(totals)[6] # slide polygon plot
    # for i in range(700):
    #     state_analysis.pc_polygons(config,k,i)

    # # # # # # Plot all full paths 
    print(f"Plotting all PC paths...")
    for i in range(len(totals)):
        print(list(totals)[i])
        state_analysis.full_path_plot(config,list(totals)[i],0,239)

    # # # # Plot PCs at every moment for top total performer
    # # k = list(totals)[0]
    # # k = 'LSTP_geo=(randNone_geo[9, 5, 3]_smNone)_N=135_IS=0.3_RS=0.1_ref=0.0_delay=1.5_U=0.6_XTrue_feedreset_IDNone'
    # # print(f"Plotting all PCs of top total performing experiment...\n {k}")
    # for t in range(config.length):
    #     state_analysis.full_pc_plot(config,k,t)

    # # # # ### Distance Analysis ###
    # # # #Determine distance metrics across states for different samples
    dist = DistanceAnalysis(config,save,show)
    dist.all_dists(config,MATs)
    dist.dist_plot(totals)
    

    # # # # ### Meta Analysis ###
    # meta = MetaAnalysis(config,save,show)
    # # k = 'Maass_smw=(randNone_geoNone_sm0.66)_N=135_IS=0.3_RS=0.3_ref=0.0_delay=1.5_U=0.6_XTrue_feedreset'
    # # k = 'Maass_rnd=(rand0.4_geoNone_smNone)_N=135_IS=0.2_RS=0.4_ref=3.0_delay=1.5_U=0.6'
    # # k = 'Maass_smw=(randNone_geoNone_sm0.0)_N=135_IS=0.2_RS=0.2_ref=0.0_delay=1.5_U=0.6_XTrue_feedreset_IDNone'
    # k = list(finals)
    # for i in range(len(finals)-10,len(finals)):
    #     print(k[i])
    #     meta.show_all(config,k[i])

    # dist = DistanceAnalysis(config,save,show)
    # dict1 = totals

    # ratio = {}
    # for k,v in dist.inter.items():
    #     ratio[k] = np.log(v)/np.log(dist.intra_mean_dist[k])

    # for k,v in dist.clust.items():
    #     if v.any() > 0:
    #         print(np.sum(v), " - ", k)

    # dict2 = dist.clust
    # dist.dist_plot(dict1)
    # I,J = meta.dict_compare(config,dict1,dict2)
    # meta.ranking_comparison_plot(I,J)
    # meta.dist_plot(finals,ratio)

if __name__ == "__main__":
    main()
