from full_analysis import PerformanceAnalysis
import itertools
import os
import pickle

def main():
    sweep = "hei_phei"
    save = False
    show = True

    directory = f'results/{sweep}/configs'
    filename = os.listdir(directory)[0]
    file = os.path.join(directory, filename)
    file_to_read = open(file, "rb")
    config = pickle.load(file_to_read)
    file_to_read.close()

    full_analysis = PerformanceAnalysis(config,save,show)
    full_analysis.performance_pull()
    full_analysis.accs_plots()
    finals, totals = full_analysis.rankings()
    full_analysis.print_rankings(finals,"Final Performance",10)
    full_analysis.print_rankings(totals,"Total Performance",10)

    top_finals=dict(itertools.islice(finals.items(),20))
    top_totals=dict(itertools.islice(totals.items(),20))
    full_analysis.accs_plots(top_totals)
    full_analysis.accs_plots(top_finals)
    full_analysis.top_plot()


if __name__ == "__main__":
    main()

