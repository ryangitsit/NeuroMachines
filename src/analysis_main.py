from full_analysis import Analysis
import itertools

def main():
    sweep = "hei_X"
    type = "Heidelberg"
    patterns = 3
    replicas = 3
    save = True
    show = False

    full_analysis = Analysis(sweep,type,patterns,replicas,save,show)
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

