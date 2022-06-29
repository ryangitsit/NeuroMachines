import argparse

def setup_argument_parser():

    parser = argparse.ArgumentParser()

    # OO implementation
    parser.add_argument("--just_input", help = " ", type = bool, default = False)
    parser.add_argument("--dir", help = " ", type = str, default = "test_sweep")
    parser.add_argument("--input_name", help = " ", type = str, default = "Poisson") #""Heidelberg")
    parser.add_argument("--input_file", help = " ", type = str, default = "shd_train.h5")
    parser.add_argument("--classes", help = " ", type = str, default = ["A","B","C"])

    # input
    parser.add_argument("--patterns", help = " ", type = int, default = 3)
    parser.add_argument("--length", help = " ", type = int, default = 100)
    parser.add_argument("--rate_low", help = " ", type = int, default = 0)
    parser.add_argument("--rate_high", help = " ", type = int, default = 5)
    parser.add_argument("--channels", help = " ", type = int, default = 40)
    parser.add_argument("--replicas", help = " ", type = int, default = 3)
    parser.add_argument("--jitter", help = " ", type = int, default = 10)

    # organizational
    parser.add_argument("--storage", help = " ", type = bool, default = True)
    parser.add_argument("--location", help = " ", type = str, default = "sweep")
    parser.add_argument("--plots", help = " ", default = True)
    parser.add_argument("--flow", help = " ", type = bool, default = False)
    parser.add_argument("--full_loc", help = " ", type = str, default=None)
    parser.add_argument("--chunk", help = " ", type = int, default = 10)

    # liquids
    parser.add_argument("--learning", help = " ", choices = ['Maass', 'STSP', 'STDP','LSTP'], type = str, default = "STSP")
    parser.add_argument("--topology", help = " ", choices = ['rnd', 'geo', 'smw'], type = str, default = "rnd")
    parser.add_argument("--input_sparsity", help = " ", type = float, default = 0.3)
    parser.add_argument("--res_sparsity", help = " ", type = float, default = .2)
    parser.add_argument("--STSP_U", help = " ", type = float, default = 0.6)
    parser.add_argument("--x_atory", help = " ", type = bool, default = False)

    
    # Each corresponds to a specific topology and will set top for the run
    # Do not select more than one
    parser.add_argument("--rndp", help = " ", type = float, default =None)   # random topology
    parser.add_argument('--dims', nargs='*', help=" ", type=int, default=None)  # geometric topology
    parser.add_argument("--beta", help = " ", type = float, default = None)     # small-world topology


    parser.add_argument("--neurons", help = " ", type = int, default = 64)
    parser.add_argument("--refractory", help = " ", type = float, default = 3.0)
    parser.add_argument("--delay", help = " ", type = float, default = 1.5)

    # Reruns
    parser.add_argument("--rerun", help = " ", type = str, default = "sweep")
    parser.add_argument("--rerun_single", help = " ", type = str, default = None)


    # storage
    parser.add_argument("--storage_liquid", help = " ", type = bool, default = True)
    parser.add_argument("--plots_liquid", help = " ", type = bool, default = True)
    parser.add_argument("--tests", help = " ", type = int, default = 1)
    parser.add_argument("--storage_output", help = " ", type = bool, default = True)
    parser.add_argument("--plots_output", help = " ", type = bool, default = True)
    parser.add_argument("--output_show", help = " ", type = bool, default = False)

    return parser.parse_args()
    