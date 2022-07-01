# NeuroMachines - A Survey on Liquid State Machines (Work in Progress)

## Summary
Liquid State Machines (LSMs) are a form of reservoir computing whereby spiking input is processed by a spiking recurrent neural network reservoir such that classification or prediction can be accomplished with a simple linear readoutmap trained on the liquid states of the reservoir (Maass2002).  Here, we survey the vast space of reservoir design possibilities and evalurate for generically good reservoir qualities on the basis of performance and representational dynamics.

## How to Use

### Prep
 - Pull the repository and install the appropriate modules
   - `pip install -r requirements.txt`
 - Download dataset from
   - https://ieee-dataport.org/open-access/heidelberg-spiking-datasets
   - Place in the `src` directory in folder named `datasets`
   - Be sure to unzip

### Single Experiments
 - Try running a single experiment with commands like:\
`python main.py --learning LSTP --topology geo --dims 15 3 3 --lamb 8 --neurons 135 --refractory 0 --delay 1.5  --res_sparsity 0.2 --input_sparsity 0.3--length 700 --replicas 3 --patterns 3 --input_name "Heidelberg" --feed "reset" --x_atory False --dir test_dir --output_show True`

 - Outputs will automatically be saved to a newly created results directory under a sub-directory given the name defined by `--dir`
 - Investigate `arg_parser.py` file to see simulation options
 - Realize that certiai pairing are required, specifically with topologies.
   - `lamb` and `dims` are for geometric and `beta` for small-world
 - `--length` should be at least 700 for "Heidelberg" experiments
 - Note the "Poisson" option is currently offline
 - Note for efficiency, only data on first sample of each pattern is saved
 - The first time running the command will generate new input data, but in the future it will load that saved data and run much faster
### Sweeps
- For windows uses, sweeps can be run from `sweep.bat` over many parameter configurations
- For non-Windows uses, sorry (for now)

### Anaylsis
Simple enter the name or you results directory (your "sweep" name) at the top of `anaylsis_main.py` and uncomment the analysis measure you are interested in


## Paper
Check out the latest version of the informal paper associated with this project by opening `paper.pdf` from the home directory

## Citations
So many --to come


