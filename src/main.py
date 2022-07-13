from LSM import *
from arg_parser import setup_argument_parser
import os
import pickle
from os.path import exists

'''
### main.py ###
    - import configuration from command line argparser
    - either create or read in a dataset
    - generate reservoir according to configuration
    - provoke liquid responses for each input pattern
    - convert liquid responses into psuedo-on-hot encoded training/test slices
    - perform logistic regression with or without chunking of slices
    - save configuration
    - note, plots and data for all of the above are saved automatically
        - plotting saves can be toggled through th argparser
'''

def main():

    ### SETUP ###
    # import arguments entered through command line as config object and define experiment name
    config = setup_argument_parser()
    full_loc=f"{config.learning}_{config.topology}=(rand{config.rndp}_geo{config.dims}_sm{config.beta})_N={config.neurons}_IS={config.input_sparsity}_RS={config.res_sparsity}_ref={config.refractory}_delay={config.delay}_U={config.STSP_U}_X{config.x_atory}_feed{config.feed}_ID{config.ID}"
    config.full_loc=full_loc

    ### INPUT ###

    # initialize input object with config parameters
    inputs = Input(config)

        # for generating new input only
    path = f'results/{config.dir}/inputs/spikes/'
    if not exists(path) or len(os.listdir(path)) < config.replicas*config.patterns:
        if config.input_name != 'MNIST':
            if config.input_name == "Heidelberg":
                '''
                - if spoken digit dataset, just input from Heidelberg
                - convert data to suitable form for experiment
                - define indices for the data of each label
                - record class names in config.classes
                '''
                inputs.get_data()
                dataset = inputs.make_dataset(config.patterns,config.replicas)
                config.classes = list(dataset.keys())

            elif config.input_name == "Poisson":
                '''
                - if spoken poisson dataset, generate new random data
                - convert data to suitable form for experiment
                - define indices for the data of each label
                '''
                print("Poisson")
                dataset = inputs.generate_data(config)            

            # in all cases, save dataset and print indices of each pattern
            inputs.save_data(config.dir)
            print(f'Dataset Generated with {config.patterns} patterns and {config.replicas} replicas.')
            for k,v in dataset.items():
                print(f"  Pattern {k} at indices {v}")
            inputs.describe()

        else:
            config,dataset = inputs.MNIST(config,"save")
        # If no new input is needed, simply read in existing input
        # Define class names depending on experiment type
        # elif config.just_input == False:

    if config.input_name != 'MNIST':
        config, dataset = inputs.read_data(config)
        print(f'Dataset Read with {config.patterns} patterns and {config.replicas} replicas.')
        for k,v in dataset.items():
            print(f"  Pattern {k} at indices {v}")
        inputs.describe()
    else:
        config,dataset = inputs.MNIST(config,"run")


    ### RESERVOIR ###
    liquids = LiquidState(config)
    liquids.describe(config)
    liquids.respond(config,inputs,dataset) # liquid response to all inputs


    ### OUTPUT ###
    output = ReadoutMap(config)
    # matrices, labels = output.heat_up(config) # legacy
    output.setup(config) # take time slices and label them, also split train/test
    output.regress(config) # fit regression model for training, and then test on unseen data

    # save final config for future use
    path = f'results/{config.dir}/configs/'
    name = f'{config.full_loc}.json'
    try:
        os.makedirs(path)    
    except FileExistsError:
        pass
    pick = f'{path}config_{name[:-5]}.pickle'
    filehandler = open(pick, 'wb') 
    pickle.dump(config, filehandler)
    filehandler.close()

if __name__ == "__main__":
    main()

