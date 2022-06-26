from LSM import *
from arg_parser import setup_argument_parser
import os
import pickle

def main():

    ### SETUP ###
    config = setup_argument_parser()

    ### INPUT ###

    inputs = Input(config)
    if config.input_name == "Heidelberg":
        names = ['ZERO','ONE','TWO','THREE','FOUR','FIVE','SIX','SEVEN','EIGHT','NINE','TEN','NULL','EINS','ZWEI','DREI','VIER','FUNF','SECHS','SEBEN','ACHT','NEUN','ZEHN']
    elif config.input_name == "Poisson":
        names = string.ascii_letters[26:52]
    config.classes=names[:config.patterns]
    print(config.classes)
    
    dataset = inputs.read_data(config)
    print(f'Dataset Read with {config.patterns} patterns and {config.replicas} replicas.')
    for k,v in dataset.items():
        print(f"  Pattern {k} at indices {v}")
    inputs.describe()


    ### SWITCH TO SAVED CONFIGS ###

    directory = f"{config.dir}_rerun"

    if config.rerun=='single':
        pick = f'results/{config.dir}/configs/{config.rerun_single}.pickle'
        file_to_read = open(pick, "rb")
        refig = pickle.load(file_to_read)
        file_to_read.close()
        refig.dir = directory

        ### RESERVOIR ###
        liquids = LiquidState(refig)
        liquids.describe()
        liquids.respond(inputs,dataset)

        ### OUTPUT ###
        output = ReadoutMap(refig)
        matrices, labels = output.heat_up(refig)
        output.setup(refig,matrices,labels)
        output.regress(refig)

    elif config.rerun=='sweep':
        path = f'results/{config.dir}/configs/'
        for filename in os.listdir(path):
            file = os.path.join(path, filename)
            file_to_read = open(file, "rb")
            refig = pickle.load(file_to_read)
            file_to_read.close()
            refig.dir = directory

            ### RESERVOIR ###
            liquids = LiquidState(refig)
            liquids.describe()
            liquids.respond(inputs,dataset)

            ### OUTPUT ###
            output = ReadoutMap(refig)
            matrices, labels = output.heat_up(refig)
            output.setup(refig,matrices,labels)
            output.regress(refig)



if __name__ == "__main__":
    main()

