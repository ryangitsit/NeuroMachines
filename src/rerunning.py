from LSM import *
from arg_parser import setup_argument_parser
import os
import pickle

def main():

    ### SETUP ###
    config = setup_argument_parser()

    ### INPUT ###

    # inputs = Input(config)    
    # config, dataset = inputs.read_data(config)
    # print(f'Dataset Read with {config.patterns} patterns and {config.replicas} replicas.')
    # for k,v in dataset.items():
    #     print(f"  Pattern {k} at indices {v}")
    # inputs.describe()


    ### SWITCH TO SAVED CONFIGS ###
    sweep = 'SuperSweep4'
    dirName = f'results/{sweep}/analysis/'
    item = 'finals'
    finals = read_json(dirName,item)

    for i,(k,v) in enumerate(finals.items()):
        if i < 100:
            for j in range(10):
                directory = f"rerun_{i}"
                pick = f'results/{sweep}/configs/config_{k}.pickle'
                file_to_read = open(pick, "rb")
                refig = pickle.load(file_to_read)
                file_to_read.close()
                refig.dir = directory
                refig.save_spikes = 0
                refig.seeding = "False"
                refig.ID = j

                dirName = f"results/{directory}/"
                try:
                    os.makedirs(dirName)    
                except FileExistsError:
                    pass

                inputs = Input(config)    
                inputs.get_data()
                dataset = inputs.make_dataset(config.patterns,config.replicas)

                ### RESERVOIR ###
                liquids = LiquidState(refig)
                liquids.describe(refig)
                liquids.respond(refig,inputs,dataset)

                ### OUTPUT ###
                output = ReadoutMap(refig)
                matrices, labels = output.heat_up(refig)
                output.setup(refig,matrices,labels)
                output.regress(refig)

    # elif config.rerun=='sweep':
    #     path = f'results/{config.dir}/configs/'
    #     for filename in os.listdir(path):
    #         file = os.path.join(path, filename)
    #         file_to_read = open(file, "rb")
    #         refig = pickle.load(file_to_read)
    #         file_to_read.close()
    #         refig.dir = directory

    #         ### RESERVOIR ###
    #         liquids = LiquidState(refig)
    #         liquids.describe()
    #         liquids.respond(inputs,dataset)

    #         ### OUTPUT ###
    #         output = ReadoutMap(refig)
    #         matrices, labels = output.heat_up(refig)
    #         output.setup(refig,matrices,labels)
    #         output.regress(refig)


if __name__ == "__main__":
    main()

