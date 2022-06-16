from LSM import *
from arg_parser import setup_argument_parser
from processing import write_dict
import pickle

def main():

    ### SETUP ###
    config = setup_argument_parser()
    full_loc=f"{config.learning}_{config.topology}=(rand{config.rndp}_geo{config.dims}_sm{config.beta})_N={config.neurons}_IS={config.input_sparsity}_RS={config.res_sparsity}_ref={config.refractory}_delay={config.delay}_U={config.STSP_U}"
    config.full_loc=full_loc

    # Display Config Settings
    print("\nConfiguration Settings")
    for k,v in (config.__dict__).items():
        print(f"  {k}: {v}")
    print("\n")

    path = f'results/{config.dir}/configs/'
    name = f'{config.full_loc}.json'
    write_dict(config.__dict__,path,name)

    pick = f'{path}{name[:-5]}.pickle'
    filehandler = open(pick, 'wb') 
    pickle.dump(config, filehandler)
    filehandler.close()


    ### INPUT ###

    inputs = Input(config)

    if config.just_input == True:

        if config.input_name == "Heidelberg":
            inputs.get_data()
            dataset = inputs.make_dataset(config.patterns,config.replicas)
            config.classes = list(dataset.keys())

        elif config.input_name == "Poisson":
            print("Poisson")
            dataset = inputs.generate_data(config)

        inputs.save_data(config.dir)
        print(f'Dataset Generated with {config.patterns} patterns and {config.replicas} replicas.')
        for k,v in dataset.items():
            print(f"  Pattern {k} at indices {v}")
        inputs.describe()

    elif config.just_input == False:
        dataset = inputs.read_data(config)
        print(f'Dataset Read with {config.patterns} patterns and {config.replicas} replicas.')
        for k,v in dataset.items():
            print(f"  Pattern {k} at indices {v}")
        inputs.describe()

        ### RESERVOIR ###
        liquids = LiquidState(config)
        liquids.describe()
        liquids.respond(inputs,dataset)

        ### OUTPUT ###
        output = ReadoutMap(config)
        matrices, labels = output.heat_up(config)
        output.setup(config,matrices,labels)
        output.regress(config)

        # save config
        path = f'results/{config.dir}/configs/'
        name = f'{config.full_loc}.json'
        write_dict(config.__dict__,path,name)

        pick = f'{path}{name[:-5]}.pickle'
        filehandler = open(pick, 'wb') 
        pickle.dump(config, filehandler)
        filehandler.close()



if __name__ == "__main__":
    main()

