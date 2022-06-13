from LSM import *
from arg_parser import setup_argument_parser
from processing import write_dict

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


    ### INPUT ###
    inputs = Input(config)

    if config.input_name == "Heidelberg":
        inputs.get_data()
        dataset = inputs.make_dataset(config.patterns,config.replicas)
        config.classes = list(dataset.keys())
        print(f'Dataset Generated with {config.patterns} patterns and {config.replicas} replicas.')
        for k,v in dataset.items():
            print(f"  Pattern {k} at indices {v}")


    # store inputs in same dataset form as Hei, or standardize and improved method
    elif config.input_name == "Poisson":
        print("Poisson")
        dataset = inputs.generate_data(config)
        #print(dataset)
        for k,v in dataset.items():
            print(f"  Pattern {k} at indices {v}")


    inputs.describe()
    if config.new_input == True:
        inputs.save_data(config.dir)


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



if __name__ == "__main__":
    main()

