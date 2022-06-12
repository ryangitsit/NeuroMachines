from LSM import *
from arg_parser import setup_argument_parser

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
    inputs = Input(config.input_name, config.input_file)
    inputs.get_data()
    inputs.describe()
    dataset = inputs.make_dataset(config.patterns,config.replicas)
    config.classes = list(dataset.keys())
    print(f'Dataset Generated with {config.patterns} patterns and {config.replicas} replicas.')
    for k,v in dataset.items():
        print(f"  Pattern {k} at indices {v}")

    if config.new_input == True:
        inputs.save_data(config.dir)


    ### RESERVOIR ###
    liquids = LiquidState(config)
    liquids.describe()
    liquids.respond(inputs,dataset)

    output = ReadoutMap(config)
    matrices, labels = output.heat_up(config)
    output.setup(config,matrices,labels)

    output.regress(config)

    js = json.dumps(config.__dict__)
    path = f'results/{config.dir}/configs/{config.full_loc}.json'
    dirName= f'results/{config.dir}/configs'
    try:
        os.makedirs(dirName)    
    except FileExistsError:
        pass
    f = open(path,"w")
    f.write(js)
    f.close()




if __name__ == "__main__":
    main()

