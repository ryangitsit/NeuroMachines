#%%
import numpy as np
from brian2 import *
import matplotlib.pyplot as plt
import tables
import os
from sklearn.linear_model import LogisticRegression
import string
from processing import *
from reservoirs import reservoir
from poisson_generator import gen_poisson_pattern,create_jitter
from plotting import performance, raster_plot, raster_save
import pickle

from collections import Counter

"""
### LSM.py ###

    - Input (class)
        - can read in existing data or generate new data
        - save all data as spikes and raster plots
            - results/input/
        - refers to dataset in future with dictionary `dataset`
            - keys are unique pattern labels
            - values are indices of replicas of that pattern

    - LiquidState (class)
        - Generates reserovirs according to defined configuration
        - Responds to each example with same reservoir initialization
        - In parallel, saves weights and pseudo-one-hot-encodes matrices
        - Saves spikes and can optionally save raster plots
            - results/{sweep}/liquids/

    - Readout (class)
        - Imports saved pseudo-one-hot-encodes matrice
        - Optionally chunks columns by `chunk_size`
        - Labels all columns (or chunks)
        - Splits into training/testing columns (or chunks)
            - The last full liquid state matrix for each pattern is test
        - Trains logistic regression on appropriate labels
        - Tests on unseen examples and tracks ratio of correct repsponse to most given response
            - records as certainty for each class
        - Automatically plots and saves certainty plots and arrays
            - results/{sweep}/performance
"""


# #pick = 'results/scale_testing/configs/Maass_rnd=(randNone_geoNone_smNone)_N=100_IS=0.2_RS=0.3_ref=0.0_delay=0.0_U=0.6.pickle'
# pick = 'results/testing/configs/config_STSP_rnd=(randNone_geoNone_smNone)_N=64_IS=0.3_RS=0.2_ref=3.0_delay=1.5_U=0.6.pickle'
# file_to_read = open(pick, "rb")
# config = pickle.load(file_to_read)
# file_to_read.close()

# print(config.__dict__)
#%%
class Input:
    '''
    - Creates new input or reads in existing input
    - Save dataset as spikes/plots
    - Maintains data as times/indices attributes of self
    '''

    def __init__(self,config):
        '''
        Initializes object with full use of .config object
        '''
        self.file = config.input_file
        self.input_name = config.input_name

    def __str__(self):
        return f"Dataset: \n{self.__dict__}"

    def get_data(self):
        '''
        - Reads in Heidelberg data
        - Adds data to self
        - Note this data is not in pushed to the git
            - Must be downloaded locally from:
        https://ieee-dataport.org/open-access/heidelberg-spiking-datasets
        '''
        if self.input_name == "Heidelberg":
            file_path = f"datasets/{self.input_name}/{self.file}"
            fileh = tables.open_file(file_path, mode='r')
            self.units = fileh.root.spikes.units
            times = fileh.root.spikes.times
            self.times = times
            self.labels = fileh.root.labels
            self.channels = np.max(np.concatenate(self.units))+1
            self.length = np.max(np.concatenate(self.times))
            self.classes = np.max(self.labels)

    def read_data(self,config):
        '''
        For existing data
        - For datasets previously saved to .txt in results/{sweep}/inputs
        - Necessary for poisson experiments because new sets will
          be different every time
        - Adds data to self
        - Note that self.dataset is used to refer to which indices carry
          the appropriate samples for a given unique pattern class
        '''
        dataset = {}
        count = 0
        UNITS = []
        TIMES = []
        labels = []
        for pattern in config.classes:
            dataset[pattern] = []
            for rep in range(config.replicas):
                location = f"results/{config.dir}/inputs/spikes/pat{pattern}_rep{rep}.txt"
                dat,indices,times = txt_to_spks(location)
                UNITS.append(indices)
                TIMES.append(times)
                labels.append(pattern)
                dataset[pattern].append(count)
                count+=1
        self.units = UNITS
        self.times = TIMES
        self.labels = labels
        self.channels = np.max(np.concatenate(self.units))+1
        self.length = config.length
        self.classes = config.patterns
        self.dataset = dataset
        return self.dataset

    def generate_data(self,config):
        '''
        For Poisson Experiments only
            - Generates class labels
            - 
        '''
        patterns = config.patterns
        replicas = config.replicas
        length = config.length
        jitter = config.jitter
        dataset = {}
        UNITS = []
        TIMES = []
        labels = []
        if patterns < 2:
            print("### Error: Minimum of 2 input patterns required.  Try again... ###")
            return
        if length < 100:
            print("### Error: Minimum of 100ms simulation length required.  Try again... ###")
            return

        # Generating class labels
        classes = []
        for letter in range(patterns):
            if patterns < 27:
                let = string.ascii_letters[letter+26]
                classes.append(let)
                for r in range(replicas):
                    dataset[let] = []
            else:
                classes.append(f"pattern_{letter}")
                for r in range(replicas):
                    dataset[str(letter)+"_"+str(r)] = []
        print(f"Pattern classes = {classes}\n")

        # Generate data form poisson_generator.py
        # add times and indices to self object
        # track location of replicas for a given pattern in dataset dict
        rates_dict = {}
        count = 0
        for i,pattern in enumerate(classes):
            print("Pattern: ", pattern)
            rand_rates, indices,times = gen_poisson_pattern(config.channels, config.rate_low, config.rate_high, config.length)
            rates_dict[pattern] = rand_rates
            for r in range(replicas):
                print(" Replica: ", r)
                jittered = create_jitter(jitter,times)
                UNITS.append(indices)
                TIMES.append(np.round(jittered,8))
                labels.append(pattern+str(r))
                dataset[pattern].append(count)
                count+=1
        self.units = UNITS
        self.times = np.array(TIMES,dtype=object)/1000
        self.labels = labels
        self.channels = config.channels
        self.length = config.length
        self.classes = config.patterns
        self.dataset = dataset
        return dataset

    def describe(self):
        # Check everything is in order
        string = f"""Dataset description:
        Dataset = {self.dataset}
        Examples = {len(self.labels)}
        Channels = {self.channels}
        Time length = {self.length} seconds
        Disctinct labels = {self.classes}
        """
        print(string)

    def make_dataset(self,patterns,replicas):
        '''
        For Heidelberg dataset only
            - creates dictionary storing correct attribute indices of
              of data for each class label
        '''
        names = ['ZERO','ONE','TWO','THREE','FOUR','FIVE','SIX','SEVEN',
                 'EIGHT','NINE','TEN','NULL','EINS','ZWEI','DREI','VIER',
                 'FUNF','SECHS','SEBEN','ACHT','NEUN','ZEHN']
        self.dataset = {}
        for i in range(patterns):
            result = np.where(np.array(self.labels)==i)
            self.dataset[names[i]]=(result[0])[:replicas]
        return self.dataset

    def save_data(self,location):
        '''
        Save all examples in unique files with inputs directory
        '''
        for k,v in self.dataset.items():
            for i,rep in enumerate(v):
                input_units = self.units[rep]
                input_times = self.times[rep]
                loc = f'{location}/inputs'
                item = f'pat{k}_rep{i}'
                save_spikes(self.channels,self.length,input_times*1000,input_units,loc,item)

inputs = Input(config)
dataset = inputs.read_data(config)
print(f'Dataset Read with {config.patterns} patterns and {config.replicas} replicas.')
for k,v in dataset.items():
    print(f"  Pattern {k} at indices {v}")
inputs.describe()

#%%

class LiquidState():
    def __init__(self, config):
        self.name = config.full_loc

    def __str__(self):
        return f"Liquid attributes: \n{self.__dict__.keys()}"
    
    def describe(self,config):
        lst = list(config.__dict__.keys())
        start = lst.index("learning")
        finish = lst.index("delay")
        print("-----------------\nLiquid attributes:")
        for k in lst[start:finish+1]:
            print(f"  {k} = {config.__dict__[k]}")
        print("-----------------\n")

    def respond(self,config,inputs,dataset):

        if inputs.input_name == 'Poisson':
            config.DT = 1
        elif inputs.input_name =='Heidelberg':
            config.DT = 100

        start_scope()
        G, S, W = reservoir(config)
        nets = Network(G, S)
        nets.store()

        dirName = f"results/{config.dir}/weights/"
        try:
            os.makedirs(dirName)    
        except FileExistsError:
            pass
        with open(f'results/{config.dir}/weights/{config.full_loc}.npy', 'wb') as f:
            np.save(f, W, allow_pickle=True)

        mats = []
        for pat,v in dataset.items():
            for rep,replica in enumerate(v):
                item = f'pat{pat}_rep{rep}'
                print(f" --- Responding to pattern {pat}, replica {rep} ---")
                loc_liq = f'{config.dir}/liquid'
                item_liq = f'{config.full_loc}_{item}'
                #################
                #################
                example = inputs.dataset[pat][rep]
                timed = inputs.times[example]*ms

                SGG = SpikeGeneratorGroup(inputs.channels, inputs.units[example], timed, dt=config.DT*us)

                SP = Synapses(SGG, G, on_pre='v+=1', dt=config.DT*us)
                SP.connect(p=config.input_sparsity)
                spikemon = SpikeMonitor(G)
                nets.add(SGG, SP, spikemon)
                nets.run((config.length)*ms)
                indices = np.array(spikemon.i)
                times = np.array(spikemon.t/ms)
                mats.append(one_hot(config.neurons,config.length,np.array(indices)[:],times[:]))
                nets.remove(SGG,SP,spikemon)
                nets.restore()
                #################
                #################
                if rep == 0:
                    save_spikes(config.neurons,inputs.length,times,indices,loc_liq,item_liq)

        storage_mats = np.array(mats)
        # print(storage_mats)
        dirName = f"results/{config.dir}/performance/liquids/encoded/"
        try:
            os.makedirs(dirName)    
        except FileExistsError:
            pass
        with open(f'results/{config.dir}/performance/liquids/encoded/mat_{config.full_loc}.npy', 'wb') as f:
            np.save(f, storage_mats, allow_pickle=True)


# liquids = LiquidState(config)
# liquids.describe(config)
# liquids.respond(config,inputs,dataset)


#%%
class ReadoutMap():
    def __init__(self,config):
        self.N = config.neurons
        self.T = config.length

    def setup(self,config):
        print(" --- PREPARE TRAIN/TEST LIQUID STATE DATA ---")
        mat_path = f'results/{config.dir}/performance/liquids/encoded/mat_{config.full_loc}.npy'
        mats = np.load(mat_path, allow_pickle=True)
        labels=[]
        
        for pat in config.classes:
            for rep in range(config.replicas):
                labels.append(pat)
        self.labels = labels
        print(labels)
        self.mats = mats
        self.full_labels = []
        self.full_train = []

        for i in range(len(labels)):
            self.full_labels += [labels[i]]*int(self.T)
            for j in range(config.length):
                self.full_train.append(np.transpose(mats[i])[j])
                # print(np.transpose(mats[i])[j])
        print(len(self.full_labels))
        print(len(self.full_train))
        self.len_labels = len(self.labels)
        self.split = config.patterns*config.replicas - config.patterns #*int(tests)
        #print(self.split)

        print(f"  Number of 1ms time slice states: {len(self.full_labels)}\n  Distinct patterns: {config.patterns}")

        self.train_range = config.length*(config.patterns*config.replicas - config.patterns) #int(self.split*len(self.full_train)/self.len_labels)
        self.test_range = config.length*config.patterns #int(config.patterns*len(self.full_train)/self.len_labels)

        print('Train/test split: ',self.train_range,self.test_range)

        ratio = self.split/(config.replicas*config.patterns)
        pat_step = int(config.length*config.replicas)
        train_rng = [(pat_step*x,int((pat_step*x+pat_step*ratio))) for x in range(3)]
        test_rng = [(int((pat_step*x+pat_step*ratio)),int((pat_step*x+pat_step*ratio)+pat_step*(1-ratio))) for x in range(3)]
        self.training = np.concatenate([self.full_train[train_rng[i][0]:train_rng[i][1]] for i in range(config.patterns)])
        self.target = np.concatenate([self.full_labels[train_rng[i][0]:train_rng[i][1]] for i in range(config.patterns)])
        self.testing = np.concatenate([self.full_train[test_rng[i][0]:test_rng[i][1]] for i in range(config.patterns)])
        self.test_target = np.concatenate([self.full_labels[test_rng[i][0]:test_rng[i][1]] for i in range(config.patterns)])
        chunk_size = 1
        if config.chunk>1:
            ########################################  
                        # CHUNKING #    
            ########################################      
            print(" --- chunking ---")
            chunk_size = config.chunk
            print(len(self.full_train))
            all_chunks = []
            count=0
            chunk_labels = []
            for sample in range(config.patterns*config.replicas):
                for t in range(int(config.length/chunk_size)):
                    chunk=[]
                    for c in range(chunk_size):
                        chunk.append(self.full_train[count])
                        count += 1
                    chunk_labels.append(self.full_labels[count-1])
                    all_chunks.append(array(np.concatenate(np.array(chunk))))

            LENGTH = config.patterns*config.replicas*config.length
            c_split = int(self.train_range/chunk_size)  
            print(c_split)
            
            ratio = self.split/(config.replicas*config.patterns)
            pat_step = int(config.length*config.replicas/chunk_size)
            train_rng = [(pat_step*x,int((pat_step*x+pat_step*ratio))) for x in range(3)]
            test_rng = [(int((pat_step*x+pat_step*ratio)),int((pat_step*x+pat_step*ratio)+pat_step*(1-ratio))) for x in range(3)]

            self.training = np.concatenate([all_chunks[train_rng[i][0]:train_rng[i][1]] for i in range(config.patterns)])
            self.target = np.concatenate([chunk_labels[train_rng[i][0]:train_rng[i][1]] for i in range(config.patterns)])
            self.testing = np.concatenate([all_chunks[test_rng[i][0]:test_rng[i][1]] for i in range(config.patterns)])
            self.test_target = np.concatenate([chunk_labels[test_rng[i][0]:test_rng[i][1]] for i in range(config.patterns)])
            self.chunk_size = chunk_size


    def regress(self,config):

        # Fit liquid states to labels for training range
        print("Fitting regression model...")
        logisticRegr = LogisticRegression(max_iter=10000)
        logisticRegr.fit(self.training, self.target)

        # Make predictions on unseen data for testing range
        print("Making predictions...")
        predictions=[]
        for i in range(len(self.testing)):
            prediction = logisticRegr.predict(self.testing[i].reshape(1, -1))
            predictions.append(prediction[0])
        print(predictions)
        print(self.test_target)

        raw_success = 0
        for i in range(len(predictions)):
            if predictions[i] == self.test_target[i]:
                raw_success += 1
        print(f"Raw accuracy = {raw_success/len(predictions)}")

        certainties = [[] for _ in range(config.patterns)]
        pre_labs = np.zeros((config.patterns,config.patterns))
        count = 0

        for i,pat in enumerate(config.classes):
            # hit = 0
            # track = 0
            run = []
            running_pred = []
            for j in range(int(config.length/self.chunk_size)):
                run.append(predictions[count])
                pred_index = config.classes.index(predictions[count])
                print(pred_index)
                pre_labs[i][pred_index] += 1
                top_pred = config.classes[np.argmax(pre_labs[i])]

                ## Running predictions predicts at each time step top class
                # running_pred.append(top_pred)
                # cert = (len([pred for pred in running_pred if pred == pat]) / len(running_pred))

                ## Ratio of correct class top class
                # if top_pred == pat:
                #     cert = 1
                # else:
                #     cert = np.clip(pre_labs[i][i]/(np.max(pre_labs[i])+1),0,1)
                cert = pre_labs[i][i]/(np.max(pre_labs[i]))

                ## how often is correct class the most common so far?
                # if max(set(run), key = run.count) == pat:
                #     hit+=1
                # #certainties[i].append(hit/(j+1))
                # if j < 1:
                #     certainties[i].append(hit)
                # else:
                #     if predictions[count] == pat:
                #         track +=1
                #     c = Counter(run)
                #     most_common = [key for key, val in c.most_common(2)]
                #     runner_up = c[most_common[1]]
                #     cert = np.clip(track/runner_up,0,1)
                #cert = np.clip(track/((j+1)/2),0,1)

                certainties[i].append(cert)
                count+=1
                
            #print(f"{pat}: {hit/(j+1)}")
        #     print(f"{pat}: {cert}")
        # print(pre_labs)




        # predictions = []
        # clean_accs = [[] for _ in range(config.patterns)]
        # clean_success = np.zeros((config.patterns,1))
        # classes = config.classes
        
        """
        for i in range(self.split,self.len_labels):
            count = 0
            success = 0
            for j in range(config.length):
                count +=1
                prediction = logisticRegr.predict(np.transpose(self.mats[i])[j].reshape(1, -1))
                predictions.append(prediction[0])

                # new acc method
                lab_index = i%3
                pred_index = classes.index(prediction[0])
                clean_success[pred_index] +=1
                if pred_index == lab_index:
                    success += 1
                clean_accs[lab_index].append(float(success/count))
        """
        # for i in range(self.split,self.len_labels):
        #     count = 0
        #     success = 0
        #     for j in range(config.length):
        #         count +=1
        #         prediction = logisticRegr.predict(np.transpose(self.mats[i])[j].reshape(1, -1))
        #         predictions.append(prediction[0])

        #         # new acc method
        #         lab_index = i%3
        #         pred_index = classes.index(prediction[0])
        #         clean_success[pred_index] +=1
        #         if pred_index == lab_index:
        #             success += 1
        #         clean_accs[lab_index].append(float(success/count))

        # for chunk in range(len(testing)):
        #     prediction = logisticRegr.predict(testing[chunk])
        #     predictions.append(prediction[0])
        # print(predictions)


        # Check prediction performance against correct labels
        # runs = np.zeros((config.patterns,config.patterns))
        # accs = []
        # for lab in range(config.patterns):
        #     accs.append([])
        #     start=config.length*lab
        #     stop=config.length*(lab+1)
        #     succ=0
        #     for i in range(start, stop):
        #         for p in range(config.patterns):
        #             if predictions[i]==config.classes[p]:
        #                 runs[lab][p] += 1
        #         if np.argmax(runs[lab][:])==lab and i>1:
        #             if i>1 and i!=start:
        #                 succ+=1            
        #         if i!=start:
        #             accs[lab].append((succ/(i-start)))

        #accs_array = np.array(accs)
        accs_array = np.array(certainties)

        #print(accs_array)
        final_mean = np.round(np.mean(accs_array,axis=0)[-1],2)
        performance(config,accs_array,final_mean)

        # Store accuracies
        dirName = f"results/{config.dir}/performance/accuracies/"
        try:
            os.makedirs(dirName)    
        except FileExistsError:
            pass
        with open(f'results/{config.dir}/performance/accuracies/{config.full_loc}.npy', 'wb') as f:
            np.save(f, accs_array, allow_pickle=True)

        print(f"***Experiment complete***\n  Config={config.full_loc}\n  Final mean accuracy:{final_mean}")


## For Debugging, ignore
# output = ReadoutMap(config)
# output.setup(config)

# output.regress(config)

# %%
