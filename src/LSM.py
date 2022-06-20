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


# #pick = 'results/scale_testing/configs/Maass_rnd=(randNone_geoNone_smNone)_N=100_IS=0.2_RS=0.3_ref=0.0_delay=0.0_U=0.6.pickle'
# pick = 'results/scale_testing/configs/Maass_rnd=(rand0.01_geoNone_smNone)_N=100_IS=0.05_RS=0.01_ref=0.0_delay=0.0_U=0.6.pickle'
# file_to_read = open(pick, "rb")
# config = pickle.load(file_to_read)
# file_to_read.close()

# print(config.__dict__)

class Input:
    def __init__(self,config):
        self.file = config.input_file
        self.input_name = config.input_name
        self.input_name = config.input_name

    def __str__(self):
        return f"Dataset: \n{self.__dict__}"

    def get_data(self):
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
        pattern_dataset = {}
        rates_dict = {}
        count = 0
        for i,pattern in enumerate(classes):
            #config.rate_low = i*60+40
            print("Pattern: ", pattern)
            pattern_dataset[pattern] = []
            rand_rates, indices,times = gen_poisson_pattern(config.channels, config.rate_low, config.rate_high, config.length)
            rates_dict[pattern] = rand_rates
            for r in range(replicas):
                print(" Replica: ", r)
                jittered = create_jitter(jitter,times)
                zipped = np.array(list(zip(indices,np.round(jittered*1000,8))))
                pattern_dataset[pattern].append(zipped)
                UNITS.append(indices)
                TIMES.append(np.round(jittered,8))
                labels.append(pattern+str(r))
                dataset[pattern].append(count)
                count+=1

        class_length = len(pattern_dataset)
        rep_len = len(pattern_dataset[classes[0]])
        spike_len = len(pattern_dataset[classes[0]][0][:,1])
        print(f"Poisson pattern dataset generated, with size: ({class_length}, {rep_len}, {spike_len})")
        self.units = UNITS
        self.times = np.array(TIMES,dtype=object)/1000
        self.labels = labels
        self.channels = config.channels
        self.length = config.length
        self.classes = config.patterns
        self.dataset = dataset
        return dataset

    def describe(self):
        string = f"""Dataset description:
        Dataset = {self.dataset}
        Subset = {self.file} 
        Examples = {len(self.labels)}
        Channels = {self.channels}
        Time length = {self.length} seconds
        Disctinct labels = {self.classes}
        """
        print(string)

    def make_dataset(self,patterns,replicas):
        names = ['ZERO','ONE','TWO','THREE','FOUR','FIVE','SIX','SEVEN','EIGHT','NINE','TEN','NULL','EINS','ZWEI','DREI','VIER','FUNF','SECHS','SEBEN','ACHT','NEUN','ZEHN']
        self.dataset = {}
        for i in range(patterns):
            result = np.where(np.array(self.labels)==i)
            self.dataset[names[i]]=(result[0])[:replicas]
        return self.dataset

    def save_data(self,location):
        for k,v in self.dataset.items():
            for i,rep in enumerate(v):
                input_units = self.units[rep]
                input_times = self.times[rep]
                loc = f'{location}/inputs'
                item = f'pat{k}_rep{i}'
                save_spikes(self.channels,self.length,input_times*1000,input_units,loc,item)

# inputs = Input(config)
# dataset = inputs.read_data(config)
# print(f'Dataset Read with {config.patterns} patterns and {config.replicas} replicas.')
# for k,v in dataset.items():
#     print(f"  Pattern {k} at indices {v}")
# inputs.describe()

class LiquidState():
    # intitializing generic *instance* attributes with parameter names
    def __init__(self, config): #N,T,learning,topology,input_sparsity,res_sparsity,refractory,delay):
        self.N = config.neurons
        self.T = config.length
        self.learning = config.learning
        self.topology = config.topology
        self.input_sparsity = config.input_sparsity
        self.res_sparsity = config.res_sparsity
        self.rndp = config.rndp
        self.dims = config.dims
        self.beta = config.beta
        self.refractory = config.refractory
        self.delay = config.delay
        self.full_loc = config.full_loc
        self.dir = config.dir
        self.STSP_U = config.STSP_U
        self.x_atory = config.x_atory

    def __str__(self):
        return f"Liquid attributes: \n{self.__dict__.keys()}"
    
    def describe(self):
        print("-----------------\nLiquid attributes:")
        for k in self.__dict__.keys():
            print(f"  {k} = {self.__dict__[k]}")
        print("-----------------\n")

    def simulate(self,inputs,example,nets,G,S):

        # # start_scope()

        # # G, S = reservoir(self)
        # # nets = Network(G, S)

        if inputs.input_name == 'Poisson':
            timed = inputs.times[example]*ms
        elif inputs.input_name =='Heidelberg':
            timed = inputs.times[example]*1000*ms
        else:
            print("Input skipped")
        
        SGG = SpikeGeneratorGroup(inputs.channels, inputs.units[example], timed, dt=1*us)
        
        SP = Synapses(SGG, G, on_pre='v+=1')
        SP.connect('i!=j', p=self.input_sparsity)
        spikemon = SpikeMonitor(G)
        nets.add(SGG, SP, spikemon)
        #nets.store()
        nets.run((self.T)*ms)
        indices = np.array(spikemon.i)
        times = np.array(spikemon.t/ms)
        nets.remove(SGG,SP,spikemon)
        nets.restore()

        return [indices,times]
        # return [inputs.units[example],timed/ms]



    def respond(self,inputs,dataset):

        start_scope()
        G, S, W = reservoir(self)
        nets = Network(G, S)
        nets.store()

        dirName = f"results/{self.dir}/weights/"
        try:
            os.makedirs(dirName)    
        except FileExistsError:
            pass
        with open(f'results/{self.dir}/weights/{self.full_loc}.npy', 'wb') as f:
            np.save(f, W, allow_pickle=True)

        for k,v in dataset.items():
            for r,rep in enumerate(v):
                item = f'pat{k}_rep{r}'
                print(f"\n --- Responding to pattern {k}, replica {r} --- \n")
                loc_liq = f'{self.dir}/liquid'
                item_liq = f'{self.full_loc}_{item}'
                # result = self.simulate(inputs,rep,nets,G,S)
                # indices = result[0]
                # times = result[1]


                #################
                #################
                example = rep
                if inputs.input_name == 'Poisson':
                    timed = inputs.times[example]*ms
                elif inputs.input_name =='Heidelberg':
                    timed = inputs.times[example]*ms
                else:
                    print("Input skipped")
                DT = 1
                print(f"--- dt = {DT} ---")
                SGG = SpikeGeneratorGroup(inputs.channels, inputs.units[example], timed, dt=DT*us)
            
                SP = Synapses(SGG, G, on_pre='v+=1', dt=DT*us)
                SP.connect('i!=j', p=self.input_sparsity)
                spikemon = SpikeMonitor(G)
                nets.add(SGG, SP, spikemon)
                #nets.store()
                nets.run((self.T)*ms)
                indices = np.array(spikemon.i)
                times = np.array(spikemon.t/ms)
                nets.remove(SGG,SP,spikemon)
                nets.restore()
                #################
                #################


                save_spikes(self.N,inputs.length,times,indices,loc_liq,item_liq)


# liquids = LiquidState(config)
# liquids.describe()
# liquids.respond(inputs,dataset)



class ReadoutMap():
    def __init__(self,config):
        self.N = config.neurons
        self.T = config.length

    def heat_up(self,config):
        # location, pat, rep, config
        print("One-hot encoding liquid state data...")
        mats = []
        labels=[]

        for rep in range(config.replicas):
            for pat in config.classes:

                path = f'results/{config.dir}/liquid/spikes/{config.full_loc}_pat{pat}_rep{rep}.txt'
                dat,indices,times = txt_to_spks(path)

                mats.append(one_hot(config.neurons,config.length,np.array(indices)[:],times[:]))
                labels.append(pat)

        self.labels = labels
        self.mats = mats

        return mats, labels

    def setup(self,config,mats,labels):
        self.full_labels = []
        self.full_train = []
        # for lab in range(len(labels)):
        #     full_labels += [labels[lab]]*int(self.T)

        # full_train = []
        for i in range(len(labels)):
            self.full_labels += [labels[i]]*int(self.T)
            for j in range(config.length):
                self.full_train.append(np.transpose(mats[i])[j])

        self.len_labels = len(self.labels)
        self.split = config.patterns*config.replicas - config.patterns #*int(tests)
        print(self.split)

        print(f"  Number of 1ms time slice states: {len(self.labels)}\n  Distinct patterns: {config.patterns}")

        self.train_range = int(self.split*len(self.full_train)/self.len_labels)
        self.test_range = int(config.patterns*len(self.full_train)/self.len_labels)

        print('Train/test split: ',self.train_range,self.test_range)
    

        # return self.train_range, self.test_range

    def regress(self,config):

        training = self.full_train[:self.train_range]
        target = self.full_labels[:self.train_range]
        testing = self.full_train[self.train_range:]
        test_target = self.full_labels[self.train_range:]
        chunk_size = 1
        chunking = True
        if chunking==True:
            ########################################  
                        # CHUNKING #    
            ########################################      
            print("--- chunking ---")
            chunk_size = 10
            train_chunks = []
            target_chunks = []
            count = 0
            for slice in range(int(len(training)/chunk_size)):
                chunk = []
                for c in range(chunk_size):
                    chunk.append(training[count])
                    count+=1
                train_chunks.append(array(np.concatenate(np.array(chunk))))
                target_chunks.append(target[count-1])

            training = train_chunks
            target = target_chunks

            test_chunks = []
            test_target_chunks = []
            count = 0
            for slice in range(int(len(testing)/chunk_size)):
                chunk = []
                for c in range(chunk_size):
                    chunk.append(testing[count])
                    count+=1
                test_chunks.append(array(np.concatenate(np.array(chunk))))
                test_target_chunks.append(test_target[count-1])

            testing = np.array(test_chunks)
            test_target = test_target_chunks
            ########################################      
            ########################################  

        # # Fit liquid states to labels for training range
        print("Fitting regression model...")
        logisticRegr = LogisticRegression(max_iter=500)
        logisticRegr.fit(training, target)


        # Make predictions on unseen data for testing range
        print("Making predictions...")
        predictions=[]
        for i in range(len(testing)):
            prediction = logisticRegr.predict(testing[i].reshape(1, -1))
            predictions.append(prediction[0])
        print(predictions)

        certainties = [[] for _ in range(config.patterns)]
        pre_labs = np.zeros((config.patterns,config.patterns))
        count = 0

        
        for i,pat in enumerate(config.classes):
            # hit = 0
            # track = 0
            run = []
            running_pred = []
            for j in range(int(config.length/chunk_size)):
                run.append(predictions[count])
                pred_index = config.classes.index(predictions[count])
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
# matrices, labels = output.heat_up(config)
# output.setup(config,matrices,labels)
# output.regress(config)
