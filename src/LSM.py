import numpy as np
from brian2 import *
import matplotlib.pyplot as plt
import tables
import os
from sklearn.linear_model import LogisticRegression
import string

from processing import *
from pop_generators import reservoir
from poisson_generator import gen_poisson_pattern,create_jitter

from plotting import performance, raster_save


class Input:
    def __init__(self,config):
        self.file = config.input_file
        self.input_name = config.input_name
        self.input_name = config.input_name

    def __str__(self):
        return f"Dataset: \n{self.__dict__}"

    def get_data(self):
        file_path = f"datasets/{self.input_name}/{self.file}"
        fileh = tables.open_file(file_path, mode='r')
        self.units = fileh.root.spikes.units
        self.times = fileh.root.spikes.times
        self.labels = fileh.root.labels
        self.channels = np.max(np.concatenate(self.units))+1
        self.length = np.max(np.concatenate(self.times))
        self.classes = np.max(self.labels)

        return self.units, self.times, self.labels, self.channels, self.length, self.classes

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
        for pattern in classes:
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
        self.times = TIMES
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
                save_spikes(self.channels,self.length,input_times,input_units,loc,item)


# class
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



    # method for describing attributs of LiquidState
    def __str__(self):
        return f"Liquid attributes: \n{self.__dict__.keys()}"
    
    def describe(self):
        print("-----------------\nLiquid attributes:")
        for k in self.__dict__.keys():
            print(f"  {k} = {self.__dict__[k]}")
        print("-----------------\n")

    # def generate_reservoir(self):

    def simulate(self,inputs,example):

        start_scope()

        G, S = reservoir(self)
        nets = Network(G, S)

        if inputs.input_name == 'Poisson':
            timed = inputs.times[example]*ms
        elif inputs.input_name =='Heidelberg':
            timed = inputs.times[example]*1000*ms
        else:
            print("Input skipped")
        
        SGG = SpikeGeneratorGroup(inputs.channels, inputs.units[example], timed, dt=100*us)
        
        SP = Synapses(SGG, G, on_pre='v+=1')
        SP.connect('i!=j', p=self.input_sparsity)
        spikemon = SpikeMonitor(G)
        nets.add(SGG, SP, spikemon)
        nets.store()
        nets.run((self.T)*ms)
        #self.zipped = np.array(list(zip(spikemon.i,spikemon.t*1000)))
        indices = np.array(spikemon.i)
        times = np.array(spikemon.t/ms)
        #nets.restore()

        return [indices,times]

    def respond(self,inputs,dataset):
        for k,v in dataset.items():
            for i,rep in enumerate(v):
                item = f'pat{k}_rep{i}'
                print(f"\n --- Responding to pattern {k}, replica {i} --- \n")
                loc_liq = f'{self.dir}/liquid'
                item_liq = f'{self.full_loc}_{item}'
                result = self.simulate(inputs,rep)
                indices = result[0]
                times = result[1]
                save_spikes(self.N,inputs.length,times,indices,loc_liq,item_liq)



class ReadoutMap():
    def __init__(self,config):
        self.N = config.neurons
        self.T = config.length

    def heat_up(self,config):
        # location, pat, rep, config
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

        print(f"  Number of examples: {len(self.labels)}\n  Distinct patterns: {config.patterns}")

        self.train_range = int(self.split*len(self.full_train)/self.len_labels)
        self.test_range = int(config.patterns*len(self.full_train)/self.len_labels)

        print('Train/test split: ',self.train_range,self.test_range)

        # return self.train_range, self.test_range

    def regress(self,config):

        logisticRegr = LogisticRegression(max_iter=500)
        logisticRegr.fit(self.full_train[:self.train_range],self.full_labels[:self.train_range])

        predictions = []
        for i in range(self.split,self.len_labels):
            for j in range(config.length):
                prediction = logisticRegr.predict(np.transpose(self.mats[i])[j].reshape(1, -1))
                predictions.append(prediction[0])


        runs = np.zeros((config.patterns,config.patterns))
        accs = []


        for lab in range(config.patterns):
            accs.append([])
            start=config.length*lab
            stop=config.length*(lab+1)
            succ=0

            for i in range(start, stop):
                for p in range(config.patterns):
                    if predictions[i]==config.classes[p]:
                        runs[lab][p] += 1
                
                if np.argmax(runs[lab][:])==lab and i>1:
                    if i>1 and i!=start:
                        succ+=1
            
                if i!=start:
                    accs[lab].append((succ/(i-start)))

        accs_array = []
        for acc in accs:
            accs_array.append(acc)
        accs_array = np.array(accs_array)

        performance(config,accs_array)


        dirName = f"results/{config.dir}/performance/accuracies/"
        try:
            os.makedirs(dirName)    
        except FileExistsError:
            pass
        with open(f'results/{config.dir}/performance/accuracies/{config.full_loc}.npy', 'wb') as f:
            np.save(f, np.array(accs), allow_pickle=True)

