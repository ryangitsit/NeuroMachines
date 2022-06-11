
#%%
from brian2 import *
from plotting import raster
import numpy as np
import matplotlib.pyplot as plt
from processing import *

#%%
print(2430/9)
#%%

def spike_loop(sweep,classes,replicas):
    """
    """

    pats = len(classes)

    directory = f'results/{sweep}/liquid/spikes'

    filename = os.listdir(directory)[1]
    file = os.path.join(directory, filename)
    # dat,indices,times = txt_to_spks(file)
    # length = int(np.ceil(np.max(times)))
    # neurons = np.max(indices) + 1

    experiments = int(len(os.listdir(directory))/(len(classes)*replicas))
    
    spikes={}
    
    # iterate through all experiments and group one hot encoded spikes
    # by class and replica, generate 3 pcs for each and store in dictionary
    # for exp in range(experiments):
    for exp in range(int(2430/9),int((2430+162)/9)):
        spike={}
        IT = []
        for pat,label in enumerate(classes):
            spikes[label]=[]
            for r in range(replicas):
                i = exp*pats*replicas + pat*replicas + r

                filename = os.listdir(directory)[i]
                file = os.path.join(directory, filename)

                if (i) % 9 == 0:
                    exp_name=file[len(directory)+1:-14]
                    print(f"{exp}-{i} experiment: {exp_name}")

                dat,indices,times = txt_to_spks(file)
                IT.append(indices)
                IT.append(times)
                spike[label] = IT

        spikes[exp_name] = spike

    return spikes

classes=["A","B","C"]
sweep = 'full_sweep'
replicas=3

spikes = spike_loop(sweep,classes,replicas)

#%%

def alt_stream(spikes):
# from plotting import raster
    length = 100
    I = []
    T = []
    classes = spikes.keys()
    colors = ['r','g','b']
    print(classes)

    #plt.figure(figsize=(24, 8))
    for iter, letter in enumerate(classes):
        i = spikes[letter][0]
        t = spikes[letter][1]
        t = np.array(t)+iter*length
        plt.plot(t,i,'.k',ms=3, color=colors[iter],label=letter)
        I.append(i)
        T.append(t/1000)
    I = np.concatenate(I)
    T = np.concatenate(T)
    
    plt.legend()
    plt.show()
    return I,T

#[2430]
exp = "STDP_smw=(randNone_geoNone_sm0.75)__N=64_IS=0.2_RS=0.3_ref=1.5_delay=1.5"

# little sweep
#exp = "STDP_smw=(randNone_geoNone_sm0.25)__N=64_IS=0.2_RS=0.15_ref=1.5_delay=1.5"


# spike = spikes['Maass_geo=(randNone_geo[8, 8, 1]_smNone)__N=64_IS=0.2_RS=0.45_ref=0.0_delay=0.0']
# spike = spikes['Maass_geo=(randTrue_geo[3, 3, 3]_smNone)__N=27_IS=0.2_RS=0.3_ref=0.0_delay=1.5']
spike = spikes[exp]
# spike = spikes["STDP_smw=(randNone_geoNone_sm0.25)__N=64_IS=0.2_RS=0.15_ref=1.5_delay=1.5"]
I,T = alt_stream(spike)

#%%
print(T*1000)
II = []
TT = []
for i in range(2):
    print(i)
    II.append(I)
    TT.append(T*1000+300*i)
II = np.concatenate(II)
TT = np.concatenate(TT)
plt.figure(figsize=(16,8))
plt.plot(TT,II,'.k')

#%%

def get_double_results(classes,WTA_N,length,spike_i,spike_t):
    spikeout = np.zeros((len(classes)*2,WTA_N))
    for p, pattern in enumerate(classes*2):
        class_low = p*length
        class_high = p*length + 100
        for t, time in enumerate(spike_t):
            if time > class_low and time < class_high:
                spikeout[p][spike_i[t]] += 1
    classifications = []
    for results in spikeout:
        classifications.append(classes[np.argmax(results)])
    return spikeout, classifications

def logic_check(spikeout,classifications):
    first_half = classifications[:len(classifications/2)]
    second_half = classifications[len(classifications/2):]
    print("first half: ",first_half, "\nSecond half: ",second_half)
    if first_half == second_half:
        print("PERFECT SCORE!")
    if (set(first_half) and set(second_half)) == 3:
        print("Diversity!")

#%%
classes=["A","B","C"]
length = 100

def get_results(classes,WTA_N,length,spike_i,spike_t):
    spikeout = np.zeros((len(classes),WTA_N))
    for p, pattern in enumerate(classes):
        class_low = p*length
        class_high = p*length + 100
        for t, time in enumerate(spike_t):
            if time > class_low and time < class_high:
                spikeout[p][spike_i[t]] += 1
    classifications = []
    for results in spikeout:
        classifications.append(classes[np.argmax(results)])
    return spikeout, classifications




#%%


"""
Note here that Brian synaptic variables cannot be passed through a function
"""
# inhibs = [4, 8, 10]
# wgains = [.5, .5, 1.25]
# cgains = [.5, 1.25, .5]
# cdecays = [140, 20, 140]

inhibs = [10]
wgains = [.5]
cgains = [.5]
cdecays = [130]

# inhibs = np.arange(2,12,2)
# wgains = np.arange(.5,2.5,.75)
# cgains = np.arange(.5,2.5,.75)
# cdecays = np.arange(20,150,30)

cycle = 1
success = 0
# inhibs = np.arange(8,12.1,2)
# wgains = np.arange(.5,2,.25)
# cgains = np.arange(.5,2,.25)
# cdecays = np.arange(10,180,30)
print(wgains)
print(cdecays)
print(inhibs)
print(len(wgains)*len(cgains)*len(cdecays)*len(inhibs))
#%%


sweep_length = len(wgains)*len(cgains)*len(cdecays)*len(inhibs)
count = 0
tracker = {}
for inhibit in inhibs:
    for wgain in wgains:
        for cgain in cgains:
            for cdecay in cdecays:
                count += 1
                inhibit = np.round(inhibit,2) 
                wgain = np.round(wgain,2)
                cgain = np.round(cgain,2)
                cdecay = np.round(cdecay,2)
                key = f"Inh = {inhibit}, W+ = {wgain}, C+ = {cgain}, C- = {cdecay}"
                #print(key)
                success=0
                for cyc in range(cycle):
                    start_scope()
                    N = 64
                    WTA_N = 3
                    sim = 100*3

                    eqs = '''
                    dv/dt = (-v)/(30*ms) : 1
                    df/dt = (-f)/(30*ms) : 1
                    '''
                    G = NeuronGroup(WTA_N, eqs, threshold='v>1', reset='v = 0', method='euler')
                    G.v = 'rand()'
                    spikemon = SpikeMonitor(G)
                    net = Network(G,spikemon)
                    SGG = SpikeGeneratorGroup(N, I, np.array(T)*1000*ms)
                    S = Synapses(SGG, G,
                                '''
                                w : 1
                                dc/dt = (-c)/(cdecay*ms) : 1
                                l : 1
                                ''',
                                on_pre='''
                                v_post += w - c
                                w = clip(w+wgain, 0, 3)
                                ''',
                                on_post='''
                                c += cgain
                                f += .011
                                ''', method='linear')
                    S.connect(p=1)
                    for pre in range(N):
                        for post in range(WTA_N):
                            S.w[pre,post]=rand()
                    #S.delay = delay_time
                    MG = StateMonitor(G, ['v','f'], record=True)
                    MS = StateMonitor(S, ['w','c','l'], record=True)
                    SGI = Synapses(G, G, 
                        on_pre=
                        '''
                        v_post += -inhibit
                        ''')
                    SGI.connect('i!=j',p=1)

                    net.add(SGG,S,SGI,MG,MS)
                    # net.store()
                    net.run(sim*ms)
                    C = MS.c
                    L = MS.l
                    V = MG.v
                    W = MS.w
                    spike_t = np.array(spikemon.t/ms)
                    spike_i = np.array(spikemon.i)
                    spikeout, classifications = get_results(classes,WTA_N,length,spike_i,spike_t)
                    # plt.figure(figsize=(14, 8))
                    # plt.plot(spike_t,spike_i,'.k')
                    # plt.show()

                    # for i in range(3):
                    #     print(classifications[i],spikeout[i])

                    if len(set(classifications))==3:
                        #print("BINGO!")
                        success += 1

                #print(success/cycle)
                tracker[key] = [success/cycle, inhibit, wgain, cgain, cdecay]

                print(f"{count}/{sweep_length}: {key} ----> {success/cycle}")

                config = "STDP_smw=(randNone_geoNone_sm0.25)__N=64_IS=0.2_RS=0.15_ref=1.5_delay=1.5"
                N=64
                plt.figure(figsize=(14, 8))
                plt.plot(spike_t,spike_i*30,'.k',ms=15,)
                plt.title(f"Prediction Spikes Over Pattern Spikes, with Synaptic Paramters\n{config}\n{key}",fontsize=18)
                color = ['r','b','g']
                plt.xlabel("Time (ms)",fontsize=16)
                plt.ylabel("Neuron index and normalized axis for synaptic parameters",fontsize=16)
                # for i in range(3):
                #     plt.plot(np.mean(W[i*N:i*N+N],axis=0)[::10],color=color[i],label='W')
                #     plt.plot(np.mean(C[i*N:i*N+N],axis=0)[::10],'-.k',color=color[i],label='C')
                #     #plt.plot(np.mean(L[i*N:i*N+N],axis=0)[::10],'--.k',color=color[i],label='L')
                #     plt.plot(MG.f[i][::10], label="Firing")

                plt.plot(np.mean(W,axis=0)[::10]*18,color='purple', linewidth=2,label='W')
                plt.plot(np.mean(C,axis=0)[::10]*18,color='orange', linewidth=2,label='C')
                I,T = alt_stream(spike)

                # plt.legend()
                # plt.show()

                for i in range(3):
                    print(classifications[i],spikeout[i])

#%%


#%%
write=False
if write == True:
    for k,v in tracker.items():
        tracker[k] = np.array(v).tolist()
    exp = "STDP_smw=(randNone_geoNone_sm0.75)__N=64_IS=0.2_RS=0.3_ref=1.5_delay=1.5"
    type = "alt_custom"
    dirName = f'results/{sweep}/performance/spiking_readout/'
    try:
        os.makedirs(dirName)    
    except FileExistsError:
        pass
    js = json.dumps(tracker)
    path = f'{dirName}{type}_{exp}.json'
    f = open(path,"w")
    f.write(js)
    f.close()

#%%

round2 = {}

for k,v in tracker.items():
    if v[0] > 0.7:
        inhibit = v[1]
        wgain = v[2]
        cgain =  v[3]
        cdecay =  v[4]
        key = f"Inh = {inhibit}, W+ = {wgain}, C+ = {cgain}, C- = {cdecay}"
        # print(key)
        success=0
        cycle = 100
        for cyc in range(cycle):
            start_scope()
            N = 64
            WTA_N = 3
            sim = 100*3

            eqs = '''
            dv/dt = (-v)/(30*ms) : 1
            df/dt = (-f)/(30*ms) : 1
            '''
            G = NeuronGroup(WTA_N, eqs, threshold='v>1', reset='v = 0', method='euler')
            G.v = 'rand()'
            spikemon = SpikeMonitor(G)
            net = Network(G,spikemon)
            SGG = SpikeGeneratorGroup(N, I, np.array(T)*1000*ms)
            S = Synapses(SGG, G,
                        '''
                        w : 1
                        dc/dt = (-c)/(cdecay*ms) : 1
                        l : 1
                        ''',
                        on_pre='''
                        v_post += w - c
                        w = clip(w+wgain, 0, 3)
                        ''',
                        on_post='''
                        c += cgain
                        f += .011
                        ''', method='linear')
            S.connect(p=1)
            for pre in range(N):
                for post in range(WTA_N):
                    S.w[pre,post]=rand()
            #S.delay = delay_time
            #MG = StateMonitor(G, ['v','f'], record=True)
            #MS = StateMonitor(S, ['w','c','l'], record=True)
            SGI = Synapses(G, G, 
                on_pre=
                '''
                v_post += -inhibit
                ''')
            SGI.connect('i!=j',p=1)

            net.add(SGG,S,SGI)#,MG,MS)
            # net.store()
            net.run(sim*ms)
            # C = MS.c
            # L = MS.l
            # V = MG.v
            # W = MS.w
            spike_t = spikemon.t/ms
            spike_i = spikemon.i
            spikeout, classifications = get_results(classes,WTA_N,length,spike_i,spike_t)

            # for i in range(3):
            #     print(classifications[i],spikeout[i])

            if len(set(classifications))==3:
                #print("BINGO!")
                success += 1

        #print(success/cycle)
        round2[key] = [success/cycle, inhibit, wgain, cgain, cdecay]

        print(f"Round2: {key} ----> {success}/{cycle}")



#%%

print(round2)

#%%
if write == True:
    for k,v in round2.items():
        round2[k] = np.array(v).tolist()
    exp = "STDP_smw=(randNone_geoNone_sm0.75)__N=64_IS=0.2_RS=0.3_ref=1.5_delay=1.5"
    type = "custom-round2"
    dirName = f'results/{sweep}/performance/spiking_readout/'
    try:
        os.makedirs(dirName)    
    except FileExistsError:
        pass
    js = json.dumps(round2)
    path = f'{dirName}{type}_{exp}.json'
    f = open(path,"w")
    f.write(js)
    f.close()

#%%
config = "STDP_smw=(randNone_geoNone_sm0.25)__N=64_IS=0.2_RS=0.15_ref=1.5_delay=1.5"
N=64
plt.figure(figsize=(14, 8))
plt.plot(spike_t,spike_i,'.k')
plt.title(f"{config}\n{key}")
color = ['r','b','g']
for i in range(3):
    plt.plot(np.mean(W[i*N:i*N+N],axis=0)[::10],color=color[i],label='W')
    plt.plot(np.mean(C[i*N:i*N+N],axis=0)[::10],'-.k',color=color[i],label='C')
    #plt.plot(np.mean(L[i*N:i*N+N],axis=0)[::10],'--.k',color=color[i],label='L')
    plt.plot(MG.f[i][::10], label="Firing")
plt.legend()
plt.show()



#%%
# C+ must = .5, W+ flexible, C- not 160
for k,v in tracker.items():
    if v[2] == 0.5 and v[3] == .5 and v[4] == 130:
        print(k, " ----> ", v[0])
    # print(k, " ----> ", v[0])

#%%

def dict_anlysis(dict,x):
    better_than_nothing = {}
    better_than_x = {}
    zero_count=0
    four_count=0
    for k,v in dict.items():
        if v[0] > 0:
            zero_count+=1
            better_than_nothing[k] = v
        if v[0] > x:
            four_count+=1
            better_than_four[k] = v
    print(f"Better than nothing = {zero_count}/{len(dict)}")
    print(f"Better than {x} = {four_count}/{len(dict)}")
    for k,v in better_than_x.items():
        print(k,v)

#%%
dict_anlysis(tracker)

for k,v in round2.items():
    # if v[0] > 0.4:
    #     print(k, " ----> ", v[0])
    print(k, " ----> ", v[0])
# counts = Counter(better_than_nothing.values())
# print(counts[''])


        

#%%

####################
inhibs = np.arange(10,14.1,1)
wgains = np.arange(.25,1.26,.25)
cgains = [0.4,0.5,0.6]
cdecays = [120,130,140]

# inhibs = [10]
# wgains = [.5]
# cgains = [.5]
# cdecays = [130]
cycle = 10
success = 0

print(len(wgains)*len(cgains)*len(cdecays)*len(inhibs))

#%%
sweep_length = len(wgains)*len(cgains)*len(cdecays)*len(inhibs)
count = 0
double = {}
for inhibit in inhibs:
    for wgain in wgains:
        for cgain in cgains:
            for cdecay in cdecays:
                count += 1
                inhibit = np.round(inhibit,2) 
                wgain = np.round(wgain,2)
                cgain = np.round(cgain,2)
                cdecay = np.round(cdecay,2)
                key = f"Inh = {inhibit}, W+ = {wgain}, C+ = {cgain}, C- = {cdecay}"
                #print(key)
                success=0
                for cyc in range(cycle):
                    start_scope()
                    N = 64
                    WTA_N = 3
                    sim = 100*3*2

                    eqs = '''
                    dv/dt = (-v)/(30*ms) : 1
                    df/dt = (-f)/(30*ms) : 1
                    '''
                    G = NeuronGroup(WTA_N, eqs, threshold='v>1', reset='v = 0', method='euler')
                    G.v = 'rand()'
                    spikemon = SpikeMonitor(G)
                    net = Network(G,spikemon)
                    SGG = SpikeGeneratorGroup(N, II, np.array(TT)*ms)
                    S = Synapses(SGG, G,
                                '''
                                w : 1
                                dc/dt = (-c)/(cdecay*ms) : 1
                                l : 1
                                ''',
                                on_pre='''
                                v_post += w - c
                                w = clip(w+wgain, 0, 3)
                                ''',
                                on_post='''
                                c += cgain
                                f += .011
                                ''', method='linear')
                    S.connect(p=1)
                    for pre in range(N):
                        for post in range(WTA_N):
                            S.w[pre,post]=rand()
                    #S.delay = delay_time
                    # MG = StateMonitor(G, ['v','f'], record=True)
                    # MS = StateMonitor(S, ['w','c','l'], record=True)
                    SGI = Synapses(G, G, 
                        on_pre=
                        '''
                        v_post += -inhibit
                        ''')
                    SGI.connect('i!=j',p=1)

                    net.add(SGG,S,SGI)#,MG,MS)
                    # net.store()
                    net.run(sim*ms)
                    # C = MS.c
                    # L = MS.l
                    # V = MG.v
                    # W = MS.w
                    spike_t = spikemon.t/ms
                    spike_i = spikemon.i
                    spikeout, classifications = get_double_results(classes,WTA_N,length,spike_i,spike_t)
                    
                    # for i in range(len(classifications)):
                    #     print(classifications[i],spikeout[i])

                    if len(set(classifications[:3]))==3:
                        letlist1 = []
                        for let in set(classifications[3:]):
                            letlist1.append(let)
                        letlist2 = []
                        for let in set(classifications[:3]):
                            letlist2.append(let)
                        if letlist1 == letlist2:
                            #print("BIG BINGO!")
                            success += 1

                #print(success/cycle)
                double[key] = [success/cycle, inhibit, wgain, cgain, cdecay]

                print(f"{count}/{sweep_length}: {key} ----> {success/cycle}")

                # for i in range(3):
                #     print(classifications[i],spikeout[i])
# %%

def dict_anlysis(dict,x):
    better_than_nothing = {}
    better_than_x = {}
    zero_count=0
    four_count=0
    for k,v in dict.items():
        if v[0] > 0:
            zero_count+=1
            better_than_nothing[k] = v
        if v[0] > x:
            four_count+=1
            better_than_x[k] = v
    print(f"Better than nothing = {zero_count}/{len(dict)}")
    print(f"Better than {x} = {four_count}/{len(dict)}")
    for k,v in better_than_x.items():
        print(k,v)

dict_anlysis(double,0.4)
# %%
