import os
# from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Plotting for .py files


def raster_plot(time, index):
    plt.figure(figsize=(12, 8))
    plt.plot(time/ms, index, '.k')
    plt.title('Raster Plot')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.show()

def raster_save(time,index,dirName,item):
    plt.figure(figsize=(22,12))
    # if len(time) > 1000:
    #     plt.plot(time, index, '.k')
    # else:
    plt.plot(time, index, '.k')
    plt.xlabel('Time (ms)',fontsize=18)
    plt.ylabel('Neuron index',fontsize=18)
    plt.title(f'Raster Plot {item}',fontsize=20)
    plt.savefig(f'{dirName}/{item}.png')
    plt.close()     

def raster_run_input(time,index,dirName,item):
    plt.figure(figsize=(8, 8))
    plt.plot(time, index, '.k',color='k',ms=14)
    max_rate = np.argmax(rates)
    plt.axhline(y=max_rate,linewidth=1,ls='--', label=f"Max rate = {rates[max_rate]} at neuron {max_rate}", color='b')
    min_rate = np.argmin(rates)
    plt.axhline(y=min_rate,linewidth=1,ls='--', label=f"Min rate = {rates[min_rate]} at neuron {min_rate}", color='g')
    half = int(len(rates)/2)
    plt.legend()
    plt.title(f'Input Pattern: {item}',y=1.07, fontsize=24)
    plt.suptitle(f'{rates[:half]}\n{rates[half:]}',y=.93, fontsize=12)
    plt.xlabel('Time (ms)', fontsize=18)
    plt.ylabel('Neuron index', fontsize=18)
    plt.savefig(f'{dirName}/{item}.png')
    plt.close() 


def performance(config,accs_array,final_mean):
    plt.figure(figsize=(14, 8))
    plt.style.use('seaborn-muted')
    plt.title(f"Certainty over time (Final Mean: {final_mean})\n{config.full_loc}", fontsize=16)
    x = np.arange(0,config.length,config.chunk)
    for i in range(len(accs_array)):
        plt.plot(x,accs_array[i],'--',label=config.classes[i])
    plt.axhline(y=1,linewidth=2,ls='--', label=f"Correct classification", color='k')
    plt.plot(x,np.mean(accs_array,axis=0),linewidth=3,label="mean")
    plt.xlabel("Time (ms), dt=1ms", fontsize=14)
    plt.ylabel("Ratio WTA of Correctness", fontsize=14)
    plt.ylim(0,1.03)
    plt.legend()
    dirName = f"results/{config.dir}/performance/plots"
    try:
        os.makedirs(dirName)    
    except FileExistsError:
        pass
    plt.savefig(f"results/{config.dir}/performance/plots/{config.full_loc}_performance.png")
    if config.output_show == True:
        plt.show()
    else:
        plt.close()


def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize=(16, 10))
    #plt.subplot(121)
    plt.plot(zeros(Ns), arange(Ns), 'ok', ms=2)
    plt.plot(ones(Nt)*(Ns/Nt)+.5*(Ns/Nt), arange(Nt), 'ok', ms=15)
    for i, j in zip(S.i, S.j):
        plt.plot([0, 1], [i, j*(Ns/Nt)+.5*(Ns/Nt)], '-k', linewidth=2*S.w[i,j])
    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(Ns, Nt))
    # plt.subplot(122)
    # plt.plot(S.i, S.j, 'ok',ms=2)
    # plt.xlim(-1, Ns)
    # plt.ylim(-1, Nt)
    # plt.xlabel('Source neuron index')
    # plt.ylabel('Target neuron index')
    plt.show()




# Inline plotting for jupyter notebooks

def sample_rand(N,n):
    figure(figsize=(24, 8))
    for samp in range(n):
        index = randint(N)
        plot(M.t/ms, M.v[index-1], label=f'Neuron {index}')
    xlabel('Time (ms)')
    ylabel('Charge mV')
    title('Random Sample of Neuron Charges')
    legend(loc='best')


def sample_connected(N):
    figure(figsize=(24, 8))
    count=0
    for pre_i in range(N):
        for post_j in range(N):
            if S.w[pre_i,post_j] > .5 and count < 1:
                count += 1
                plot(M.t/ms, M.v[pre_i], label=f'Presynaptic Neuron {pre_i}')
                plot(M.t/ms, M.v[post_j], label=f'Postsynaptic Neuron {post_j}')
    xlabel('Time (ms)')
    ylabel('Charge mV')
    title('Sample of Two Connected Neurons')
    legend(loc='best')


def scatter_weights(S):
    figure(figsize=(24, 8))
    scatter(S.x_pre/um, S.x_post/um, S.w*15)
    title('Synaptic Connections and Weights (dot size)')
    xlabel('Source neuron position (um)')
    ylabel('Target neuron position (um)');


def raster(time, index):
    figure(figsize=(24, 8))
    plot(time/ms, index, '.k')
    title('Raster Plot')
    xlabel('Time (ms)')
    ylabel('Neuron index');