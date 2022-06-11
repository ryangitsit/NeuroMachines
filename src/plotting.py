import os
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



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




# Plotting for .py files

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

def raster_plot(time, index):
    plt.figure(figsize=(24, 8))
    plt.plot(time/ms, index, '.k')
    plt.title('Raster Plot')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.show()


def raster_run(time,index,pop,top,ref,delay,pat,rep,location):
    plt.figure(figsize=(24, 8))
    plt.plot(time, index, '.k')
    plt.title(f'Raster Plot {pop}_{top}_ref={ref}_delay={delay}_pat{pat}_rep{rep}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.savefig(f'{location}/{pop}_{top}_ref={ref}_delay={delay}_pat{pat}_rep{rep}.png')
    plt.close()

def raster_run_input(time,index,pop,top,ref,delay,pat,rep,rates,location):
    plt.figure(figsize=(8, 8))
    plt.plot(time, index, '.k',color='k',ms=14)

    max_rate = np.argmax(rates)
    plt.axhline(y=max_rate,linewidth=1,ls='--', label=f"Max rate = {rates[max_rate]} at neuron {max_rate}", color='b')

    # nonzero = []
    # for i in rates:
    #     if i == 0:
    #         nonzero.append(rates[max_rate]) 
    #     else:
    #         nonzero.append(i)
    min_rate = np.argmin(rates)
    plt.axhline(y=min_rate,linewidth=1,ls='--', label=f"Min rate = {rates[min_rate]} at neuron {min_rate}", color='g')
    half = int(len(rates)/2)
    plt.legend()
    plt.title(f'Input Pattern: {pat}, Replica: {rep}',y=1.07, fontsize=24)
    plt.suptitle(f'{rates[:half]}\n{rates[half:]}',y=.93, fontsize=12)
    plt.xlabel('Time (ms)', fontsize=18)
    plt.ylabel('Neuron index', fontsize=18)
    plt.savefig(f'{location}/{pop}_{top}_ref={ref}_delay={delay}_pat{pat}_rep{rep}.png')
    plt.close()

def raster_runner(time,index,pat,rep,configs,location):

    dic = configs
    for key, value in dic.items():
        str = key
        globals()[str] = value

    plt.figure(figsize=(24, 8))
    plt.plot(time, index, '.k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')

    plt.title(f'Raster Plot'+full_loc+f'_pat{pat}_rep{rep}')
    plt.savefig(f'{location}/'+full_loc+f'_pat{pat}_rep{rep}.png')
    plt.close()             

def double_raster_plot(t_1,i_1,t_2,i_2):
    fig, axs = plt.subplots(2,figsize=(24,16))
    fig.suptitle('Pattern Input and Liquid Response')
    axs[0].plot(t_1/ms, i_1, '.k')
    axs[1].plot(t_2/ms, i_2, '.k')
    # plt.title('Raster Plot')
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Neuron index')
    plt.show()

def raster_save(time,index,dirName,item):
    plt.figure(figsize=(16, 8))
    plt.plot(time, index, '.k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')

    plt.title(f'Raster Plot {item}')
    plt.savefig(f'{dirName}/{item}.png')
    plt.close()         
    
def avg_performance(accs,rule,top,ref,delay,location,save):

    avg = np.mean(np.array([accs[0], accs[1], accs[2]]), axis=0 )
    sns.color_palette()
    plt.figure(figsize=(24, 16))
    plt.title(f"Average accuracy over time (How often correct lable is most frequently predicted class)\n{rule}_{top}_ref={ref}_delay={delay}", fontsize=24)
    plt.plot(avg)
    plt.ylim(0,1)
    plt.xlabel("Time (ms), dt=1ms", fontsize=20)
    plt.ylabel("Ratio of Correctness", fontsize=20)
    plt.legend()
    if save:
        plt.savefig(f'{location}/EACH_{rule}_{top}_ref={ref}_delay={delay}.png')
    else:
        plt.show()


def performance_plot(accs,patterns,len_patterns,rule,top,ref,delay,location,save):
    sns.color_palette()
    #sns.set_theme()
    plt.figure(figsize=(24, 16))
    plt.title(f"{rule} Accuracy over time (How often correct lable is most frequently predicted class)\n{rule}_{top}_ref={ref}_delay={delay}", fontsize=24)
    for i in range(len_patterns):
        plt.plot(accs[i],label=patterns[i],linewidth=2)
        plt.xlabel("Time (ms), dt=1ms", fontsize=20)
        plt.ylabel("Ratio of Correctness", fontsize=20)
        plt.legend()
    if save:
        plt.savefig(f'{location}/EACH_{rule}_{top}_ref={ref}_delay={delay}.png')
    else:
        plt.show()


# def output_avg(configs, accs_array):

#     dic = configs
#     for key, value in dic.items():
#         str = key
#         globals()[str] = value

#     avg = np.mean(accs_array, axis=0)
#     plt.title(f"Average accuracy over time")
#     plt.plot(avg)
#     plt.ylim(0,1)
#     plt.xlabel("Time (ms), dt=1ms")
#     plt.ylabel("Ratio of Correctness")

#     plt.show()

def output_multi(configs, classes, accs):

    dic = configs
    for key, value in dic.items():
        str = key
        globals()[str] = value

    plt.figure(figsize=(24, 16))
    plt.title(f"Accuracy over time\n"+full_loc+f"\n{patterns} Patterns, {replicas-tests} Train, {tests} Test", fontsize=22)
    for i in range(len(accs)):
        plt.plot(accs[i],label=classes[i])

    plt.xlabel("Time (ms), dt=1ms", fontsize=18)
    plt.ylabel("Ratio of Correctness", fontsize=18)
    plt.legend()

    if plots_output == True:

        dirName = f"results/{location}/performance/plots"
        try:
            os.makedirs(dirName)    
        except FileExistsError:
            pass

        plt.savefig(f"results/{location}/performance/plots/"+full_loc+".png")
    if output_show == True:
        plt.show()
    else:
        plt.close()

def performance(config,accs):

    plt.figure(figsize=(16, 8))
    plt.style.use('seaborn-muted')
    plt.title(f"Accuracy over time\n{config.full_loc}", fontsize=22)
    for i in range(len(accs)):
        plt.plot(accs[i],'--',label=config.classes[i])
    plt.plot(np.mean(accs,axis=0),linewidth=3,label="mean")
    plt.xlabel("Time (ms), dt=1ms", fontsize=18)
    plt.ylabel("Ratio of Correctness", fontsize=18)
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