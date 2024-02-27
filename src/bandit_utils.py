import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

def play_mab(environment, agent, N, T, disable_tqdm_T = False, seed = 42):
    """
    Play N independent runs of length T for the specified agent.

    :param environment: a MAB instance
    :param agent: a bandit algorithm
    :param N: number of independent simulations
    :param T: decision horizon
    :return: the agent's name, and the collected data in numpy arrays
    """

    rewards = np.zeros((N, T))
    regrets = np.zeros((N, T))
    pseudo_regrets = np.zeros((N, T))
    avg_rewards = np.zeros((N, T))

    for n in tqdm(range(N), desc=f'Training agent {agent.name()}'):
        # print("WARNING ON SEED")
        agent.reset(seed = n)
        environment.reset(seed = n)
        for t in tqdm(range(T), disable = disable_tqdm_T):
            action = agent.predict()
            reward = environment.get_reward(action)
            agent.update(action, reward)
            
            rew = max(reward)
            rewards[n,t] = rew
            # compute instantaneous reward  and (pseudo) regret
            # rewards[n,t] = max(rewards[action])
            # best_reward = max(rewards[environment.best_action])
            # regrets[n,t]= best_reward - reward # this can be negative due to the noise, but on average it's positive
            #avg_rewards[n,t] = means[action]
            #pseudo_regrets[n,t] = best_reward - means[action]
    return agent.name(), rewards #, regrets, avg_rewards, pseudo_regrets

def play_all(env, methods, T, N, disable_tqdm_T = False, seed = 42):
    return [play_mab(env, learner, N, T, disable_tqdm_T, seed) for learner in methods]

def play_all_parallel(env, methods, T, N, disable_tqdm_T = False):
    num_processes = len(methods)
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(_play_mab_util, [(env, learner, N, T, disable_tqdm_T) for learner in methods])
    return results

def _play_mab_util(attributes):
    return play_mab(attributes[0],attributes[1],attributes[2],attributes[3], attributes[4])


def plot_results(plt_arr, title = "Results", path = None):
    # Customize the font properties for the legend
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(figsize = (7,6))
    for name, runs in plt_arr:
        if name == 'MSE3':
            line = 'solid'
        else:
            line =  'dashdot'
        avg = np.mean(runs, axis = 0)
        runs_sum = np.cumsum(runs, axis = 1) 
        std = np.std(runs_sum, axis = 0)
        std_err_z_score = std / np.sqrt(len(runs_sum)) * 1.96
        plotting = np.cumsum(avg)
        ax.plot(plotting, label = name, linestyle = line, lw = 2)
        ax.fill_between(np.arange(plotting.shape[0]), plotting - std_err_z_score, plotting + std_err_z_score, alpha=0.2)
    # ax.set_xlabel('Time', fontsize = 21, fontweight='360')
    # ax.set_ylabel('Cumulative Reward', fontsize = 21, fontweight='360')
    # ax.title(title)
    ax = plt.gca()

    # Increase the font size of the tick labels on both the x and y axes
    ax.tick_params(axis='both', labelsize=18)  # Adjust the fontsize to your preference
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
    # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True,useOffset=False))
    # ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True,useOffset=False))
    ax.legend(prop={'size': '21', 'weight': '400', 'family': 'Times New Roman'})#, title_fontweight='bold')
    ax.grid()
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)

def boxplots(res, path = None):
    fig, ax = plt.subplots()
    tot = []
    for label, run_dict in res:
        data = np.cumsum(run_dict, axis = 1)[:,-1]
        tot += [data]

    x_positions = np.arange(len(tot)) + 3
    ax.boxplot(tot, positions=x_positions)
    ax.set_ylabel("Cumulative Reward",fontsize=15)
    ax.set_xticks(x_positions)  # Set the tick positions
    labels = [a for a,_ in res]
    ax.set_xticklabels(labels, fontsize=15) 
    if path is not None: 
        fig.savefig(path)

def plot_multi_res(res_dict, num_rows, num_cols, figx = 12, figy = 15):
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(figx, figy))
    for i, (key, sub_dict) in enumerate(res_dict.items()):
        row = i // num_cols
        col = i % num_cols
        
        for sub_key, sub_data in sub_dict:
            sub_data = sub_data.mean(axis = 0)
            axs[row, col].plot(np.cumsum(sub_data), label=sub_key, linestyle = 'solid', lw = 2)

        axs[row, col].set_title(f"K:{key[0]}, M:{key[1]},delta:{key[2]},G:{key[3]}")
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('Mistakes') 
        axs[row, col].grid()
    axs[0, 0].legend()
    for j in range(i + 1, num_rows * num_cols):
        row = j // num_cols
        col = j % num_cols
        fig.delaxes(axs[row, col])


