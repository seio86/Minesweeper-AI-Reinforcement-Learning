import argparse, pickle
from tqdm import tqdm
from keras.models import load_model
from DQN_agent import *
from minesweeper_env import MinesweeperEnv
import os

import matplotlib.pyplot as plt
import csv
import pandas as pd



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# intake MinesweeperEnv parameters, beginner mode by default
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DQN to play Minesweeper')
    parser.add_argument('--width', type=int, default=9,
                        help='width of the board')
    parser.add_argument('--height', type=int, default=9,
                        help='height of the board')
    parser.add_argument('--n_mines', type=int, default=10,
                        help='Number of mines on the board')
    parser.add_argument('--episodes', type=int, default= 400, #100_000, #gherla seio putting 300 instead of 100_000
                        help='Number of episodes to train on')
    parser.add_argument('--model_name', type=str, default=f'{MODEL_NAME}',
                        help='Name of model')

    return parser.parse_args()

params = parse_args()

AGG_STATS_EVERY = 100 # calculate stats every 100 games for tensorboard
SAVE_MODEL_EVERY =100 #10_000 # save model and replay every 10,000 episodes


def get_nr_discovered_tiles(current_state):
    nr_discov_tiles= 0 #how many tiles discovered until now
    for idx_row in range(0, len(current_state)):
        for idx_col in range(0, len(current_state)):
            if current_state[idx_row][idx_col]!= -0.125:
                nr_discov_tiles+= 1
    return nr_discov_tiles
def write_csv(csv_name, episode_list, extra_field_name, row_data_list):
    field_names = ['episode_number', extra_field_name]
    data_total = [episode_list, row_data_list]
    data_array = np.transpose(np.array(data_total))
    df = pd.DataFrame(data_array, columns= field_names)
    df.to_csv('the_plots/'+extra_field_name+'_' + str(params.model_name) + '_' + str(params.episodes) + 'episodes' + '.csv',
              encoding='utf-8', index=False, float_format='%.5f')
    return


def main():
    env = MinesweeperEnv(params.width, params.height, params.n_mines)
    agent = DQNAgent(env, params.model_name)

    progress_list, wins_list, ep_rewards, nr_discov_tiles_list = [], [], [], []
    n_clicks = 0

    episode_list= []
    med_progress_list= []
    med_reward_list= []
    win_rate_list= []
    med_discov_tiles_list= []
    learn_rate_list=[]
    epsilon_list= []

    for episode in tqdm(range(1, params.episodes+1), unit='episode'):
        agent.tensorboard.step = episode

        env.reset()
        episode_reward = 0
        past_n_wins = env.n_wins

        done = False
        nr_discov_tiles = 0

        while not done:
            current_state = env.state_im

            action = agent.get_action(current_state)

            new_state, reward, done = env.step(action)

            episode_reward += reward
            nr_discov_tiles= get_nr_discovered_tiles(current_state)
            agent.update_replay_memory((current_state, action, reward, new_state, nr_discov_tiles, done))
            agent.train(done)

            n_clicks += 1

        #def func_catecasute_descoperite():#nr casute pastrate in o lista


        progress_list.append(env.n_progress) # n of non-guess moves
        nr_discov_tiles_list.append(nr_discov_tiles)
        ep_rewards.append(episode_reward)

        if env.n_wins > past_n_wins:
            wins_list.append(1)
        else:
            wins_list.append(0)

        if len(agent.replay_memory) < MEM_SIZE_MIN:
            continue

        if not episode % AGG_STATS_EVERY:
            med_progress = round(np.median(progress_list[-AGG_STATS_EVERY:]), 2)
            win_rate = round(np.sum(wins_list[-AGG_STATS_EVERY:]) / AGG_STATS_EVERY, 2)
            med_reward = round(np.median(ep_rewards[-AGG_STATS_EVERY:]), 2)
            med_discovtiles= round(np.median(nr_discov_tiles_list[-AGG_STATS_EVERY:]), 2)

            agent.tensorboard.update_stats(
                progress_med = med_progress,
                winrate = win_rate,
                reward_med = med_reward,
                learn_rate = agent.learn_rate,
                epsilon = agent.epsilon,
                med_dicoveries= med_discovtiles)

            print(f'Episode: {episode}, Median progress: {med_progress}, Median reward: {med_reward}, Win rate : {win_rate}, Median discovered tiles : {med_discovtiles}')

            episode_list.append(episode)
            med_progress_list.append(med_progress)
            med_reward_list.append(med_reward)
            win_rate_list.append(win_rate)
            med_discov_tiles_list.append(med_discovtiles)
            learn_rate_list.append(agent.learn_rate)
            epsilon_list.append(agent.epsilon)

        if not episode % SAVE_MODEL_EVERY:
            with open(f'replay/{MODEL_NAME}.pkl', 'wb') as output:
                pickle.dump(agent.replay_memory, output)

            agent.model.save(f'models/{MODEL_NAME}.h5')

    return episode_list, med_progress_list,med_reward_list,win_rate_list, med_discov_tiles_list, learn_rate_list, epsilon_list


if __name__ == "__main__":
    episode_list, med_progress_list, med_reward_list, win_rate_list, med_discov_tiles_list, learn_rate_list, epsilon_list= main()

    if not os.path.isdir('the_plots'):
        os.mkdir('the_plots')





    plt.plot(episode_list, med_progress_list, 'rs--')
    plt.ylabel('Median progress')
    plt.xlabel('Max Number of episodes= ' +str(params.episodes))
    plt.title('Progress; '+'Model= '+ str(params.model_name))
    plt.savefig('the_plots/Progress_'+str(params.model_name)+'_'+str(params.episodes)+'episodes'+'.png')
    plt.close()
    csv_name= 'the_plots/Progress_'+str(params.model_name)+'_'+str(params.episodes)+'episodes'+'.csv'
    extra_field_name= 'Progress'
    row_data_list= med_progress_list
    write_csv(csv_name, episode_list, extra_field_name, row_data_list)


    plt.plot(episode_list, med_reward_list, 'gs--')
    plt.ylabel('Median reward')
    plt.xlabel('Max Number of episodes= ' +str(params.episodes))
    plt.title('Reward; '+'Model= '+ str(params.model_name))
    plt.savefig('the_plots/Reward_'+str(params.model_name)+'_'+str(params.episodes)+'episodes'+'.png')
    plt.close()
    csv_name = 'the_plots/Reward_' + str(params.model_name) + '_' + str(params.episodes) + 'episodes' + '.csv'
    extra_field_name = 'Reward'
    row_data_list = med_reward_list
    write_csv(csv_name, episode_list, extra_field_name, row_data_list)

    plt.plot(episode_list, win_rate_list, 'bs--')
    plt.ylabel('Median win rate')
    plt.xlabel('Max Number of episodes= ' +str(params.episodes))
    plt.title('Win rate; '+'Model= '+ str(params.model_name))
    plt.savefig('the_plots/Win_rate_'+str(params.model_name)+'_'+str(params.episodes)+'episodes'+'.png')
    plt.close()
    csv_name = 'the_plots/Win_rate_' + str(params.model_name) + '_' + str(params.episodes) + 'episodes' + '.csv'
    extra_field_name = 'Win_rate'
    row_data_list = win_rate_list
    write_csv(csv_name, episode_list, extra_field_name, row_data_list)

    plt.plot(episode_list, med_discov_tiles_list, 'bs-')
    plt.ylabel('Median discovered tiles before win/loose')
    plt.xlabel('Max Number of episodes= ' +str(params.episodes))
    plt.title('Number of tiles discovered; '+'Model= '+ str(params.model_name)+' ')
    plt.savefig('the_plots/Discovtiles_+'+str(params.model_name)+'_'+str(params.episodes)+'episodes'+'.png')
    plt.close()
    csv_name = 'the_plots/Discovtiles_' + str(params.model_name) + '_' + str(params.episodes) + 'episodes' + '.csv'
    extra_field_name = 'Discovtiles'
    row_data_list = med_discov_tiles_list
    write_csv(csv_name, episode_list, extra_field_name, row_data_list)

    plt.plot(episode_list, learn_rate_list, 'rs--')
    plt.ylabel('Median learn rate')
    plt.xlabel('Max Number of episodes= ' +str(params.episodes))
    plt.title('Learn rate; '+'Model= '+ str(params.model_name))
    plt.savefig('the_plots/Learn_rate_'+str(params.model_name)+'_'+str(params.episodes)+'episodes'+'.png')
    plt.close()
    csv_name = 'the_plots/Learn_rate_' + str(params.model_name) + '_' + str(params.episodes) + 'episodes' + '.csv'
    extra_field_name = 'Learn_rate'
    row_data_list = learn_rate_list
    write_csv(csv_name, episode_list, extra_field_name, row_data_list)

    plt.plot(episode_list, epsilon_list, 'g^--')
    plt.ylabel('Median epsilon')
    plt.xlabel('Max Number of episodes= ' +str(params.episodes))
    plt.title('Epsilon; '+'Model= '+ str(params.model_name))
    plt.savefig('the_plots/Epsilon_'+str(params.model_name)+'_'+str(params.episodes)+'episodes'+'.png')
    plt.close()
    csv_name = 'the_plots/Epsilon_' + str(params.model_name) + '_' + str(params.episodes) + 'episodes' + '.csv'
    extra_field_name = 'Epsilon'
    row_data_list = epsilon_list
    write_csv(csv_name, episode_list, extra_field_name, row_data_list)

print('Return 0')