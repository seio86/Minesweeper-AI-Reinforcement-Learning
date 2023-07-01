import argparse
from tqdm import tqdm
from keras.models import load_model
from MinesweeperAgentWeb import *

def parse_args():
    parser = argparse.ArgumentParser(description='Play Minesweeper online using a DQN')
    parser.add_argument('--model', type=str, default='conv128x4_dense512x2_y0.1_minlr0.001',
                        help='name of model')
    #parser.add_argument('--model', type=str, default='conv64x4_dense512x2_y0.1_minlr0.001',
    #                    help='name of model')
    parser.add_argument('--episodes', type=int, default= 500, #was 100
                        help='Number of episodes to play')

    return parser.parse_args()

params = parse_args()

my_model = load_model(f'models/{params.model}.h5')
list_moves= []
def main():
    print("Model used: ", params)
    pg.FAILSAFE = True
    agent = MinesweeperAgentWeb(my_model)
    count_wins=0
    for episode in tqdm(range(1, params.episodes+1)):
        agent.reset()
        count_moves = 0
        done = False
        while not done:
            current_state = agent.state
            action = agent.get_action(current_state)

            new_state, done = agent.step(action)
            count_moves +=1
            if pg.locateOnScreen('WIN.png', region=agent.loc) != None:
                count_wins += 1
        print("Game ", str(episode), " had ", str(count_moves), " clicks before win/loose!")
        list_moves.append(count_moves)
    print("Model used: ", params)
    print("List of clicks per game in order: ",str(list_moves))
    print("Minimum clicks per game= ", min(list_moves), " in ", len(list_moves), " games")
    print("Average clicks per game= ", sum(list_moves) / len(list_moves), " in ", len(list_moves), " games")
    print("Maximum clicks per game= ", max(list_moves), " in ", len(list_moves), " games")
    print("Number of wins= ", str(count_wins))

    return

if __name__ == "__main__":
    main()
