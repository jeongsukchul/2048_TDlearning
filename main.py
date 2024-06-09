import numpy as np
from game import Board
from game import IllegalAction, GameOver
from agent import nTupleNetwork
import pickle
import random
from collections import namedtuple
from collections import deque
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import os 
"""
Vocabulary
--------------

Transition: A Transition shows how a board transfromed from a state to the next state. It contains the board state (s), the action performed (a), 
the reward received by performing the action (r), the board's "after state" after applying the action (s_after), and the board's "next state" (s_next) after adding a random tile to the "after state".

Gameplay: A series of transitions on the board (transition_history). Also reports the total reward of playing the game (game_reward) and the maximum tile reached (max_tile).
"""
Transition = namedtuple("Transition", "step, s, a, r, s_after, s_next")
Gameplay = namedtuple("Gameplay", "transition_history game_reward max_tile, replay_buffer")

def play(agent, board, replay_buffer,spawn_random_tile=False, alpha=0.1,beta=1.0,lambd=0.5,mode='TD0',buffer_size=100,model_learning_step=20):
    "Return a gameplay of playing the given (board) until terminal states."
    b = Board(board)
    r_game = 0
    transition_history = []
    step = 0
    while True:
        a_best = agent.best_action(b.board)
        s = b.copyboard()
        # print("state s : ",s)
        s_after = None
        try:
            r = b.act(a_best)
            r_game += r
            s_after = b.copyboard()
            b.spawn_tile(random_tile=spawn_random_tile)
            s_next = b.copyboard()

            # agent.update(transition_history, step, mode, alpha=alpha)
            transition_history.append(
                Transition(step = step, s=s, a=a_best, r=r, s_after=s_after, s_next=s_next)
            )
            step+=1

        except (IllegalAction, GameOver) as e:
            # game ends when agent makes illegal moves 
            r = None
            # if e == GameOver:
            #     s_after = None
            # elif e == IllegalAction:
            #     s_after = s
            s_after = None
            s_next = None
            break

    # agent.termination_update(transition_history, step, mode, alpha=alpha)
    replay_buffer +=transition_history
    replay_buffer = replay_buffer[-100:]
    gp = Gameplay(
        transition_history=transition_history,
        game_reward=r_game,
        max_tile=2 ** max(b.board),
        replay_buffer = replay_buffer
    )
    
    learn_from_gameplay(agent, gp, mode, alpha=0.1, beta=1.0, lambd=0.5, model_learning_step=20)
    return gp


def learn_from_gameplay(agent, gp, mode, alpha=0.1, beta=1.0,lambd= 0.5, model_learning_step=20):
    "Learn transitions in reverse order except the terminal transition"
    # for i in range(len(gp.transition_history)-1):
    #     agent.update(gp.transition_history[:(i+1)], i+1, mode, alpha=alpha, lambd=lambd)
    # agent.termination_update(gp.transition_history,len(gp.transition_history),mode,alpha=alpha)
    for tr in gp.transition_history[::-1]:
        delta = agent.GetDelta(tr.s_after, tr.s_next)
        # agent.V(tr.s_after,alpha*delta)
        agent.update(gp.transition_history, delta, tr.step, mode, alpha=0.1, beta=1.0, lambd=0.5)
    #Dyna-Q loading
    if len(gp.replay_buffer)==0:
        return
    for i in range(model_learning_step):
        item = random.randint(0, len(gp.replay_buffer)-1)
        delta = agent.GetDelta(gp.replay_buffer[item].s_after, gp.replay_buffer[item].s_next)
        agent.V(gp.replay_buffer[item].s_after,alpha*delta)

    
def load_agent(path):
    return pickle.load(path.open("rb"))


# map board state to LUT
TUPLES = [
    # # horizontal 4-tuples
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
    # vertical 4-tuples
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
    # all 4-tile squares
    [0, 1, 4, 5],
    [4, 5, 8, 9],
    [8, 9, 12, 13],
    [1, 2, 5, 6],
    [5, 6, 9, 10],
    [9, 10, 13, 14],
    [2, 3, 6, 7],
    [6, 7, 10, 11],
    [10, 11, 14, 15],
    # [[0,1,2,3],[0,4,8,12],[3,7,11,15],[12,13,14,15]],
    # [[4,5,6,7],[8,9,10,11],[1,5,9,13],[2,6,10,14]],
    # [[0,1,2,4,5,6],[1,2,3,5,6,7],[8,9,10,12,13,14],[9,10,11,13,14,15],[0,1,4,5,8,9],[2,3,6,7,10,11],[4,5,8,9,12,13],[6,7,10,11,14,15]],
    # [[4,5,6,8,9,10],[5,6,7,9,10,11],[1,2,5,6,9,10],[5,6,9,10,13,14]],
]
TUPLES_sym = [
    [[0,1,2,3],[0,4,8,12],[3,7,11,15],[12,13,14,15]],
    [[4,5,6,7],[8,9,10,11],[1,5,9,13],[2,6,10,14]],
    [[0,1,2,4,5,6],[1,2,3,5,6,7],[8,9,10,12,13,14],[9,10,11,13,14,15],[0,1,4,5,8,9],[2,3,6,7,10,11],[4,5,8,9,12,13],[6,7,10,11,14,15]],
    [[4,5,6,8,9,10],[5,6,7,9,10,11],[1,2,5,6,9,10],[5,6,9,10,13,14]],
]

def plot(log,mode_name,path):
    nb_rows = 2
    nb_cols = 2
    fig, axs = plt.subplots(nb_rows, nb_cols)
    for key, value in log.items():
        games = np.linspace(0, len(value)*100, len(value))
    a = axs[0,0]
    a.plot(games, log["reward"])
    a.set(xlabel='games', ylabel='mean rewards', title='Mean Rewards')

    a = axs[0,1]
    a.plot(games, log["mean_max_tile"])
    a.set(xlabel='games', ylabel='mean max tile', title='Mean Max tile')

    a = axs[1,0]
    a.plot(games, log["2048_rate"])
    a.set(xlabel='games', ylabel='2048 rates', title='2048 success rate')

    a = axs[1,1]
    a.plot(games, log["maxtile"])
    a.set(xlabel='games', ylabel='maximum tile', title='Maximum Tile')
    plt.suptitle(mode_name, fontsize=20)
    PNG_PATH = os.path.join(SAVE_PATH, 'plot.png')
    plt.savefig(PNG_PATH)
    plt.show()



def append_to_csv(file_path, games, result_name, return_value, isfirstappend = False):
    # Create a DataFrame with the new data
    data = {'games': games, result_name: return_value}
    df = pd.DataFrame(data)
    # Append the DataFrame to the CSV file
    df.to_csv(file_path, mode='w', header=isfirstappend, index=False)

if __name__ == "__main__":
    import numpy as np

    agent = None
    # prompt to load saved agents
    from pathlib import Path

    
    n_session = 5000
    n_episode = 100
    alpha = 0.1
    beta = 1.0
    lambd = 0.5
    mode = 'TDlambda'
    DynaQ = False
    if DynaQ:
        model_learning_step = 20
        buffer_size=100
    else:
        model_learning_step = 0
        buffer_size = 0
    symmetric_sampling = True
    after_state=True 

    path = Path("tmp")
    saves = list(path.glob("*.pkl"))
    if len(saves) > 0:
        print("Found %d saved agents:" % len(saves))
        for i, f in enumerate(saves):
            print("{:2d} - {}".format(i, str(f)))
        k = input(
            "input the id to load an agent, input nothing to create a fresh agent:"
        )
        if k.strip() != "":
            k = int(k)
            n_games, agent = load_agent(saves[k])
            print("load agent {}, {} games played".format(saves[k].stem, n_games))
    if agent is None:
        print("initialize agent")
        n_games = 0
        if symmetric_sampling:
            agent = nTupleNetwork(TUPLES_sym, symmetric_sampling=symmetric_sampling, after_state=after_state,lambd=lambd)
        else:
            agent = nTupleNetwork(TUPLES, symmetric_sampling=symmetric_sampling, after_state=after_state, lambd=lambd)
   


    log = defaultdict(list)
    print("training")
    try:
        for i_se in range(n_session):
            gameplays = []
            replay_buffer = []
            for i_ep in range(n_episode):
                gp = play(agent, None, replay_buffer, spawn_random_tile=True,alpha=alpha,beta=beta,lambd=lambd,mode=mode,buffer_size=buffer_size, model_learning_step=model_learning_step)
                gameplays.append(gp)
                n_games += 1
            n2048 = sum([1 for gp in gameplays if gp.max_tile == 2048])
            mean_maxtile = np.mean([gp.max_tile for gp in gameplays])
            maxtile = max([gp.max_tile for gp in gameplays])
            mean_gamerewards = np.mean([gp.game_reward for gp in gameplays])
            print(
                "Game# %d, tot. %dk games, " % (n_games, n_games / 1000)
                + "mean game rewards {:.0f}, mean max tile {:.0f}, 2048 rate {:.0%}, maxtile {}".format(
                    mean_gamerewards, mean_maxtile, n2048 / len(gameplays), maxtile
                ),
            )
            #log values
            log["reward"].append(mean_gamerewards)
            log["mean_max_tile"].append(mean_maxtile)
            log["2048_rate"].append(n2048 / len(gameplays))
            log["maxtile"].append(maxtile)
    except KeyboardInterrupt:
        print("training interrupted")
        # print("{} games played by the agent".format(n_games))
        # if input("save the agent? (y/n)") == "y":
        #     fout = "tmp/{}_{}games.pkl".format(agent.__class__.__name__, n_games)
        #     pickle.dump((n_games, agent), open(fout, "wb"))
        #     print("agent saved to", fout)
        # mode_name = mode +"+alpha_"+str(alpha)
        # if symmetric_sampling:
        #     mode_name = 'sym_'+mode_name
        # if ~after_state:
        #     mode_name+='_no_after_state'
        # if DynaQ:
        #     mode_name+='_dyanq_'+str(model_learning_step)
        # if input("save history with csv file? (y/n)")=="y":
        #     for key, value in log.items():
        #         games = np.linspace(0, len(value)*100, len(value))
        #     SAVE_PATH = os.path.abspath(os.path.dirname(__file__))

        #     SAVE_PATH = os.path.join(SAVE_PATH, mode_name)
        #     if not os.path.exists(SAVE_PATH):
        #         os.makedirs(SAVE_PATH)
        #     CSV_PATH = os.path.join(SAVE_PATH, 'reward.csv')
        #     append_to_csv(CSV_PATH, games=list(games), return_value=list(log["reward"]), result_name='Mean reward', isfirstappend=True)
        #     CSV_PATH = os.path.join(SAVE_PATH, 'mean_max_tile.csv')
        #     append_to_csv(CSV_PATH, games=list(games), return_value=list(log["mean_max_tile"]), result_name='Mean Max Tile', isfirstappend=True)
        #     CSV_PATH = os.path.join(SAVE_PATH, '2048_rate.csv')
        #     append_to_csv(CSV_PATH, games=list(games), return_value=list(log["2048_rate"]), result_name='2048 rate', isfirstappend=True)
        #     CSV_PATH = os.path.join(SAVE_PATH, 'max_tile.csv')
        #     append_to_csv(CSV_PATH, games=list(games), return_value=list(log["maxtile"]), result_name='Max Tile', isfirstappend=True)
        # if input("plot log (y/n)")=="y":
        #     plot(log,mode_name)
mode_name = mode +"+alpha_"+str(alpha)
if symmetric_sampling:
    mode_name = 'sym_'+mode_name
if not after_state:
    mode_name+='_no_after_state'
if DynaQ:
    mode_name+='_dyanq_'+str(model_learning_step)
for key, value in log.items():
    games = np.linspace(0, len(value)*100, len(value))

SAVE_PATH = os.path.abspath(os.path.dirname(__file__))
SAVE_PATH = os.path.join(SAVE_PATH, mode_name)
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


MODEL_PATH = os.path.join(SAVE_PATH,mode_name+'.pkl')
pickle.dump(agent, open(MODEL_PATH, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
print("agent saved to", MODEL_PATH)


CSV_PATH = os.path.join(SAVE_PATH, 'reward.csv')
append_to_csv(CSV_PATH, games=list(games), return_value=list(log["reward"]), result_name='Mean reward', isfirstappend=True)
CSV_PATH = os.path.join(SAVE_PATH, 'mean_max_tile.csv')
append_to_csv(CSV_PATH, games=list(games), return_value=list(log["mean_max_tile"]), result_name='Mean Max Tile', isfirstappend=True)
CSV_PATH = os.path.join(SAVE_PATH, '2048_rate.csv')
append_to_csv(CSV_PATH, games=list(games), return_value=list(log["2048_rate"]), result_name='2048 rate', isfirstappend=True)
CSV_PATH = os.path.join(SAVE_PATH, 'max_tile.csv')
append_to_csv(CSV_PATH, games=list(games), return_value=list(log["maxtile"]), result_name='Max Tile', isfirstappend=True)
plot(log,mode_name,SAVE_PATH)

