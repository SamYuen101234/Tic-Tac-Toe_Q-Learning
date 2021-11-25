from game import *
from agent import *

import sys
import pickle
import os
import argparse
from tqdm import tqdm


class GameLearning(object):
    """
    A class that holds the state of the learning process. Learning
    agents are created/loaded here, and a count is kept of the
    games that have been played.
    """
    def __init__(self, args, alpha=0.5, gamma=0.9, epsilon=0.1, eps_decay=0., encourage_explore=True):

        if args.load:
            # load an existing agent and continue training
            if not os.path.isfile(args.path):
                raise ValueError("Cannot load agent: file does not exist.")
            with open(args.path, 'rb') as f:
                agent = pickle.load(f)
                agent.eps=0 # no random action during in during competition
        else:
            # check if agent state file already exists, and ask
            # user whether to overwrite if so
            if os.path.isfile(args.path):
                print('An agent is already saved at {}.'.format(args.path))
                while True:
                    response = input("Are you sure you want to overwrite? [y/n]: ")
                    if response.lower() in ['y', 'yes']:
                        break
                    elif response.lower() in ['n', 'no']:
                        print("OK. Quitting.")
                        sys.exit(0)
                    else:
                        print("Invalid input. Please choose 'y' or 'n'.")

            agent = QLearner(alpha,gamma,epsilon,eps_decay=eps_decay,encourage_explore=encourage_explore)


        self.games_played = 0
        self.path = args.path
        self.agent = agent
        self.agent_num_win = 0
        self.challenger_num_win = 0

    def beginPlaying(self):
        """ Loop through game iterations with a human player. """
        print("Welcome to Tic-Tac-Toe. You are 'X' and the computer is 'O'.")

        def play_again():
            print("Games played: %i" % self.games_played)
            while True:
                play = input("Do you want to play again? [y/n]: ")
                if play == 'y' or play == 'yes':
                    return True
                elif play == 'n' or play == 'no':
                    return False
                else:
                    print("Invalid input. Please choose 'y' or 'n'.")

        while True:
            game = Tic_Tac_Toe(self.agent)
            game.start()
            self.games_played += 1
            self.agent.save(self.path)
            if not play_again():
                print("OK. Quitting.")
                break

    def beginLearning(self, episodes, alpha=0.5, gamma=0.9, epsilon=0.1, eps_decay=0., encourage_explore=True, debug_mode=False):
        """ Loop through game iterations with a teaching agent. """
        self.challenger = QLearner(alpha, gamma, epsilon, eps_decay=eps_decay, encourage_explore=encourage_explore) # another agent
        # Train for alotted number of episodes
        for i in tqdm(range(episodes)):
            game = Tic_Tac_Toe(player1=self.agent, player2=self.challenger, debug_mode=debug_mode)
            agent_num_win, challenger_num_win = game.start()
            self.agent_num_win+=agent_num_win
            self.challenger_num_win+=challenger_num_win
            self.games_played += 1
            # Monitor progress
            if self.games_played % 10000 == 0:
                print("Games played: %i" % self.games_played)
                num_draw = self.games_played - self.agent_num_win - self.challenger_num_win
                print('Agent 1 won ' + str(self.agent_num_win) + ', ' + 'Agent 2 won ' + str(self.challenger_num_win) 
                      + ', ' + 'Num draw: ' + str(num_draw) + ', Esplion: ' + str(self.agent.eps) )
                
                # save best agent
        if self.agent_num_win > self.challenger_num_win:
            self.agent.save(self.path)
        else:
            self.challenger.save(self.path)
    
    def test(self, test_episodes=100000):
        self.agent_num_win = 0
        self.challenger_num_win = 0
        # load the best agent as agent1 for test
        with open(args.path, 'rb') as f:
            self.agent = pickle.load(f)
        self.agent.eps = 0 # no random
        self.challenger.eps = 1 # all random
        self.challenger.eps_decay=0. # prevent eps_decay during testing, the agent in random
        for i in tqdm(range(test_episodes)):
            game = Tic_Tac_Toe(player1=self.agent, player2=self.challenger)
            agent_num_win, challenger_num_win = game.start()
            self.agent_num_win+=agent_num_win
            self.challenger_num_win+=challenger_num_win
        print("Games played: %i" % self.games_played)
        num_draw = test_episodes - self.agent_num_win - self.challenger_num_win
        print('Agent 1 won ' + str(self.agent_num_win) + ', ' + 'Agent 2 won ' + str(self.challenger_num_win) 
                + ', ' + 'Num draw: ' + str(num_draw))
        
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Play Tic-Tac-Toe.")
    parser.add_argument("-p", "--path", type=str, required=False,
                        help="Specify the path for the agent pickle file. "
                             "Defaults to q_agent.pkl")
    parser.add_argument("-l", "--load", action="store_true",
                        help="whether to load trained agent")
    parser.add_argument("-n", "--num_episodes", default=None, type=int,
                        help="employ teacher agent who knows the optimal "
                             "strategy and will play for TEACHER_EPISODES games")
    
    parser.add_argument("-d", "--debug", default=0, type=int,
                        help="Debug mode. Print out the board in each step")
    
    args = parser.parse_args()
    
    args.path = 'q_agent.pkl'
    args.alpha=0.3
    args.gamma=0.9
    args.eps_decay=0.
    args.epsilon=0.2 # randomness
    args.encourage_explore=True # if the Q-value is 0, we set it as 1, encourage the agents to explore it and optimize it
    
    gl = GameLearning(args=args, 
                      alpha=args.alpha, 
                      gamma=args.gamma, 
                      epsilon=args.epsilon, 
                      eps_decay=args.eps_decay,
                      encourage_explore=args.encourage_explore) # Create env and initialize the agent
    
    if args.num_episodes is not None:
        # initialize agent2 and start the game
        gl.beginLearning(episodes=args.num_episodes, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, 
                         eps_decay=args.eps_decay, encourage_explore=args.encourage_explore, debug_mode=args.debug)
        print('Testing')
        gl.test()
    else:
        gl.beginPlaying()
        