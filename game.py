import numpy as np
import random



class Tic_Tac_Toe:
    def __init__(self, player1, player2=None, debug_mode=0):
        self.board = np.array([' ' for x in range(9)]).reshape(3,3)
        self.agent = player1
        self.challenger = player2
        self.debug_mode = debug_mode
        self.agent_num_win = 0
        self.challenger_num_win = 0
    
    def agentMove(self, action):
        self.board[action[0], action[1]] = 'O'
        if self.debug_mode:
            print("Agent 1's Action")
            self.printBoard()
    
    def challengerMove(self, action=None):
        # agent 2
        if action is not None:
            self.board[action[0], action[1]] = 'X'
            if self.debug_mode:
                print("Agent 2's Action")
                self.printBoard()
        else: # human player's term
            self.printBoard()
            while True:
                move = input("Your move! Please select a row and column from 0-2 "
                                    "in the format row,col: ")
                print('\n')
                try:
                        row, col = int(move[0]), int(move[2])
                except ValueError:
                    print("INVALID INPUT! Please use the correct format.")
                    continue
                
                # the inputs are out of range (0-2)
                if row not in range(3) or col not in range(3) or not self.board[row][col] == ' ':
                    print("INVALID MOVE! Choose again.")
                    continue
                
                self.board[row][col] = 'X'
                break
    
    def checkWin(self, key):
        ''' 
        Check whether the player X or player O wins the game
        Key: X or O
        '''
         # check for player win on diagonals
        a = [self.board[0][0], self.board[1][1], self.board[2][2]]
        b = [self.board[0][2], self.board[1][1], self.board[2][0]]
        if a.count(key) == 3 or b.count(key) == 3:
            return True
        # check for player win on rows/columns
        for i in range(3):
            col = [self.board[0][i], self.board[1][i], self.board[2][i]]
            row = [self.board[i][0], self.board[i][1], self.board[i][2]]
            if col.count(key) == 3 or row.count(key) == 3:
                return True
        return False
    
    def checkDraw(self):
        ''' No one win the game '''
        draw = True
        for row in self.board:
            for elt in row:
                if elt == ' ':
                    draw = False
        return draw
    
    def checkEnd(self, key):
        if self.checkWin(key):
            if self.challenger is None:
                self.printBoard()
                if key == 'X':
                    print("Challenger/Human wins!")
                else:
                    print("Q agent wins!")
            return 1
        elif self.checkDraw():
            if self.challenger is None:
                self.printBoard()
                print("It's a draw!")
            return 0.5 # reward of draw
        return -1 # not end yet
    
    def playGame(self, player_first):
        challenger_prev_action=None
        if player_first: # agent2 or player first
            if self.challenger is not None:
                prev_state = self.getStateKey(self.board) # check old state
                challenger_prev_action = self.challenger.get_action(prev_state) # make decision base on the old state
                self.challengerMove(challenger_prev_action) # make a move
                # first step only, no need to check, game continues. 0 reward
                challenger_reward = 0
                new_state = self.getStateKey(self.board) # check new state
                # challenger_new_action = self.challenger.get_action(new_state) # check new action (Q-learning not need this)
                #self.challenger.update(new_state, None, challenger_reward) # update the Q-values
            else:
                self.challengerMove()
            
        # iterate until game is over
        while True:
            prev_state = self.getStateKey(self.board)
            agent_prev_action = self.agent.get_action(prev_state)
            # execute oldAction, observe reward and state
            self.agentMove(agent_prev_action)
            agent_check = self.checkEnd('O') # check agent1
            if not agent_check == -1: # ended, agent1 wins or draw, agent lose or draw
                # game is over. +1 reward if win, 0 if draw
                if agent_check == 1:
                    self.agent_num_win += agent_check # count the number of games agent1 has won
                    agent_reward = agent_check
                    challenger_reward = -1*agent_check
                else: # draw, both can receive 0.5 reward
                    agent_reward = agent_check
                    challenger_reward = agent_check
                break
            else:
                challenger_reward = 0 # game continues. agent2 0 reward
                #challenger_new_action = None
                if self.challenger is not None:
                    new_state = self.getStateKey(self.board)
                    # challenger_new_action = self.challenger.get_action(new_state) # check new action (Q-learning not need this)
                    self.challenger.update(new_state, None, challenger_reward) # update the agent2' Q-value
            
            
            if self.challenger is not None:
                prev_state = self.getStateKey(self.board)
                challenger_prev_action = self.challenger.get_action(prev_state)
                self.challengerMove(challenger_prev_action) # agent move
            else:
                self.challengerMove() # player move
            challenger_check = self.checkEnd('X') # check agent2 or player
            if not challenger_check == -1: # ended
                # game is over. -1 reward if lose, 0 if draw
                if challenger_check == 1:
                    self.challenger_num_win += challenger_check # count the number of games agent1 has won
                    agent_reward = -1*challenger_check
                    challenger_reward = challenger_check
                else:
                    agent_reward = challenger_check
                    challenger_reward = challenger_check
                break
            else:
                agent_reward = 0 # not end yet, agent1 reward 0
                new_state = self.getStateKey(self.board)
                #agent_new_action = self.agent.get_action(new_state) # determine new action (epsilon-greedy)
                self.agent.update(new_state, None, agent_reward) # update agent1' Q-values
        
        # Game over. Update the two agents
        new_state = self.getStateKey(self.board)
        self.agent.update(new_state, None, agent_reward)
        if self.challenger is not None:
            self.challenger.update(new_state, None, challenger_reward)
    
    def start(self):
        if self.challenger is not None:
            # During learning, chose who goes first randomly with equal probability
            if random.random() < 0.5:
                self.playGame(player_first=False) # agent1 first
            else:
                self.playGame(player_first=True) # agent2 first
        else:
            while True:
                response = input("Would you like to go first? [y/n]: ")
                print('')
                if response == 'n' or response == 'no':
                    self.playGame(player_first=False)
                    break
                elif response == 'y' or response == 'yes':
                    self.playGame(player_first=True)
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
        
        return self.agent_num_win, self.challenger_num_win
    
    def getStateKey(self, board):

        key = ''
        for row in board:
            for elt in row:
                key += elt
        return key


    def printBoard(self):
        # "board" is a list of 10 strings representing the board (ignore index 0)
        print(' ' + self.board[0][0] + ' | ' + self.board[0][1] + ' | ' + self.board[0][2])
        print('-----------')
        print(' ' + self.board[1][0] + ' | ' + self.board[1][1] + ' | ' + self.board[1][2])
        print('-----------')
        print(' ' + self.board[2][0] + ' | ' + self.board[2][1] + ' | ' + self.board[2][2])
    
    
