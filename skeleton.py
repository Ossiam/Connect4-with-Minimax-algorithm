from nis import match
import gym
import random
import requests
import numpy as np
import argparse
import sys
import time
from gym_connect_four import ConnectFourEnv

env: ConnectFourEnv = gym.make("ConnectFour-v0")

#SERVER_ADRESS = "http://localhost:8000/"
SERVER_ADRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["example_user"]
ROWS = 6
COLUMNS = 7
DEPTH = 3

PLAYER = 1
AI = -1


def call_server(move):
   res = requests.post(SERVER_ADRESS + "move",
                       data={
                           "stil_id": STIL_ID,
                           "move": move, # -1 signals the system to start a new game. any running game is counted as a loss
                           "api_key": API_KEY,
                       })
   # For safety some respose checking is done here
   if res.status_code != 200:
      print("Server gave a bad response, error code={}".format(res.status_code))
      exit()
   if not res.json()['status']:
      print("Server returned a bad status. Return message: ")
      print(res.json()['msg'])
      exit()
   return res

def check_stats():
   res = requests.post(SERVER_ADRESS + "stats",
                       data={
                           "stil_id": STIL_ID,
                           "api_key": API_KEY,
                       })

   stats = res.json()
   return stats

"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""
def opponents_move(env):
   env.change_player() # change to oppoent
   avmoves = env.available_moves()
   if not avmoves:
      env.change_player() # change back to student before returning
      return -1

   # TODO: Optional? change this to select actions with your policy too
   # that way you get way more interesting games, and you can see if starting
   # is enough to guarrantee a win
   action = random.choice(list(avmoves))

   state, reward, done, _ = env.step(action)
   if done:
      if reward == 1: # reward is always in current players view
         reward = -1
   env.change_player() # change back to student before returning
   return state, reward, done
      
def get_valid_placement(state):
   return [col for col in range(COLUMNS) if state[0][col] == 0]

def get_available_row(state, col):
   return next(row for row in reversed(range(ROWS)) if state[row][col] == 0)

def check_win(state):
   
   lineD = np.zeros([4], dtype=int)
   lineU = np.zeros([4], dtype=int)
   
   # Calculate horizontal score
   for row in range(ROWS):
      for col in range(COLUMNS-3):
         line = state[row, col:col+4]
         if abs(np.sum(line)) == 4:
            return True
         
   # Calculate vertical Score
   for row in range(ROWS-3):
      for col in range(COLUMNS):
         line = state[row:row+4, col]
         if abs(np.sum(line)) == 4:
            return True
         
   # Calculate diagonal score, downwards and upwards
   for row in range(ROWS-3):
      for col in range(COLUMNS-3):
         for n in range(4):
            lineD[n] = state[row+n][col+n]
            lineU[n] = np.fliplr(state)[row+n][col+n]
            if abs(np.sum(lineD)) == 4 or abs(np.sum(lineU)) == 4:
               return True
   
def line_scoring(line):

   """
   Evaluates a given line of 4 discs and assigns it a score
   Prioritization of the mini-max algorithm can be tuned by changing the values
   """
   score = 0
   player_discs = np.sum(line == 1)
   AI_discs = np.sum(line == -1)
   
   if player_discs == 4:
      score += np.Inf

   if player_discs == 3:
      score += 6000
   if player_discs == 2:
      score += 2000

   if AI_discs == 4:
      score -= np.Inf

   if AI_discs == 3:
      score -= 8000
   if AI_discs == 2:
      score -= 2000

   return score
      
def score_state(state):
   
   score = 0
   
   lineD = np.zeros([4], dtype=int)
   lineU = np.zeros([4], dtype=int)
   
   # Calculate horizontal score
   score += np.sum([line_scoring(state[row, col:col+4]) 
      for row in range(ROWS) for col in range(COLUMNS-3)])
   # Calculate vertical Score
   score += np.sum([line_scoring(state[row:row+4, col]) 
      for row in range(ROWS-3) for col in range(COLUMNS)])
   
   # Calculate diagonal score, downwards and upwards
   for row in range(ROWS-3):
      for col in range(COLUMNS-3):
         for n in range(4):
            lineD[n] = state[row+n][col+n]
            lineU[n] = np.fliplr(state)[row+n][col+n]
         score += line_scoring(lineD) + line_scoring(lineU)

   return score
   
def minimax(state, depth, alpha, beta, maximizingPlayer):
   
   valid_placements = get_valid_placement(state)

   if len(valid_placements) == 0:
      return 0

   if depth == 0 or check_win(state):
      return score_state(state)
      
   if maximizingPlayer:
      maxEval = -np.inf
      for col in valid_placements:
         row = get_available_row(state,col)
         copy_state = state.copy()
         copy_state[row][col] = 1
         score = minimax(copy_state, depth-1, alpha, beta, False)
         if score > maxEval:
            maxEval = score
         alpha = max(alpha, maxEval)
         if beta <= alpha:
            break
      return maxEval

   else:
      minEval = np.inf
      for col in valid_placements:
         row = get_available_row(state,col)
         copy_state = state.copy()
         copy_state[row][col] = -1
         score = minimax(copy_state, depth-1, alpha, beta, True)
         if score < minEval:
            minEval = score
         beta = min(beta, minEval)
         if beta <= alpha:
            break
      return minEval

def student_move(state):

   start_time = time.time()
   best_placement = []
   for col in get_valid_placement(state):
      row = get_available_row(state, col)
      copy_state = state.copy()
      copy_state[row][col] = 1
      new_score = minimax(copy_state, DEPTH, -np.inf, np.inf, False)
      best_placement.append((new_score, col))
   sorted_best_placement = sorted(best_placement)

   print(sorted_best_placement)
   end_time = time.time()
   totaltime = end_time - start_time

   print("Duration: ", float("{:.3f}".format(totaltime)), "seconds")
   return sorted_best_placement[len(sorted_best_placement)-1][1]
   
def play_game(vs_server = False):
   """
   The reward for a game is as follows. You get a
   botaction = random.choice(list(avmoves)) reward from the
   server after each move, but it is 0 while the game is running
   loss = -1
   win = +1
   draw = +0.5
   error = -10 (you get this if you try to play in a full column)
   Currently the player always makes the first move
   """

   # default state
   state = np.zeros((6, 7), dtype=int)

   # setup new game
   if vs_server:
      # Start a new game
      res = call_server(-1) # -1 signals the system to start a new game. any running game is counted as a loss

      # This should tell you if you or the bot starts
      print(res.json()['msg'])
      botmove = res.json()['botmove']
      state = np.array(res.json()['state'])
   else:
      # reset game to starting state
      env.reset(board=None)
      # determine first player
      student_gets_move = random.choice([True, False])
      if student_gets_move:
         print('You start!')
         print()
      else:
         print('Bot starts!')
         print()

   # Print current gamestate
   print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
   print(state)
   print()

   done = False
   while not done:
      # Select your move
      stmove = student_move(state)

      # make both student and bot/server moves
      if vs_server:
         # Send your move to server and get response
         res = call_server(stmove)
         print(res.json()['msg'])

         # Extract response values
         result = res.json()['result']
         botmove = res.json()['botmove']
         state = np.array(res.json()['state'])
      else:
         if student_gets_move:
            # Execute your move
            avmoves = env.available_moves()
            if stmove not in avmoves:
               print("You tied to make an illegal move! Games ends.")
               break
            state, result, done, _ = env.step(stmove)

         student_gets_move = True # student only skips move first turn if bot starts

         # print or render state here if you like

         # select and make a move for the opponent, returned reward from students view
         if not done:
            state, result, done = opponents_move(env)

      # Check if the game is over
      if result != 0:
         done = True
         if not vs_server:
            print("Game over. ", end="")
         if result == 1:
            print("You won!")
         elif result == 0.5:
            print("It's a draw!")
         elif result == -1:
            print("You lost!")
         elif result == -10:
            print("You made an illegal move and have lost!")
         else:
            print("Unexpected result result={}".format(result))
         if not vs_server:
            print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
         print(state)
         print()
         return result
      else:
         print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

      # Print current gamestate
      print(state)
      print()

def main():
   # Parse command line arguments
   parser = argparse.ArgumentParser()
   group = parser.add_mutually_exclusive_group()
   group.add_argument("-l", "--local", help = "Play locally", action="store_true")
   group.add_argument("-o", "--online", help = "Play online vs server", action="store_true")
   parser.add_argument("-s", "--stats", help = "Show your current online stats", action="store_true")
   parser.add_argument("-r", "--rounds", help = "Choose how many rounds to play", type=int, default=1)
   args = parser.parse_args()

   wins = 0
   losses = 0
   totaltime = 0
   round_number = 1

   # Print usage info if no arguments are given
   if len(sys.argv)==1:
      parser.print_help(sys.stderr)
      sys.exit(1)

   start_time = time.time()

   for i in range(args.rounds):
      print("Round number: ", round_number)
      if args.local:
         result = play_game(vs_server = False)
      elif args.online:
         result = play_game(vs_server = True)
      if result == 1:
         wins += 1
      if result == -1:
         losses += 1
      round_number += 1

   end_time = time.time()
   totaltime += end_time - start_time

   print("Wins: ", wins)
   print("Losses: ", losses)
   print("Total time: ", float("{:.3f}".format(totaltime)), "seconds")
   
   if args.stats:
      stats = check_stats()
      print(stats)

   # Run program with "--online" when you are ready to play against the server
   # the results of your games there will be logged
   # you can check your stats bu running the program with "--stats"

if __name__ == "__main__":
    main()
