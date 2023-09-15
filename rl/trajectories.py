""" A trajectory can be defined as the sequence:

    Tau = (s_0, a_0, s_1, a_1, ... , s_T, a_T)

    Where s_t is the state at time t, and a_t the action taken by
    the policy in this state. Because the reward only occurs at
    the end of the game we will do trajectories until the game is over,
    which means both players pass (draw), one resigns (un-implemented
    as of 14/09/23) or there are no more legal moves.

    Our RL agent will always play white for now (14/09/23), this gives it
    a slight disadvantage as black plays first. """

import numpy as np

from nanoAlphaGo.game.board import GoBoard
from nanoAlphaGo.config import BLACK, WHITE
from nanoAlphaGo.rl.policy import PolicyNN
from nanoAlphaGo.graphics.rendering import display_board


def collect_trajectories(policyNN, n_trajectories):
    trajectories = [play_game(policyNN) for _ in range(n_trajectories)]
    return trajectories


def play_game(policy):
    board = GoBoard()
    adversary = PolicyNN(BLACK)
    players = {WHITE: policy,
               BLACK: adversary}

    trajectory = []
    consecutive_passes = 0

    player = BLACK
    while not game_is_over(board, consecutive_passes, player):
        board_state = np.copy(board.board)
        move = players[player].generate_move(board)
        if move == "pass":
            consecutive_passes += 1
        else:
            consecutive_passes = 0
            board.apply_move(move, player)
        trajectory.append({
            "board_state": board_state,
            "move": move})
        player = -player
    import pdb;pdb.set_trace() 
    return trajectory


def game_is_over(board, consecutive_passes, turn):
    players_both_passed = consecutive_passes > 1
    no_legal_moves = len(board.legal_moves(turn)) == 0
    return players_both_passed or no_legal_moves


if __name__ == '__main__':
    policy = PolicyNN(color=WHITE)
    t = play_game(policy)

