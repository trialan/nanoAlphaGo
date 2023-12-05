""" A trajectory can be defined as the sequence:

    Tau = (s_0, a_0, s_1, a_1, ... , s_T, a_T)

    Where s_t is the state at time t, and a_t the action taken by
    the policy in this state. Because the reward only occurs at
    the end of the game we will do trajectories until the game is over,
    which means both players pass (draw), one resigns (un-implemented
    as of 14/09/23) or there are no more legal moves.

    Our RL agent will always play white for now (14/09/23), this gives it
    a slight disadvantage as black plays first (this is typically adjusted
    by imposing a komi). """


from tqdm import tqdm
import multiprocessing
import numpy as np
import torch
import wandb

from nanoAlphaGo.config import BLACK, WHITE, PASS, BOARD_SIZE
from nanoAlphaGo.game.board import GoBoard, assert_board_is_self_consistent
from nanoAlphaGo.game.scoring import calculate_outcome_for_player
from nanoAlphaGo.graphics.rendering import display_board
from nanoAlphaGo.rl.policy import PolicyNN


def collect_trajectories(policyNN, n_trajectories):
    state_dict = policyNN.state_dict()
    with multiprocessing.Pool(7) as pool:
        args = [(state_dict, _) for _ in range(n_trajectories)]
        trajectories = list(pool.starmap(play_game_wrapper, args))
    wandb.log({"Mean score": get_mean_score(trajectories)})
    return trajectories


def st_collect_trajectories(policyNN, n_trajectories):
    trajectories = [play_game(policyNN) for _ in tqdm(range(n_trajectories))]
    wandb.log({"Mean score": get_mean_score(trajectories)})
    return trajectories


def get_mean_score(trajectories):
    scores = [t['rewards'][-1] for t in trajectories]
    return np.mean(scores)


def play_game(policy):
    board = GoBoard()
    adversary = PolicyNN(BLACK)
    game_data = initialise_game_data()

    while not game_is_over(board, game_data):
        play_turn(board, policy, adversary, game_data)
        switch_player(game_data)

    game_outcome = calculate_outcome_for_player(board, policy.color)
    trajectory = build_trajectory(game_data, game_outcome, policy.device)
    return trajectory


def play_game_wrapper(state_dict, _):
    policyNN = PolicyNN(WHITE)
    policyNN.load_state_dict(state_dict)
    policyNN.eval()
    return play_game(policyNN)


def initialise_game_data():
    game_data = {
        'moves': [],
        'rewards': [],
        'board_states': [],
        'policy_probs': [],
        'consecutive_passes': 0,
        'player': BLACK,
        'turn': 0,
    }
    return game_data


def game_is_over(board, game_data):
    """ Two consecutive passes, or only legal move is PASS, or too many turns. """
    players_both_passed = game_data["consecutive_passes"] > 1
    no_legal_moves = board.legal_moves(game_data["player"]) == []
    game_too_long = too_many_turns(game_data)
    return players_both_passed or no_legal_moves or game_too_long


def too_many_turns(game_data):
    """ 101 was chosen as turn limit because 9x9 = 81. I round up for PASS
        move, but realistically the agent should rarely pass, and it would
        seem odd for there to be over 20 passes in a game. """
    threshold = BOARD_SIZE * BOARD_SIZE + 20
    too_many_turns = game_data["turn"] - game_data["moves"].count(PASS) > threshold
    return too_many_turns


def play_turn(board, policy, adversary, game_data):
    player = game_data["player"]
    network = get_player_network(policy, adversary, player)
    move, board_state, probs = compute_move(network, board)
    update_consecutive_passes(move, game_data)
    board.apply_move(move, player)
    game_data = update_game_data(game_data, move, board_state, probs)
    return move, board_state, probs


def switch_player(game_data):
    game_data["player"] = -game_data["player"]


def build_trajectory(game_data, game_outcome, device):
    game_data["rewards"][-1] = game_outcome
    trajectory = {"rewards": torch.tensor(game_data["rewards"]).to(device),
                  "moves": torch.tensor(game_data["moves"]).to(device),
                  "board_states": torch.stack(game_data["board_states"]).to(device),
                  "move_probs": torch.stack(game_data["policy_probs"]).to(device)}
    return trajectory


def update_game_data(game_data, move, board_state, probs):
    game_data["moves"].append(move)
    game_data["rewards"].append(0)
    game_data["board_states"].append(board_state)
    game_data["policy_probs"].append(probs)
    game_data["turn"] += 1
    return game_data


def get_player_network(policy, adversary, player):
    players = {WHITE: policy, BLACK: adversary}
    return players[player]


def compute_move(network, board):
    board_state = board.tensor.clone()
    assert_board_is_self_consistent(board)
    batch = board_state.unsqueeze(0)
    probs = network.forward(batch).detach() #for serialization (mp)
    move = network.get_move_as_int_from_prob_dist(probs, board_state)
    return move, board_state, probs


def update_consecutive_passes(move, game_data):
    if move == PASS:
        game_data["consecutive_passes"] += 1
    else:
        game_data["consecutive_passes"] = 0


