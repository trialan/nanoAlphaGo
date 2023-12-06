from nanoAlphaGo.rl.policy import PolicyNN
import wandb
from nanoAlphaGo.config import WHITE
from nanoAlphaGo.rl.trajectories import st_collect_trajectories
import pstats
import cProfile

policy = PolicyNN(WHITE)
cProfile.run("st_collect_trajectories(policy, 50)", "results.prof"); p = pstats.Stats("results.prof"); p.sort_stats("cumulative").print_stats(25)
