from functools import partial

from .multiagentenv import MultiAgentEnv
from .stag_hunt import StagHunt
from smac.env import MultiAgentEnv, StarCraft2Env
from .matrix_game.matrix_game_simple import Matrixgame
from .ctf.ctf import CTF, CTF_v2

# TODO: Do we need this?
def env_fn(env, **kwargs) -> MultiAgentEnv: # TODO: this may be a more complex function
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)


REGISTRY = {}
REGISTRY["matrix_game"] = partial(env_fn, env=Matrixgame)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["ctf"] = partial(env_fn, env=CTF)
REGISTRY["ctf2"] = partial(env_fn, env=CTF_v2)


import multiagent
from .MAPredatorPreyWrappers import RandomPreyActions,PredatorPreyTerminator
import gym

def env_fn2( **kwargs):
    print(kwargs)
    env = gym.make(kwargs.get('env_args').get("name"))
    env = RandomPreyActions(env)
    env = PredatorPreyTerminator(env)
    return env

REGISTRY["predator_prey"] = partial(env_fn2)
