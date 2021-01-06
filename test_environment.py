from smac.env import MultiAgentEnv, StarCraft2Env
import torch as th
e=StarCraft2Env()
o = e.reset()
print(o)
e.close()
