## Mario-ML

Contains (some of) the work for my reinforcement learning project, teaching an agent to play the original Super Mario Bros. game purely
from the game's state using a CNN.

The majority of the code is within Bot.py. This is where the agent's training and testing routines are. The prioritized_replay.py and SumTree.py
contain the logic for the prioritized replay memory of the agent.


random_agent.py is simply a random agent for playing Mario, it selects random actions. This was used as a comparison when getting results.
