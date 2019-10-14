import random

from keras.layers import *

from skimage import color

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

class Random_Agent(object):
    """
    Agent Class - contains all necessary function for building, training & querying Neural Network.
    """

    # Constructor
    def __init__(self, state_size, action_size, learning_rate):
        # Initialize Hyperparams & Structure
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

    # Choose action based on current models prediction
    def rand_act(self, _):
        # Randomly choose an action occasionally
        return random.randrange(self.action_size)



# Mario screen is 256 (width) x 240 (height)
# img = crop(img, [(125, 209), (84, 170)]) -> [[y1, y2], [x1, x2]] DEFAULT CROP
def crop(img, dimensions):
    return img[dimensions[0][0]:dimensions[0][1], dimensions[1][0]:dimensions[1][1]]

# Find Mario's x location on screen
def find(current_x, max_x):
    # Ensure max_x updated before calling this function
    min_x = max_x - 128
    # Returns either 0 or the pixel value slightly behind where mario is located
    return max(0, int(((current_x-min_x)/(max_x-min_x))*128)-15)  # cap at 0

# Test Current Model
def test(env2, bot, y1, y2, x1, x2):
    print("Testing..")
    # Reset Test Environment & Game Over Flag
    state = env2.reset()
    # Change State size from 240 x 256 x 3 -> (y2-y1) x (x2-x1) x 1 (height, width, channels)
    state = color.rgb2gray(state)
    # Crop Image to dimensions
    state = crop(state, [(y1, y2), (x1, x2)])

    # Reshape initial state so can be fed into NN
    state = np.reshape(state, [1, state_size[0], state_size[1], 1])

    done = False
    # Keeps track of the reward over the last 100 frames
    reward_avg = []
    cul_reward = 0
    max_x = 128

    while not done:

        action = bot.rand_act(state)

        # Carry out action
        next_state, reward, done, info = env2.step(action)

        if info['x_pos'] >= max_x:
            max_x = info['x_pos']

        x1 = find(info['x_pos'], max_x)
        x2 = x1 + x_size

        # Convert new state to greyscale
        next_state = color.rgb2gray(next_state)
        # Crop new state
        next_state = crop(next_state, [(y1, y2), (x1, x2)])

        # Reshape new state so can be fed into NN
        next_state = np.reshape(next_state, [1, state_size[0], state_size[1], 1])
        # Old state = new state
        state = next_state

        # Average Reward
        cul_reward += reward
        reward_avg.append(reward)
        if len(reward_avg) > frames:
            reward_avg.remove(reward_avg[0])
        if len(reward_avg) == frames:
            t = sum(reward_avg) / frames
            # Exit condition to prevent algorithm getting stuck and never progressing
            if t < 0:
                break

    print("Testing Max X:", max_x)
    print("End Of Test.")
    return max_x


# Frame average before reset
frames = 100

# Main Loop
if __name__ == "__main__":
    # NB: Environment Step function returns 4 values: observation (object), reward (float), done (bool), info (dict)
    # Obs -> Environment specific object e.g. pixel data from camera, joint angles, board game state etc.
    # Reward -> Amount of reward from previous action
    # Done -> Whether in end of episode state
    # Info -> Diagnostic info useful for debugging/learning improvement e.g. raw probabilities behind last environment
    # state change: Note - agent cant use this info for learning


    # Frame average before reset
    frames = 100

    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v1')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

    moveList = ["NOP", "Right", "Right + A", "Right + B", "Right + A + B", "A", "Left"]
    moveHist = [[0, 0] for x in moveList]  # First item is current avg, second is counter -> avg = mH[0]/mH[1]

    # Environment Crop Parameters
    y1, y2 = 123, 209
    x1, x2 = 36, 122
    x_size = x2 - x1
    y_size = y2 - y1

    # State & Action Size (for network input/output)
    state_size = (y_size, x_size, 1)
    action_size = env.action_space.n

    # Create Agent -> input build as param for different architecture style
    bot = Random_Agent(state_size, action_size, 0.0005)

    max_score = 0
    vals = []
    for i in range(800):
        print(i)
        x = test(env, bot, y1, y2, x1, x2)
        vals.append(x)

    print("800 Iterations:")
    print("Mean:", np.mean(vals))
    print("Median:", np.median(vals))
    print("Max:", max(vals))
