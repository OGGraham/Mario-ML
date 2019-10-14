import numpy as np
import random
import os
from contextlib import redirect_stdout

from cv2 import imwrite, waitKey, destroyWindow, imshow

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.utils import plot_model

import tensorflow as tf

from skimage import color

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import matplotlib.pyplot as plt

from prioritized_replay import Memory

import time

# Set True when using Colab, False when not
colab = False

# Set True when want to load from prev
load = True

# Colab or Nah
if not colab:
    import visdom

# Agent Settings:
ddqn = True
prioritized = False

t_avg = []

class Agent(object):
    """
    Agent Class - contains all necessary function for building, training & querying Neural Network.
    """
    # Constructor
    def __init__(self, state_size, action_size, learning_rate):
        # Initialize Hyperparams & Structure
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.batch_size = 32
        self.memory_size = 10000
        self.epsilon = 1
        self.epsilon_min = 0.1  # At min, 1 in 10 moves are random # Originally 0.05
        self.epsilon_decay = 0.98  # Originally 0.95
        self.gamma = 0.95  # Discount Factor
        self.model = self.build()
        # --- Prioritized ---
        if prioritized:
            self.memory = Memory(self.memory_size)
        # --- Non Prioritized ----
        else:
            self.memory = []
        # --- Double DQN ---
        if ddqn:
            self.target_model = self.build()
            self.update_target_frequency = 10000  # Number of Epochs which target_model is updated to
            self.t = 0

    # Huber Loss - For Reinforcement Learning
    # clip_delta defines boundary where the loss function goes from quadratic -> linear
    def huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        # Condition - decides which loss to use
        cond = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    # Build NN Architecture
    def build(self):
        # # Model Type
        model = Sequential()

        # # Add Layers
        # # Final NN
        model.add(Conv2D(32, kernel_size=8, strides=4, padding="same", input_shape=self.state_size))
        model.add(ReLU())
        model.add(Conv2D(64, kernel_size=4, strides=2, padding="same"))
        model.add(ReLU())
        model.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
        model.add(ReLU())
        # Flatten to Pass to Non-Convolution Layers
        model.add(Flatten())
        # Dense Layer
        model.add(Dense(512))
        model.add(LeakyReLU())
        model.add(Dense(self.action_size))


        # Compile Model
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, epsilon=K.epsilon()/100))
        model.compile(loss=self.huber_loss, optimizer=Adam(lr=self.learning_rate))

        # Prints Network Outputs
        print("Network Structure:")
        for layer in model.layers:
            print(layer.output_shape)

        # Save network model - uncomment to generate model graph
        # plot_model(model, to_file="model.png")
        # print("Drawn Model.")

        # Return model
        return model

    # Load Other Model
    def load(self, name):
        self.model.load_weights(name)

    # Save Current Model
    def save(self, path):
        self.model.save_weights(path)

    # DDQN - Updates target network weights
    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    # Remember previous iterations so can learn from them
    def remember(self, state, action, reward, next_state, done):
        if not prioritized:
            # --- NON PRIORITIZED REPLAY ---
            self.memory.append([state, action, reward, next_state, done])
        else:
            # --- PRIORITIZED REPLAY ----
            # Get value before update
            q_vals = self.model.predict(state)
            old_val = q_vals[0][action]

            # Get new value
            if not done:
                # q_update = reward + gamma * np.amax(self.model.predict(next_state)[0])  # Standard Q
                q_update = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])  # Double Q
            # Otherwise, just reward
            else:
                q_update = reward

            # Calculate error
            error = abs(old_val - q_update)

            # Add to memory
            self.memory.add(error, (state, action, reward, next_state, done))

        if ddqn:
            # Update Target Model every target_freq steps
            if self.t >= self.update_target_frequency:
                print("Updating Target Model..")
                self.t = 0
                self.update_target_model()
                print("Current Q Vals (for prev state):", self.model.predict(state)[0])

    # Choose action based on current models prediction
    def act(self, state):
        # Randomly choose an action occasionally
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), True
        # Otherwise predict an action
        else:
            values = self.model.predict(state)[0]
            # Take argmax so returning singular action
            return np.argmax(values), False

    # Reset Epsilon -> Prevents 'burning in' of non-optimal behaviour
    # Useful article - https://www.alexirpan.com/2018/02/14/rl-hard.html
    def reset_epsilon(self):
        self.epsilon = 1

    # Choose action based on current models prediction - for testing (no epsilon)
    def test_act(self, state):
        # Take argmax so returning singular action
        action = self.model.predict(state)[0]
        return np.argmax(action)

    # Using info in memory as training data -> L E A R N
    def learn(self):
        history = None  # Stores epoch info

        # --- NON PRIORITIZED REPLAY ---------------------------------------------------------------------------------
        if not prioritized:
            # Only run if enough samples in memory
            if len(self.memory) < self.batch_size:
                return
            # Returns memory in a random sequence
            # Add on the death memory of each time died
            batch = random.sample(self.memory, self.batch_size)

            # Loop through batch
            # start = time.time()

            for state, action, reward, next_state, done in batch:
                # If still alive -> reward + predicted future reward
                if not done:
                    if not ddqn:
                        q_update = reward + self.gamma * np.amax(self.model.predict(next_state)[0])  # Standard Q
                    else:
                        q_update = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])  # Double Q
                # Otherwise, just reward
                else:
                    q_update = reward

                # Q value tings
                q_vals = self.model.predict(state)
                q_vals[0][action] = q_update

                # Actually Learn
                history = self.model.fit(state, q_vals, epochs=1, verbose=0)


                if ddqn:
                    # Increment t (double q learn update param)
                    self.t += 1

            # t_avg.append(time.time() - start)
            # print("Current t avg:", np.mean(t_avg))
            # To prevent running out of memory
            while len(self.memory) > self.memory_size:
                self.memory.remove(self.memory[0])

        # --- PRIORITIZED REPLAY ---------------------------------------------------------------------------------------
        else:
            batch, idxs, is_weights = self.memory.sample(self.batch_size)

            # Loop through batch
            for state, action, reward, next_state, done in batch:
                # If still alive -> reward + predicted future reward
                if not done:
                    if not ddqn:
                        q_update = reward + self.gamma * np.amax(self.model.predict(next_state)[0])  # Standard Q
                    else:
                        q_update = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])  # Double Q
                # Otherwise, just reward
                else:
                    q_update = reward

                # Q value tings
                q_vals = self.model.predict(state)
                # Get old val
                old_val = q_vals[0][action]
                # Update with new value
                q_vals[0][action] = q_update

                # Actually Learn
                history = self.model.fit(state, q_vals, epochs=1, verbose=0)

                # Update priorities in memory
                idx = idxs[self.t % self.batch_size]
                self.memory.update(idx, abs(old_val - q_update))

                if ddqn:
                    # Increment t (double q learn update param)
                    self.t += 1

        # --------------------------------------------------------------------------------------------------------------

        # Update epsilon (exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history


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

        # img = env2.render('rgb_array')
        # img = crop(img, [(y1, y2), (x1, x2)])
        # imwrite("prev_frame.png", img)
        # imshow("crop", img)
        # waitKey(0)
        # destroyWindow("crop")

        # Given the state choose an action
        action = bot.test_act(state)

        # Carry out action
        next_state, reward, done, info = env2.step(action)

        if info['x_pos'] >= max_x:
            max_x = info['x_pos']

        x1 = find(info['x_pos'], max_x)
        x2 = x1 + x_size

        # y1 = min(123, info['y_pos'])
        # y2 = min(209, y_size + info['y_pos'])

        # If reached end of level, break
        if info['flag_get']:
            break

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


# Main Loop
if __name__ == "__main__":
    # NB: Environment Step function returns 4 values: observation (object), reward (float), done (bool), info (dict)
    # Obs -> Environment specific object e.g. pixel data from camera, joint angles, board game state etc.
    # Reward -> Amount of reward from previous action
    # Done -> Whether in end of episode state
    # Info -> Diagnostic info useful for debugging/learning improvement e.g. raw probabilities behind last environment

    # ------------------------------------------------------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------------------------------------------------------

    # Model Name (for saving the current model)
    name = "otherworldmodel"

    # Set path to save (either /drive or /models depending on method)
    if colab:
        path = "drive/My Drive/Project/"
    else:
        path = "./models/"


    # Training Params
    epochs = 10000  # basically inf
    # Frame average before reset
    frames = 100

    # Number Of Frames/Actions taken
    i = 0

    # Starting epoch (0 by default if nothing loaded)
    start_epoch = 0

    # Stores best test score currently (save model as current best if new val >)
    top_score = -1
    train_mean_score = [0, 0]  # First Val is culumative max x, Second Val is number of iterations -> mean = [0]/[1]
    test_mean_score = [0, 0]

    # Make Mario Bros Levels (Train & Test Environments)
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v1')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    env2 = gym_super_mario_bros.make('SuperMarioBros-1-2-v1')
    env2 = BinarySpaceToDiscreteSpaceEnv(env2, SIMPLE_MOVEMENT)

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

    # Setup visdom
    if not colab:
        vis = visdom.Visdom()

    # Create Agent
    bot = Agent(state_size, action_size, 0.0001)  # initial good ddqn used 0.0001

    # Save network structure to txt
    with open(path + name + '-network_structure.txt', 'w') as f:
        with redirect_stdout(f):
            bot.model.summary()

    # ------------------------------------------------------------------------------------------------------------------
    # LOAD INITIAL
    # ------------------------------------------------------------------------------------------------------------------
    if load:
        # Load best model
        try:
            bot.load(path + name + "-current-best-ddqn.h5")
            print("Best Model Loaded..")
        except Exception as e:
            print("Failure to load.")
            print(e)
            quit(-1)

        max_x = test(env2, bot, y1, y2, x1, x2)

        print("Loaded Best Max X:", max_x)
        top_score = max_x

        # Load current model
        try:
            bot.load(path + name + "-ddqn.h5")
            print("Loaded most recent model successfully - Continuing..")
        except Exception:
            print("Failure to load.")
            quit(-1)

    # ------------------------------------------------------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------------------------------------------------------

    # Loop for # of Epochs
    for e in range(start_epoch, epochs):
        # Print epoch
        print("Iteration number - ", e)

        # Reset Environment & Game Over Flag
        state = env.reset()
        # Change State size from 240 x 256 x 3 -> (y2-y1) x (x2-x1) x 1 (height, width, channels)
        state = color.rgb2gray(state)
        # Crop Image to dimensions
        state = crop(state, [(y1, y2), (x1, x2)])

        # imshow("crop", state)
        # waitKey(0)

        # Reshape initial state so can be fed into NN
        state = np.reshape(state, [1, state_size[0], state_size[1], 1])

        # Reset Epsilon
        bot.reset_epsilon()

        # Display visually
        # env.render('human')
        # env.render()

        done = False
        # Keeps track of the reward over the last 100 frames
        reward_avg = []
        cul_reward = 0
        # Keeps track of avg loss
        loss_avg = []
        # Keep track of centre of screen
        max_x = 128
        print("Starting Iteration:")

        n = 32

        while not done:
            # Given the state choose an action
            action, r_bool = bot.act(state)

            # Increment i
            i += 1

            # Carry out action
            next_state, reward, done, info = env.step(action)

            # (as this gives us Mario's x position)
            if info['x_pos'] >= max_x:
                max_x = info['x_pos']

            x1 = find(info['x_pos'], max_x)
            x2 = x1 + x_size

            if r_bool:
                print("Action", i, "(Random):", moveList[action], "| Reward Associated:", reward, "| Current X Pos:", info['x_pos'])
            else:
                print("Action", str(i) + ":", moveList[action], "| Reward Associated:", reward, "| Current X Pos:", info['x_pos'])

            # Update Histogram vals
            if reward != -15:  # exclude penalty from dying
                moveHist[action][1] += 1
                moveHist[action][0] += reward  # calcs avg reward for that action

            # Convert new state to greyscale
            next_state = color.rgb2gray(next_state)
            # Crop new state
            next_state = crop(next_state, [(y1, y2), (x1, x2)])
            # Reshape new state so can be fed into NN
            next_state = np.reshape(next_state, [1, state_size[0], state_size[1], 1])

            # Add observation to bot's memory
            bot.remember(state, action, reward, next_state, done)

            # Update NN (will only do anything when has more samples than the batch size)
            # --- Non Prioritized ---
            if not prioritized:
                history = bot.learn()
            # --- Prioritized ---
            else:
                if bot.memory.tree.n_entries >= bot.batch_size:
                    history = bot.learn()
                else:
                    history = None

            # Old state = new state
            state = next_state

            # For graphically representing loss on graphs
            # With loss, we can see that initially loss -> V High, decreases over time; as new areas are encountered,
            # loss will spike as encountering unknown states -> then tends back towards 0 with more time
            if history is not None:
                loss_avg.append(sum(history.history['loss'])/len(history.history['loss']))

            # Average Reward
            cul_reward += reward
            reward_avg.append(reward)
            if len(reward_avg) > frames:
                reward_avg.remove(reward_avg[0])
            if len(reward_avg) == frames:
                t = sum(reward_avg)/frames
                # Exit condition to prevent algorithm getting stuck and never progressing
                if t < 0:
                    break

        # --------------------------------------------------------------------------------------------------------------
        # TRAINING (continued - end of iteration)
        # --------------------------------------------------------------------------------------------------------------
        if done:
            print("Agent Died.")
        else:
            print("Agent Got Stuck.")

        # Update train mean score
        train_mean_score[0] += max_x
        train_mean_score[1] += 1

        print("Iteration Results:")
        print("Max Distance:", max_x)
        print("Current Iteration Max X Average:", train_mean_score[0]/train_mean_score[1])
        print("Avg Loss:", sum(loss_avg) / len(loss_avg))

        # If not using colab, update realtime graphs
        if not colab:
            # Output training results to visdom graph
            vis.scatter(X=[[e, max_x]], win="Train Graph", update="append",
                        opts=dict(title="Train Graph",
                                  name="Train Graph",
                                  xlabel="Iteration",
                                  ylabel="Max X"))

            vis.scatter(X=[[e, sum(loss_avg)/len(loss_avg)]], win="Loss Graph", update="append",
                        opts=dict(title="Loss Graph",
                                  name="Loss Graph",
                                  xlabel="Iteration",
                                  ylabel="Avg Loss"))

            vis.scatter(X=[[e, train_mean_score[0]/train_mean_score[1]]], win="Train Mean Score Graph", update="append",
                        opts=dict(title="Train Mean Score Graph",
                                  name="Train Mean Score Graph",
                                  xlabel="Iteration",
                                  ylabel="Mean Max X"))

            vis.bar(X=[x[0]/x[1] for x in moveHist], win="Avg Move Reward",
                    opts=dict(title="Avg Move Reward",
                              name="Avg Move Reward",
                              xlabel="",
                              ylabel="Avg Reward",
                              legend=moveList,
                              )

                    )

            vis.bar(X=[x[1] for x in moveHist], win="Moves Chosen",
                    opts=dict(title="Moves Chosen",
                              name="Moves Chosen",
                              xlabel="",
                              ylabel="Number Of Times Chosen",
                              legend=moveList,
                              )

                    )

        # Save all data to text docs
        with open(path + name + '-train_max_x.txt', 'a+') as file:
            file.write(str(max_x) + ",")

        with open(path + name + '-train_avg_max_x.txt', 'a+') as file:
            file.write(str(train_mean_score[0]/train_mean_score[1]) + ",")

        with open(path + name + '-train_loss.txt', 'a+') as file:
            file.write(str(sum(loss_avg)/len(loss_avg)) + ",")

        with open(path + name + '-move_reward.txt', 'w') as file:
            file.write(str([x[0] for x in moveHist]))

        with open(path + name + '-move_count.txt', 'w') as file:
            file.write(str([x[1] for x in moveHist]))

        # --------------------------------------------------------------------------------------------------------------
        # TEST (after end of iteration)
        # --------------------------------------------------------------------------------------------------------------

        print("End Of Iteration:")

        max_x = test(env2, bot, y1, y2, x1, x2)

        # Update Test Mean Score
        test_mean_score[0] += max_x
        test_mean_score[1] += 1

        print("Testing Avg Distance:", test_mean_score[0]/test_mean_score[1])

        if not colab:
            # Output test result to visdom graph
            vis.scatter(X=[[e, max_x]], win="Test Graph", update="append",
                        opts=dict(title="Test Graph",
                                  name="Test Graph",
                                  xlabel="Iteration",
                                  ylabel="Max X"))

            vis.scatter(X=[[e, test_mean_score[0]/test_mean_score[1]]], win="Test Avg Dist Graph", update="append",
                        opts=dict(title="Test Avg Dist Graph",
                                  name="Test Avg Dist Graph",
                                  xlabel="Iteration",
                                  ylabel="Avg Max X"))

        # Output results into .txt
        with open(path + name + '-test_max_x.txt', 'a+') as file:
            file.write(str(max_x) + ",")

        with open(path + name + '-test_avg_max_x.txt', 'a+') as file:
            file.write(str(test_mean_score[0]/test_mean_score[1]) + ",")

        # --------------------------------------------------------------------------------------------------------------
        # Saving Models/Other Stuff
        # --------------------------------------------------------------------------------------------------------------

        if max_x > top_score:
            print("Current Model Produced Best Score! Saving Current Best!")
            top_score = max_x
            bot.save(path + name + "-current-best-ddqn.h5")

        # Save model every 10 iterations
        if e % 10 == 0:
            print("Saving Model.")
            bot.save(path + name + "-ddqn.h5")
