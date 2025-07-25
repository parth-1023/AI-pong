Behavioral Cloning for Atari Pong with PyTorch

In this game, each player can execute one of 6 possible actions at each timestep (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, and LEFTFIRE).

The goal is to execute the best action at each timestep based on the current state of the game.

This project implements a behavioral cloning agent for the Atari game Pong. We train a convolutional neural network (CNN) to imitate an expert player by mapping raw game frames to expert actions. After training, the model is evaluated by “playing” Pong against the built-in AI opponent. Full credit requires the agent to reach 21 points first.
