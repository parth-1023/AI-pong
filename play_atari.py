import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


class CNNPolicyNet(nn.Module):
    def __init__(self):
        super(CNNPolicyNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 16, 4)  # Input channels = 4 (stacked frames), output channels = 16
        self.bn1 = nn.BatchNorm2d(16)  # BatchNorm for conv1
        self.pool = nn.MaxPool2d(2, 2)  # MaxPooling with 2x2 kernel
        
        self.conv2 = nn.Conv2d(16, 32, 5)  # Second conv layer
        self.bn2 = nn.BatchNorm2d(32)  # BatchNorm for conv2
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 18 * 18, 120)  # Flattened input size = 32 * 18 * 18
        self.bn3 = nn.BatchNorm1d(120)  # BatchNorm for fc1
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)  # BatchNorm for fc2
        self.fc3 = nn.Linear(84, 10)  # Output layer (10 possible actions in Pong)

    def forward(self, x):
        # Convolutional layers with ReLU, pooling, and BatchNorm
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)
        
        # Fully connected layers with ReLU and BatchNorm
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)  # No activation for the output layer
        return x

    def predict(self, x):
        # Prediction method for inference
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            policy_logits = self.forward(x)
            action_probs = F.softmax(policy_logits, dim=-1)  # Convert logits to probabilities
        return action_probs


def render_env(env, policy, max_steps=500):
    obs = env.reset()
    for _ in range(max_steps):
        obs = torch.tensor(obs).permute(0, 3, 1, 2).float().to(device)
        with torch.no_grad():
            actionProbs = policy.predict(obs)
        action = torch.multinomial(actionProbs, 1).item()  # randomly pick an action according to policy
        obs, _, done, _ = env.step([action])
        env.render()
        if done:
            break  # game over
    env.close()



# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the Atari environment
env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=2, env_kwargs={'render_mode': "human"})
env = VecFrameStack(env, n_stack=4)  # Stack 4 frames for better context

# Initialize the model
model = CNNPolicyNet().to(device)

# Load the pre-trained model
model.load_state_dict(torch.load('pong_model.pth', weights_only=True))

# Render the environment with the model
render_env(env, model, max_steps=500)
