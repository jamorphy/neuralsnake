import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from snake import SnakeEnv
import os
from pathlib import Path
import json
from datetime import datetime

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        # Ensure n_actions matches all possible actions including NO_ACTION
        assert n_actions == 5, "Network must support 5 actions (UP, RIGHT, DOWN, LEFT, NO_ACTION)"
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.cat(state), 
                torch.tensor(action),
                torch.tensor(reward),
                torch.cat(next_state),
                torch.tensor(done))
    
    def __len__(self):
        return len(self.buffer)

class SnakeAgent:
    def __init__(self, env, save_dir="training_data", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        print(f"\nInitializing SnakeAgent...")
        print(f"Save directory: {self.save_dir}")
        print(f"Using device: {device}")
        print(f"Grid size: {env.GRID_SIZE}x{env.GRID_SIZE}")
        self.env = env
        self.device = device
        
        # Network initialization
        channels = env.observation_space.shape[2]  # Number of input channels
        self.state_shape = (channels, env.GRID_SIZE, env.GRID_SIZE)
        self.n_actions = env.action_space.n
        
        self.policy_net = DQN(self.state_shape, self.n_actions).to(device)
        self.target_net = DQN(self.state_shape, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.batch_size = 128
        self.gamma = 0.99
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.target_update = 10
        self.learning_rate = 1e-4
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(10000)
        
        self.eps = self.eps_start
        self.steps_done = 0
    
    def select_action(self, state, training=True):
        if training and random.random() < self.eps:
            # Random action with NO_ACTION included
            current_dir = self.env.direction
            # Include NO_ACTION (4) in valid actions
            valid_actions = [a for a in range(4) if abs(a - current_dir) != 2]
            valid_actions.append(self.env.NO_ACTION)  # Add NO_ACTION as a possible choice
            return random.choice(valid_actions)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            
            # Filter invalid actions (180-degree turns)
            current_dir = self.env.direction
            valid_actions = [a for a in range(4) if abs(a - current_dir) != 2]
            valid_actions.append(self.env.NO_ACTION)  # Add NO_ACTION
            
            # Get Q-values only for valid actions
            valid_q_values = q_values[0][valid_actions]
            
            return valid_actions[valid_q_values.argmax().item()]
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            if len(self.memory) % 100 == 0:  # Log every 100 experiences
                print(f"Collecting experiences: {len(self.memory)}/{self.batch_size}")
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.float().to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.float().to(self.device)  # Convert to float tensor
        
        # Compute Q(s_t, a)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()

    def save_episode_data(self, episode_num, frames, actions, rewards, q_values):
        """Save episode data for diffusion model training"""
        episode_dir = self.save_dir / f"episode_{episode_num:06d}"
        episode_dir.mkdir(exist_ok=True)
        
        # Save frames as numpy array
        frames = np.array(frames)
        np.save(episode_dir / "frames.npy", frames)
        
        # Save episode metadata
        episode_data = {
            "actions": [int(a) for a in actions],
            "rewards": [float(r) for r in rewards],
            "q_values": [[float(q) for q in qv] for qv in q_values],
            "total_reward": float(sum(rewards)),
            "episode_length": len(frames),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(episode_dir / "episode_data.json", "w") as f:
            json.dump(episode_data, f)
            
        print(f"\nSaved episode {episode_num:06d}:")
        print(f"  Frames shape: {frames.shape}")
        print(f"  Total reward: {episode_data['total_reward']:.2f}")
        print(f"  Episode length: {episode_data['episode_length']}")
        print(f"  Directory: {episode_dir}")

    def train(self, num_episodes, max_steps=10000, save_interval=10):
        print("\nStarting training...")
        print(f"Number of episodes: {num_episodes}")
        print(f"Max steps per episode: {max_steps}")
        print(f"Save interval: {save_interval} episodes")
        print(f"Batch size: {self.batch_size}")
        print(f"Initial epsilon: {self.eps}")
        print(f"Minimum replay buffer size before training: {self.batch_size}")
        
        scores = []
        losses = []
        start_time = time.time()
        best_score = float('-inf')
        
        for episode in range(num_episodes):
            if episode == 0:
                print(f"\nStarting episode {episode}...")
            
            state = self.env.reset()
            state = np.transpose(state, (2, 0, 1))  # Channel-first format for PyTorch
            episode_reward = 0
            episode_loss = 0
            
            # Data collection for diffusion model
            episode_frames = []
            episode_actions = []
            episode_rewards = []
            episode_q_values = []
            
            # Collect initial frame
            episode_frames.append(self.env.render(mode='rgb_array'))
            
            if episode == 0:
                print("Initial frame collected")
            
            for step in range(max_steps):
                if episode == 0 and step % 100 == 0:
                    print(f"Step {step}, Memory buffer size: {len(self.memory)}/{self.memory.buffer.maxlen}")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            state = np.transpose(state, (2, 0, 1))  # Channel-first format for PyTorch
            episode_reward = 0
            episode_loss = 0
            
            # Data collection for diffusion model
            episode_frames = []
            episode_actions = []
            episode_rewards = []
            episode_q_values = []
            
            # Collect initial frame
            episode_frames.append(self.env.render(mode='rgb_array'))
            
            for step in range(max_steps):
                # Get action and Q-values
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(state_tensor).cpu().numpy()[0]
                
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.transpose(next_state, (2, 0, 1))
                episode_reward += reward
                
                # Collect data
                episode_frames.append(self.env.render(mode='rgb_array'))
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_q_values.append(q_values)
                
                # Store transition in memory
                self.memory.push(
                    torch.FloatTensor(state).unsqueeze(0),
                    action,
                    reward,
                    torch.FloatTensor(next_state).unsqueeze(0),
                    done
                )
                
                state = next_state
                
                # Perform optimization step
                loss = self.optimize_model()
                if loss is not None:
                    episode_loss += loss
                
                if done:
                    break
            
            # Update target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.eps = max(self.eps_end, self.eps * self.eps_decay)
            
            scores.append(episode_reward)
            losses.append(episode_loss / (step + 1))
            
            # Print progress
            # Save episode data periodically
            # Update best score
            if episode_reward > best_score:
                best_score = episode_reward
                print(f"\nNew best score: {best_score:.2f} (Episode {episode})")

            # Save episode data periodically
            if episode % save_interval == 0:
                self.save_episode_data(episode, episode_frames, episode_actions, 
                                     episode_rewards, episode_q_values)
            
            # Print progress
            if episode % 10 == 0:
                avg_score = np.mean(scores[-10:])
                avg_loss = np.mean(losses[-10:])
                elapsed_time = time.time() - start_time
                eps_per_sec = (episode + 1) / elapsed_time
                
                print(f"\nEpisode {episode}/{num_episodes} ({elapsed_time:.1f}s, {eps_per_sec:.2f} eps/s)")
                print(f"  Score: {episode_reward:.2f} (Avg {avg_score:.2f}, Best {best_score:.2f})")
                print(f"  Loss: {episode_loss:.4f} (Avg {avg_loss:.4f})")
                print(f"  Epsilon: {self.eps:.3f}")
                print(f"  Memory: {len(self.memory)}/{self.memory.buffer.maxlen}")
                print(f"  Steps this episode: {step + 1}")
                
                # Save checkpoint
                if episode % 100 == 0:
                    checkpoint_path = self.save_dir / f"checkpoint_episode_{episode}.pth"
                    self.save(checkpoint_path)
                    print(f"  Saved checkpoint: {checkpoint_path}")
        
        return scores, losses
    
    def save(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'eps': self.eps,
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.eps = checkpoint['eps']

if __name__ == "__main__":
    import time
    
    # Training example
    print("\nStarting Snake RL Training")
    print("=" * 50)
    
    # Initialize environment and agent
    env = SnakeEnv(grid_size=20)
    agent = SnakeAgent(env)
    
    # Train the agent
    try:
        start_time = time.time()
        scores, losses = agent.train(num_episodes=1000)
        total_time = time.time() - start_time
        
        print("\nTraining Complete!")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Episodes per second: {1000/total_time:.2f}")
        print(f"Final average score (last 100): {np.mean(scores[-100:]):.2f}")
        print(f"Best score: {max(scores):.2f}")
        
        # Save the trained agent
        final_path = "snake_agent_final.pth"
        agent.save(final_path)
        print(f"\nSaved final model to: {final_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving checkpoint...")
        agent.save("snake_agent_interrupted.pth")
        print("Checkpoint saved to: snake_agent_interrupted.pth")
