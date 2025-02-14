import numpy as np
import os
from pathlib import Path
from snake import SnakeEnv
import json
import time
from datetime import datetime

class DataCollector:
    def __init__(self, save_dir="collected_data", frame_skip=2):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.frame_skip = frame_skip
        
        # Initialize environment
        self.env = SnakeEnv(grid_size=20, pixel_size=20)
        
        # Storage for current episode
        self.reset_episode_storage()
        
        # Metadata for collection
        self.episode_count = 0
        self.total_frames = 0
        self.total_score = 0
        self.start_time = time.time()
        
    def reset_episode_storage(self):
        """Reset storage for a new episode"""
        self.current_frames = []
        self.current_actions = []
        self.current_rewards = []
        self.current_dones = []
        
    def collect_episode(self, max_steps=1000):
        """Collect a single episode using simple action policy"""
        obs = self.env.reset()
        done = False
        step_count = 0
        frame_count = 0
        
        while not done and step_count < max_steps:
            step_count += 1
            
            # Simple policy: 80% continue same direction, 20% random turn
            if np.random.random() < 0.2:
                # Get valid turns (can't turn 180 degrees)
                current_dir = self.env.direction
                valid_actions = [a for a in range(4) if abs(a - current_dir) != 2]
                action = np.random.choice(valid_actions)
            else:
                action = self.env.direction
            
            # Store frame (with frame skip)
            if step_count % self.frame_skip == 0:
                frame = self.env.render(mode='rgb_array')
                self.current_frames.append(frame)
                self.current_actions.append(action)
                frame_count += 1
            
            # Step environment
            obs, reward, done, info = self.env.step(action)
            self.current_rewards.append(reward)
            self.current_dones.append(done)
        
        return frame_count, info['score']
    
    def save_episode(self, frame_count, score):
        """Save the current episode to disk"""
        episode_dir = self.save_dir / f"episode_{self.episode_count:06d}"
        episode_dir.mkdir(exist_ok=True)
        
        # Save frames as numpy array
        frames = np.array(self.current_frames)
        np.save(episode_dir / "frames.npy", frames)
        
        # Convert numpy types to native Python types for JSON serialization
        episode_data = {
            "actions": [int(a) for a in self.current_actions],
            "rewards": [float(r) for r in self.current_rewards],
            "dones": [bool(d) for d in self.current_dones],
            "episode_length": int(frame_count),
            "total_reward": float(sum(self.current_rewards)),
            "score": int(score),
            "frame_skip": int(self.frame_skip)
        }
        
        with open(episode_dir / "episode_data.json", "w") as f:
            json.dump(episode_data, f)
            
        # Update counters
        self.episode_count += 1
        self.total_frames += frame_count
        self.total_score += score
        
        # Log progress
        elapsed = time.time() - self.start_time
        avg_score = self.total_score / self.episode_count
        print(f"Episode {self.episode_count}: frames={frame_count}, score={score}, "
              f"avg_score={avg_score:.1f}, total_frames={self.total_frames}, "
              f"time={elapsed:.1f}s")
        
    def collect_data(self, num_episodes=1000):
        """Main collection loop"""
        try:
            for _ in range(num_episodes):
                self.reset_episode_storage()
                frame_count, score = self.collect_episode()
                self.save_episode(frame_count, score)
                
        except KeyboardInterrupt:
            print("\nCollection interrupted by user")
        finally:
            self.env.close()
            
        # Save collection metadata
        metadata = {
            "total_episodes": self.episode_count,
            "total_frames": self.total_frames,
            "average_score": self.total_score / self.episode_count,
            "collection_time": time.time() - self.start_time,
            "date": datetime.now().isoformat(),
            "frame_skip": self.frame_skip,
            "env_config": {
                "grid_size": self.env.GRID_SIZE,
                "pixel_size": self.env.PIXEL_SIZE
            }
        }
        
        with open(self.save_dir / "collection_metadata.json", "w") as f:
            json.dump(metadata, f)

if __name__ == "__main__":
    # Start with a small test run
    collector = DataCollector(frame_skip=2)  # Only save every 2nd frame
    collector.collect_data(num_episodes=5)
