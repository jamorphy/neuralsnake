import pygame
import numpy as np
import gym
from gym import spaces

class SnakeEnv(gym.Env):
    """Custom Snake Environment that follows gym interface"""
    
    # Action mappings
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    NO_ACTION = 4  # Indicates no new direction input
    
    def __init__(self, grid_size=20, pixel_size=20):
        super(SnakeEnv, self).__init__()
        
        # Constants
        self.GRID_SIZE = grid_size
        self.PIXEL_SIZE = pixel_size
        self.SCREEN_SIZE = self.GRID_SIZE * self.PIXEL_SIZE
        
        # Colors optimized for diffusion model training
        self.BACKGROUND = (32, 32, 32)    # Dark gray background for better contrast
        self.SNAKE_HEAD = (0, 191, 255)   # Bright blue head - very distinct
        self.SNAKE_BODY = (50, 205, 50)   # Lime green body - clear but not too bright
        self.FOOD = (255, 69, 0)          # Bright orange-red food - high contrast
        self.GRID_LINES = (64, 64, 64)    # Subtle grid lines for spatial reference
        
        # Initialize screen as None - will be created on first human render
        self.screen = None
        
        # Action space (0: up, 1: right, 2: down, 3: left)
        self.action_space = spaces.Discrete(5)
        
        # Observation space (grid_size x grid_size x 3)
        # Channel 1: Snake body
        # Channel 2: Snake head
        # Channel 3: Food
        self.observation_space = spaces.Box(low=0, high=1,
                                         shape=(self.GRID_SIZE, self.GRID_SIZE, 3),
                                         dtype=np.float32)
        
        # Initialize game state
        self.reset()
        
    def reset(self):
        """Reset the game state and return observation"""
        # Initialize snake in the middle of the grid, moving right
        self.snake = [(self.GRID_SIZE//2, self.GRID_SIZE//2)]
        self.direction = self.RIGHT
        
        # Place food in random location
        self.place_food()
        
        # Reset score and steps
        self.score = 0
        self.steps = 0
        
        return self._get_observation()
    
    def place_food(self):
        """Place food in random empty location"""
        while True:
            self.food = (np.random.randint(0, self.GRID_SIZE),
                        np.random.randint(0, self.GRID_SIZE))
            if self.food not in self.snake:
                break
    
    def _get_observation(self):
        """Convert game state to observation array"""
        obs = np.zeros((self.GRID_SIZE, self.GRID_SIZE, 3), dtype=np.float32)
        
        # Snake body
        for segment in self.snake[1:]:
            obs[segment[1], segment[0], 0] = 1
        
        # Snake head
        obs[self.snake[0][1], self.snake[0][0], 1] = 1
        
        # Food
        obs[self.food[1], self.food[0], 2] = 1
        
        return obs
    
    def _is_valid_direction(self, new_direction):
        """Check if the new direction is valid (can't turn 180 degrees)"""
        if abs(new_direction - self.direction) == 2:
            return False
        return True
    
    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        self.steps += 1
        
        # Update direction if valid and not NO_ACTION
        if action != self.NO_ACTION and self._is_valid_direction(action):
            self.direction = action
        # NO_ACTION means continue in current direction
            
        # Get new head position based on current direction
        x, y = self.snake[0]
        if self.direction == self.UP:
            y = (y - 1) % self.GRID_SIZE
        elif self.direction == self.RIGHT:
            x = (x + 1) % self.GRID_SIZE
        elif self.direction == self.DOWN:
            y = (y + 1) % self.GRID_SIZE
        elif self.direction == self.LEFT:
            x = (x - 1) % self.GRID_SIZE
            
        new_head = (x, y)
        
        # Check if snake hits itself
        if new_head in self.snake[:-1]:
            return self._get_observation(), -1.0, True, {}
        
        # Move snake
        self.snake.insert(0, new_head)
        
        # Check if snake gets food
        reward = 0
        if new_head == self.food:
            self.score += 1
            reward = 1.0
            self.place_food()
        else:
            self.snake.pop()
            reward = -0.01  # Small negative reward to encourage efficient paths
            
        # Check if game should end (optional: can add max steps)
        done = False
        if self.steps >= self.GRID_SIZE * self.GRID_SIZE * 10:
            done = True
            
        return self._get_observation(), reward, done, {"score": self.score}
    
    def render(self, mode='human'):
        """Render the game state"""
        if self.screen is None and mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
            pygame.display.set_caption('Snake RL')
            
        # Create a surface to render on (either screen or new surface for rgb_array mode)
        if mode == 'human':
            surface = self.screen
        elif mode == 'rgb_array':
            surface = pygame.Surface((self.SCREEN_SIZE, self.SCREEN_SIZE))
        else:
            raise ValueError(f"Invalid mode: {mode}")
            
        # Draw everything
        surface.fill(self.BACKGROUND)
        
        # Draw grid lines for better spatial understanding
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(surface, self.GRID_LINES, 
                           (i * self.PIXEL_SIZE, 0),
                           (i * self.PIXEL_SIZE, self.SCREEN_SIZE))
            pygame.draw.line(surface, self.GRID_LINES,
                           (0, i * self.PIXEL_SIZE),
                           (self.SCREEN_SIZE, i * self.PIXEL_SIZE))
        
        # Draw snake body
        for segment in self.snake[1:]:
            pygame.draw.rect(surface, self.SNAKE_BODY,
                           (segment[0]*self.PIXEL_SIZE + 1, 
                            segment[1]*self.PIXEL_SIZE + 1,
                            self.PIXEL_SIZE - 2, self.PIXEL_SIZE - 2))
        
        # Draw head
        pygame.draw.rect(surface, self.SNAKE_HEAD,
                       (self.snake[0][0]*self.PIXEL_SIZE + 1,
                        self.snake[0][1]*self.PIXEL_SIZE + 1,
                        self.PIXEL_SIZE - 2, self.PIXEL_SIZE - 2))
        
        # Draw food
        pygame.draw.rect(surface, self.FOOD,
                       (self.food[0]*self.PIXEL_SIZE,
                        self.food[1]*self.PIXEL_SIZE,
                        self.PIXEL_SIZE, self.PIXEL_SIZE))
        
        if mode == 'human':
            pygame.display.flip()
            return None
        elif mode == 'rgb_array':
            return pygame.surfarray.array3d(surface).transpose(1, 0, 2)
            
    def close(self):
        """Clean up resources"""
        if self.screen is not None:
            pygame.quit()

# Example usage and human playable mode
if __name__ == "__main__":
    env = SnakeEnv()
    obs = env.reset()
    done = False
    clock = pygame.time.Clock()
    
    while not done:
        env.render()
        
        # Process input
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    action = env.UP
                elif event.key == pygame.K_d:
                    action = env.RIGHT
                elif event.key == pygame.K_s:
                    action = env.DOWN
                elif event.key == pygame.K_a:
                    action = env.LEFT
        
        # If no key pressed, use NO_ACTION to continue in current direction
        if action is None:
            action = env.NO_ACTION
            
        # Step environment
        obs, reward, done, info = env.step(action)
        
        # Control game speed
        clock.tick(10)  # 10 FPS
        
    env.close()
