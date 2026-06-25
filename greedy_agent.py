import gymnasium as gym
import numpy as np
from collections import deque
from env.factory_escape_env import FactoryEscapeEnv

def get_action(env):
    agent_pos = env.unwrapped.agent_pos
    agent_dir = env.unwrapped.agent_dir
    
    # 1. Determine Target
    if env.unwrapped.carrying:
        target_pos = env.unwrapped.exit_pos
        # If next to exit and facing it, drop
        dx = abs(env.unwrapped.front_pos[0] - target_pos[0])
        dy = abs(env.unwrapped.front_pos[1] - target_pos[1])
        if dx <= 1 and dy <= 1:
            return env.unwrapped.actions.drop
    else:
        # Find nearest alive worker
        workers = [w for w in env.unwrapped.workers if not w.is_rescued and w.cur_pos is not None and tuple(w.cur_pos) != (-1, -1)]
        if not workers:
            return env.unwrapped.actions.done 
        
        # Simple Manhattan distance to find the closest one
        workers.sort(key=lambda w: abs(w.cur_pos[0] - agent_pos[0]) + abs(w.cur_pos[1] - agent_pos[1]))
        target_pos = tuple(workers[0].cur_pos)
        
        # If next to worker and facing it, pickup
        if tuple(env.unwrapped.front_pos) == target_pos:
            return env.unwrapped.actions.pickup

    # 2. BFS Pathfinding to navigate the factory walls
    grid = env.unwrapped.grid
    queue = deque([[agent_pos]])
    seen = {agent_pos}
    
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        
        if (x, y) == target_pos:
            if len(path) > 1:
                next_step = path[1]
                # Calculate required direction to take the next step
                dx = next_step[0] - agent_pos[0]
                dy = next_step[1] - agent_pos[1]
                
                if dx == 1: req_dir = 0
                elif dy == 1: req_dir = 1
                elif dx == -1: req_dir = 2
                else: req_dir = 3
                
                if agent_dir == req_dir:
                    return env.unwrapped.actions.forward
                else:
                    # Turn towards the required direction
                    diff = (req_dir - agent_dir) % 4
                    if diff == 1:
                        return env.unwrapped.actions.right
                    else:
                        return env.unwrapped.actions.left
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                cell = grid.get(nx, ny)
                # Walkable if Empty, Floor, or our Target
                if (nx, ny) not in seen:
                    if cell is None or cell.type == 'floor' or (nx, ny) == target_pos:
                        seen.add((nx, ny))
                        queue.append(path + [(nx, ny)])
                        
    # If no path is found (e.g., trapped by lava), take a random action to try and unstick
    return env.unwrapped.action_space.sample()

def main():
    print("Initializing Greedy Baseline Agent...")
    env = FactoryEscapeEnv(size=16, render_mode=None)
    
    episodes = 100
    successes = 0
    total_workers_saved = []
    total_rewards = []
    total_steps = []
    
    print(f"Running {episodes} episodes. Please wait...")
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        steps = 0
        
        while not done and not truncated:
            action = get_action(env)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            
        if info.get('success'):
            successes += 1
        total_workers_saved.append(env.unwrapped.saved_workers)
        total_rewards.append(ep_reward)
        total_steps.append(steps)
        
    print("\n" + "="*35)
    print("    GREEDY AGENT BASELINE RESULTS")
    print("    (Evaluated over 100 episodes)")
    print("="*35)
    print(f"Success Rate           : {successes/episodes * 100:.1f}%")
    print(f"Average Workers Saved  : {np.mean(total_workers_saved):.2f} out of {env.unwrapped.n_workers}")
    print(f"Average Reward         : {np.mean(total_rewards):.2f}")
    print(f"Average Episode Length : {np.mean(total_steps):.1f} steps")
    print("="*35)

if __name__ == "__main__":
    main()
