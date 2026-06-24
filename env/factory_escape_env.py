import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Lava, Floor
from minigrid.core.mission import MissionSpace
from env.objects import Worker

class FactoryEscapeEnv(MiniGridEnv):
    def __init__(self, size=16, n_workers=3, max_steps=1000, spread_rate=10, render_mode=None):
        self.n_workers = n_workers
        self.spread_rate = spread_rate
        self.saved_workers = 0
        
        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=False,
            render_mode=render_mode
        )

    @staticmethod
    def _gen_mission():
        return "Rescue all workers and escape before the factory collapses."

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.grid.vert_wall(8, 0, height)
        self.grid.horz_wall(0, 8, width)
        
        self.grid.set(8, 4, None)
        self.grid.set(4, 8, None)
        self.grid.set(8, 12, None)
        self.grid.set(12, 8, None)

        self.exit_pos = (width - 2, height - 2)
        self.put_obj(Goal(), *self.exit_pos)

        self.ignition_pos = (1, 1)
        self.put_obj(Lava(), *self.ignition_pos)
        self.lava_positions = {self.ignition_pos}

        self.workers = []
        self.saved_workers = 0
        worker_positions = [(3, 3), (12, 4), (5, 12)]
        for pos in worker_positions:
            worker = Worker(color="blue", health=200) 
            self.put_obj(worker, *pos)
            self.workers.append(worker)

        self.agent_pos = (7, 7)
        self.agent_dir = 0
        self.step_count_custom = 0

    def step(self, action):
        was_carrying = self.carrying

        obs, reward, terminated, truncated, info = super().step(action)
        self.step_count_custom += 1

        reward = -0.05  

        if action == self.actions.drop and was_carrying and not self.carrying:
            dx = abs(self.front_pos[0] - self.exit_pos[0])
            dy = abs(self.front_pos[1] - self.exit_pos[1])
            if dx <= 1 and dy <= 1:
                self.saved_workers += 1
                reward += 1.0 * (1.0 - (self.step_count_custom / self.max_steps))
                self.grid.set(*self.front_pos, None) 
                was_carrying.is_rescued = True

        if self.agent_pos == self.exit_pos:
            if self.saved_workers < self.n_workers:
                terminated = False

        if self.step_count_custom % self.spread_rate == 0:
            new_lava = set()
            for lx, ly in self.lava_positions:
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = lx + dx, ly + dy
                    if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                        cell = self.grid.get(nx, ny)
                        if cell is None or isinstance(cell, Floor) or isinstance(cell, Worker):
                            new_lava.add((nx, ny))
            
            for nx, ny in new_lava:
                self.grid.set(nx, ny, Lava()) 
                self.lava_positions.add((nx, ny))

        alive_workers = 0
        for worker in self.workers:
            if not worker.is_rescued:
                worker.decay_health()
                
                # Safely convert numpy array to tuple for set checking
                in_lava = False
                if worker.cur_pos is not None:
                    pos_tuple = tuple(worker.cur_pos)
                    if pos_tuple in self.lava_positions:
                        in_lava = True

                if worker.health <= 0 or in_lava:
                    reward -= 0.5
                    worker.is_rescued = True 
                    
                    if worker.cur_pos is not None:
                        pos_tuple = tuple(worker.cur_pos)
                        # Only overwrite with lava if they aren't currently being carried (-1, -1)
                        if pos_tuple != (-1, -1) and pos_tuple not in self.lava_positions:
                            self.grid.set(*worker.cur_pos, Lava()) 
                else:
                    alive_workers += 1

        if self.saved_workers == self.n_workers:
            terminated = True
            info['success'] = True
        elif alive_workers == 0 and self.saved_workers < self.n_workers:
            terminated = True
            info['success'] = False

        return obs, reward, terminated, truncated, info
