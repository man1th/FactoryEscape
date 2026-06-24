from minigrid.core.world_object import Ball

class Worker(Ball):
    def __init__(self, color="blue", health=100):
        super().__init__(color)
        self.health = health
        self.is_rescued = False

    def decay_health(self):
        if self.health > 0:
            self.health -= 1
            
    def can_pickup(self):
        return True
