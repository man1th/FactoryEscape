import gymnasium as gym
import pygame
from minigrid.manual_control import ManualControl
from env.factory_escape_env import FactoryEscapeEnv

class CustomManualControl(ManualControl):
    def __init__(self, env):
        super().__init__(env)

    def key_handler(self, event):
        key = event.key

        # 1. Handle String Keys (MiniGrid 2.3.0+)
        if isinstance(key, str):
            if key == 'escape':
                self.env.close()
                import sys
                sys.exit(0)
            elif key == 'backspace':
                self.reset(self.env.unwrapped.seed)
            elif key == 'left':
                self.step(self.env.unwrapped.actions.left)
            elif key == 'right':
                self.step(self.env.unwrapped.actions.right)
            elif key == 'w':
                self.step(self.env.unwrapped.actions.forward)
            elif key == 'e':
                self.step(self.env.unwrapped.actions.pickup)
            elif key == 'r':
                self.step(self.env.unwrapped.actions.drop)
                
        # 2. Handle Integer Keys (Standard Pygame)
        elif isinstance(key, int):
            if key == pygame.K_ESCAPE:
                self.env.close()
                import sys
                sys.exit(0)
            elif key == pygame.K_BACKSPACE:
                self.reset(self.env.unwrapped.seed)
            elif key == pygame.K_LEFT:
                self.step(self.env.unwrapped.actions.left)
            elif key == pygame.K_RIGHT:
                self.step(self.env.unwrapped.actions.right)
            elif key == pygame.K_w:
                self.step(self.env.unwrapped.actions.forward)
            elif key == pygame.K_e:
                self.step(self.env.unwrapped.actions.pickup)
            elif key == pygame.K_r:
                self.step(self.env.unwrapped.actions.drop)

def main():
    print("Starting FactoryEscape-v0 Custom Manual Verification...")
    print("CONTROLS:")
    print("  W           : Move Forward")
    print("  Left/Right  : Turn Left / Turn Right")
    print("  E           : Pick Up Worker")
    print("  R           : Drop Worker at Exit")
    print("  Backspace   : Reset Episode")
    print("  Escape      : Quit")
    
    env = FactoryEscapeEnv(size=16, render_mode="human")
    manual_control = CustomManualControl(env)
    manual_control.start()

if __name__ == "__main__":
    main()
