import gym
import pygame
from gym.utils.play import play

mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
play(gym.make("LunarLander-v2"), keys_to_action=mapping)
