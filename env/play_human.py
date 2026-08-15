"""
this file lets humans play the game and do manual testing of levels.
Uses pygame for rendering and keyboard input, not used by RL agent
"""

import pygame
import env.constants as C
from env.game import Game


def get_action_from_keys():
    """
    Check what keys are being pressed and return that action
    """
    keys = pygame.key.get_pressed()

    # supporting wsad and arrows
    if keys[pygame.K_SPACE] or keys[pygame.K_w] or keys[pygame.K_UP]:
        return C.ACTION_JUMP
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        return C.ACTION_LEFT
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return C.ACTION_RIGHT

    return C.ACTION_NOOP



def draw(screen, game):
    """
    used by pygame to draw the actual level
    """
    screen.fill((30,30,40)) # dark background


def main():
    pygame.init()


