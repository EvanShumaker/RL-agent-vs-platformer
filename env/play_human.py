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

    state = game.get_state()

    # now draw platforms using pos and dimensions. 
    # draw.rect() takes screen, color, and pos / dimensions
    for (x,y,w,h) in state["platforms"]:
        pygame.draw.rect(screen, (100,100,110), (x,y,w,h))

    # this is the goal
    pygame.draw.rect(screen, (240,200,60), *state["goal"])

    # Draw pit (visual of this helps with testing)
    pxmin, pxmax = C.PIT_X_RANGE
    pygame.draw.rect(screen, (60,20,20), (pxmin, C.SCREEN_HEIGHT - 10,pxmax - pxmin, 10))

    # now for the player icon
    if state["alive"]:
        color = (80,180,240)
    else:
        color = (200,60,60)

    pygame.draw.rect(screen,color, (state["player_x"], state["player_y"], 
                                    C.PLAYER_WIDTH, C.PLAYER_HEIGHT) )
    
    pygame.display.flip()


def main():
    pygame.init()
    screen = pygame.display.set_mode((C.SCREEN_WIDTH, C.SCREEN_HEIGHT))
    pygame.display.set_caption("Platformer RL agent, human play mode")
    clock = pygame.time.Clock()

    game = Game()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = get_action_from_keys()
        reward, done = game.step(action)

        draw(screen, game)

        if done: # messages for every possible ending scenario
            state = game.get_state()
            if state["alive"] and game.player.check_goal_reached(game.goal):
                print(f"Reached the goal!! steps: {state['steps_elapsed']}")
            elif not state["alive"]:
                print("Died. Resetting...")
            else:
                print("Max steps reached. Resseting...")
            pygame.time.wait(500) # 500ms
            game.reset()

        clock.tick(60) #60 fps

    pygame.quit()


if __name__ == "__main__":
    main()
