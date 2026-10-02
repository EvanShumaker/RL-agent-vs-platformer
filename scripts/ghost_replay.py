"""
Plays back recorded ghost trajectories all at once. many
"ghost" characters moving simultaneously along their recorded paths,
colored by training stage (blue is early, red is later). The top-N
highest-reward runs should stand out
"""

import json
import pygame
import env.constants as C

TRAJECTORY_PATH = "logs/ghost_trajectories.json"
TOP_N_HIGHLIGHT = 10
FPS = 60


def load_runs():
    with open(TRAJECTORY_PATH, "r") as f:
        return json.load(f)

def get_color_for_checkpoint(checkpoint_episode, min_ep, max_ep):
    """Based on how high the reward was, the color will be closer to
    blue than red. This will probably get called for every run in main()"""

    # min and max ep used to base the colors off of. Fraction math to get what the gradient should be
    if max_ep == min_ep:
            fraction = 1.0
    else:
        fraction = (checkpoint_episode - min_ep) / (max_ep - min_ep)
    r = int(255 * fraction)
    b = int(255 * (1 - fraction))
    return (r, 80, b)

def draw_static_level(screen):
    """self explanatory, recreating the level based on what was used in training"""
    for (x, y, w, h) in C.PLATFORMS:
        pygame.draw.rect(screen, (100, 100, 110), (x, y, w, h))

    gx, gy, gw, gh = C.GOAL
    pygame.draw.rect(screen, (240, 200, 60), (gx, gy, gw, gh))
    pit_x_min, pit_x_max = C.PIT_X_RANGE
    pygame.draw.rect(screen, (60, 20, 20), (pit_x_min, C.SCREEN_HEIGHT - 10, pit_x_max - pit_x_min, 10))


def main():
    """Big func: get all the runs as json, sort by reward, init pygame, then
     add all the run records with loop. """
    
    runs = load_runs()
    if not runs:
        print("No recorded runs found — run scripts/record_ghost_trajectories.py first.")
        return

    episodes = [r["checkpoint_episode"] for r in runs]
    min_ep, max_ep = min(episodes), max(episodes)

    sorted_indices = sorted(range(len(runs)), key=lambda i: runs[i]["total_reward"], reverse=True)
    top_indices = set(sorted_indices[:TOP_N_HIGHLIGHT])

    max_length = max(len(r["trajectory"]) for r in runs)

    pygame.init()
    screen = pygame.display.set_mode((C.SCREEN_WIDTH, C.SCREEN_HEIGHT))
    pygame.display.set_caption("Ghost Replay — All Training Stages")
    clock = pygame.time.Clock()

    ghost_surface = pygame.Surface((C.SCREEN_WIDTH, C.SCREEN_HEIGHT), pygame.SRCALPHA)

    frame = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # do the important stuff here. 2 for loops, one for most runs, the other for top runs
        screen.fill((30, 30, 40))
        draw_static_level(screen)
        ghost_surface.fill((0, 0, 0, 0))  # clear to fully transparent

        # Pass 1: faint ghosts onto the transparent surface
        for i, run in enumerate(runs):
            if i in top_indices or frame >= len(run["trajectory"]):
                continue
            x, y = run["trajectory"][frame]
            color = get_color_for_checkpoint(run["checkpoint_episode"], min_ep, max_ep)
            center = (int(x + C.PLAYER_WIDTH / 2), int(y + C.PLAYER_HEIGHT / 2))
            pygame.draw.circle(ghost_surface, (*color, 60), center, 5)

        screen.blit(ghost_surface, (0, 0))

        # Pass 2: solid top-N runs drawn on top, so they stay crisp
        for i in top_indices:
            run = runs[i]
            if frame >= len(run["trajectory"]):
                continue
            x, y = run["trajectory"][frame]
            color = get_color_for_checkpoint(run["checkpoint_episode"], min_ep, max_ep)
            center = (int(x + C.PLAYER_WIDTH / 2), int(y + C.PLAYER_HEIGHT / 2))
            pygame.draw.circle(screen, color, center, 6)

        pygame.display.flip()

        frame = (frame + 1) % max_length  # loop back once everyone's finished
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()