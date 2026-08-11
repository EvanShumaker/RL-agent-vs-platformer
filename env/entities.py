"""
Game entities for player, platform, and goal.
Just dimensions for platform and goal. Collision logic will go in player class
"""

import env.constants as C


#AABB = axis aligned bounding box. used to check rectangle overlap in a simple platformer like this
def rects_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    """AABB collsiion check between two rectanges"""
    return(
        x1 < x2 + w2 and
        x1 + w1 > x2 and
        y1 < y2 + h2 and
        y1 + h1 > y2
    )


class Platform:
    """Just a solid rectangle the player can stand on. This one doesnt move"""

    def __init__(self, x, y, width, height):
        # introducing and setting all the elements of this class in one go
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class Player:
    """This is what the player moves, it has its own physics and collision stuff"""

    def __init__(self):
        '''
        i actually move the 'creation' logic to reset(), because its something I need to call
        multiple times, while still keeping the data of all the trials. Remember, init gets 
        called upon creation.
        '''
        self.reset()

    def reset(self):
        self.x = C.PLAYER_START_X
        self.y = C.PLAYER_START_Y
        self.vs = 0.0 #velocity
        self.vy = 0.0
        self.width = C.PLAYER_WIDTH
        self.height = C.PLAYER_HEIGHT
        self.on_ground = False #starts in the air
        self.alive = True

    def apply_action(self, action):
        """Translate an action into velocity"""
        if action == C.ACTION_LEFT:
            self.vx = -C.MOVE_SPEED
        elif action == C.ACTION_RIGHT:
            self.vx = C.MOVE_SPEED
        else:
            self.vx = 0.0

        if action == C.ACTION_JUMP and self.on_ground:
            self.vy = C.JUMP_VELOCITY
            self.on_ground = False

    
    def apply_physics(self):
        """Apply gravity and update position based on current velocity."""
        self.vy += C.GRAVITY
        self.vy = min(self.vy, C.TERMINAL_VELOCITY)

        self.x += self.vx
        self.y += self.vy

        # Keep player within horizontal screen bounds
        self.x = max(0, min(self.x, C.SCREEN_WIDTH - self.width))


    def resolve_platform_collisions(self, platforms):
        """
        Check collision against all platforms and snap the player to the
        top of any platform it's landing on. Right now it only resolves
        vertical (landing) collisions, since that's what a platformer needs most.
        """
        self.on_ground = False

        for platform in platforms:
            if rects_overlap(self.x, self.y, self.width, self.height,
                platform.x, platform.y, platform.width, platform.height):
                # Only treat it as "landing" if the player was falling
                # and is above the platform's top edge
                falling = self.vy >= 0
                was_above = (self.y + self.height - self.vy) <= platform.y + 1

                if falling and was_above:
                    self.y = platform.y - self.height
                    self.vy = 0
                    self.on_ground = True

