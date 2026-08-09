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