"""
This is the main game loop and logic. 
"""

import env.constants as C
from env.entities import Player, Platform, Goal


class Game:
    """Other than init, ill need a function to call the player reset(), a step()
    to move it forwards a frame, function to call check_goal_reached(), and a 
    function to update the rewards? like -.01 every turn"""
    def __init__(self):
        # this loop SHOULD work, C.PLATFORMS is the list defined in constants.
        # the Platform init() needs 4 args, so the * should separate them
        self.platforms = [Platform(*p) for p in C.PLATFORMS]
        # same extrapolation for this?
        self.goal = Goal(*C.GOAL)
        self.player = Player()
        self.steps_elapsed = 0
        self.done = False
        self.reset()

    
    def reset(self):
        """resets the game"""
        self.player.reset()
        self.done = False
        self.steps_elapsed = 0
        # leading _ is convention for variables that arent meant to be used outside of class
        self._prev_dist_to_goal = self._distance_to_goal()

    
    def step(self, action):
        """
        loads the next frame of the game based on what the player does.
        I just need to call the code in the player class in entities to 
        apply action, apply physics, etc.
        """
        if self.done:
            raise RuntimeError("Step() called after episode ended. Call reset() again first")

        self.player.apply_action(action)
        self.player.apply_physics()
        # need to give Game platforms info so it can pass it here
        self.player.resolve_platform_collisions()
        self.player.check_pit_fall(C.PIT_X_RANGE)

        self.steps_elapsed += 1

        return self._compute_reward(), self._check_done()


    def _compute_reward(self):
        """
        Just like with varibales, leading _ means this func isnt called outside
        of this class. This func just calculates distance and progress to get the current reward.
        Its called once per step/frame, so i just add one -.01 at the start
        """
        reward = C.REWARD_STEP

        dist = self._distance_to_goal()
        progress = self._prev_dist_to_goal - dist
        reward += progress * C.REWARD_PROGRESS_SCALE
        self._prev_dist_to_goal = dist

        if not self.player.alive:
            reward += C.REWARD_DEATH
        elif self.player.check_goal_reached(self.goal):
            reward += C.REWARD_GOAL
        
        return reward


    def _check_done(self):
        """
        basically all the cases in which the agent / player needs to reset
        """
        if not self.player.alive:
            return True
        if self.player.check_goal_reached(self.goal):
            return True
        if self.steps_elapsed >= C.MAX_EPISODE_STEPS:
            return True
        return False


    # not correct, fix later
    def _distance_to_goal():
        """
        check the distance to the nearest edge. Theres 3 possibilities:
        1. player center y val is within the y range of the goal, only need to measure x dist
        2. player center x val is within goals x range
        3. player center is outside of both ranges (diagonal)"""
        #TODO: This measures against the top left point... i need additional cases for each side of the goal box
        xdist = self.goal.x - self.player.x
        ydist = self.goal.y - self.player.y
        if(self.player.y > self.goal.y and self.player.y < self.goal.y+self.goal.height):
            return abs(xdist)
        if(self.player.x < self.goal.x and self.player.y < self.goal.y+self.goal.height):
            return abs(ydist)
        


    def get_state(self):
        """
        maybe useful later. use this to print out all the game information
        should i print? or return as a list or dict?
        """
