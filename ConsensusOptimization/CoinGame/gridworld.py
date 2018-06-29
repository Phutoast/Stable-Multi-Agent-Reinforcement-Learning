from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
import sys
import random
import copy
import warnings

from pycolab import ascii_art
from pycolab import human_ui
from pycolab import rendering
from pycolab.prefab_parts import sprites as prefab_sprites

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

EMPTY_MAP = ['#######',
             '#     #',
             '#     #',
             '#     #',
             '#     #',
             '#     #',
             '#######']

# Adding a Color to the map
COLOR_MAP = {'#': (0, 0, 0), ' ': (1, 1, 1), 'P': (1, 0, 0), 'Q': (0, 1, 0), '1':(0, 0, 1), '2':(1, 1, 0)}

class Game(object):
    def __init__(self):
        self.reward_p1 = 0
        self.reward_p2 = 0
        self.is_p1_in = False
        self.is_p2_in = False
        
        self.p1_played = False
        self.p2_played = False
        
        self.old_board = []

    def reset(self):
        self.game = make_game() 
        obs, reward, discount = self.game.its_showtime()
        obs = renderer(obs)
        
        self.reward_p1 = 0
        self.reward_p2 = 0
        self.is_p1_in = False
        self.is_p2_in = False

        self.p1_played = False
        self.p2_played = False
        return obs, reward, discount

    def step(self, actions):
        self.p1_played = False
        self.p2_played = False

        if self.is_p1_in or self.is_p2_in:
            raise RuntimeError("Game already ended")
        
        p1_action, p2_action = actions
        obs, _, _ = self.game.play((1, p1_action))
        obs = renderer(obs)
        
        obs, _, _ = self.game.play((2, p2_action))
        obs = renderer(obs)
        return obs, (self.reward_p1, self.reward_p2), self.is_p1_in or self.is_p2_in

# Have to be global
game = Game()

# Since pytorch accept [batch_size x channel x width x height]
def renderer(obs, scale=8):
    renderer_normal = rendering.ObservationToArray(COLOR_MAP, dtype=np.float32)
    obs_normal = renderer_normal(obs)
    obs_normal = trans_image(obs_normal)
    new_size = obs_normal.shape[1]*scale
    
    # Scale up the observation.
    scaled_img = np.stack([scipy.misc.imresize(obs_normal[:, :, i], [new_size, new_size, 1], interp="nearest")
                           for i in range(3)], axis=2)
                           
    return np.transpose(scaled_img, (2, 1, 0))

def replace_string(string, index, val):
    return string[:index] + val + string[index + 1:]

def random_map(empty_map, num_p1_reward=1, num_p2_reward=1, size=10):
    """
    Stupid way to random a map - Need to optimize this.
    """
    
    # I want to have "lower" number of X compare to $ and there is a chance that $ will replace X.
    # But I also want P to exists
    empty_map = copy.deepcopy(empty_map)
    
    if random.random() < 0.5:
        for _ in range(num_p1_reward):
            row = random.randint(1, size-2)
            col = random.randint(1, size-2)
            empty_map[row] = replace_string(empty_map[row], col, '1')
    else: 
        for _ in range(num_p2_reward):
            row = random.randint(1, size-2)
            col = random.randint(1, size-2)
            empty_map[row] = replace_string(empty_map[row], col, '2')

    while empty_map[row][col] != ' ': 
        row = random.randint(1, size-2)
        col = random.randint(1, size-2)
    empty_map[row] = replace_string(empty_map[row], col, 'P')
    
    while empty_map[row][col] != ' ': 
        row = random.randint(1, size-2)
        col = random.randint(1, size-2)
    empty_map[row] = replace_string(empty_map[row], col, 'Q')

    return empty_map

def make_game():
    return ascii_art.ascii_art_to_game(
                                       random_map(EMPTY_MAP, size=7),
                                       what_lies_beneath=' ',
                                       sprites={'P': PlayerSprite1, 'Q': PlayerSprite2})

class PlayerSprite1(prefab_sprites.MazeWalker):
    def __init__(self, corner, position, character):
        super().__init__(corner, position, character, impassable='#')
    
    def update(self, actions, board, layers, backdrop, things, the_plot):
        del layers, backdrop, things
        
        global game
        if not game.p2_played:
            game.old_board = copy.deepcopy(board)
        game.p1_played = True
         
        if not actions is None and actions[0] == 1:
            if actions == (1, 0): 
                self._north(board, the_plot)
            elif actions == (1, 1):
                self._south(board, the_plot)
            elif actions == (1, 2):
                self._west(board, the_plot)
            elif actions == (1, 3):
                self._east(board, the_plot)

            # Checking for reward
            if ord('1') == board[self.position]:
                game.reward_p1 += 1
                game.is_p1_in = True
            elif ord('2') == board[self.position]:
                game.reward_p1 += 1
                game.reward_p2 -= 2
                game.is_p1_in = True
            
class PlayerSprite2(prefab_sprites.MazeWalker):
    def __init__(self, corner, position, character):
        super().__init__(corner, position, character, impassable='#')
    
    def update(self, actions, board, layers, backdrop, things, the_plot):
        del layers, backdrop, things
        
        global game 
        if not game.p1_played:
            game.old_board = copy.deepcopy(board)
        game.p2_played = True
        
        if not actions is None and actions[0] == 2:
            if actions == (2, 0): 
                self._north(board, the_plot)
            elif actions == (2, 1):
                self._south(board, the_plot)
            elif actions == (2, 2):
                self._west(board, the_plot)
            elif actions == (2, 3):
                self._east(board, the_plot)
            
            if ord('1') == game.old_board[self.position]:
                game.reward_p1 -= 2
                game.reward_p2 += 1 
                game.is_p2_in = True 
            elif ord('2') == game.old_board[self.position]:
                game.reward_p2 += 1
                game.is_p2_in = True 

def trans_image(img):
    return np.transpose(img, (1, 2, 0))

def display_image(img):
    """
        Getting the output from renderer
        then display it
        
        Args:
        img(numpy array) - image that we want to display
        """
    
    # Just transpose the image back !!
    plt.imshow(trans_image(img))
    plt.show()
