import logging
import sys
from collections import defaultdict
from enum import Enum
from typing import Dict, Optional, Sequence, List

import numpy as np
from gym import Env, spaces
from pyhtzee import Pyhtzee
from pyhtzee.classes import Category, Rule
from pyhtzee.utils import CATEGORY_ACTION_OFFSET

log = logging.getLogger(__name__)


class GameType(Enum):
    SUDDEN_DEATH = 0,
    RETRY_ON_WRONG_ACTION = 1


def get_score(score: Optional[int]) -> int:
    return score if score is not None else -1


def get_dice_face_counts(dice: Sequence[int]) -> Dict[int, int]:
    faces: Dict[int, int] = defaultdict(int)
    for die in dice:
        faces[die] += 1
    return faces


class YahtzeeSingleEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 rule: Rule = Rule.YAHTZEE,
                 game_type: GameType = GameType.RETRY_ON_WRONG_ACTION,
                 init_dices_list: List = None,
                 init_action_list: List = None,
                 seed=None):
        self.pyhtzee = Pyhtzee(seed=seed, rule=rule)
        self.rule = rule
        self.game_type = game_type
        self.action_space = spaces.Discrete(44)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(13),  # round
            spaces.Discrete(4),  # sub-round
            spaces.Box(low=1, high=6, shape=(1,), dtype=np.uint8),  # die 1
            spaces.Box(low=1, high=6, shape=(1,), dtype=np.uint8),  # die 2
            spaces.Box(low=1, high=6, shape=(1,), dtype=np.uint8),  # die 3
            spaces.Box(low=1, high=6, shape=(1,), dtype=np.uint8),  # die 4
            spaces.Box(low=1, high=6, shape=(1,), dtype=np.uint8),  # die 5
            spaces.Box(low=-1, high=5, shape=(1,), dtype=np.int16),  # aces
            spaces.Box(low=-1, high=10, shape=(1,), dtype=np.int16),  # twos
            spaces.Box(low=-1, high=15, shape=(1,), dtype=np.int16),  # threes
            spaces.Box(low=-1, high=20, shape=(1,), dtype=np.int16),  # fours
            spaces.Box(low=-1, high=25, shape=(1,), dtype=np.int16),  # fives
            spaces.Box(low=-1, high=30, shape=(1,), dtype=np.int16),  # sixes
            spaces.Box(low=-1, high=30, shape=(1,), dtype=np.int16),  # three of a kind
            spaces.Box(low=-1, high=30, shape=(1,), dtype=np.int16),  # four of a kind
            spaces.Box(low=-1, high=25, shape=(1,), dtype=np.int16),  # full house
            spaces.Box(low=-1, high=30, shape=(1,), dtype=np.int16),  # small straight
            spaces.Box(low=-1, high=40, shape=(1,), dtype=np.int16),  # large straight
            spaces.Box(low=-1, high=30, shape=(1,), dtype=np.int16),  # chance
            spaces.Box(low=-1, high=50, shape=(1,), dtype=np.int16),  # yahtzee
            spaces.Box(low=-1, high=35, shape=(1,), dtype=np.int16),  # upper bonus
            spaces.Box(low=-1, high=1200, shape=(1,), dtype=np.int16),  # yahtzee bonus
        ))

        self.init_dices_list = init_dices_list if init_dices_list is not None else []
        self.init_action_list = init_action_list if init_action_list is not None else []
        assert len(self.init_action_list) == len(self.init_dices_list)
        for action in self.init_action_list:
            assert action >= CATEGORY_ACTION_OFFSET, "No re-roll permitted while re-setting"

    def get_observation_space(self):
        pyhtzee = self.pyhtzee
        return (
            pyhtzee.round,
            pyhtzee.sub_round,
            pyhtzee.dice[0],
            pyhtzee.dice[1],
            pyhtzee.dice[2],
            pyhtzee.dice[3],
            pyhtzee.dice[4],
            get_score(pyhtzee.scores.get(Category.ACES)),
            get_score(pyhtzee.scores.get(Category.TWOS)),
            get_score(pyhtzee.scores.get(Category.THREES)),
            get_score(pyhtzee.scores.get(Category.FOURS)),
            get_score(pyhtzee.scores.get(Category.FIVES)),
            get_score(pyhtzee.scores.get(Category.SIXES)),
            get_score(pyhtzee.scores.get(Category.THREE_OF_A_KIND)),
            get_score(pyhtzee.scores.get(Category.FOUR_OF_A_KIND)),
            get_score(pyhtzee.scores.get(Category.FULL_HOUSE)),
            get_score(pyhtzee.scores.get(Category.SMALL_STRAIGHT)),
            get_score(pyhtzee.scores.get(Category.LARGE_STRAIGHT)),
            get_score(pyhtzee.scores.get(Category.CHANCE)),
            get_score(pyhtzee.scores.get(Category.YAHTZEE)),
            get_score(pyhtzee.scores.get(Category.UPPER_SECTION_BONUS)),
            get_score(pyhtzee.scores.get(Category.YAHTZEE_BONUS)),
        )

    def sample_action(self):
        action = self.pyhtzee.sample_action()
        log.info(f'Sampled action: {action}')
        return action

    def step(self, action: int):
        pyhtzee = self.pyhtzee
        reward = pyhtzee.take_action(action)
        finished = pyhtzee.is_finished()

        return self.get_observation_space(), reward, finished, {}

    def render(self, mode='human', close=False):
        dice = self.pyhtzee.dice
        outfile = sys.stdout
        outfile.write(f'Dice: {dice[0]} {dice[1]} {dice[2]} {dice[3]} {dice[4]} '
                      f'Round: {self.pyhtzee.round}.{self.pyhtzee.sub_round} '
                      f'Score: {self.pyhtzee.get_total_score()}\n')

    # Custom utilities

    def set_dice(self, dices: List[int]):
        assert len(dices) == 5
        for d in dices:
            assert d in [1, 2, 3, 4, 5, 6], f"Dices values should be in 1,...6; {d} found"
        self.pyhtzee.dice = dices.copy()

    def reset_internal_state(self, init_dices_list: List, init_action_list: List):
        assert len(init_dices_list) == len(init_action_list) or len(init_dices_list) == len(init_action_list) + 1

        self.pyhtzee = Pyhtzee(rule=self.rule)
        init_dices_list = self.init_dices_list + init_dices_list
        init_action_list = self.init_action_list + init_action_list
        for dices, action in zip(init_dices_list, init_action_list):
            self.set_dice(dices)
            self.pyhtzee.take_action(action)
        return self.get_observation_space(), self.pyhtzee.get_total_score()

    def get_available_actions(self):
        return self.pyhtzee.get_possible_actions()
