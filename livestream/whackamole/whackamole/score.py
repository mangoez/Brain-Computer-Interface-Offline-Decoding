# -*- coding: utf-8 -*-

"""
Whack a Mole
~~~~~~~~~~~~~~~~~~~
A simple Whack a Mole game written with PyGame
:copyright: (c) 2018 Matt Cowley (IPv4)
"""

from .constants import LevelConstants, GameConstants
from .text import Text


class Score:
    """
    Handles the scoring for the player
    """

    def __init__(self, text: Text):
        self.hits = 0
        self.misses = 0
        self.text = text

    @property
    def score(self):
        return (self.hits - (self.misses / 2)) * 2

    @property
    def level(self):
        if self.score < 0:
            return 1
        else:
            return int(1 + (self.score // LevelConstants.LEVELGAP))

    @property
    def attempts(self):
        return self.hits + self.misses

    def disp_score(self, timer, debug):
        # Generate hit/miss data
        hits = [self.hits, 0 if self.attempts == 0 else self.hits / self.attempts * 100]
        misses = [self.misses, 0 if self.attempts == 0 else self.misses / self.attempts * 100]

        # Generate score text
        text = "Score: {:,.0f} / Hits: {:,} ({:,.1f}%) / Misses: {:,} ({:,.1f}%) / Level: {:,.0f}".format(
            self.score, hits[0], hits[1], misses[0], misses[1], self.level
        )

        # Display timer
        if timer:
            display = None
            if timer == -1: display = "Click to begin..."
            if timer < 0: timer = 0
            if not display: display = "{:,.0f}s".format(timer)
            text += " / Time Remaining: {}".format(display)

        # Add any extra readout data
        if debug:
            ext_data_comp = []
            for key, val in debug.items():
                ext_data_comp.append("{}: {}".format(key, val))
            ext_data = " / ".join(ext_data_comp)
            text += " / {}".format(ext_data)

        return text

    def label(self, *, timer=None, debug={}, size=1):
        return self.text.get_label(self.disp_score(timer, debug), "/", scale=size,
                                   width=GameConstants.GAMEWIDTH, background=(0, 0, 0, 0.4 * 255))

    def hit(self):
        self.hits += 1

    def miss(self):
        self.misses += 1
