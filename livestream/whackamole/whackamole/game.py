# -*- coding: utf-8 -*-

"""
Whack a Mole
~~~~~~~~~~~~~~~~~~~
A simple Whack a Mole game written with PyGame
:copyright: (c) 2018 Matt Cowley (IPv4)
"""

from pygame import init, quit, display, image, transform, time, mouse, event, Surface, \
    SRCALPHA, QUIT, MOUSEBUTTONDOWN, KEYDOWN, \
    K_e, K_r, K_t, K_y, K_u, K_i, K_o, K_p, K_SPACE, K_ESCAPE
import socket

from .constants import Constants
from .mole import Mole
from .score import Score
from .text import Text
from datetime import datetime



class Game:
    """
    Handles the main game
    Takes :time: in seconds for game timer
    """

    def __init__(self, *, timer: int = None, autostart: bool = True):
        # Init pygame
        init()

        # Create pygame screen
        self.screen = display.set_mode((Constants.GAMEWIDTH, Constants.GAMEHEIGHT))
        display.set_caption(Constants.TEXTTITLE)

        # Load background
        self.img_background = image.load(Constants.IMAGEBACKGROUND)
        self.img_background = transform.scale(self.img_background, (Constants.GAMEWIDTH, Constants.GAMEHEIGHT))

        # Load hole
        self.img_hole = image.load(Constants.IMAGEHOLE)
        self.img_hole = transform.scale(self.img_hole, (Constants.HOLEWIDTH, Constants.HOLEHEIGHT))

        # Load mallet
        self.img_mallet = image.load(Constants.IMAGEMALLET)
        self.img_mallet = transform.scale(self.img_mallet, (Constants.MALLETWIDTH, Constants.MALLETHEIGHT))

        # Set timer
        self.timer = timer

        # Reset/initialise data
        self.reset()

        # Run
        if autostart:
            self.run()

    def reset(self):
        # Load moles
        self.moles = [Mole() for _ in range(Constants.MOLECOUNT)]

        # Generate hole positions
        self.holes = []
        self.used_holes = []
        base_row = Constants.GAMEHEIGHT / Constants.HOLEROWS
        base_column = Constants.GAMEWIDTH / Constants.HOLECOLUMNS
        for row in range(Constants.HOLEROWS):
            rowY = base_row * row
            rowY += (base_row - Constants.HOLEHEIGHT) / 2
            for column in range(Constants.HOLECOLUMNS):
                thisX = base_column * column
                thisX += (base_column - Constants.HOLEWIDTH) / 2
                self.holes.append((int(thisX), int(rowY)))

        # Get the text object
        self.text = Text()

        # Get the score object
        self.score = Score(self.text)

        # Indicates whether the HUD indicators should be displayed
        self.show_hit = 0
        self.show_miss = 0

        # Allow for game timer
        self.timer_start = 0

    @property
    def timerData(self):
        if self.timer is not None and self.timer_start != 0:
            remain = (time.get_ticks() - self.timer_start) / 1000
            remain = self.timer - remain
            endGame = True if remain <= 0 else False
            return (remain, endGame)
        return (None, False)

    def loop_events(self):

        hit = False
        miss = False
        clicked = False
        pos = (0, 0)

        # Handle PyGame events
        for e in event.get():

            # Handle quit
            if e.type == QUIT:
                self.loop = False
                break

            gameTime, endGame = self.timerData

            if not endGame:
                # Handle click (in our case )
                if e.type == MOUSEBUTTONDOWN and e.button == Constants.LEFTMOUSEBUTTON:

                    # Start timer if not started
                    if self.timer is not None and self.timer_start == 0:
                        self.timer_start = time.get_ticks()

                    else:
                        # Handle hit/miss
                        clicked = True
                        miss = True
                        for mole in self.moles:
                            pos = mole.get_hammer_pos()
                            if mole.is_hit(pos) == 1:  # Hit
                                hit = True
                                miss = False
                            if mole.is_hit(pos) == 2:  # Hit but stunned
                                miss = False

                        if hit:
                            self.score.hit()
                        if miss:
                            self.score.miss()

                if e.type == KEYDOWN:

                    # Allow escape to abort attempt
                    if e.key == K_ESCAPE:
                        self.reset()
                        break

                    # Handle cheats (for dev work)
                    if Constants.DEBUGMODE:
                        if e.key == K_e:
                            hit = True
                            miss = False
                            self.score.hit()
                        if e.key == K_r:
                            hit = False
                            miss = True
                            self.score.miss()

                        if e.key == K_t:
                            self.score.misses = 0
                        if e.key == K_y:
                            self.score.misses += 5
                        if e.key == K_u:
                            self.score.misses -= 5

                        if e.key == K_i:
                            self.score.hits = 0
                        if e.key == K_o:
                            self.score.hits += 5
                        if e.key == K_p:
                            self.score.hits -= 5

            # End game screen
            else:
                if e.type == KEYDOWN:
                    if e.key == K_SPACE:
                        # Restart
                        self.reset()
                        break

        return (clicked, hit, miss)

    def loop_display(self, clicked, hit, miss):
        gameTime, endGame = self.timerData
        if not gameTime and self.timer:
            gameTime = -1

        # Display bg
        self.screen.blit(self.img_background, (0, 0))

        # Display holes
        for position in self.holes:
            self.screen.blit(self.img_hole, position)

        # Display moles
        for mole in self.moles:
            holes = [f for f in self.holes if f not in self.used_holes]
            mole_display = mole.do_display(holes, self.score.level, not endGame)

            # If new/old hole given
            if len(mole_display) > 1:
                if mole_display[1] == 0:  # New hole
                    self.used_holes.append(mole_display[2])
                else:  # Old hole
                    if mole_display[2] in self.used_holes:
                        self.used_holes.remove(mole_display[2])

            # If should display
            if mole_display[0]:
                # Get pos and display
                pos = mole.get_hole_pos(not endGame)
                self.screen.blit(mole.image, pos)

            # Hammer
            thisHammer = transform.rotate(self.img_mallet.copy(),
                                        (Constants.MALLETROTHIT if clicked else Constants.MALLETROTNORM))
            hammer_x, hammer_y = mole.get_hammer_pos()
            hammer_x -= thisHammer.get_width() / 5
            hammer_y -= thisHammer.get_height() / 4
            self.screen.blit(thisHammer, (hammer_x, hammer_y))

        # Fade screen if not started or has ended
        if self.timer and (endGame or gameTime == -1):
            overlay = Surface((Constants.GAMEWIDTH, Constants.GAMEHEIGHT), SRCALPHA, 32)
            overlay = overlay.convert_alpha()
            overlay.fill((100, 100, 100, 0.9 * 255))
            self.screen.blit(overlay, (0, 0))

        # Debug data for readout
        debug_data = {}
        if Constants.DEBUGMODE:
            debug_data = {
                "DEBUG": True,
                "FPS": int(self.clock.get_fps()),
                "MOLES": "{}/{}".format(Constants.MOLECOUNT, Constants.HOLEROWS * Constants.HOLECOLUMNS),
                "KEYS": "E[H]R[M]T[M0]Y[M+5]U[M-5]I[H0]O[H+5]P[H-5]"
            }

        # Display data readout
        data = self.score.label(timer=gameTime, debug=debug_data, size=(1.5 if endGame else 1))
        self.screen.blit(data, (5, 5))

        # Display hit/miss indicators
        if not endGame:

            # Hit indicator
            if hit:
                self.show_hit = time.get_ticks()
            if self.show_hit > 0 and time.get_ticks() - self.show_hit <= Constants.MOLEHITHUD:
                hit_label = self.text.get_label("BONK!", scale=3, color=(255, 50, 0))
                hit_x = (Constants.GAMEWIDTH - hit_label.get_width()) / 2
                hit_y = (Constants.GAMEHEIGHT - hit_label.get_height()) / 2
                self.screen.blit(hit_label, (hit_x, hit_y))
            else:
                self.show_hit = 0

            # Miss indicator
            if miss:
                self.show_miss = time.get_ticks()
            if self.show_miss > 0 and time.get_ticks() - self.show_miss <= Constants.MOLEMISSHUD:
                miss_label = self.text.get_label("Miss!", scale=2, color=(0, 150, 255))
                miss_x = (Constants.GAMEWIDTH - miss_label.get_width()) / 2
                miss_y = (Constants.GAMEHEIGHT + miss_label.get_height()) / 2
                self.screen.blit(miss_label, (miss_x, miss_y))
            else:
                self.show_miss = 0

        # Click to start indicator
        if self.timer and gameTime == -1:
            timer_label = self.text.get_label("Click to begin...", scale=2, color=(0, 255, 255))
            timer_x = (Constants.GAMEWIDTH - timer_label.get_width()) / 2
            timer_y = (Constants.GAMEHEIGHT - timer_label.get_height()) / 2
            self.screen.blit(timer_label, (timer_x, timer_y))

        # Time's up indicator
        if self.timer and endGame:
            timer_label_1 = self.text.get_label("Time's up!", scale=3, color=(0, 150, 255))
            timer_label_2 = self.text.get_label("Press space to restart...", scale=2, color=(0, 150, 255))

            timer_x_1 = (Constants.GAMEWIDTH - timer_label_1.get_width()) / 2
            timer_x_2 = (Constants.GAMEWIDTH - timer_label_2.get_width()) / 2

            timer_y_1 = (Constants.GAMEHEIGHT / 2) - timer_label_1.get_height()
            timer_y_2 = (Constants.GAMEHEIGHT / 2)

            self.screen.blit(timer_label_1, (timer_x_1, timer_y_1))
            self.screen.blit(timer_label_2, (timer_x_2, timer_y_2))

    def start(self):
        # UDP_IP = "127.0.0.1"
        # UDP_PORT = 1234

        # sock = socket.socket(socket.AF_INET, # Internet
        #                     socket.SOCK_DGRAM) # UDP
        # sock.bind((UDP_IP, UDP_PORT))
        
        self.clock = time.Clock()
        self.loop = True

        while self.loop:
            # Checking we have received the data from UDP!!!
            # data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
            # current_time = datetime.now()
            # current_time = current_time.strftime("%M:%S")
            # print("Time: ", current_time, "received message: %s" % data)
            
            # Do all events
            clicked, hit, miss = self.loop_events() 

            # Do all render
            self.loop_display(clicked, hit, miss)

            # Update display
            self.clock.tick(Constants.GAMEMAXFPS)
            display.flip()

    def run(self):
        self.start()
        quit()

