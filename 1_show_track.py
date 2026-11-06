"""1 - Draw Monza with its corners named -> results/1_monza_track.png"""
import os

from sim import plotting as P
from sim.track import get_track

if __name__ == "__main__":
    P.dark_style()
    path = P.track_figure(get_track())
    if os.name == "nt":
        os.startfile(path)
