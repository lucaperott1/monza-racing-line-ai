"""3 - Watch the best AI so far drive a lap of Monza in a game window.

Keys: SPACE pause | R restart | + / - zoom | B benchmark line | 1-4 speed | ESC quit
(Optional: --name seed2 to watch an AI trained under another name.)
"""
import argparse

from sim import viewer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    viewer.run(ap.parse_args().name)
