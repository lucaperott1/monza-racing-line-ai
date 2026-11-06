"""Monza's corners, as distances (metres) along the centre line from the start line.
Found from the track's curvature peaks."""

CORNERS = [
    # name,          start, end
    ("Rettifilo",      880, 1030),
    ("Curva Grande",  1250, 1660),
    ("Roggia",        2100, 2260),
    ("Lesmo 1",       2480, 2640),
    ("Lesmo 2",       2840, 2930),
    ("Ascari",        3920, 4180),
    ("Parabolica",    5100, 5500),
]


def corner_at(s):
    """Name of the corner at distance s, or the straight/stretch before it."""
    for name, a, b in CORNERS:
        if a <= s <= b:
            return name
    for name, a, b in CORNERS:
        if s < a:
            return f"before {name}"
    return "the final straight"
