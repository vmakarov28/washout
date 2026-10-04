"""The aircraft in three dimensions: B-rep solids and STEP files.

`bspline` is numpy and scipy only, like the rest of the program: it
fits each part's smooth surfaces as tensor-product B-splines and
measures how far they are from the printed geometry. `brep` and
`export` need the OpenCASCADE bindings (`pip install -e .[cad]`, which
brings `cadquery-ocp`), and nothing in the search imports them -- the
CAD export runs at export time and in `check --conformance`, never
inside the loop, exactly as tiers 1 and 2 do. See docs/ROADMAP-CAD.md.
"""


def available() -> bool:
    """Whether the OpenCASCADE bindings are installed."""
    try:
        import OCP  # noqa: F401
    except ImportError:
        return False
    return True
