# docs

- **[ROADMAP.md](ROADMAP.md)** — what is left between here and a flying
  wing nobody touched, in value order, with the evidence for each item.
  Start here. It supersedes the old `next-session-prompt.md` and carries
  its hard constraints forward unchanged.
- **[how-washout-designs-aircraft.html](how-washout-designs-aircraft.html)**
  — the long-form narrative: the pipeline, every model in it, and every
  bug found so far with what it cost. Open it in a browser.

Elsewhere in the repo, and worth reading in this order:

1. `README.md` — the idea, the print-orientation trick, and the gates.
2. `CLAUDE.md` — the non-negotiable rules. Short, and each line is there
   because breaking it cost a session.
3. `results/README.md` — the fleet as searched, and why its masses are
   not the masses.
4. `data/polars/README.md` — the only measured data in the project, and
   what is wrong with using both files together.
5. `tests/test_structure_in_search.py` — the house style for pinning a
   fix. A bug found becomes a test, with the physical reason in the
   docstring.
