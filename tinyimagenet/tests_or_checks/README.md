# Verification location

These tests MUST run on Kaggle. They were not executed on the laptop.

1. `python -m pytest tests_or_checks -q` checks NC identities, protocol values,
   deterministic splits, frozen-manifest rejection, exact lag pairing and
   degenerate correlation handling.
2. `python scripts/run_smoke.py --data-root <detected-root>` checks all four
   methods on actual Tiny ImageNet smoke subsets, including teacher/buffer
   persistence and interrupted versus uninterrupted results.
3. Results from these engineering checks live only in `smoke/` or pytest
   temporary directories. They cannot enter scientific analyses.

Local checks are limited to source inspection, JSON/notebook structure,
file inventories, checksums and ZIP integrity. Runtime correctness remains
unverified until these Kaggle gates pass. Never call the package validated
solely because the source archive was created.
