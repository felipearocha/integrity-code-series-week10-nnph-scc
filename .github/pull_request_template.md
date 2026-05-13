## Summary

<!-- What does this PR change and why? Keep it to 2-3 sentences. -->

## Type of change

- [ ] `feat`     New physics, mechanism, or capability
- [ ] `fix`      Bug or incorrect coefficient
- [ ] `docs`     Documentation only
- [ ] `test`     Adding or correcting tests
- [ ] `refactor` No functional change
- [ ] `chore`    Build / CI / housekeeping

## Physics references

<!-- For new mechanisms or corrections: list the paper, standard, or report. DOI when available. -->

## Verification checklist

- [ ] `pytest tests/ -q` is green locally
- [ ] `python -m validation.benchmarks` reproduces textbook values
- [ ] `python run_all.py` completes end-to-end
- [ ] `ruff check .` is clean
- [ ] New `[SOURCE]` or `[ASSUMED]` tags added where applicable
- [ ] Updated `CHANGELOG.md` (Unreleased section)
- [ ] Updated `equations.html` if a new governing equation was added

## Notes for the reviewer

<!-- Anything unusual: numerical regressions, parameter recalibration, performance impact. -->
