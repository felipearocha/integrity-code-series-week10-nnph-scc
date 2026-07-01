# Contributing

Thank you for considering a contribution. This is a research-grade package; technical rigor is more important than feature velocity.

## Ground rules

1. **Cite your sources.** Every new physical mechanism, coefficient, or correlation must include a `[SOURCE: Author Year]` tag in the code, ideally peer-reviewed. If a value is engineering judgment, mark it `[ASSUMED]`.
2. **No FFS claims.** This is a research tool. Pull requests that imply use as a fitness-for-service authority will be declined.
3. **Tests must pass.** `pytest tests/ -q` must be green before opening a PR.
4. **No drift in benchmarks.** `python -m validation.benchmarks` must reproduce the textbook values (Faraday 1.163 mm/yr, Tada K_I within 15%, hoop stress band).

## Workflow

1. Fork the repository.
2. Create a topic branch: `git checkout -b feature/short-description`.
3. Make commits using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):
   - `feat:` new physics or mechanism
   - `fix:` bug or incorrect coefficient
   - `docs:` documentation only
   - `test:` adding or correcting tests
   - `refactor:` no functional change
   - `chore:` build / CI / housekeeping
4. Run the full pipeline locally before pushing:
   ```bash
   pip install -e .[dev]
   pytest tests/ -q
   python run_all.py
   ```
5. Open a pull request against `main`. Fill out the PR template.
6. CI must pass on all supported Python versions (3.11, 3.12).

## Adding a new damage mechanism or correlation

When you add new physics, please include:

- A docstring with the governing equation in plain-text or LaTeX.
- A `[SOURCE]` tag with author + year (and DOI when available).
- At least one analytical benchmark in `validation/benchmarks.py`.
- At least three unit tests covering nominal, boundary, and degenerate input.
- An entry in `equations.html` if the equation deserves visible documentation.

## Reporting bugs

Open an issue using the **Bug report** template. Include:
- Python version (`python --version`).
- OS.
- Minimal reproducing snippet.
- Expected vs. observed numerical result.

## Code style

- PEP 8 with line length 110.
- Run `ruff check .` before pushing.
- Use NumPy-style docstrings for public functions.
- Avoid em dashes in user-facing output (legacy stylistic constraint).

## Author

Felipe Rocha
