# ADR-0005: Adopt ruff for formatting and linting

- Status: Accepted
- Date: 2026-06-22

## Context

The project had no formatter or linter. Indentation had drifted to two styles
(2-space in the encoders and candidate agents, 4-space elsewhere), and there was
no enforcement to keep style consistent or to catch dead code.

## Decision

Add `ruff` as a dev dependency with a conservative, behavior-neutral config in
`pyproject.toml`:

- `line-length = 100`, `target-version = "py311"`.
- Lint `select = ["E", "F", "I"]` (pycodestyle errors, pyflakes, isort),
  `ignore = ["E501"]` — the formatter owns line length and leaves unsplittable
  strings long.
- Apply `ruff format` + import-sort across `src` + `tests` in one mechanical
  commit (normalizing all indentation to 4-space), and remove the one genuinely
  unused local it surfaced.

Opinionated modernization rules (`UP`, `B`) are intentionally **excluded** for now
to keep the change behavior-neutral; they can be enabled later as a separate,
reviewed pass.

## Consequences

- An enforceable, green baseline: `ruff check` and `ruff format --check` pass.
- One large but purely mechanical reformat commit (whitespace + import order);
  review with `git diff --ignore-all-space`.
- ~30 `UP`/`B` modernization suggestions remain unaddressed by design.
