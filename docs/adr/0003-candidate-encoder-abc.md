# ADR-0003: Enforce the screen-encoder contract with an ABC

- Status: Accepted
- Date: 2026-06-22
- Related: ADR-0001

## Context

`CandidateEncoder`'s shared plumbing (`candidate_action_vectors`,
`valid_action_mask`) calls four primitives — `valid_action_candidates`,
`encode_state`, `encode_action`, `action_key` — that the base does **not** define;
concrete encoders supply them (the Template Method pattern). The contract lived
only in the class docstring: it was an informal interface. A subclass that forgot
a primitive, or a direct instantiation of the base, failed only at *call time*
with an `AttributeError`. This was also inconsistent with the agent side, where
`ScreenAgent` is a real `abc.ABC`.

## Decision

Make `CandidateEncoder` inherit `abc.ABC` and mark the four primitives
`@abstractmethod`.

## Consequences

- The contract is enforced at instantiation (fail-fast), documented in code, and
  visible to type checkers/IDEs.
- Consistent with the `ScreenAgent` interface in `agents/base.py`.
- No behavior change: all five concrete encoders already implement the four
  methods, so they remain instantiable; only direct instantiation of the abstract
  base is now blocked.
