# Project Workflow

## Guiding Principles

1.  **The Plan is the Source of Truth:** All work must be tracked in `plan.md`.
2.  **The Tech Stack is Deliberate:** Changes to the tech stack must be documented in `tech-stack.md` *before* implementation.
3.  **Test-Driven Development:** Write unit tests before implementing functionality.
4.  **High Code Coverage:** Aim for >80% code coverage for all modules.
5.  **User Experience First:** Every decision should prioritize developer UX, determinism, and safety.
6.  **Non-Interactive & CI-Aware:** Prefer non-interactive commands. Use `CI=true` for watch-mode tools.
7.  **Financial & Operational Safety:** No remote training dispatch or external cloud resource creation is executed without explicit dry-run flags and verification.

## Task Workflow

All tasks follow a strict lifecycle:

### Standard Task Workflow

1.  **Select Task:** Choose the next available task from `plan.md` in sequential order.
2.  **Mark In Progress:** Before beginning work, edit `plan.md` and change the task from `[ ]` to `[~]`.
3.  **Write Failing Tests (Red Phase):**
    - Create a new test file for the feature or module.
    - Write unit tests clearly defining expected behavior, type contracts, and acceptance criteria.
    - **CRITICAL:** Run the tests and confirm that they fail as expected.
4.  **Implement to Pass Tests (Green Phase):**
    - Write the minimum amount of code necessary to make the failing tests pass.
    - Run the test suite again and confirm that all tests now pass.
5.  **Refactor (Optional but Recommended):**
    - Refactor implementation and test code to improve clarity, eliminate duplication, and enhance performance without changing external behavior.
    - Rerun tests to ensure they still pass.
6.  **Verify Coverage:** Run coverage reports: `pytest --cov=symforge --cov-report=term-missing`. Target: >80% coverage for new code.
7.  **Document Deviations:** If implementation differs from tech stack:
    - **STOP** implementation.
    - Update `tech-stack.md` with new design.
    - Add dated note explaining the change.
    - Resume implementation.
8.  **Commit Code Changes:** Stage code changes and commit with conventional commit format (e.g., `feat(synth): implement typed hole expansion`).
9.  **Attach Task Summary with Git Notes:**
    - Step 9.1: Obtain commit hash (`git log -1 --format="%H"`).
    - Step 9.2: Draft task summary.
    - Step 9.3: Attach note: `git notes add -m "<note content>" <commit_hash>`.
10. **Record Task Commit SHA in Plan:** Update task status in `plan.md` from `[~]` to `[x]` and append the short commit SHA.
11. **Commit Plan Update:** Stage and commit `plan.md` (`conductor(plan): Mark task '...' as complete`).

### Phase Completion Verification and Checkpointing Protocol

1.  **Announce Protocol Start:** Announce completion of the phase.
2.  **Ensure Test Coverage:**
    - Identify modified files: `git diff --name-only <previous_checkpoint_sha> HEAD`.
    - Ensure every Python code file has a matching test file in `tests/`.
3.  **Execute Automated Tests:**
    - Execute test command: `pytest -v tests/`.
4.  **Actionable Verification Report:**
    - Produce structured summary of automated test outputs and verification instructions.
5.  **Attach Verification Report to Git Notes:**
    - Attach report to checkpoint commit with `git notes`.
6.  **Update Plan with Checkpoint:**
    - Update phase header with `[checkpoint: <sha>]`.
    - Commit `plan.md` with `conductor(plan): Mark phase '<PHASE NAME>' as complete`.

### Quality Gates

- [ ] All tests pass (`pytest -v tests/`)
- [ ] Code coverage >80%
- [ ] Code follows Google Python Style Guide & Ruff rules (`ruff check .`)
- [ ] Static typing verified (`mypy symforge`)
- [ ] Asynchronous calls enforce hard timeouts and iteration caps
- [ ] All public APIs documented with Google-style docstrings
