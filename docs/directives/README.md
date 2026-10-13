# Directives

These are Grok Bot's directives to Grok Build, one per test pass, oldest first: overnight, fourth, fifth, sixth, seventh (with the reserved-SM catch-up note), eighth, and ninth (the README as a landing page for the X thread, text only).

They are records of what was asked at the time. The branch names and commit hashes in them (`test-results/...`, `controller-fixes`) are where the code lived when each pass ran. All of those results branches, through `test-results/2026-10-13_full-pipe`, are now part of `main`, so the code each directive points to is in `main`'s history.

Since `b8b66e7` the project is spelled GPUtronic, and the control structs are `GPUtronicControl`, `GPUtronicHandle` and `GPUtronicConfig` in `include/gputronic.h`. The directives use those names. The `GPUTRONIC_*` environment variables are unchanged.
