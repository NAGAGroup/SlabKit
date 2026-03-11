# Observation: DocWriter step limits on large doc rewrites

**Observed:** 2026-03-10, session remove-allocator-registry-singleton, subtask 07

DocWriter hit its maximum step limit twice when given a single task covering all 6 sections of a ~300-line design document. The first run completed §2 + partially §3 (7 edits). The second run completed §3 remainder + partially §4 (3 edits). HeadWrench completed §4–§6 directly.

**Pattern:** For design doc updates with many surgical edits across 5+ sections, break the task into two DocWriter delegations:
1. First run: §1–§3 (conceptual/architecture sections)
2. Second run: §4–§6 (file state, remaining work, out-of-scope)

This keeps each delegation bounded to ~3-4 sections and avoids step-limit truncation.
