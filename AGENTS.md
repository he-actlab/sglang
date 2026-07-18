# Speculative-decoding fork agent instructions

## Repository boundary

This repository is the SGLang **implementation fork** for the
`parallel_sd_inference` research project. Keep runtime code and tests here. All
research-project documentation lives in the `parallel_sd_inference` repository,
which is the parent directory in the combined checkout used by this project.

Do not create project design notes, review reports, bug/TODO trackers,
experiment records, result tables, provenance records, runbooks, paper drafts,
or status summaries in this fork. If the research repository is not available
at `..`, locate its checkout before writing documentation; do not create a local
substitute in this repository.

`docs_new/` contains upstream SGLang product documentation and has its own
`AGENTS.md`. Modify it only when the project owner explicitly requests an
upstream/public SGLang documentation change. It is not the documentation home
for the research project.

## Where to record project information

Paths below are relative to the `parallel_sd_inference` repository root (`..`
in the combined checkout):

The canonical routing table lives in the parent repository's `AGENTS.md`
(`../AGENTS.md` in the combined checkout) — do not duplicate it here. The one
fork-side rule: code changes land on `spec-colo`; documentation about them lands
in the parent repository per that table.

The complete recording, status, result-certification, and cross-document rules
are in `../AGENTS.md`. Follow them whenever a fork change needs to be recorded.

## Code-review findings

- Record new findings in `../design/KNOWN-BUGS-AND-TODOS.md` as `REPORTED`
  until an independent reproducer or archived evidence confirms them.
- Cite the exact fork commit and code paths. Keep the implementation fix and
  regression tests in this repository.
- After a fix, update the tracker with the fixing commit, regression gate, and
  impact on historical results. Do not duplicate the tracker inside this fork.
