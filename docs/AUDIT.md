# Repository audit

## Critical

- A plaintext MongoDB credential was committed in the legacy README. It has been removed; rotate it immediately because working-tree removal does not remove Git history.
- The legacy transformer resampled the test split, producing invalid evaluation. The new pipeline fits preprocessing and applies SMOTETomek only to training data.
- Legacy evaluation concatenated train and test data. The replacement keeps test data untouched through selection and threshold setting.

## High

- The repository contains a copied Windows/Python runtime (`DLLs`, `conda-meta`, executables and standard-library files). They are environment artefacts, not application source. They are ignored; remove them from Git with `git rm -r --cached` after reviewing the index. Use `git-filter-repo` separately for historical cleanup—no history rewrite was performed.
- Training was MongoDB/AWS dependent. The supported path is local CSV; cloud storage is optional and disabled by default.

## Medium

- Data validation had an unimplemented zero-variance function and KS drift treated all columns alike. The replacement reports data quality and numeric-only KS drift with FDR correction.
- Training lacked deterministic splits, baseline comparison, probability metrics and traceable metadata/versioning.

## Low

- Dependencies included stale transitive/runtime packages. They were replaced by bounded direct dependencies.
