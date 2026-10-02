# Public release structure

```text
.
├── README.md
├── README_zh.md
├── PUBLIC_RELEASE_NOTES.md
├── Dockerfile
├── run_ssn.py
├── run_xpinn.py
├── run_pipeline.py
├── ssn/
├── xpinn/
├── interface/
├── tests/
└── figures/
    └── showcase/
```

The public release intentionally excludes the private repository's `docs/`, `data/`, `experiments/`, `logs/`, `legacy/`, agent configuration, roadmaps, and unpublished research notes.

The main reusable entry point is `run_ssn.py`. `run_xpinn.py` and `run_pipeline.py` expose the experimental SSN-guided XPINN workflow.
