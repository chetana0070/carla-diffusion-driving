# Experiment Matrix

| ID | Policy | Temporal input | Action chunk | DAgger | Residual SAC | Purpose |
|---|---|---:|---:|---:|---:|---|
| E00 | Expert agent | N/A | N/A | N/A | N/A | Upper-bound and collection audit |
| E01 | Behavioral cloning | No | No | No | No | Original-style baseline |
| E02 | Behavioral cloning | Yes | Yes | No | No | Isolate temporal-context effect |
| E03 | Diffusion | Yes | Yes | No | No | Isolate diffusion effect |
| E04 | Diffusion | Yes | Yes | Yes | No | Measure corrective-data value |
| E05 | Diffusion | Yes | Yes | Yes | Yes | Final bounded refinement |

Every learned condition is evaluated using seeds 17, 29, and 43 on the same
frozen route suite.

## Required ablations

- image only vs. image plus scalar state;
- single frame vs. four-frame history;
- one action vs. sixteen-action chunk;
- temporal BC vs. temporal diffusion;
- before vs. after DAgger;
- before vs. after residual SAC.

