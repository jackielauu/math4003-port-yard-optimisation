| Scenario | Data | Ops | Multi-op assets | Iters | Feas / Opt cuts | Stopped | Master total (s) | Master max (s) | 10 s hits | Master @10 cuts (s) | Master @20 cuts (s) | Median nnz feas / opt | Median opt-cut coef ratio | Final gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1_system_scale_L | Old data | 8 | 1 | 24 | 18 / 5 | converged | 0.05 | 0.00 | 0 | 0.001 | 0.004 | 1 / 121 | 2.0e+00 | -0.00% |
| F1_system_scale_L | New data | 30 | 9 | 56 | 12 / 44 | wall_cap | 245.57 | 10.00 | 22 | 0.638 | 0.087 | 13 / 982 | 1.3e+00 | 10.99% |
| F1_system_scale_L | New data, no-overlap constraint removed | 30 | 9 | 60 | 59 / 1 | max_iters | 0.62 | 0.03 | 0 | 0.002 | 0.010 | 6 / 976 | 1.3e+00 | 33.13% |
| F1_system_scale_M | Old data | 12 | 1 | 60 | 33 / 27 | max_iters | 2.13 | 0.12 | 0 | 0.006 | 0.012 | 3 / 199 | 1.3e+00 | 0.51% |
| F1_system_scale_M | New data | 60 | 16 | 53 | 37 / 16 | wall_cap | 236.34 | 10.00 | 16 | 0.170 | 0.235 | 19 / 2176 | 2.0e+00 | 21.12% |
| F1_system_scale_M | New data, no-overlap constraint removed | 60 | 16 | 60 | 60 / 0 | max_iters | 0.76 | 0.02 | 0 | 0.019 | 0.005 | 4 / – | – | –% |
| F4_operational_conflict_M | Old data | 12 | 1 | 60 | 28 / 32 | max_iters | 3.59 | 0.29 | 0 | 0.006 | 0.008 | 2 / 198 | 1.3e+00 | 0.37% |
| F4_operational_conflict_M | New data, no-overlap constraint removed | 60 | 17 | 60 | 60 / 0 | max_iters | 1.74 | 0.05 | 0 | 0.020 | 0.031 | 6 / – | – | –% |
| F1_system_scale_H | Old data | 16 | 1 | 60 | 25 / 35 | max_iters | 3.52 | 1.41 | 0 | 0.016 | 0.014 | 3 / 241 | 2.0e+00 | 1.98% |
| F1_system_scale_H | New data | 100 | 24 | 60 | 19 / 41 | max_iters | 128.79 | 5.94 | 0 | 1.278 | 1.562 | 3042 / 4115 | 1.5e+04 | 26.97% |
| F1_system_scale_H | New data, no-overlap constraint removed | 100 | 24 | 60 | 60 / 0 | max_iters | 1.71 | 0.05 | 0 | 0.009 | 0.008 | 6 / – | – | –% |
