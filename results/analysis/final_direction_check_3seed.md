# Final Direction Check: CIFAR-100 Seeds 42/43/44

| Method | Avg. Acc. | Forgetting | BWT | ETF--Forget Corr. |
|---|---:|---:|---:|---:|
| LwF | 0.6801 $\pm$ 0.0107 | 0.0117 $\pm$ 0.0100 | -0.0107 $\pm$ 0.0091 | -- |
| LwF+ETF Strong | 0.6639 $\pm$ 0.0100 | 0.0304 $\pm$ 0.0017 | -0.0300 $\pm$ 0.0022 | 0.6118 $\pm$ 0.2695 |

## Interpretation

- LwF remains the stronger accuracy/forgetting baseline across three seeds.
- LwF+ETF Strong is close in average accuracy, but it does not beat LwF on forgetting.
- The ETF drift/forgetting correlation is positive on all three hybrid seeds, but variable; the three-seed mean is moderate rather than consistently above 0.8.
- Best paper framing: ETF geometry is a diagnostic and auxiliary geometric regularizer, not currently a standalone SOTA replacement.
