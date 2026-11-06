# Comparative SBDD benchmark summary

Corpus: 3D-SynTree V2 test-split constructions (33 forward-replay-verified target ligands of the frozen test manifest); the TargetDiff/DiffSBDD/SyntheMol baselines await operator-exported structures (their checkpoints/environments are external) — recorded as not_provided, never fabricated
Frozen targets: 33 (reference method: `syntree`)

| method | structures | validity | Fsp3 | diversity  | retro[stock_aligned]  | retro[independent_external] |
|---|---|---|---|---|---|---|
| syntree | 33 | 1.000 [1.000, 1.000] | 0.536 [0.437, 0.638] | 0.875 | 0.250 [0.000, 0.500] | 0.667 [0.417, 0.917] |

> Retrosynthesis tracks are reported separately (#44); no merged headline solve rate exists.
