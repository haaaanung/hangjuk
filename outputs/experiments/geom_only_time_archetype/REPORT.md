# Geom-only forward-time route-archetype ensemble

Past labels define smoothed joint posteriors over fishing-ground tiles and trajectory-behavior bins. Prediction still accepts geom only.

## June metrics

| variant                         |   rows |   accuracy |   macro_f1 |
|:--------------------------------|-------:|-----------:|-----------:|
| v1_existing                     | 773403 |   0.700474 |   0.680533 |
| spatial_previous                | 773403 |   0.701378 |   0.680939 |
| archetype_only                  | 773403 |   0.549710 |   0.332268 |
| may_selected_archetype_ensemble | 773403 |   0.702870 |   0.681547 |

## Final class metrics

| class   |   precision |   recall |   f1-score |       support |
|:--------|------------:|---------:|-----------:|--------------:|
| 연안복합    |    0.715325 | 0.772481 |   0.742805 | 331353.000000 |
| 자망      |    0.744973 | 0.662446 |   0.701290 | 282165.000000 |
| 채낚기연승낚시 |    0.649079 | 0.742731 |   0.692754 |  45847.000000 |
| 통발      |    0.594154 | 0.584603 |   0.589340 | 114038.000000 |

## Frozen May rule

|   archetype_weight |   spatial_weight |   coastal |   gillnet |   longline |     trap |   macro_f1 |   accuracy |
|-------------------:|-----------------:|----------:|----------:|-----------:|---------:|-----------:|-----------:|
|           0.050000 |         0.200000 |  1.400000 |  1.300000 |   0.900000 | 1.100000 |   0.676774 |   0.698165 |

| variant                         |   rows |   accuracy |   macro_f1 |
|:--------------------------------|-------:|-----------:|-----------:|
| may_selected_archetype_ensemble | 761846 |   0.697827 |   0.675603 |

This is forward-time evaluation on a mostly recurring vessel population, not unseen-vessel evaluation.
