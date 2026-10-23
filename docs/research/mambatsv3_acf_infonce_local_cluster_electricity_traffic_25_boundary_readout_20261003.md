# Electricity and Traffic 25% Boundary Readout

This matrix was registered as a validation-only applicability boundary. No
held-out test metrics were produced, and no checkpoint was selected after
seeing the results.

| Target | Seed | Active validation | Exact-null validation | Active vs null |
| --- | ---: | ---: | ---: | ---: |
| Electricity | 5550 | 8.072209 | 7.910865 | +2.040% |
| Electricity | 5551 | 8.076884 | 7.937118 | +1.761% |
| Electricity | 5552 | 8.108902 | 7.900408 | +2.639% |
| Traffic | 5560 | 0.825434 | 0.822318 | +0.379% |
| Traffic | 5561 | 0.818671 | 0.816161 | +0.307% |
| Traffic | 5562 | 0.814301 | 0.813476 | +0.101% |

Positive values mean active is worse. The active arm lost all six paired
validation comparisons. This is a retained negative transfer boundary for the
current source-pool and 25%-label protocol; it is not folded into the positive
PEMS-BAY, METR-LA, or Weather confirmation claims.

The result argues against a universal claim. The paper should state that
train-only ACF contrastive pretraining is beneficial for some source/target
periodic matches, while high-dimensional Electricity and Traffic require a
different alignment or should be reported as failure domains.
