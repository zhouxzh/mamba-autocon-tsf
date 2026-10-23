# Weather 25% Label Gate Readout

| Seed | Active validation | Exact-null validation | Active vs null |
| --- | ---: | ---: | ---: |
| 5590 | 2.517228 | 2.516675 | +0.022% |
| 5591 | 2.555773 | 2.555105 | +0.026% |
| 5592 | 2.529494 | 2.530351 | -0.034% |

The predeclared all-seed validation gate failed: active is worse in two of
three seeds. No Weather 25% test reader was launched, so no held-out test
metric exists. The source policy was fixed from the Weather 10%-prefix
selection and was not reselected after this result.

This is a retained label-regime boundary. The current method has held-out
confirmation for Weather at 10% labels, but not at 25%; the manuscript must
not imply a universal monotone benefit as target labels increase.
