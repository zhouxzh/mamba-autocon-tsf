# Official PatchTST Weather H192 Extension Plan

Pinned source: `yuqinie98/PatchTST` commit
`204c21efe0b39603ad6e2ca640ef5896646ab1a9`, imported from the official
supervised PatchTST model through the local chronological adapter.

Weather H192 uses seeds 5450--5452, a 10% target prefix, `seq_len=96`, five
epochs, and 64 training windows. Validation result: active `4.487684` versus
official PatchTST `4.647161` (`-3.432%`, 3/3). The test extension may read each
validation-selected official checkpoint once and may not alter candidate,
source, split, or adapter parameters.
