# AutoCon Pretraining Traffic Pilot

_Authorized validation-only direction pilot; not a confirmatory or held-out experiment._

---

## Scope

The source pool is ETTh1, Weather, and PEMS03. Traffic is excluded from pretraining and adapted using 10% of its original chronological training prefix. All evaluations use Traffic's validation partition; the target test split is closed.

Three arms share the same one-scan channel-count agnostic backbone (`d_model=16`, `d_ff=32`): random initialization, masked reconstruction pretraining, and masked reconstruction plus train-only AutoCon ACF relations. Each pretraining arm uses three source epochs and 512 maximum source windows per dataset. Target fine-tuning uses five epochs and 256 sampled target training windows.

This is a low-cost directional pilot. It may only decide whether the registered 898-unit protocol merits execution; it cannot support a performance, robustness, or publication claim.

## Superseded implementation check

The first pilot exposed an implementation defect: its reconstruction loss masked selected patch errors but the encoder still saw the unmasked input patch. It therefore was not a valid masked-patch pretraining test. Its numbers must not be interpreted as a negative transfer result or used to stop the revised channelwise protocol.

The defect was corrected by replacing selected patch tokens with a shared mask token before the Mamba scan. A new guarded cross-channel smoke passed on ETTh1, Weather, and PEMS03. The corrected channelwise Traffic pilot is recorded separately and is the only pilot eligible for a direction decision.

## Historical invalid result

The guarded pilot completed normally with no target test artifact. Traffic used `1052/10526` rows of the original chronological training prefix (10%), then evaluated the following validation block (`3509` rows). The three fine-tuning arms used the same target windows, architecture, seed, batch size, five epochs, and one frozen-encoder warm-up epoch.

| Arm | Best validation scaled MSE | Relative to random initialization |
| --- | ---: | ---: |
| Random initialization | 2.312522 | reference |
| Reconstruction pretraining | 2.340754 | +1.221% |
| AutoCon + reconstruction pretraining | 2.340742 | +1.221% |

The AutoCon-plus-reconstruction and reconstruction-only arms are effectively identical in this invalid pilot and both are worse than random initialization. The values are retained for audit only. Do not use them for an accuracy/robustness claim.

## Execution

The shared source encoder was trained only on ETTh1, Weather, and PEMS03 training prefixes; Traffic was excluded from the source pool. The GPU guard completed with no trigger and recorded `42C` maximum temperature, `32%` maximum fan, `74W` maximum power, and `907MB` maximum memory. These small pilot telemetry values are not an inference-speed result.

## Corrected channelwise result: stop

The corrected model applied a single shared Mamba module independently to each channel's patch sequence, preserving per-sensor temporal states while keeping parameter shapes independent of channel count. Selected patch tokens were replaced with a shared mask token before the Mamba scan. The source pool, target prefix, seed, fine-tuning budget, and target validation block matched the historical pilot.

| Arm | Best validation scaled MSE | Relative to random initialization |
| --- | ---: | ---: |
| Random initialization | .963972 | reference |
| Masked reconstruction pretraining | .983536 | +2.029% |
| Masked reconstruction + AutoCon pretraining | .985084 | +2.190% |

Both corrected pretrained arms are worse than random initialization, and AutoCon is also slightly worse than reconstruction-only. The corrected directional pilot fails. Stop the channelwise pretraining candidate and do not launch the 898-unit low-data/shift/missing-sensor matrix, tune pretraining duration/lambda/masking rate, or read any target test split.

The corrected guarded run completed without a temperature or fan trigger. It is a valid negative transfer result, unlike the historical pooled implementation check.

## Channel-context lag-pair result: stop

A final preregistered structural probe preserved the same single shared Mamba scan but injected a channel-count agnostic DeepSets context (per-patch channel mean and standard deviation) before scanning. The AutoCon source batches used continuous ACF lag-pair sampling, with `94.5--96.7%` global contrastive-active anchors across the three pretraining epochs. The target Traffic split and adaptation budget remained unchanged.

| Arm | Best validation scaled MSE | Relative to random initialization |
| --- | ---: | ---: |
| Random initialization | .971181 | reference |
| Channel-context masked reconstruction | .983376 | +1.256% |
| Channel-context masked reconstruction + AutoCon | .980755 | +0.985% |

Lag-pair AutoCon improves over reconstruction-only by `.266%`, showing that the relation objective is active and directionally useful within this encoder. It nevertheless remains worse than random initialization. This rejects the final single-scan, channel-count agnostic pretraining candidate under the registered direction pilot. Do not launch the 898-unit matrix or search context rank, context initialization, source composition, mask rate, lambda, or epoch count after this result.

The guarded run completed with no trigger; GPU telemetry peaked at `54C`, `32%` fan, `242W`, and `1089MB` memory.

## Predictive next-patch lag-pair result: stop

The final objective-alignment probe replaced masked reconstruction with a next-patch task: 72 observed points predicted the next 24-point normalized patch under Smooth L1 loss. The same channel-count agnostic context, source exclusion, continuous ACF lag-pair batches, and target 10% Traffic adaptation protocol were used. All pretrained checkpoints restored 28 compatible shared encoder keys into the 96-to-96 target forecaster; no target test metric was produced.

| Arm | Best validation scaled MSE | Relative to random initialization |
| --- | ---: | ---: |
| Random initialization | .967596 | reference |
| Predictive pretraining | .993730 | +2.701% |
| Predictive pretraining + AutoCon | .992696 | +2.594% |

The predictive task activates AutoCon lag-pair relations (`95.7--96.7%` global active anchors) and AutoCon improves over predictive-only pretraining by `.104%`. Both arms remain worse than random initialization. This rejects the prediction-aligned pretraining mechanism in the registered Traffic 10% direction pilot. Stop the pretraining line as a source of a new SCI-Q1 main claim; do not run the full low-data/robustness matrix or tune the candidate after this result. The GPU guard completed without a trigger, peaking at `56C`, `32%` fan, `250W`, and `1089MB`.

## Physical-interval conditioned result: stop

The final pilot added registered physical sampling intervals (`ETTh1=60`, `Weather=10`, `PEMS03=5`, target Traffic=60 minutes) to the shared context while retaining next-patch Smooth L1 pretraining and continuous ACF lag-pair batches. Relation anchors remained active at roughly `95.7--96.7%`, but the target validation results were:

| Arm | Best validation scaled MSE | Relative to random initialization |
| --- | ---: | ---: |
| Random initialization | .970218 | reference |
| Predictive pretraining | .974532 | +0.444% |
| Predictive pretraining + AutoCon | .976465 | +0.644% |

The physical time-scale condition does not rescue transfer. This is the final registered direction pilot; the 898-unit matrix is stopped and no further context, source composition, masking, interval, or loss tuning is permitted.

## Evidence

- [Pilot summary](../../artifacts/exploratory/mambats_autocon_pretraining_traffic_pilot_20261002/summary.json)
- [Corrected channelwise summary](../../artifacts/exploratory/mambats_autocon_pretraining_traffic_pilot_20261002_channelwise_masked/summary.json)
- [Channel-context lag-pair summary](../../artifacts/exploratory/mambats_autocon_pretraining_traffic_pilot_20261002_channelwise_context_lagpairs/summary.json)
- [Predictive lag-pair summary](../../artifacts/exploratory/mambats_autocon_pretraining_traffic_pilot_20261002_channelwise_nextpatch/summary.json)
- [AutoCon pretraining log](../../artifacts/exploratory/mambats_autocon_pretraining_traffic_pilot_20261002/logs/autocon_pretrain.log)
- [GPU guard event](../../artifacts/exploratory/mambats_autocon_pretraining_traffic_pilot_20261002/gpu_guard/event.json)
