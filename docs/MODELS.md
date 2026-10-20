# Model setup

The public repository includes the application and model definitions. It does not bundle trained signature-verification checkpoints or signature datasets.

You can open the interface, explore the workflows and inspect preprocessing without weights. Comparisons require compatible checkpoints.

## Inference checkpoints

Run the application from the repository root. Place the checkpoint files under `models/`:

| Engine | Required files | Compatible model |
| --- | --- | --- |
| General v1 | `models/last_model_cedar_selected_utsig.pth` | `SiameseNetworkLegacy` in `src/models/siamese_network_legacy.py` |
| Low-resolution scan v2 | `models/archive_v2_binary_head.pth`, `models/archive_v2_gray_head.pth` | `ArchivePairHead` with a shared MobileNetV3 encoder |
| High-resolution mono v3 | `models/utsig_v3_binary_head.pth` | `PairHead` with a MobileNetV3 encoder |

The low-resolution and monochrome engines also instantiate torchvision's pretrained MobileNetV3 Small backbone. Its first use may download the weights if they are not already cached. The general engine does not require that backbone.

Use checkpoints you have permission to use. There is currently no public checkpoint download attached to this repository. Do not rename an arbitrary model file to match these names: the architecture and calibration must also match.

If you only have the General v1 checkpoint, select **Advanced settings → General v1** so automatic routing does not choose an engine whose files are absent.

## Calibration

`best_threshold.json` contains the General v1 prototype's distance threshold and uncertainty bands. The other engines define their score boundaries in `verify_archive_v2.py` and `verify_utsig_v3.py`.

The included values document the current prototype configuration. They are not a claim of universal reliability. Recalibrate and evaluate against the model and data used in your own deployment.

`calibrate_legacy_threshold.py` targets the active legacy checkpoint. Inspect its dataset configuration and validation protocol before running it.

## Training and checkpoint compatibility

There are two distinct Siamese architectures in the source:

- `SiameseNetworkLegacy` is the application's default General v1 architecture and produces 64-dimensional embeddings.
- `SiameseNetwork` is used by the newer `train.py` / `evaluate.py` pipeline and produces 128-dimensional, L2-normalized embeddings.

Running `python train.py` does not automatically provision the application's default checkpoint. To use a newly trained model, explicitly align the architecture, checkpoint, preprocessing and thresholds in the inference configuration, then validate that complete combination.

The `scripts/` directory also contains MobileNet transfer-learning, calibration and evaluation experiments. Read each script's arguments and dataset expectations before use. Several scripts are dataset-specific experiments rather than a single automated end-to-end training command.

## Data layout

Datasets are supplied separately. General dataset-loading code and the experiment scripts describe their expected directory layouts. Portable defaults live under `dataset/`; adjust each script's dataset configuration to match your local copy.

The archive and skilled-forgery experiments also accept local dataset locations through `SIGNIFY_ARCHIVE_DATASET` and `SIGNIFY_SKILLED_DATASET`, respectively.

Dataset names such as CEDAR and UTSig identify the research workflows supported by the source. Their appearance here does not grant a license to obtain or redistribute those datasets.

## Screenshot examples

The README screenshots were captured with an existing local General v1 checkpoint and artificial pen-stroke inputs from `examples/synthetic/`. They illustrate the UI and actual example inference. The displayed confidence is a score mapping used by the application, not a benchmark accuracy or a calibrated identity probability.
