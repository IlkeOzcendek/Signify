# Third-party notes

The root MIT License covers original Signify source code and original project assets included in this repository.

## Software dependencies

Signify uses PyTorch, torchvision, OpenCV, Streamlit, streamlit-cropper, NumPy, pandas, scikit-learn, Plotly, SQLAlchemy, Alembic, ReportLab, Pillow and supporting Python packages. These packages are installed separately and retain their respective licenses.

## Pretrained backbones

The MobileNet-based engines request torchvision's pretrained MobileNetV3 Small weights. These weights are retrieved separately or read from the local cache. Their terms are not replaced by Signify's source-code license.

## Signature datasets and trained checkpoints

Research code references CEDAR, UTSig and locally organized signature collections. No signature dataset, personal signature records or trained checkpoint binaries are included in this distribution. Users must obtain the appropriate data and permissions independently.

## Demonstration assets

The included signature marks in `examples/synthetic/` are artificial demonstration inputs. The screenshots use those inputs and contain no real person's signature or registered identity. The SVG illustrations are static project graphics.
