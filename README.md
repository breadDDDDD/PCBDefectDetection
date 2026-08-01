# PCB Defect Detection with YOLOv5

Detecting defects on printed circuit boards across 6 defect classes using YOLOv5, with evaluation of the trained model.

## Overview

This project fine-tunes the YOLOv5s object detector on a PCB defect dataset to locate and classify manufacturing defects in board images. The workflow — data setup, training, validation, results visualization, and weight export — is contained in a single Colab-style notebook, with the trained weights (`best.pt`) included in the repo.

## Defect Classes

| ID | Class |
|----|-------|
| 0 | `missing_hole` |
| 1 | `mouse_bite` |
| 2 | `open_circuit` |
| 3 | `short` |
| 4 | `spur` |
| 5 | `spurious_copper` |

## Repository Contents

| File | Description |
|------|-------------|
| `YOLO_V5_PCB_Detection.ipynb` | Main notebook — full end-to-end pipeline |
| `yolo_v5_pcb_detection.py` | Script export of the notebook |
| `best.pt` | Trained YOLOv5 weights from the best run |
| `README.md` | This file |

## Dataset

- **Original data:** [PCB Defects on Kaggle](https://www.kaggle.com/datasets/akhatova/pcb-defects) (akhatova)
- **Modified for YOLOv5:**
  - [Google Drive](https://drive.google.com/file/d/1P3AxgjNOhwa4EetJONs4EqR3vnjX4iIz/view?usp=sharing)
  - [Kaggle](https://www.kaggle.com/datasets/breaddddd/pcb-defect-modified/data)

The modified version restructures the annotations and folder layout into the YOLOv5 format (`images/train`, `images/val` with matching label files).

## Requirements

Running this on GPU is strongly recommended — YOLOv5 training is computationally expensive. Google Colab or a Kaggle notebook with GPU/TPU acceleration works well.

Dependencies come from the YOLOv5 repo itself, plus:

```
pillow
pandas
matplotlib
```

## Setup & Usage

The notebook is written for Google Colab with data stored in Google Drive. Adjust paths if you're using Kaggle or a local machine.

**1. Upload the data**

Place `yoloData.zip` (the modified dataset) into your Google Drive root.

**2. Enable GPU**

Runtime → Change runtime type → GPU/TPU acceleration.

**3. Clone YOLOv5 and install dependencies**

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

**4. Mount Drive and unzip the dataset**

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
unzip /content/drive/MyDrive/yoloData.zip -d /content/drive/MyDrive/yolodatasetprocessed/yoloData
```

**5. Generate `data.yaml`**

The notebook writes this automatically:

```yaml
path: /content/drive/MyDrive/yolodatasetprocessed/yoloData
train: images/train
val: images/val

names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
```

**6. Train**

```bash
python train.py --img 416 --batch 16 --epochs 275 --data data.yaml --weights yolov5s.pt --cache --name pcb_3rd
```

**7. Validate**

```bash
python val.py --weights runs/train/pcb_3rd4/weights/best.pt --data data.yaml
```

> Update the run directory name (`pcb_3rd4`) to match your actual run — YOLOv5 appends a number when a run name already exists.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base weights | `yolov5s.pt` |
| Image size | 416 |
| Batch size | 16 |
| Epochs | 275 |
| Caching | Enabled |
| W&B logging | Disabled (`WANDB_MODE=disabled`) |

## Results & Evaluation

After training, the notebook:

- Copies all generated `.png` and `.jpg` artifacts (confusion matrix, PR curve, F1 curve, label distributions, sample batches) into `outputpng/` and `outputjpg/` folders in Drive, and collects them into arrays for inline display.
- Reads `results.csv` into a pandas DataFrame and plots **train vs. validation box loss** and **train vs. validation classification loss** across epochs, side by side.
- Copies the best weights to a `saved_weights/` folder in Drive for reuse.

To view a specific result image inline, index into the array:

```python
Image.open(png_arr[0])
```

## Inference with the Included Weights

```python
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
results = model('path/to/pcb_image.jpg')
results.show()
```

## Notes

- All intermediate and final outputs are written back to Google Drive so nothing is lost when the Colab session ends.
- Paths in the notebook are hardcoded to `/content/drive/MyDrive/yolodatasetprocessed/` — change these to match your own directory structure.

## Credits

- Model: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- Dataset: [akhatova/pcb-defects](https://www.kaggle.com/datasets/akhatova/pcb-defects)
