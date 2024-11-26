# YOLOv5 Bubble Detection for Hot Air Balloon Viles

This project uses a YOLOv5 model trained to detect bubbles in hot air balloon viles. The model can detect bubbles in video frames, making it useful for automated bubble detection in various hot air balloon scenarios.

## Table of Contents

- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Training](#training)
- [Acknowledgements](#acknowledgements)

---

## Installation

### Step 1: Clone the repository

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/noahwons/Hot-air-Baloon-Bubble-Detection.git
cd Hot-air-Baloon-Bubble-Detection
```

This will create a local copy of the repository and navigate into the project directory.

### Step 2: Install dependencies

Ensure you have Python 3.7+ installed and then install the required dependencies using `pip`:

```bash
pip install -U -r requirements.txt
```

This will install all the necessary libraries and dependencies, including PyTorch, OpenCV, and YOLOv5.

---

## Requirements

- **Python 3.7+** (recommended version)
- **PyTorch** (for deep learning framework)
- **OpenCV** (for image/video processing)
- **YOLOv5** (for object detection)

Make sure your environment has these dependencies installed for smooth execution.

---

## Usage

### Running the Model

To run inference and detect bubbles in a video, use the following command:

```bash
python detect.py --weights runs/train/exp3/weights/best.pt --device cpu --source /Users/noahwons/Downloads/bubbles_test.MP4 --view-img --img-size 1280
```

### Arguments Explained:
- **`--weights`**: The path to the trained YOLOv5 model weights file (in this case, `best.pt`).
- **`--device`**: Specifies the device to run the model on (`cpu` for CPU or `cuda` for GPU).
- **`--source`**: The path to the input video file (`/Users/noahwons/Downloads/bubbles_test.MP4` in this example).
- **`--view-img`**: Displays the detection results in an OpenCV window.
- **`--img-size`**: Input image size for the model (set to `1280` for higher accuracy with smaller objects).

### Example Usage:

Run the model on your own video file by changing the `--source` parameter to point to the video you want to analyze. For example:

```bash
python detect.py --weights runs/train/exp3/weights/best.pt --device cpu --source /path/to/your/video_file.MP4 --view-img --img-size 1280
```

---

## Training

To train the model on your own dataset, use the following command:

```bash
python train.py --data bubbles_dataset.yaml --cfg yolov5s.yaml --weights '' --batch-size 16 --epochs 300
```

### Arguments Explained:
- **`--data`**: Path to the dataset configuration YAML file (`bubbles_dataset.yaml`), which should define your data and class labels.
- **`--cfg`**: Specifies the model configuration (e.g., `yolov5s.yaml` for a small model).
- **`--weights`**: Pretrained weights file or leave empty (`''`) to train from scratch.
- **`--batch-size`**: The number of samples per batch during training.
- **`--epochs`**: The number of epochs (iterations) to train the model (300 epochs in this case).

### Training a Custom Dataset:

To train the model on a custom dataset, ensure that you have a valid dataset configuration file (`bubbles_dataset.yaml`) and appropriate labeled data (images with annotations). The dataset file should define the paths to your training and validation data, as well as the class labels.

---

## Acknowledgements

- This project uses [YOLOv5](https://github.com/ultralytics/yolov5) for object detection, a state-of-the-art model for real-time object detection.
- Special thanks to the open-source community and contributors who have made object detection models like YOLOv5 freely available.

---

This `README.md` file is now formatted with markdown annotations to help with clarity and organization. You can copy and paste this directly into your project's README file, making any adjustments as necessary for your project specifics.
