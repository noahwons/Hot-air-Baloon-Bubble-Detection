YOLOv5 Bubble Detection for Hot Air Balloon Viles
This project uses a YOLOv5 model trained to detect bubbles in hot air balloon viles. The model can detect bubbles in video frames, making it useful for automated bubble detection in various hot air balloon scenarios.

Table of Contents
Installation
Requirements
Usage
Training
Acknowledgements
Installation
Clone this repository:

bash
Copy code
git clone https://github.com/noahwons/Hot-air-Baloon-Bubble-Detection.git
cd Hot-air-Baloon-Bubble-Detection
Install the required dependencies using pip:

bash
Copy code
pip install -U -r requirements.txt
Requirements
Python 3.7+
torch (PyTorch)
opencv-python (for video/image processing)
yolov5 (for detection)
Ensure that you have a working environment with these dependencies installed.

Usage
To run inference and detect bubbles in a video file, use the following command:

bash
Copy code
python detect.py --weights runs/train/exp3/weights/best.pt --device cpu --source /Users/noahwons/Downloads/bubbles_test.MP4 --view-img --img-size 1280
Arguments:
--weights: Path to the trained YOLOv5 model weights (in this case, best.pt).
--device: Set the device for running the model (cpu or cuda for GPU).
--source: Path to the video file for bubble detection (/Users/noahwons/Downloads/bubbles_test.MP4 in this case).
--view-img: Displays the detection results in a window.
--img-size: Input image size for the model. In this case, we use 1280 to improve accuracy for small or distant bubbles.
Example:
For example, if your video file is located at /Users/noahwons/Downloads/bubbles_test.MP4, you can use the command above to start detection. The model will detect and display bubbles found in the video.

Training
The model was trained using the YOLOv5 architecture on a custom dataset of hot air balloon viles with labeled bubble annotations. To train the model, use the following command:

bash
Copy code
python train.py --data bubbles_dataset.yaml --cfg yolov5s.yaml --weights '' --batch-size 16 --epochs 300
--data: Path to the dataset configuration YAML file.
--cfg: Model configuration (e.g., yolov5s.yaml for a small model).
--weights: Specify pretrained weights or leave it empty for training from scratch.
--batch-size: The number of samples per batch during training.
--epochs: The number of training epochs (300 was used for this model).
For a more detailed training setup, refer to YOLOv5 documentation or adjust the configuration to suit your needs.

Acknowledgements
This project uses the YOLOv5 model for object detection.
Special thanks to the open-source community for creating and improving object detection models and frameworks.
