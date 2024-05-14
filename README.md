
# deepstream-yolov9-qat
This project focuses on implementing the YOLOv9 model on the Jetson Orin Nano, exploring various configurations to enhance real-time object detection performance.

<div align="center">
  <img src="examples/yolov9-qat.gif" alt="YOLOv9 Tracker">
    <p>YOLOv9 Tracker in Action</p>
</div>

## Acknowledgments

This project builds upon ideas and code from two other projects. For more insights and related code, please refer to the following repositories:

- [yolov9](https://github.com/WongKinYiu/yolov9)
- [yolov9-qat](https://github.com/levipereira/yolov9-qat)
- [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

I also use Docker images for the Jetson provided by:

- [jetson-containers](https://github.com/dusty-nv/jetson-containers)

These Docker images help simplify setting up and running our project on Jetson devices.
## Usage

### Running the Pipeline

To run the GStreamer pipeline, execute the script with the necessary arguments. Below is an example command:

```bash

python3 deepstream-yolov9.py /path/to/media/file --gpu-id 0 --onnx-file /path/to/onnx/file --precision fp32
```
#### Command-line Arguments

-   `media_file`: Path to the media file or URI.
-   `--gpu-id`: GPU ID to use (default: 0).
-   `--onnx-file`: Path to the ONNX model file (default: "default.onnx").
-   `--precision`: Model precision mode, one of `fp32`, `fp16`, or `int8` (default: `fp32`).


## Model Export to ONNX

To export the model to the ONNX format, I utilized code from the following repositories: [[1]](https://github.com/levipereira/yolov9-qat), [[2]](https://github.com/marcoslucianops/DeepStream-Yolo/tree/master), which I modified to include the `DeepStreamOutput` class  in the Python file [export_qat.py](https://github.com/levipereira/yolov9-qat/blob/master/export_qat.py):

```python
import torch
import torch.nn as nn

class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.transpose(1, 2)
        boxes = x[:, :, :4]
        scores, classes = torch.max(x[:, :, 4:], 2, keepdim=True)
        classes = classes.float()
        return boxes, scores, classes

# Integrate the DeepStreamOutput class with the model
model = nn.Sequential(model, DeepStreamOutput())

# Modify input and output names to match the nvdsinfer_custom_impl_Yolo lib requirements
torch.onnx.export(
    model, 
    input_tensor, 
    "model.onnx", 
    input_names=["input"], 
    output_names=["boxes", "scores", "classes"]
)
```
To do the export, I used the Docker image nvcr.io/nvidia/pytorch:23.02-py3, which has everything needed for PyTorch and ONNX.
### Debugging

To enable latency measurement for debugging, export the following environment variables:

```bash
export NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1
export NVDS_ENABLE_LATENCY_MEASUREMENT=1
```
For more information on latency measurement, please refer to the [DeepStream SDK forum](https://forums.developer.nvidia.com/t/deepstream-sdk-faq/80236/10).
