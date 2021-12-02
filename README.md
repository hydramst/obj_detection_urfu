### Test repository for object detection basis on TF model faster_rcnn/resnet101

# Overview
Faster R-CNN with Resnet-101 (v1) initialized from Imagenet classification checkpoint. Trained on [COCO 2017](https://cocodataset.org/) dataset (images scaled to 1024x1024 resolution).

Model created using the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

An example detection result is shown below.![od_no_keypoints](https://user-images.githubusercontent.com/44744458/144443625-71ecfc4e-1e4b-4274-8177-ec406035f7fb.png)


## Example use
### Apply image detector on a single image.

```python
  detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_1024x1024/1")
  detector_output = detector(image_tensor)
  class_ids = detector_output["detection_classes"]
```

## Inputs
A three-channel image of variable size - the model does NOT support batching. The input tensor is a tf.uint8 tensor with shape [1, height, width, 3] with values in [0, 255].

## Outputs
The output dictionary contains:

- num_detections: a tf.int tensor with only one value, the number of detections [N].
- detection_boxes: a tf.float32 tensor of shape [N, 4] containing bounding box coordinates in the following order: [ymin, xmin, ymax, xmax].
- detection_classes: a tf.int tensor of shape [N] containing detection class index from the label file.
- detection_scores: a tf.float32 tensor of shape [N] containing detection scores.
- raw_detection_boxes: a tf.float32 tensor of shape [1, M, 4] containing decoded detection boxes without Non-Max suppression. M is the number of raw detections.
- raw_detection_scores: a tf.float32 tensor of shape [1, M, 90] and contains class score logits for raw detection boxes. M is the number of raw detections.
- detection_anchor_indices: a tf.float32 tensor of shape [N] and contains the anchor indices of the detections after NMS.
- detection_multiclass_scores: a tf.float32 tensor of shape [1, N, 90] and contains class score distribution (including background) for detection boxes in the image including background class.

## Suitable use cases
This model is suitable for localizing the most prominent objects in an image.

## Unsuitable use cases
This model is unsuitable for standalone use in mission-critical applications such as obstacle and human detection for autonomous driving.

# Source
The model's checkpoints are [publicly available](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) as a part of the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

## Metrics
| Metric | Value | Outputs |
| :---         | :---         | :---         | 
| mAP on COCO 2017 test   | 37.1     | Boxes    |

