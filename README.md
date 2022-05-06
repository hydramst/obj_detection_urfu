/
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

```python
[0.8288508 , 0.80556387, 0.48192698, 0.3948713 , 0.38885146,
         0.36356452, 0.149957  , 0.11586377, 0.11364764, 0.09704131,
         0.0828484 , 0.07247633, 0.0532763 , 0.04258242, 0.03404129,
         0.03302562, 0.03075191, 0.03046039, 0.02866995, 0.02826792,
         0.02518162, 0.02476639, 0.02434707, 0.02354911, 0.02333108,
         0.02302751, 0.02274698, 0.02229893, 0.02165616, 0.02052507,
         0.01953048, 0.01929945, 0.01902393, 0.01879928, 0.01862326,
         0.01763749, 0.01762882, 0.01752466, 0.01738989, 0.01717171,
         0.01700711, 0.01647753, 0.01614243, 0.01611474, 0.01609552,
         0.01569015, 0.01563403, 0.0155161 , 0.01551434, 0.01540104,
         0.01530248, 0.01519424, 0.01491171, 0.01485869, 0.01439488,
         0.01423746, 0.01421249, 0.01411673, 0.01380104, 0.01372972,
         0.01360917, 0.0135833 , 0.01353732, 0.01349267, 0.0132508 ,
         0.01311269, 0.01295835, 0.01286185, 0.01269022, 0.01224661,
         0.0122228 , 0.01213494, 0.01198593, 0.01171231, 0.01149833,
         0.01122445, 0.01112813, 0.01088971, 0.01065397, 0.0106535 ,
         0.01046035, 0.01034549, 0.01031911, 0.01024702, 0.01022914,
         0.0102281 , 0.01004824, 0.01003751, 0.01000172, 0.00995308,
         0.00992376, 0.00990987, 0.00987318, 0.00984126, 0.00982562,
         0.00979355, 0.00956395, 0.00956208, 0.00947487, 0.00943458]
```
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

