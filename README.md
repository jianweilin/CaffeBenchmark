Caffe Classification
====================

## Classification Speed Benchmark

Table lists average classification times per image (milliseconds). Averages are
computed over classification of 100 images from MS-COCO validation set. For the VGG networks
a very high variance in classification times was observed, some images were classified fast
while most of them took more processing time that other network configurations.

| Model                             | Input Size | CPU (ms) | GPU (ms) |
|-----------------------------------|------------|----------|----------|
| `bvlc_alexnet.caffemodel`         | 227x227    | 314.7    | 20.1     |
| `bvlc_googlenet.caffemodel`       | 224x224    | 490.7    | 33.6     |
| `VGG_ILSVRC_16_layers.caffemodel` | 224x224    | 758.2    | 35.7     |
| `VGG_ILSVRC_19_layers.caffemodel` | 224x224    | >1000    | 34.3     |
| `ResNet-50-model.caffemodel`      | 224x224    | >1000    | 50.4     |
| `ResNet-101-model.caffemodel`     | 224x224    | >1000    | 84.6     |
| `ResNet-152-model.caffemodel`     | 224x224    | >1000    | 116.1    |
