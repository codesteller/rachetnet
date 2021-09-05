# RachetNet
Highly Accurate Transformer based Object Detection Network

## Create Dataset
```
python3 ./utils/dataset/coco2tfrecord.py --image_dir /dataset/coco2017/train2017 --object_annotations_file /dataset/coco2017/annotations/instances_train2017.json --output_file_prefix /dataset/coco2017/tfrecords/train
```