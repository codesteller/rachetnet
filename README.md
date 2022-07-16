# RachetNet
Highly Accurate Transformer based Object Detection Network

## Start Docker
```
docker run --gpus '"device=<gpu-id>"' -it --rm --shm-size=1g --ulimit memlock=-1 \
-p 9000:8888 -p 0.0.0.0:9006:6006 \
-v $(pwd):/workspace/myspace \
-v <coco2017-local-path>:/dataset/coco2017 \
-w /workspace/myspace \
codesteller/rachetnet:22.02
```

## Create Dataset
```
python3 ./utils/dataset/create_coco_tfrecord.py --image_dir /dataset/coco2017/train2017 --object_annotations_file /dataset/coco2017/annotations/instances_train2017.json --output_file_prefix /dataset/coco2017/tfrecords/train
```
