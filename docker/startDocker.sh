docker run --gpus '"device=0"' -it --rm --shm-size=1g --ulimit memlock=-1 \
-p 9000:8888 -p 0.0.0.0:9006:6006 \
-v $(pwd):/workspace/myspace \
-v /mnt/ws_drive/05_Datasets/mscoco/coco_dataset/coco2017:/dataset/coco2017 \
-w /workspace/myspace \
codesteller/rachetnet:22.02
