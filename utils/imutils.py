from unittest import main
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import nvidia.dali.math as math
import numpy as np
from time import time
import os.path

import random
random.seed(1231231)   # Random is used to pick colors


def build_pipeline(file_root, anno_file, num_gpus=1, device_id=0, batch_size=32, num_threads=4):
    pipe = Pipeline(batch_size=batch_size,
                    num_threads=num_threads, device_id=device_id)
    with pipe:
        jpegs, bboxes, labels, polygons, vertices = fn.readers.coco(
            file_root=file_root,
            annotations_file=anno_file,
            polygon_masks=True,
            ratio=True)
        images = fn.decoders.image(
            jpegs, device="mixed", output_type=types.RGB)
        pipe.set_outputs(images, bboxes, labels, polygons, vertices)

    pipe.build()
    return pipe


def build_run_pipeline(file_root, anno_file, num_gpus=1, device_id=0, batch_size=32, num_threads=4):
    pipe = Pipeline(batch_size=batch_size,
                    num_threads=num_threads, device_id=device_id)
    with pipe:
        jpegs, bboxes, labels, polygons, vertices = fn.readers.coco(
            file_root=file_root,
            annotations_file=anno_file,
            polygon_masks=True,
            ratio=True)
        images = fn.decoders.image(
            jpegs, device="mixed", output_type=types.RGB)
        pipe.set_outputs(images, bboxes, labels, polygons, vertices)

    pipe.build()

    pipe_out = pipe.run()

    images_cpu = pipe_out[0].as_cpu()
    bboxes_cpu = pipe_out[1]
    labels_cpu = pipe_out[2]
    polygons_cpu = pipe_out[3]
    vertices_cpu = pipe_out[4]

    bboxes = bboxes_cpu.at(4)
    labels = labels_cpu.at(4)

    # for bbox, label in zip(bboxes, labels):
    #     x, y, width, height = bbox
    #     print(
    #         f"Bounding box (x={x}, y={y}, width={width}, height={height}), label={label}")

    return pipe, (images_cpu, bboxes_cpu, labels_cpu, polygons_cpu, vertices_cpu)


def plot_sample(data_cpu, img_index, ax):
    images_cpu, bboxes_cpu, labels_cpu, polygons_cpu, vertices_cpu = data_cpu
    img = images_cpu.at(img_index)

    H = img.shape[0]
    W = img.shape[1]

    ax.imshow(img)
    bboxes = bboxes_cpu.at(img_index)
    labels = labels_cpu.at(img_index)
    polygons = polygons_cpu.at(img_index)
    vertices = vertices_cpu.at(img_index)
    categories_set = set()
    for label in labels:
        categories_set.add(label)

    category_id_to_color = dict([(cat_id, [random.uniform(0, 1), random.uniform(
        0, 1), random.uniform(0, 1)]) for cat_id in categories_set])

    for bbox, label in zip(bboxes, labels):
        rect = patches.Rectangle((bbox[0] * W, bbox[1] * H), bbox[2] * W, bbox[3] * H,
                                 linewidth=1, edgecolor=category_id_to_color[label], facecolor='none')
        ax.add_patch(rect)

    for polygon in polygons:
        mask_idx, start_vertex, end_vertex = polygon
        polygon_vertices = vertices[start_vertex:end_vertex]
        polygon_vertices = polygon_vertices * [W, H]
        poly = patches.Polygon(polygon_vertices, True,
                               facecolor=category_id_to_color[label], alpha=0.7)
        ax.add_patch(poly, )


def test_function1():
    test_score = True
    DALI_EXTRA_PATH = os.environ['DALI_EXTRA_PATH']
    data_root = r"/dataset/coco2017"
    train_file_root = os.path.join(data_root, 'train2017')
    train_anno_file = os.path.join(
        data_root, 'annotations', 'instances_train2017.json')
    valid_file_root = os.path.join(data_root, 'test2017')
    valid_anno_file = os.path.join(
        data_root, 'annotations', 'instances_test2017.json')

    num_gpus = 1     # Single GPU for this example
    device_id = 0
    batch_size = 32
    num_threads = 4  # Number of CPU threads

    train_pipe, train_data_cpu = build_run_pipeline(
        train_file_root, train_anno_file, num_gpus, device_id, batch_size, num_threads)

    # valid_images_cpu, valid_bboxes_cpu, valid_labels_cpu, valid_polygons_cpu, valid_vertices_cpu = build_run_pipeline(
    #     valid_file_root, valid_anno_file, num_gpus, device_id, batch_size, num_threads)

    test_values = [(0.0, 0.07831250131130219,
                    0.951517641544342, 0.6724218726158142, 26),
                   (0.34839916229248047, 0.2545156180858612,
                    0.6457588076591492, 0.7268593907356262, 1)]

    _, bboxes_cpu, labels_cpu, _, _ = train_data_cpu
    bboxes = bboxes_cpu.at(4)
    labels = labels_cpu.at(4)

    id = 0
    for bbox, label in zip(bboxes, labels):
        x, y, width, height = bbox
        if label == test_values[id][4]:
            assert abs(x - test_values[id][0]) < 0.001
            assert abs(y - test_values[id][1]) < 0.001
            assert abs(width - test_values[id][2]) < 0.001
            assert abs(height - test_values[id][3]) < 0.001
            print("Test case {} passed".format(id))
        else:
            print("Test case {} failed".format(id))
            test_score = False
        id += 1

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    fig.tight_layout()
    plot_sample(train_data_cpu, 3, ax[0, 0])
    plot_sample(train_data_cpu, 0, ax[0, 1])
    plot_sample(train_data_cpu, 5, ax[1, 0])
    plot_sample(train_data_cpu, 12, ax[1, 1])
    plt.savefig("test_results/test.png")

    return test_score


def test_function2():
    test_score = True
    DALI_EXTRA_PATH = os.environ['DALI_EXTRA_PATH']
    data_root = r"/dataset/coco2017"
    train_file_root = os.path.join(data_root, 'train2017')
    train_anno_file = os.path.join(
        data_root, 'annotations', 'instances_train2017.json')
    valid_file_root = os.path.join(data_root, 'test2017')
    valid_anno_file = os.path.join(
        data_root, 'annotations', 'instances_test2017.json')

    num_gpus = 1     # Single GPU for this example
    device_id = 0
    batch_size = 32
    num_threads = 4  # Number of CPU threads

    train_pipe = build_pipeline(
        train_file_root, train_anno_file, num_gpus, device_id, batch_size, num_threads)

    train_pipe.run()

    pipe_out = train_pipe.run()

    images_cpu = pipe_out[0].as_cpu()
    bboxes_cpu = pipe_out[1]
    labels_cpu = pipe_out[2]
    polygons_cpu = pipe_out[3]
    vertices_cpu = pipe_out[4]

    bboxes = bboxes_cpu.at(4)
    labels = labels_cpu.at(4)

    test_values = [(0.7889999747276306, 0.8648958206176758,
                    0.15845313668251038, 0.13510416448116302, 44),
                   (0.001687500043772161, 0.08539583534002304,
                    0.9983124732971191, 0.9011250138282776, 54)]

    id = 0
    for bbox, label in zip(bboxes, labels):
        x, y, width, height = bbox
        if label == test_values[id][4]:
            assert abs(x - test_values[id][0]) < 0.001
            assert abs(y - test_values[id][1]) < 0.001
            assert abs(width - test_values[id][2]) < 0.001
            assert abs(height - test_values[id][3]) < 0.001
            print("Test case {} passed".format(id))
        else:
            print("Test case {} failed".format(id))
            test_score = False
        id += 1

    # for bbox, label in zip(bboxes, labels):
    #     x, y, width, height = bbox
    #     print(
    #         f"Bounding box (x={x}, y={y}, width={width}, height={height}), label={label}")

    return test_score


if __name__ == "__main__":
    if test_function1():
        print("Test Function 1 passed")
    else:
        print("Test Function 1 failed")
    if test_function2():
        print("Test Function 2 passed")
    else:
        print("Test Function 2 failed")
