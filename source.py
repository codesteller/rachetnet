import configparser
import os
from utils.imutils import build_pipeline, plot_sample
from utils.models import Model


def main():
    config = configparser.ConfigParser()
    config.read("config/config.ini")

    # Build Data pipeline
    data_root = config["DATASET"]["root_path"]
    train_file_root = os.path.join(data_root, 'train2017')
    train_anno_file = os.path.join(
        data_root, 'annotations', 'instances_train2017.json')
    valid_file_root = os.path.join(data_root, 'val2017')
    valid_anno_file = os.path.join(
        data_root, 'annotations', 'instances_val2017.json')

    print(f"Train file root: {train_file_root}")
    print(f"Train anno file: {train_anno_file}")

    train_pipe = build_pipeline(train_file_root, train_anno_file)
    valid_pipe = build_pipeline(valid_file_root, valid_anno_file)

    # Build Model
    model = Model(os.path.join(
        config["EXPERIMENT"]["experiment_path"], config["EXPERIMENT"]["experiment_name"]), config["EXPERIMENT"]["architecture"])

    print(model.model_dir)
    # Train Model
    # model.train(train_pipe, valid_pipe)


if __name__ == main():
    main()
