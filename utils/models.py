import tensorflow as tf
import numpy as np
import os


class Model:
    def __init__(self, exp_dir, arch_name) -> None:
        # Create Directories for Experimenation
        self.exp_dir = self._create_dirs(exp_dir)
        self.model_dir = self._create_dirs(os.path.join(exp_dir, 'model'))
        self.log_dir = self._create_dirs(os.path.join(exp_dir, 'log'))
        self.checkpoint_dir = self._create_dirs(os.path.join(exp_dir, 'checkpoint'))
        self.arch_name = arch_name
        self.model_arch = None
    
    def train(self, train_pipe, valid_pipe):
        # Build Model
        self.model_arch = self.build_model()
        # Train Model
    
    def build_model(self):
        # Build Model
        model = tf.keras.models.Sequential()
        

    @staticmethod
    def _create_dirs(c_dir):
        try:
            if os.path.exists(c_dir):
                print(f"Experiment directory {c_dir} already exists")
                return c_dir
            else:
                os.makedirs(c_dir)
                print(f"Experiment directory {c_dir} created")
                return c_dir
        except Exception as e:
            print(f"Error creating experiment directory {c_dir} -> {e}")
            return False

        