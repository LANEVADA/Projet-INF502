"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import tensorflow as tf
import numpy as np
from PIL import Image  
from io import BytesIO  # For Python 3.x

class Logger(object):
    def __init__(self, log_dir, suffix=None):
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def image_summary(self, tag, images, step):
        with self.writer.as_default():
            for i, img in enumerate(images):
                img = tf.convert_to_tensor(img, dtype=tf.uint8)  # Ensure it's a tensor
                tf.summary.image(f"{tag}/{i}", img[None], step=step)  # Add batch dimension
            self.writer.flush()

    def video_summary(self, tag, videos, step):
        sh = list(videos.shape)
        sh[-1] = 1
        separator = np.zeros(sh, dtype=videos.dtype)
        videos = np.concatenate([videos, separator], axis=-1)

        with self.writer.as_default():
            for i, vid in enumerate(videos):
                v = vid.transpose(1, 2, 3, 0)  # Reorder axes
                v = [np.squeeze(f) for f in np.split(v, v.shape[0], axis=0)]
                img = np.concatenate(v, axis=1)[:, :-1, :]

                img = tf.convert_to_tensor(img, dtype=tf.uint8)
                tf.summary.image(f"{tag}/{i}", img[None], step=step)  # Add batch dimension
            self.writer.flush()

