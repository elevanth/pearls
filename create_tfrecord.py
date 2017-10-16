# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.
Example usage:
    ./create_pascal_tf_record --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import hashlib
import io
import logging
import os
import sys
import glob
import cv2

from lxml import etree
import PIL.Image
from random import shuffle
import numpy as np
import tensorflow as tf

import dataset_util
# from object_detection.utils import label_map_util

logging.basicConfig(level=logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('dataset_name', 'ai-challenger', 'convert different dataset')
flags.DEFINE_string('data_dir', '../dataset', 'Root directory to raw dataset.')
flags.DEFINE_string('mode', 'test', 'Convert training set, validation set or merged set.')
flags.DEFINE_string('annotations_formate', 'json', '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', 'results/input_data', 'Path to output TFRecord')
flags.DEFINE_integer('joint_num', 14, 'number of body joints')
flags.DEFINE_boolean('shuffle_flag', False, 'shuffle data of the tfrecords')
FLAGS = flags.FLAGS

MODE = ['train', 'valid', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']
JOINTS = ['r_shoulder', 'r_elbow', 'r_wrist', \
          'l_shoulder', 'l_elbow', 'l_wrist', \
          'r_hip', 'r_knee', 'r_ankle', \
          'l_hip', 'l_knee', 'l_ankle', \
          'head', 'neck']
x_list = [elem+'_x' for elem in JOINTS]
y_list = [elem+'_y' for elem in JOINTS]
z_list = [elem+'_z' for elem in JOINTS]
ALL_LIST = x_list + y_list + z_list


def load_label_data(data_dir):
    if data_dir.endswith("json"):
        with open(data_dir) as json_file:
            data = json.load(json_file)
            return data


def read_images(path):
    filenames = next(os.walk(path))[2]
    num_files = len(filenames)
    images = np.zeros((num_files,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS),dtype=np.uint8)
    labels = np.zeros((num_files, ), dtype=np.uint8)
    f = open('label.txt')
    lines = f.readlines()
    # 遍历所有的图片和label，将图片resize到[227,227,3]
    for i, filename in enumerate(filenames):
        img = cv2.imread(os.path.join(path, filename))
        img = cv2.imresize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
        images[i] = img
        labels[i] = int(lines[i])
    f.close()
    return images,labels


def dict_to_tf_example(js,
                       dataset_directory,
                       image_subdirectory='JPEGImages'):
    """Convert XML derived dict to tf.Example proto.
    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.
    Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
        dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
        PASCAL dataset directory holding the actual image data.
    Returns:
    example: The converted tf.Example.
    Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    if isinstance(js, dict):
        img_id = js['image_id']
        img_name = img_id + '.jpg'
    else:
        img_name = js
        img_id = js.split('.')[0]
    # full_path = os.path.join(dataset_directory, img_name)
    # with tf.gfile.GFile(full_path, 'rb') as fid:
    #     encoded_jpg = fid.read()
    # # write into memory in binary
    # encoded_jpg_io = io.BytesIO(encoded_jpg)
    # image = PIL.Image.open(encoded_jpg_io)
    # width = image.size[0]
    # height = image.size[1]
    # if image.format != 'JPEG':
    #     raise ValueError('Image format not JPEG')
    # # hash--convert image bytes to short string
    # key = hashlib.sha256(encoded_jpg).hexdigest()
    images, labels = read_images(dataset_directory)
    img_raw = images[i].tostring()

    example_features = {
        'image/id': dataset_util.bytes_feature(img_id.encode('utf8')),
        'image/filename': dataset_util.bytes_feature(img_name.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(full_path.encode('utf8')),
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(img_raw),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),}

    kp_human_id = []
    box_human_id = []
    key_points = []
    human_box = []
    if FLAGS.mode != 'test':
        # arrange key_points lists
        for key, val in js["keypoint_annotations"].items():
            kp_human_id.append(key)
            key_points.extend(val)

        # arrange human_box lists
        if "human_annotations" in js:
            for key, val in js["human_annotations"].items():
                box_human_id.append(key)
                human_box.extend(val)

        if len(box_human_id) != len(kp_human_id):
            print('WARNING: annotation human of key_points and box are different!')

    # save human_box and key_points features
    example_features['image/human_box/name'] = dataset_util.bytes_list_feature([b.encode('utf8') for b in box_human_id])
    example_features['image/human_box/value'] = dataset_util.int64_list_feature(human_box)
    example_features['image/key_points/name'] = dataset_util.bytes_list_feature([k.encode('utf8') for k in kp_human_id])
    example_features['image/key_points/value'] = dataset_util.int64_list_feature(key_points)

    example = tf.train.Example(features=tf.train.Features(feature=example_features))
    return example


def main(_):
    if FLAGS.mode not in MODE:
        raise ValueError('mode must be in : {}'.format(MODE))

    logging.info('Reading from %s dataset.', FLAGS.dataset_name)

    if FLAGS.dataset_name == 'ai-challenger':
        image_root = os.path.join(FLAGS.data_dir, FLAGS.dataset_name, FLAGS.mode)
        img_dir_list = os.listdir(image_root)
        for i in range(len(img_dir_list)):
            if os.path.isdir(os.path.join(image_root, img_dir_list[i])):
                dataset_directory = os.path.join(image_root, img_dir_list[i])
            elif img_dir_list[i].endswith(FLAGS.annotations_formate):
                annotations_dir = os.path.join(image_root, img_dir_list[i])

    # dataset_img_directory = os.path.join(dataset_directory, '*.jpg')
    # img_addrs = glob.glob(dataset_img_directory)

    output_file = FLAGS.dataset_name + '_' + FLAGS.mode + '.tfrecords'
    output_file = os.path.join(FLAGS.output_path, output_file)
    writer = tf.python_io.TFRecordWriter(output_file)

    if FLAGS.mode == 'test':
        data = os.listdir(dataset_directory)
        # if FLAGS.shuffle_flags:
        #     shuffle(data)
    else:
        data = load_label_data(annotations_dir)
        # if FLAGS.shuffle_flags:
        #     addr_label = list(zip(img_addrs, data))
        #     shuffle(addr_label)
        #     img_addrs, data = zip(*addr_label)
    if FLAGS.shuffle_flag:
            shuffle(data)

    length = len(data)
    for idx in range(length):
        if idx%1000 == 0:
            logging.info('On image %d of %d', idx, length)
        tf_example = dict_to_tf_example(data[idx], dataset_directory)
        writer.write(tf_example.SerializeToString())
    writer.close()
    sys.stdout.flush()


if __name__ == '__main__':
    tf.app.run()
