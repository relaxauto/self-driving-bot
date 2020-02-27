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

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

import hashlib
import io
import logging
import os
import random
import re
import fnmatch
import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf
import sys

flags = tf.compat.v1.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', './data/label_map.pbtxt', 'Path to label map proto')
flags.DEFINE_string('tfod', '', 'Path to tensorflow/models/research/')
flags.DEFINE_integer('num_shards', 1, 'Number of TFRecord shards')

FLAGS = flags.FLAGS

sys.path.append(FLAGS.tfod)

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

def get_class_name_from_filename(file_name):
    """Gets the class name from a file.

    Args:
      file_name: The file name to get the class name from.
                 ie. "american_pit_bull_terrier_105.jpg"

    Returns:
      A string of the class name.
    """
    match = re.match(r'([A-Za-z-]+)(_[0-9]+\.jpg)', file_name, re.I)
    return match.groups()[0]


def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
      mask_path: String path to PNG encoded mask.
      label_map_dict: A map from string label names to integers ids.
      image_subdirectory: String specifying subdirectory within the
        Pascal dataset directory holding the actual image data.


    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    print(data['filename'])
    img_path = os.path.join(image_subdirectory, data['filename'])
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    masks = []
    if 'object' in data:
        for obj in data['object']:

            xmin = float(obj['bndbox']['xmin'])
            xmax = float(obj['bndbox']['xmax'])
            ymin = float(obj['bndbox']['ymin'])
            ymax = float(obj['bndbox']['ymax'])

            xmins.append(xmin / width)
            ymins.append(ymin / height)
            xmaxs.append(xmax / width)
            ymaxs.append(ymax / height)
            #class_name = get_class_name_from_filename(data['filename'])
            class_name = obj['name']
            classes_text.append(class_name.encode('utf8'))
            ## this is the class id
            classes.append(label_map_dict[class_name])



    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     annotations_dir,
                     image_dir,
                     xml_files,
                     faces_only=True,
                     mask_type='png'):
    """Creates a TFRecord file from examples.

    Args:
      output_filename: Path to where output file is saved.
      num_shards: Number of shards for output file.
      label_map_dict: The label map dictionary.
      annotations_dir: Directory where annotation files are stored.
      image_dir: Directory where image files are stored.
      examples: Examples to parse and save to tf record.
      faces_only: If True, generates bounding boxes for pet faces.  Otherwise
        generates bounding boxes (as well as segmentations for full pet bodies).
      mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
        smaller file sizes.
    """
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_filename, num_shards)
        for idx, example in enumerate(xml_files):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(xml_files))
            xml_path = os.path.join(annotations_dir, example)

            with tf.io.gfile.GFile(xml_path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            try:
                tf_example = dict_to_tf_example(
                    data,
                    label_map_dict,
                    image_dir)

                if tf_example:
                    shard_idx = idx % num_shards
                    output_tfrecords[shard_idx].write(tf_example.SerializeToString())
            except ValueError:
                logging.warning('Invalid example: %s, ignoring.', xml_path)

# the script assuming the following directory structure:
# under the data_dir, there should be two subdirectories: images and annotations
# all the xml files are in the annotations directory.


def main(_):
    data_dir = FLAGS.data_dir
    print(FLAGS.label_map_path)
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    #label_map_dict = FLAGS.label_map_path

    logging.info('Reading from dataset.')
    image_dir = os.path.join(data_dir, 'images')
    annotations_dir = os.path.join(data_dir, 'annotations')
    xml_dir = annotations_dir
    xml_files = fnmatch.filter(os.listdir(xml_dir), '*.xml')

    random.seed(42)
    random.shuffle(xml_files)
    num_xml_files = len(xml_files)
    num_train = int(0.7 * num_xml_files)
    train_annot_files = xml_files[:num_train]
    val_annot_files = xml_files[num_train:]

    logging.info('%d training and %d validation examples.',
                 len(train_annot_files), len(val_annot_files))

    train_output_path = os.path.join(FLAGS.output_dir, 'train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'validation.record')

    create_tf_record(
        train_output_path,
        FLAGS.num_shards,
        label_map_dict,
        annotations_dir,
        image_dir,
        train_annot_files)

    create_tf_record(
        val_output_path,
        FLAGS.num_shards,
        label_map_dict,
        annotations_dir,
        image_dir,
        val_annot_files)

if __name__ == '__main__':
    tf.compat.v1.app.run()
