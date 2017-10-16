from create_tfrecord import FLAGS, JOINTS, ALL_LIST
import dataset_util

import  hashlib
import io
import PIL.Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_path = 'results/input_data/ai-challenger_valid.tfrecords'  # address to save the hdf5 file
shuffle_batch = False
feature = {
        'image/id': tf.FixedLenFeature([], tf.string),
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/source_id': tf.FixedLenFeature([], tf.string),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/key/sha256': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/human_box/name': tf.FixedLenFeature([], tf.string),
        'image/human_box/value': tf.FixedLenFeature([], tf.int64),
        'image/key_points/name': tf.FixedLenFeature([], tf.string),
        'image/key_points/value': tf.FixedLenFeature([], tf.int64),}


# Read TFRecords using queue structs
def ReadTFRecord(sess, tfrecords, shuffle_batch):
    # Create a list of filenames and pass it to a queue
    record_queue = tf.train.string_input_producer([tfrecords], num_epochs=1)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_ex = reader.read(record_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_ex, features=feature)
    # Convert the image data from string back to the numbers
    # ori_key = features['image/key/sha256']# .decode('utf8')
    image = tf.decode_raw(features['image/encoded'], tf.uint8)

    # encoded_jpg_io = io.BytesIO(img)
    # encoded_image = PIL.Image.open(encoded_jpg_io)
    # key = hashlib.sha256(encoded_image).hexdigest()
    # if ori_key != key:
    #     raise ValueError('origin image damaged!')

    label = {}
    # width = tf.decode_raw(features['image/width'], tf.int32)
    # height = tf.decode_raw(features['image/height'], tf.int32)
    label['width'] = features['image/width']
    label['height'] = features['image/height']
    height = tf.cast(features['image/height'], tf.int64)
    width = tf.cast(features['image/width'], tf.int64)
    print(height)
    image = tf.reshape(image, [height, width, 3])

    box_human_id = features['image/human_box/name']
    human_box = features['image/human_box/value']
    kp_human_id = features['image/key_points/name']
    key_points = features['image/key_points/value']

    if shuffle_batch:
        images, box_human_ids, human_boxs = tf.train.shuffle_batch([image, box_human_id, human_box],
                                                batch_size=4,
                                                capacity=2000,
                                                num_threads=2,
                                                min_after_dequeue=1000)
    else:
        images, box_human_ids, human_boxs = tf.train.batch([image, box_human_id, human_box],
                                                batch_size=4,
                                                capacity=2000,
                                                num_threads=2)

    # label['width'] = features['image/width']
    # label['height'] = features['image/height']
    # # print('width and height is ', width, height)
    #
    # label['box_human_id'] = features['image/human_box/name']
    # label['human_box'] = features['image/human_box/value']
    # label['kp_human_id'] = features['image/key_points/name']
    # label['key_points'] = features['image/key_points/value']

    return images, box_human_ids, human_boxs


def main():
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:

        sess.run(init_op)

        imgs, label = ReadTFRecord(sess, data_path, shuffle_batch)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        for i in range(2):
            image, b, c = sess.run([imgs, label])
            image = tf.reshape(image, [label['width'], label['height'], 3])
            print(image.shape)
            box_human_id = [f.decode('utf8') for f in label['box_human_id']]
            if len(label['human_box'])%4 != 0:
                raise ValueError('human_box metrics length error!')
            for idx in len(box_human_id):
                label[box_human_id[idx]] = label['human_box'][idx:(4*idx-1)]
            kp_human_id = [f.decode('utf8') for f in label['kp_human_id']]
            if len(label['human_box'])%FLAGS.joint_num != 0:
                raise ValueError('key_points metrics length error!')
            for idx in len(kp_human_id):
                label[kp_human_id[idx]] = label['key_points'][idx:(FLAGS.joint_num*idx-1)]
            for key, val in label:
                print('key', key)
                print('val', len(val))
        # images, labels = tf.train.shuffle_batch([image, label],
        #                                         batch_size=10,
        #                                         capacity=30,
        #                                         num_threads=1,
        #                                         min_after_dequeue=10)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
