# --------------------------------------------------------
# Graph as Label
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import argparse
import glob

import cv2
import numpy as np
import os
import tensorflow as tf
import time

list_dir = '/nfs.yoda/xiaolonw/judy_folder/pred_vid_data/bair_push/'
# data_dir = '/scratch/snpowers/GoogleBrainPush/'
data_dir = '/scratch/yufeiy2/softmotion30_44k/'

save_dir = '/scratch/yufeiy2/bair/images/'
# save_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/bair/images/'


"""Code for building the input for the prediction model."""

import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from google.protobuf.json_format import MessageToJson
from google.protobuf.json_format import MessageToDict
import json

FLAGS = flags.FLAGS

# Original image dimensions
# ORIGINAL_WIDTH = 640
# ORIGINAL_HEIGHT = 512
ORIGINAL_WIDTH = 128
ORIGINAL_HEIGHT = 128
COLOR_CHAN = 3
STATE_DIM = 3
ACION_DIM = 4


def inspect(obj, depth):
    max_num = 0
    if isinstance(obj, dict):
        for key in obj:
            if '0/' in key:
                print('\t' * depth, key)
            if 'encoded' in key:
                num = int(key.split('/')[0])
                if max_num < num:
                    max_num = num
            num = inspect(obj[key], depth + 1)
            max_num = max(max_num, num)
    elif isinstance(obj, list):
        # print('\t' * depth + 'list: ', len(obj))
        pass
    elif isinstance(obj, str):
        # print('\t' * depth + 'str ')
        pass
    else:
        print('\t' * depth + type(obj))
    return max_num

def read_out_dir(tf_dir):
    basename = os.path.basename(tf_dir)

    tf_list = glob.glob(tf_dir + '/*.tfrecords')
    image_aux_batch, action_batch, endeff_pos_batch = build_bair_pipeline(tf_list)
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    tf.train.start_queue_runners(sess)

    cnt = 0
    while True:
        try:
            image_aux, actions, endeff = sess.run([image_aux_batch, action_batch, endeff_pos_batch])
            N = image_aux.shape[0]
            for n in range(N):
                vid_dir = os.path.join(save_dir, '%s/traj_%05d/' % (basename, cnt))
                if not os.path.exists(vid_dir):
                    os.makedirs(vid_dir)
                    # print('## Make Direcotry: ', vid_dir)
                save_video(vid_dir, image_aux[n] * 255)
                # print(actions[n].shape, endeff[n].shape)
                # action: (dt, 4), end_pos: (dt, 3)
                np.savez_compressed(vid_dir + 'meta.npz', action=actions[n], end_pos=endeff[n])
                cnt += 1

        except tf.errors.OutOfRangeError:
            break
        print(cnt)


def build_bair_pipeline(filenames):
    """Stolen from https://github.com/febert/visual_mpc/blob/357ba4df88dc5ab07ddb67a6cab260fc24a43a01/python_visual_mpc/video_prediction/read_tf_record_sawyer12.py"""
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False, num_epochs=1)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_aux1_seq, image_main_seq, endeffector_pos_seq, action_seq, object_pos_seq, init_pix_distrib_seq = [], [], [], [], [], []
    init_pix_pos_seq = []

    load_indx = range(0, 30, conf['skip_frame'])
    load_indx = load_indx[:conf['sequence_length']]

    for i in load_indx:
        if 'single_view' not in conf:
            image_main_name = str(i) + '/image_main/encoded'
        image_aux1_name = str(i) + '/image_aux1/encoded'
        action_name = str(i) + '/action'
        endeffector_pos_name = str(i) + '/endeffector_pos'

        features = {

                    image_aux1_name: tf.FixedLenFeature([1], tf.string),
                    action_name: tf.FixedLenFeature([ACION_DIM], tf.float32),
                    endeffector_pos_name: tf.FixedLenFeature([STATE_DIM], tf.float32),
        }
        if 'single_view' not in conf:
            (features[image_main_name]) = tf.FixedLenFeature([1], tf.string)

        features = tf.parse_single_example(serialized_example, features=features)

        COLOR_CHAN = 3
        if '128x128' in conf:
            ORIGINAL_WIDTH = 128
            ORIGINAL_HEIGHT = 128
            IMG_WIDTH = 128
            IMG_HEIGHT = 128
        else:
            ORIGINAL_WIDTH = 64
            ORIGINAL_HEIGHT = 64
            IMG_WIDTH = 64
            IMG_HEIGHT = 64

        if 'single_view' not in conf:
            image = tf.decode_raw(features[image_main_name], tf.uint8)
            image = tf.reshape(image, shape=[1,ORIGINAL_HEIGHT*ORIGINAL_WIDTH*COLOR_CHAN])
            image = tf.reshape(image, shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
            if IMG_HEIGHT != IMG_WIDTH:
                raise ValueError('Unequal height and width unsupported')
            crop_size = min(ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
            image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
            image = tf.reshape(image, [1, crop_size, crop_size, COLOR_CHAN])
            image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
            image = tf.cast(image, tf.float32) / 255.0
            image_main_seq.append(image)

        image = tf.decode_raw(features[image_aux1_name], tf.uint8)
        image = tf.reshape(image, shape=[1, ORIGINAL_HEIGHT * ORIGINAL_WIDTH * COLOR_CHAN])
        image = tf.reshape(image, shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
        if IMG_HEIGHT != IMG_WIDTH:
            raise ValueError('Unequal height and width unsupported')
        crop_size = min(ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
        image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
        image = tf.reshape(image, [1, crop_size, crop_size, COLOR_CHAN])
        image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
        image = tf.cast(image, tf.float32) / 255.0
        image_aux1_seq.append(image)

        endeffector_pos = tf.reshape(features[endeffector_pos_name], shape=[1, STATE_DIM])
        endeffector_pos_seq.append(endeffector_pos)
        action = tf.reshape(features[action_name], shape=[1, ACION_DIM])
        action_seq.append(action)

    if 'single_view' not in conf:
        image_main_seq = tf.concat(values=image_main_seq, axis=0)

    image_aux1_seq = tf.concat(values=image_aux1_seq, axis=0)

    if conf['visualize']: num_threads = 1
    else: num_threads = np.min((conf['batch_size'], 32))

    if 'ignore_state_action' in conf:
        [image_main_batch, image_aux1_batch] = tf.train.batch(
                                    [image_main_seq, image_aux1_seq],
                                    conf['batch_size'],
                                    num_threads=num_threads,
                                    capacity=100 * conf['batch_size'])
        return image_main_batch, image_aux1_batch, None, None
    elif 'single_view' in conf:
        endeffector_pos_seq = tf.concat(endeffector_pos_seq, 0)
        action_seq = tf.concat(action_seq, 0)
        [image_aux1_batch, action_batch, endeffector_pos_batch] = tf.train.batch(
            [image_aux1_seq, action_seq, endeffector_pos_seq],
            conf['batch_size'],
            num_threads=num_threads,
            capacity=100 * conf['batch_size'], allow_smaller_final_batch=True)
        return image_aux1_batch, action_batch, endeffector_pos_batch

    else:
        endeffector_pos_seq = tf.concat(endeffector_pos_seq, 0)
        action_seq = tf.concat(action_seq, 0)
        [image_main_batch, image_aux1_batch, action_batch, endeffector_pos_batch] = tf.train.batch(
                                    [image_main_seq,image_aux1_seq, action_seq, endeffector_pos_seq],
                                    conf['batch_size'],
                                    num_threads=num_threads,
                                    capacity=100 * conf['batch_size'], allow_smaller_final_batch=True)
        return image_main_batch, image_aux1_batch, action_batch, endeffector_pos_batch


def build_pipeline(tf_list):
    def _parse_function(serialized_example):
        image_seq = []
        for i in range(30):
            image_name =  str(i) + '/image_aux1/encoded'
            # 'endeffector_pos'
            # 'action'
            # 'image_aux1/encoded'
            # 'image_main/encoded'
            features = {image_name: tf.FixedLenFeature([1], tf.string)}
            features = tf.parse_single_example(serialized_example, features=features)

            image_buffer = tf.reshape(features[image_name], shape=[])
            image = tf.image.decode_jpeg(image_buffer, channels=COLOR_CHAN)
            image.set_shape([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
            image = tf.reshape(image, [1, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
            # image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
            image_seq.append(image)

        image_seq = tf.concat(image_seq, 0)
        return image_seq
    # filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)
    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)

    dataset = tf.data.TFRecordDataset(tf_list)
    dataset = dataset.map(_parse_function)
    # dataset.repeat(1)

    return dataset


def save_video(vid_dir, video):
    video = video.astype(np.uint8)
    for i in range(len(video)):
        img_file = os.path.join(vid_dir, '%03d.jpg' % i)
        # if os.path.exists(img_file):
        #     continue
        cv2.imwrite(img_file, cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR))


def inspect_tfrecord(tf_file, sess):
    save_dir = '/nfs.yoda/xiaolonw/judy_folder/transfer/tmp/'
    cnt = 0

    # binary = tf.placeholder(tf.)
    for example in tf.python_io.tf_record_iterator(tf_file):
        # result = tf.train.Example.FromString(example)
        jsonMessage = MessageToJson(tf.train.Example.FromString(example))
        msg = json.loads(jsonMessage)
        # print(list(find_key(msg)))
        key_list = msg['features']['feature'].keys()
        obj = msg['features']['feature']
        for k in key_list:
            if not 'image/encoded' in k or not 'move' in k:
                continue
            print(k)
            cnt += 1
            image_name = k
            features = {image_name: tf.FixedLenFeature([1], tf.string)}
            features = tf.parse_single_example(example, features=features)
            image_buffer = tf.reshape(features[image_name], shape=[])
            image = tf.image.decode_jpeg(image_buffer, channels=COLOR_CHAN)
            image = tf.reshape(image, [1, ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
            out = sess.run(image)
            print(out)

        # cnt += 1
        break
        # print(len(result))
    print(cnt)


def find_key(dictionary):
    keys = []
    if isinstance(dictionary, dict):
        keys.append(dictionary.keys())
        for k in dictionary.keys():
            keys.append(find_key(dictionary[k]))
    if isinstance(dictionary, list):
        for l in dictionary:
            keys.append(find_key(l))
    if not isinstance(dictionary, dict) or not isinstance(dictionary, list):
        return keys
    return keys


parser = argparse.ArgumentParser()
# Optimization hyperparameters
parser.add_argument('--name', default='test', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    # tf_dir = data_dir + 'push/push_%s' % args.name
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    tf_dir = data_dir + '/%s' % args.name
    conf = {}
    conf['skip_frame'] = 1
    conf['sequence_length'] = 30  # 'sequence length, including context frames.'
    conf['use_state'] = True
    conf['batch_size'] = 32
    conf['visualize'] = True
    conf['single_view'] = ''


    # read_out_dir(tf_dir)

    word = ['test', 'train', '*']
    name_list = ['bair_testimglist', 'bair_trainimglist', 'bair_imglist']
    for i in range(len(word)):
        base = word[i]
        img_list = glob.glob(save_dir + word[i] + '/*/*.jpg')
        img_list = sorted(img_list)
        print(len(img_list))
        if not os.path.exists(list_dir):
            os.makedirs(list_dir)
        with open(list_dir + name_list[i] + '.txt', 'w') as fp:
            for img_path in img_list:
                base = img_path.split('/')[-3]
                img = img_path.split('/')[-2] + '/' + img_path.split('/')[-1].split('.')[0]
                fp.write('%s\n' % (base + '/' + img))



