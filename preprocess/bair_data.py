import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt

import pdb

from PIL import Image
import imp

import cPickle

# Dimension of the state and action.
STATE_DIM = 3
ACION_DIM = 4

def build_tfrecord_input(conf, training=True):
    """Create input tfrecord tensors.

    Args:
      training: training or validation data_files.
      conf: A dictionary containing the configuration for the experiment
    Returns:
      list of tensors corresponding to images, actions, and states. The images
      tensor is 5D, batch x time x height x width x channels. The state and
      action tensors are 3D, batch x time x dimension.
    Raises:
      RuntimeError: if no files found.
    """
    filenames = gfile.Glob(os.path.join(conf['data_dir'], '*'))
    if not filenames:
        raise RuntimeError('No data_files files found.')

    index = int(np.floor(conf['train_val_split'] * len(filenames)))
    if training:
        filenames = filenames[:index]
    else:
        filenames = filenames[index:]

    if conf['visualize']:
        filenames = gfile.Glob(os.path.join(conf['data_dir'], '*'))
        print 'using input file', filenames
        shuffle = False
    else: shuffle = True


    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_aux1_seq, image_main_seq, endeffector_pos_seq, action_seq, object_pos_seq, init_pix_distrib_seq = [], [], [], [], [], []
    init_pix_pos_seq = []

    load_indx = range(0, 30, conf['skip_frame'])
    load_indx = load_indx[:conf['sequence_length']]
    print 'using frame sequence: ', load_indx

    for i in load_indx:
        if 'single_view' not in conf:
            image_main_name = str(i) + '/image_main/encoded'
        image_aux1_name = str(i) + '/image_aux1/encoded'
        action_name = str(i) + '/action'
        endeffector_pos_name = str(i) + '/endeffector_pos'
        # state_name = 'move/' +str(i) + '/state'

        if 'canon_ex' in conf:
            init_pix_pos_name = '/init_pix_pos'
            init_pix_distrib_name = str(i) +'/init_pix_distrib'

        features = {

                    image_aux1_name: tf.FixedLenFeature([1], tf.string),
                    action_name: tf.FixedLenFeature([ACION_DIM], tf.float32),
                    endeffector_pos_name: tf.FixedLenFeature([STATE_DIM], tf.float32),
        }
        if 'single_view' not in conf:
            (features[image_main_name]) = tf.FixedLenFeature([1], tf.string)

        if 'canon_ex' in conf:
            (features[init_pix_distrib_name]) = tf.FixedLenFeature([1], tf.string)
            (features[init_pix_pos_name]) = tf.FixedLenFeature([2], tf.float32)

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

        if 'canon_ex' in conf:
            init_pix_distrib = tf.decode_raw(features[init_pix_distrib_name], tf.uint8)
            init_pix_distrib = tf.reshape(init_pix_distrib, shape=[1, ORIGINAL_HEIGHT * ORIGINAL_WIDTH])
            init_pix_distrib = tf.reshape(init_pix_distrib, shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 1])
            crop_size = min(ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
            init_pix_distrib = tf.image.resize_image_with_crop_or_pad(init_pix_distrib, crop_size, crop_size)
            init_pix_distrib = tf.reshape(init_pix_distrib, [1, crop_size, crop_size, 1])
            init_pix_distrib = tf.image.resize_bicubic(init_pix_distrib, [IMG_HEIGHT, IMG_WIDTH])
            init_pix_distrib = tf.cast(init_pix_distrib, tf.float32) / 255.0
            init_pix_distrib_seq.append(init_pix_distrib)

            init_pix_pos = tf.reshape(features[init_pix_pos_name], shape=[1, 2])
            init_pix_pos_seq.append(init_pix_pos)

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
    elif 'canon_ex' in conf:
        endeffector_pos_seq = tf.concat(endeffector_pos_seq, 0)
        action_seq = tf.concat(action_seq, 0)

        init_pix_pos_seq = tf.concat(init_pix_pos_seq, 0)
        init_pix_distrib_seq = tf.concat(init_pix_distrib_seq, 0)

        [image_aux1_batch, action_batch, endeffector_pos_batch, init_pix_distrib_batch, init_pix_pos_batch] = tf.train.batch(
            [image_aux1_seq, action_seq, endeffector_pos_seq, init_pix_distrib_seq, init_pix_pos_seq],
            conf['batch_size'],
            num_threads=num_threads,
            capacity=100 * conf['batch_size'])
        return image_aux1_batch, action_batch, endeffector_pos_batch, init_pix_distrib_batch, init_pix_pos_batch

    elif 'single_view' in conf:
        endeffector_pos_seq = tf.concat(endeffector_pos_seq, 0)
        action_seq = tf.concat(action_seq, 0)
        [image_aux1_batch, action_batch, endeffector_pos_batch] = tf.train.batch(
            [image_aux1_seq, action_seq, endeffector_pos_seq],
            conf['batch_size'],
            num_threads=num_threads,
            capacity=100 * conf['batch_size'])
        return image_aux1_batch, action_batch, endeffector_pos_batch

    else:
        endeffector_pos_seq = tf.concat(endeffector_pos_seq, 0)
        action_seq = tf.concat(action_seq, 0)
        [image_main_batch, image_aux1_batch, action_batch, endeffector_pos_batch] = tf.train.batch(
                                    [image_main_seq,image_aux1_seq, action_seq, endeffector_pos_seq],
                                    conf['batch_size'],
                                    num_threads=num_threads,
                                    capacity=100 * conf['batch_size'])
        return image_main_batch, image_aux1_batch, action_batch, endeffector_pos_batch


##### code below is used for debugging

def add_visuals_to_batch(image_data, action_data, state_data, action_pos = False):
    batchsize, sequence_length = state_data.shape[0], state_data.shape[1]

    img = np.uint8(255. * image_data)

    image__with_visuals = np.zeros((32, 15, 64, 64, 3), dtype=np.uint8)

    for b in range(batchsize):
        for t in range(sequence_length):
            actions = action_data[b, t]
            state = state_data[b, t, :2]
            sel_img = img[b,t]
            image__with_visuals[b, t] = get_frame_with_visual(sel_img, actions, state, action_pos= action_pos)

    return image__with_visuals.astype(np.float32) / 255.0


def get_frame_with_posdata(img, pos):
    """
    visualizes the actions in the frame
    :param img:
    :param action:
    :param state:
    :param action_pos:
    :return:
    """
    pos = pos.squeeze().reshape(4,2)

    fig = plt.figure(figsize=(1, 1), dpi=64)
    fig.add_subplot(111)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    axes = plt.gca()
    plt.cla()
    axes.axis('off')
    plt.imshow(img, zorder=0)
    axes.autoscale(False)

    for i in range(4):
        pos_img = mujoco_to_imagespace(pos[i])
        plt.plot(pos_img[1], pos_img[0], zorder=1, marker='o', color='b')

    fig.canvas.draw()  # draw the canvas, cache the renderer

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # plt.show()
    Image.fromarray(data).show()
    pdb.set_trace()

    return data


def get_frame_with_visual(img, action, state, action_pos= False):
    """
    visualizes the actions in the frame
    :param img:
    :param action:
    :param state:
    :param action_pos:
    :return:
    """
    fig = plt.figure(figsize=(1, 1), dpi=64)
    fig.add_subplot(111)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    axes = plt.gca()
    plt.cla()
    axes.axis('off')
    plt.imshow(img, zorder=0)
    axes.autoscale(False)

    if action_pos:
        p = mujoco_to_imagespace(action)
    else:
        p = mujoco_to_imagespace(state + .05 * action)

    state = mujoco_to_imagespace(state)

    plt.plot(state[1], state[0], zorder=1, marker='o', color='r')

    yaction = np.array([state[0], p[0]])
    xaction = np.array([state[1], p[1]])
    plt.plot(xaction, yaction, zorder=1, color='y', linewidth=3)

    fig.canvas.draw()  # draw the canvas, cache the renderer

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # plt.show()
    # Image.fromarray(data).show()
    # pdb.set_trace()

    return data

def mujoco_to_imagespace(mujoco_coord, numpix=64):
    viewer_distance = .75  # distance from camera to the viewing plane
    window_height = 2 * np.tan(75 / 2 / 180. * np.pi) * viewer_distance  # window height in Mujoco coords
    pixelheight = window_height / numpix  # height of one pixel
    pixelwidth = pixelheight
    window_width = pixelwidth * numpix
    middle_pixel = numpix / 2
    pixel_coord = np.array([-mujoco_coord[1], mujoco_coord[0]])/pixelwidth + \
                  np.array([middle_pixel, middle_pixel])
    return pixel_coord



def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def write_tf_records(images, actions, states, filepath, init_pix_distrib, init_pix_pos):
    filename = os.path.join(dir, filepath + '/tfrecords/canon_examples.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    feature = {}

    for ex in range(images.shape[0]):
        sequence_length = 15

        for tind in range(sequence_length):
            image_raw = (images[ex,tind]*255.).astype(np.uint8)
            image_raw = image_raw.tostring()

            feature[str(tind) + '/action'] = _float_feature(actions[ex, tind].tolist())
            feature[str(tind) + '/endeffector_pos'] = _float_feature(states[ex,tind].tolist())
            feature[str(tind) + '/image_aux1/encoded'] = _bytes_feature(image_raw)

            pix_raw = (init_pix_distrib[ex][tind]*255.).astype(np.uint8)
            pix_raw = pix_raw.tostring()
            feature[str(tind) + '/init_pix_distrib'] = _bytes_feature(pix_raw)

            if tind == 0:
                feature['/init_pix_pos'] = _float_feature(init_pix_pos[ex].tolist())

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    # for debugging only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print 'using CUDA_VISIBLE_DEVICES=', os.environ["CUDA_VISIBLE_DEVICES"]
    conf = {}

    current_dir = os.path.dirname(os.path.realpath(__file__))
    # DATA_DIR = '/'.join(str.split(current_dir, '/')[:-3]) + '/pushing_data/canonical_examples/tfrecords'
    DATA_DIR = '/'.join(str.split(current_dir, '/')[:-3]) + '/pushing_data/softmotion30_44k/test'
    # conf['canon_ex'] = ""

    # DATA_DIR = '/'.join(str.split(current_dir, '/')[:-3]) + '/pushing_data/softmotion30/test'

    conf['schedsamp_k'] = -1  # don't feed ground truth
    conf['data_dir'] = DATA_DIR  # 'directory containing data_files.' ,
    conf['skip_frame'] = 1
    conf['train_val_split']= 0.95
    conf['sequence_length']= 15      # 'sequence length, including context frames.'
    conf['use_state'] = True
    conf['batch_size']= 32
    conf['visualize']= True
    conf['single_view'] = ''
    conf['context_frames'] = 2

    print '-------------------------------------------------------------------'
    print 'verify current settings!! '
    for key in conf.keys():
        print key, ': ', conf[key]
    print '-------------------------------------------------------------------'

    print 'testing the reader'

    # image_main_batch, image_aux_batch, action_batch, endeff_pos_batch  = build_tfrecord_input(conf, training=False)
    if 'canon_ex' in conf:
        image_aux_batch, action_batch, endeff_pos_batch, pix_distrib_batch, pix_pos_batch = build_tfrecord_input(conf, training=False)
    else:
        image_aux_batch, action_batch, endeff_pos_batch = build_tfrecord_input(conf, training=False)

    sess = tf.InteractiveSession()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    from video_prediction.utils_vpred.create_gif import comp_single_video

    for i in range(2):
        print 'run number ', i

        # image_main, image_aux, actions, endeff = sess.run([image_main_batch, image_aux_batch, action_batch, endeff_pos_batch])
        if 'canon_ex' in conf:
            image_aux, actions, endeff, init_pix_distrib, init_pix_pos  = sess.run([image_aux_batch, action_batch, endeff_pos_batch, pix_distrib_batch, pix_pos_batch])
        else:
            image_aux, actions, endeff = sess.run([image_aux_batch, action_batch, endeff_pos_batch])

        # file_path = '/'.join(str.split(DATA_DIR, '/')[:-1]+['/preview'])
        # comp_single_video(file_path, image_aux)

        # show some frames
        for i in range(10,15):

            print actions[i]
            print endeff[i]

            if 'canon_ex' in conf:
                img = np.uint8(255. *init_pix_distrib[i, 2])
                img = Image.fromarray(np.squeeze(img))
                img.show()

            image_aux = np.squeeze(image_aux)
            img = np.uint8(255. * image_aux[i, 0])
            img = Image.fromarray(img, 'RGB')
            # img.save(file_path,'PNG')
            img.show()
            print i
            pdb.set_trace()

            # pdb.set_trace()