# import gflags
import numpy as np
import os
import sys
# import glob
# from random import randint
# from sklearn import metrics
import cv2

from keras import backend as K
import tensorflow as tf

import utils
import img_utils
from img_utils import central_image_crop
from constants import TEST_PHASE
from common_flags import FLAGS
import socket
import fcntl
import time

FLAGS(sys.argv)

# 按需分配显存，防止Keras吃满显存
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需分配显存
K.tensorflow_backend.set_session(tf.Session(config=config))

COLLISION_THRESHOLD = 0.5

def main(argv):
    # 碰撞概率低通滤波
    global img, TCP_socket
    collision_pre = 0   # 碰撞概率低通滤波前次数值
    collision_filter_parameters = 0.5   # 碰撞概率低通滤波参数，越大新进来的值影响越小
    # 加载路径
    img_path = "../../dataset/hostData/realTimeImg/realTimeImg.jpg"
    img_file = open(img_path, 'r')
    img_grayscale = FLAGS.img_mode == 'grayscale'

    # 图片大小
    img_width, img_height = FLAGS.img_width, FLAGS.img_height
    target_size = (img_width, img_height)
    crop_img_width, crop_img_height = FLAGS.crop_img_width, FLAGS.crop_img_height
    crop_size = (crop_img_width, crop_img_height)

    # 设置测试模式
    K.set_learning_phase(TEST_PHASE)

    # 加载 json 并创建模型
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    model = utils.jsonToModel(json_model_path)

    # 加载权重
    weights_load_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.weights_fname)
    model.load_weights(weights_load_path)

    model.compile(loss='mse', optimizer='adam')

    # TCP socket
    print("TCP socket connecting")
    while True:
        try:
            TCP_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            TCP_socket.connect(("127.0.0.1", 5556))  # 127.0.0.1为本机IP地址
            break
        except:
            continue

    while True:
        # 读取图像
        fcntl.flock(img_file, fcntl.LOCK_EX)
        img = cv2.imread(img_path)
        fcntl.flock(img_file, fcntl.LOCK_UN)

        if img_grayscale:
            if len(img.shape) != 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if target_size:
            if (img.shape[0], img.shape[1]) != target_size:
                img = cv2.resize(img, target_size)

        if crop_size:
            img = central_image_crop(img, crop_size[0], crop_size[1])

        if img_grayscale:
            img = img.reshape((img.shape[0], img.shape[1], 1))

        img = np.asarray(img, dtype=np.float32) * np.float32(1.0 / 255.0)

        # 预测结果
        outs = model.predict_on_batch(img[None])
        steer = outs[0][0][0]  # 转向数据
        collision_predict = outs[1][0][0]  # 预测碰撞概率
        collision_probability = collision_pre * collision_filter_parameters + collision_predict * (1 - collision_filter_parameters)
        collision_pre = collision_probability

        # TCP socket 发送数据
        seq = '{:.10f}'.format(collision_probability)
        TCP_socket.send(seq.encode('utf-8'))  # send datas

        print("\rsteer: {:<+.10f}, collision_predict: {:<.10f}   ".format(steer, collision_predict), end='')


if __name__ == "__main__":
    main(sys.argv)
