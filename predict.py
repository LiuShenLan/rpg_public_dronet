import gflags
import numpy as np
import os
import sys
import glob
from random import randint
from sklearn import metrics

from keras import backend as K

import utils
import img_utils
from constants import TEST_PHASE
from common_flags import FLAGS

FLAGS(sys.argv)
import socket


def main(argv):
    # 加载路径
    COLLISION_THRESHOLD = 0.5
    img_path = "../../dataset/hostData/realTimeImg.jpg"
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
        img = img_utils.load_img(img_path,
                                 grayscale=img_grayscale,
                                 crop_size=crop_size,
                                 target_size=target_size) * np.float32(1.0 / 255.0)
        # 预测结果
        outs = model.predict_on_batch(img[None])
        steer = outs[0][0][0]  # 转向数据
        collision_pred = outs[1][0][0]  # 碰撞概率
        collision_label = collision_pred >= COLLISION_THRESHOLD
        print("\rsteer: {:<+.10f}, collision_pred: {:<.10f} collision_label: {}   "
              .format(steer, collision_pred, collision_label), end='')

        # TCP socket 发送数据
        seq = '%f' % collision_pred
        TCP_socket.send(seq.encode('utf-8'))  # send datas

if __name__ == "__main__":
    main(sys.argv)
