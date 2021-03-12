# -*- coding:utf-8 -*-
from random import random, shuffle

import tensorflow as tf
import easydict
import numpy as np
import os

FLAGS = easydict.EasyDict({"img_size": 224,
                           
                           "load_size": 266,
                           
                           "batch_size": 32,
                           
                           "epochs": 500,
                           
                           "n_classes": 55,

                           "lr": 0.001,
                           
                           "tr_img_path": "D:/[1]DB/[4]etc_experiment/UTK_face/drive-download-20210310T064015Z-001/UTKFace/",
                           
                           "tr_txt_path": "D:/[1]DB/[4]etc_experiment/UTK_face/crop_face_train.txt",
                           
                           "re_img_path": "D:/[1]DB/[4]etc_experiment/UTK_face/drive-download-20210310T063149Z-001/ALL/",
                           
                           "re_txt_path": "D:/[1]DB/[4]etc_experiment/UTK_face/full_face_train.txt",

                           "te_img_path": "D:/[1]DB/[4]etc_experiment/UTK_face/drive-download-20210310T063149Z-001/ALL/",

                           "te_txt_path": "D:/[1]DB/[4]etc_experiment/UTK_face/full_face_test.txt",
                           
                           "train": True,
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "save_checkpoint": ""})


tr_optim = tf.keras.optimizers.Adam(FLAGS.lr)
re_optim = tf.keras.optimizers.Adam(FLAGS.lr)

def func(tr_list, re_list):

    tr_img = tf.io.read_file(tr_list)
    tr_img = tf.image.decode_jpeg(tr_img, 3)
    tr_img = tf.image.resize(tr_img, [FLAGS.img_size, FLAGS.img_size])
    tr_img = tf.image.per_image_standardization(tr_img)

    re_img = tf.io.read_file(re_list)
    re_img = tf.image.decode_jpeg(re_img, 3)
    re_img = tf.image.resize(re_img, [FLAGS.img_size, FLAGS.img_size])
    re_img = tf.image.per_image_standardization(re_img)


    return tr_img, tr_list, re_img, re_list

def te_func(img, lab):

    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)


    return img, lab

def feature_cal_loss(tr_logits, re_logits):

    tr_logits = tf.nn.sigmoid(tr_logits)
    re_logits = tf.nn.sigmoid(re_logits)
    energy_ft = tf.reduce_sum(tf.abs(tr_logits - re_logits), 1)
 
    Q = 10
    total_loss = tf.reduce_mean(2*energy_ft*energy_ft/(Q))

    return total_loss

@tf.function
def train_step(tr_img, tr_lab, re_img, re_lab, tr_model, re_model):

    # tr model 에는 crop된 이미지들, re model 에는 full shot 이미지들을 넣어서 full shot 이미지에 대한것을 나이 추정에 써보자
    # 각 age는 동일하다고 가정하자
    with tf.GradientTape(persistent=True) as tape:
        tr_logits, tr_class_logits = tr_model(tr_img, True)
        re_logits, re_class_logits = re_model(re_img, True)

        tr_class_softmax = tf.nn.softmax(tr_class_logits, 1)
        re_class_softmax = tf.nn.softmax(re_class_logits, 1)

        diff_softmax = tf.abs(tr_class_softmax - re_class_softmax)

        feature_loss = feature_cal_loss(tr_logits, re_logits)

        cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(re_lab, diff_softmax)

        total_loss = feature_loss + cross_entropy_loss

    tr_grads = tape.gradient(total_loss, tr_model.trainable_variables)
    re_grads = tape.gradient(total_loss, re_model.trainable_variables)
    tr_optim.apply_gradients(zip(tr_grads, tr_model.trainable_variables))
    re_optim.apply_gradients(zip(re_grads, re_model.trainable_variables))

    return total_loss, feature_loss, cross_entropy_loss

def age_label(tr_name_list, re_name_list):

    age_buf = []
    for i in range(FLAGS.batch_size):
        tr_name = tf.compat.as_str_any(tr_name_list[i].numpy())
        re_name = tf.compat.as_str_any(re_name_list[i].numpy())
        
        tr_name = tr_name.split("/")
        tr_name = tr_name[-1].split("_")
        age = int(tr_name[0]) - 16
        age_buf.append(age)
    age_buf = tf.convert_to_tensor(age_buf)

    return age_buf

@tf.function
def test_step(input, label, model):
    _, logits = model(input, False)
    logits = tf.nn.softmax(logits, 1)
    logits = tf.argmax(logits, 1, tf.int32)
    
    predict = tf.abs(logits - label)

    return predict[0]

def main():
    tr_model = tf.keras.applications.MobileNetV2(input_shape=(FLAGS.img_size, FLAGS.img_size, 3),
                                                 include_top=False) # 잠시 임시로 해놓은것
    re_model = tf.keras.applications.ResNet50V2(input_shape=(FLAGS.img_size, FLAGS.img_size, 3),
                                                 include_top=False) # 잠시 임시로 해놓은것

    tr_model.summary()
    re_model.summary()

    h = tr_model.output
    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    r = re_model.output
    r = tf.keras.layers.GlobalAveragePooling2D()(r)

    h_logits = tf.keras.layers.Dense(1024)(h)
    r_logits = tf.keras.layers.Dense(1024)(r)

    h = tf.keras.layers.Dense(FLAGS.n_classes)(h_logits)
    r = tf.keras.layers.Dense(FLAGS.n_classes)(r_logits)

    tr_model = tf.keras.Model(inputs=tr_model.input, outputs=[h_logits, h])
    re_model = tf.keras.Model(inputs=re_model.input, outputs=[r_logits, r])

    tr_model.summary()
    re_model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("===============")
            print("* Restored!!! *")
            print("===============")

    if FLAGS.train:
        count = 0

        tr_img = np.loadtxt(FLAGS.tr_txt_path, dtype="<U200", skiprows=0, usecols=0)
        tr_img = [FLAGS.tr_img_path + img for img in tr_img]

        re_img = np.loadtxt(FLAGS.re_txt_path, dtype="<U200", skiprows=0, usecols=0)
        re_img = [FLAGS.re_img_path + img for img in re_img]

        test_data = np.loadtxt(FLAGS.te_txt_path, dtype="<U200", skiprows=0, usecols=0)
        test_data = [FLAGS.te_img_path + img for img in test_data]
        test_lab = np.loadtxt(FLAGS.te_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_gener = tf.data.Dataset.from_tensor_slices((test_data, test_lab))
        te_gener = te_gener.map(te_func)
        te_gener = te_gener.batch(1)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        for epoch in range(FLAGS.epochs):

            T = list(zip(tr_img, re_img))
            shuffle(T)
            tr_img, re_img = zip(*T)
            tr_img, re_img = np.array(tr_img), np.array(re_img)

            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img, re_img))
            tr_gener = tr_gener.map(func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)


            tr_idx = len(tr_img) // FLAGS.batch_size
            tr_iter = iter(tr_gener)
            for step in range(tr_idx):
                tr_images, tr_name, re_images, re_name = next(tr_iter)

                labels = age_label(tr_name, re_name)

                total_loss, feature_loss, cross_entropy_loss = train_step(tr_images, labels, 
                                  re_images, labels,
                                  tr_model, re_model)

                if count % 10 == 0:
                    print("Epoch: {} [{}/{}]\ntotal loss = {}\nfeature loss = {}\ncross entropy loss = {}".format(epoch, step + 1, tr_idx, total_loss,
                                                                                                                  feature_loss, cross_entropy_loss))

                if count % 50 == 0 and count != 0:
                    te_iter = iter(te_gener)
                    te_idx = len(test_data) // 1
                    ae = 0
                    for i in range(te_idx):
                        test_images, test_labels = next(te_iter)

                        ae += test_step(test_images, test_labels, re_model)
                    print("===========================")
                    print("{} steps test MAE = {}".format(count, ae / te_idx))
                    print("===========================")

                count += 1


if __name__ == "__main__":
    main()