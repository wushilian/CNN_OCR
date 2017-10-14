import tensorflow as tf
import numpy as np
import os
import cv2
import model
import random
import matplotlib.pyplot as plt
import config as cfg

charset=cfg.charset
num_class=cfg.num_class
width=cfg.width
height=cfg.height
epoch=cfg.epoch
batch=cfg.batch
model_ckpt = cfg.model_ckpt
train_dir=cfg.train_dir
test_dir=cfg.test_dir
pretrained_path=cfg.pretrained_path

def read_data(dir_name):
    train_dir =dir_name
    img_names = os.listdir(train_dir)
    data=[]
    x=[]
    y=[]
    for img_name in img_names:
        img=cv2.imread(train_dir+'/'+img_name,0)
        img=cv2.resize(img,(width,height))
        #plt.imshow(img,cmap='gray')
        #plt.show()
        img=img[:,:,np.newaxis]

        label = img_name.split('.')[0].split('_')[1]#img_name:0_A.jpg
        data.append([img,label])
    #random.shuffle(data)
    return data
def shuffle(data):
    x=[]
    y=[]
    random.shuffle(data)
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(data[i][1])
    length = len(y)
    label = np.zeros([length, num_class])
    for i in range(length):
        index = charset.index(y[i])
        label[i][index] = 1
    return  np.array(x),label
def read_test(dirs):
    test_dir = dirs
    img_names = os.listdir(test_dir)
    x = []
    y = []
    for img_name in img_names:
        img = cv2.imread(test_dir + '/' + img_name, 0)
        img = cv2.resize(img, (width,height))
        img = img[:, :, np.newaxis]
        label = img_name.split('.')[0].split('_')[1]
        x.append(img)
        y.append(label)
    length = len(y)
    label = np.zeros([length, num_class])
    for i in range(length):
        index = charset.index(y[i])
        label[i][index] = 1
        #print(y[i],label[i])
    return np.array(x),np.array(label)
def label_img(testdir):#recognize images from an dir
    img_names = os.listdir(testdir)
    x = []
    for img_name in img_names:
        img = cv2.imread(testdir + '/' + img_name, 0)
        img = cv2.resize(img, (width, height))
        img = img[:, :, np.newaxis]
        x.append(img)
    x=np.array(x)
    print('have load image',len(img_names))
    sess = tf.InteractiveSession()

    cnn = model.cnn_ocr()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    path =pretrained_path
    saver.restore(sess, path)
    print('model has been restored')
    predict=sess.run(cnn.predict,feed_dict={cnn.x:x})
    #decode label
    for i in range(len(img_names)):
        temp=np.argwhere(predict[i]==1.0)
        if(len(temp)==1):
            index=int(np.argwhere(predict[i]==1.0))
            os.rename(testdir + '/' + img_names[i],
                      testdir + '/' + str(i) + '_' + charset[index] + '.jpg')
def train(Is_restore=False):
    sess = tf.InteractiveSession()
    cnn = model.cnn_ocr()
    optimizer=tf.train.AdadeltaOptimizer(0.01)
    train_step = optimizer.minimize(cnn.loss)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    if Is_restore:
        saver.restore(sess,pretrained_path)
        print('model has been restored')
    data=read_data(train_dir)
    print('have load image:',len(data))
    x_test,y_test=read_test(test_dir)
    print('begin training---------------------------------------------------------------')
    for i in range(epoch):
        x, y = shuffle(data)
        x = x / 255
        num_data = len(y)
        for j in range(int(num_data/batch)):
            _,trainloss,trainacc=sess.run([train_step,cnn.loss,cnn.acc],feed_dict={cnn.x:x[j*batch:(j+1)*batch],cnn.y:y[j*batch:(j+1)*batch]})
            if j%50==0:
                #trainacc=sess.run(cnn.acc,feed_dict={cnn.x:x[0:500],cnn.y:y[0:500]})
                acc=sess.run(cnn.acc,feed_dict={cnn.x:x_test,cnn.y:y_test})
                print('epoch:',i,' trainloss:',trainloss,' trainacc:',trainacc,'  acc:',acc)
        saver.save(sess,model_ckpt,global_step=i)
train(Is_restore=False)
#label_img('')
