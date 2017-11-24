import tensorflow as tf
slim=tf.contrib.slim
class cnn_ocr:

    # Create model
    def __init__(self):
        width=50
        height=50
        self.num_class=34
        self.x = tf.placeholder(tf.float32, [None, width,height,1])
        self.predict=self.network(self.x)
        self.y = tf.placeholder(tf.float32, [None,self.num_class])
        self.loss = self.loss_with_step()
        self.acc=self.acc_cal()
    def network(self, x):
        conv1=slim.conv2d(x, 6,[5,5], scope='conv1')
        pool1=slim.max_pool2d(conv1, [2, 2], scope='pool1')
        conv2=slim.conv2d(pool1, 16, [5,5], scope='conv2')
        pool2=slim.max_pool2d(conv2, [2, 2], scope='pool2')
        conv3=slim.conv2d(pool2, 120, [5,5], scope='conv3')
        conv4 = slim.conv2d(conv3, 240, [3,3], scope='conv4')
        flat=slim.flatten(conv4)
        fc1=slim.fully_connected(flat, 2048, scope='fc1')
        drop1=slim.dropout(fc1,0.5,scope='dropout1')
        fc2=slim.fully_connected(drop1, 1024, scope='fc2')
        fc3 = slim.fully_connected(fc2, 512, scope='fc3')
        predict = slim.fully_connected(fc3, self.num_class, activation_fn=tf.nn.softmax, scope='pr0')
        #safe_exp = tf.clip_by_value(fc4, 1e-10, 10)

        return predict


    def loss_with_step(self):#may be exist some error

        safe_log=tf.clip_by_value(self.predict,1e-5,1e100)

        cross_entropy = -tf.reduce_sum(self.y*tf.log(safe_log))
        return cross_entropy
    def acc_cal(self):
        correct_prediction = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return accuracy
