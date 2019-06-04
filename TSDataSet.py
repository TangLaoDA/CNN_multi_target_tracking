import os
import tensorflow as tf
from PIL import Image
import numpy as np

class MyDataset:
    def __init__(self,path,batch_size):
        self.path = path
        self.filenames = os.listdir(path)
        self.labels = list(map(lambda filename:int(filename.split(".")[5]),self.filenames))

        self.dataset = tf.data.Dataset.from_tensor_slices((self.filenames,self.labels))
        self.dataset = self.dataset.map(
            lambda filename,label:tuple(tf.py_func(
                self._read_py_function,[filename,label],[tf.float32,label.dtype])))
        self.dataset = self.dataset.shuffle(buffer_size=3)
        self.dataset = self.dataset.repeat()
        self.dataset = self.dataset.batch(batch_size)

        iterator = self.dataset.make_one_shot_iterator()
        self.next_element = iterator.get_next()
        pass
    def get_bacth(self,sess):
        return sess.run(self.next_element)
    #将传入的图片路径转为图片对象
    def _read_py_function(self,filename, label):
        _filename=bytes.decode(filename)
        img_path = os.path.join(self.path,_filename)
        im = Image.open(img_path)
        im = im.convert("L")
        im = im.resize((300,300))
        image_data = np.array(im,dtype=np.float32)/255.0
        return image_data,label
init=tf.global_variables_initializer()
if __name__ == '__main__':
    mydata = MyDataset("JsamplePic",10)
    with tf.Session() as sess:
        xs,ys = mydata.get_bacth(sess)

        bn = tf.contrib.layers.batch_norm(xs, decay=0.9, updates_collections=None, epsilon=1e-5,
                                          scale=True,
                                          is_training=True)
        sess.run(init)
        print(sess.run(bn))



