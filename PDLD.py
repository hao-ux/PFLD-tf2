import tensorflow as tf
from nets.PFLD import PFLDInference
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


class FaceKeyPointsNet(object):
    _defaults = {
        "model_path": "./model_data/model.h5", # 权重路径
        "input_shape": [112, 112, 3],  # 输入图片大小
    }
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
        
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        
        # 使用gpu
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        self.generate()
        print("导入模型成功！！！")
        
    def generate(self):
        self.model = PFLDInference(self.input_shape, is_train=False)
        self.model.load_weights(self.model_path, by_name=True)
        
    def detect_image(self, img):
        
        img_copy = np.copy(img)
        img_copy_h, img_copy_w = img_copy.shape[:2]
        
        img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
        img_rgb = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)
        img_data = np.expand_dims(np.array(img)/255.0, 0)
        landmark = self.model.predict(img_data)[0] * self.input_shape[0]
        landmark[0::2] = landmark[0::2] * img_copy_w / self.input_shape[0]
        landmark[1::2] = landmark[1::2] * img_copy_h / self.input_shape[0]
        self.show(img_rgb, landmark)
    
    def show(self, img, landmark):
        plt.imshow(img)
        for i in range(0, len(landmark),2):
            plt.scatter(landmark[i], landmark[i+1], s=20, marker='.', c='m')
        plt.show()
        
    def fps(self, img, n=100):
        start = time.time()
        img = np.array(img)
        img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
        img_data = np.expand_dims(np.array(img)/255.0, 0)
        for _ in range(n):
            landmark = self.model.predict(img_data)[0]
        end = time.time()
        avg_time = (end - start)/n
        return avg_time
        
