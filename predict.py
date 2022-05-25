
import cv2
import tensorflow as tf

from PDLD import FaceKeyPointsNet

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ----------------------- #
# mode=1 测试单张图片预测时间
# mode=0 测试单张图片效果展示

mode = 0
# 测试100次，仅在mode=1有效
n = 100

if __name__ == "__main__":
    facekeypoints = FaceKeyPointsNet()
    if mode == 0:
        while True:
            img = input('Input image filename:')
            try:
                image = cv2.imread(img)
            except:
                print('打开错误！')
            # 检测图片
            facekeypoints.detect_image(image)
    elif mode == 1:
        img = cv2.imread('./img/1.png')
        t = facekeypoints.fps(img, n=100)
        print("FPS:{}, n: {}, time: {}".format(1/t, n, t))