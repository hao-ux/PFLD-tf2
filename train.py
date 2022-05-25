

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
from utils.callback import LossHistory
from utils.loss import PFLDLoss
from tensorflow.keras.optimizers import Adam
from datasets import PFLDDatasets
from nets.PFLD import PFLDInference
import warnings
warnings.filterwarnings('ignore')


# 使用gpu
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
# -------------------------------- #
# batch_size -> 批次
# epochs -> 轮次
# model_path -> 预训练权重路径
# input_shape -> 输入图片大小
# lr -> 学习率
# -------------------------------- #
batch_size = 32
epochs = 100
model_path = ''
input_shape = [112,112,3]
lr=1e-3

def adjust_lr(epoch, lr=lr):
    print("Seting to %s" % (lr))
    if epoch < 3:
        return lr
    else:
        return lr * 0.93
    
if __name__ == '__main__':
    traindatasets = PFLDDatasets('./train_data/list.txt', batch_size)
    validdatasets = PFLDDatasets('./test_data/list.txt', batch_size)

    
    model = PFLDInference(input_shape, is_train=True)
    if model_path != '':
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
    
    callback = [
            EarlyStopping(monitor='loss', patience=15, verbose=1),
            ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',monitor='val_loss',
                            save_weights_only=True, save_best_only=False, period=1),
            TensorBoard(log_dir='./logs1'),
            LossHistory('./logs1'),
            LearningRateScheduler(adjust_lr)
        ]
    
    model.compile(
        loss={'train_out': PFLDLoss()}, optimizer=Adam(learning_rate=lr)
    )
    history = model.fit(
            x                      = traindatasets,
            validation_data        = validdatasets,
            workers                = 1,
            epochs                 = epochs,
            callbacks              = callback,
            steps_per_epoch        = len(traindatasets),
            validation_steps       = len(validdatasets),
            verbose=1
        )
    
    
    
