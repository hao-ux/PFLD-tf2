
from nets.PFLD import PFLDInference

# 输出网络结构

if __name__ == '__main__':
    inputs = [112, 112, 1]
    model = PFLDInference(inputs=inputs, is_train=True)
    model.summary()
    print(model.output)
    
    
    