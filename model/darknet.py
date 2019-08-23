import keras
from keras.layers import Conv2D,UpSampling2D,ZeroPadding2D,BatchNormalization,LeakyReLU,Add
import tensorflow as tf
from keras.models import Model

def DBL(inputs,filters,kernel_size,downsample=False,activate=True,bn=True):
    if downsample:
        padding='valid'
        strides=2
        inputs=ZeroPadding2D(((1,0),(1,0)))(inputs)#stride为2，输入边长为偶数，补一个零
    else:
        padding='same'
        strides=1
    inputs=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,
                padding=padding,use_bias=not bn,kernel_regularizer=keras.regularizers.l2(0.0005),
                kernel_initializer=keras.initializers.random_normal(stddev=0.01),
                bias_initializer=keras.initializers.constant(0.))(inputs)
    if bn: inputs=BatchNormalization()(inputs)
    if activate: inputs=LeakyReLU(alpha=0.1)(inputs)
    return inputs

def res_unit(inputs,filters_1,filters_out):
    short=inputs
    conv=DBL(inputs,filters=filters_1,kernel_size=(1,1))
    conv=DBL(conv,filters=filters_out,kernel_size=(3,3))
    outputs=Add()([short,conv])
    return outputs


def Darknet53(inputs):
    inputs=DBL(inputs,32,(3,3))

    inputs=DBL(inputs,64,(3,3),downsample=True)
    for i in range(1):
        inputs=res_unit(inputs,32,64)

    inputs=DBL(inputs,128,(3,3),downsample=True)
    for i in range(2):
        inputs=res_unit(inputs,64,128)

    inputs=DBL(inputs,256,(3,3),downsample=True)
    for i in range(8):
        inputs=res_unit(inputs,128,256)
    branch1=inputs

    inputs=DBL(inputs,512,(3,3),downsample=True)
    for i in range(8):
        inputs=res_unit(inputs,256,512)
    branch2=inputs

    inputs=DBL(inputs,1024,(3,3),downsample=True)
    for i in range(4):
        outputs=res_unit(inputs,512,1024)

    return branch1,branch2,outputs


