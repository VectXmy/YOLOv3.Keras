from keras.layers import Layer,Lambda
import tensorflow as tf
from . import model_config

class Upsample(Layer):
    def __init__(self,height_factor,width_factor,method=1,**kwargs):
        """0：双线性差值。1：最近邻居法。2：双三次插值法。3：面积插值法"""
        self.height_factor=height_factor
        self.width_factor=width_factor
        self.method=method
        super(Upsample,self).__init__(**kwargs)
    def _upsample(self,inputs,height_factor,width_factor,method):
        # input_shape=inputs.shape.as_list()
        input_shape=tf.shape(inputs)
        return tf.image.resize_images(inputs,(input_shape[1]*height_factor,input_shape[2]*width_factor),method=method)
    def call(self,inputs):
        return self._upsample(inputs,self.height_factor,self.width_factor,self.method)
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[1]*self.height_factor,input_shape[2]*self.width_factor,input_shape[3])
    def get_config(self):
        config={'height_factor':self.height_factor,
                'width_factor':self.width_factor,
                'method':self.method}
        base_config=super(Upsample,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class Decode(Layer):
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_num_per_grid, 5 + num_classes]
            contains (x, y, w, h, score, probability)
            x,y为中心
    """
    def __init__(self,class_num,stride,anchor_size,anchor_num,**kwargs):
        self.class_num=class_num
        self.stride=stride
        self.anchor_size=anchor_size
        self.anchor_num=anchor_num
        super(Decode,self).__init__(**kwargs)
    def call(self,inputs):
        input_shape=tf.shape(inputs)
        batch_size,out_size=input_shape[0],input_shape[1]

        inputs=tf.reshape(inputs,(batch_size,out_size,out_size,self.anchor_num,5+self.class_num))

        raw_dxdy=inputs[...,0:2]
        raw_dwdh=inputs[...,2:4]
        raw_conf=inputs[...,4:5]
        raw_prob=inputs[...,5:]

        #         >>> np.tile(np.arange(2)[:,np.newaxis],[1,2])
        #           array([[0, 0],
        #                   [1, 1]])
        #         >>> np.tile(np.arange(2)[np.newaxis,:],[2,1])
        #             array([[0, 1],
        #                   [0, 1]])
        grid_y=tf.tile(tf.range(out_size,dtype=tf.int32)[:,tf.newaxis],[1,out_size])
        grid_x=tf.tile(tf.range(out_size,dtype=tf.int32)[tf.newaxis,:],[out_size,1])
        grid_xy=tf.concat([grid_x[:, :, tf.newaxis], grid_y[:, :, tf.newaxis]],axis=-1)
        # shape(2,2,2)
        # [[[0,0],[0,1]],
        # [[1,0],[1,1]]]
        grid_xy=tf.tile(grid_xy[tf.newaxis,:,:,tf.newaxis,:],[batch_size,1,1,self.anchor_num,1])
        # shape(batch_size,2,2,anchor_num,2),最后一维为grid的x和y坐标,和inputs的格式一样
        grid_xy=tf.cast(grid_xy,tf.float32)
        pred_xy=(tf.sigmoid(raw_dxdy)+grid_xy)*self.stride
        pred_wh=(tf.exp(raw_dwdh)*self.anchor_size)*self.stride
        # pred_xywh=tf.concat([pred_xy,pred_wh],axis=-1)

        pred_conf=tf.sigmoid(raw_conf)
        pred_prob=tf.sigmoid(raw_prob)

        return tf.concat([pred_xy,pred_wh,pred_conf,pred_prob],axis=-1)

    def compute_output_shape(self,input_shape):
        batch_size,out_size=input_shape[0],input_shape[1]
        return (batch_size,out_size,out_size,self.anchor_num,5+self.class_num)
    def get_config(self):
        config={'class_num':self.class_num,
                'stride':self.stride,
                'anchor_size':self.anchor_size,
                'anchor_num':self.anchor_num}
        base_config=super(Decode,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
'''metrics layers'''
from .metrics import m_iou
def m_iou_lambda_func(inputs):
    branch_and_decodedbranch=inputs[0:6]
    label_and_bboxes=inputs[6:]
    return m_iou(label_and_bboxes,branch_and_decodedbranch)
MeanIou=Lambda(function=m_iou_lambda_func,name="m_iou")

'''loss layers'''
from .losses import conf_loss,prob_loss,giou_loss,bbox_loss
def conf_loss_lambda_func(inputs):
    branch_and_decodedbranch=inputs[0:6]
    label_and_bboxes=inputs[6:]
    return conf_loss(label_and_bboxes,branch_and_decodedbranch)
def prob_loss_lambda_func(inputs):
    branch_and_decodedbranch=inputs[0:6]
    label_and_bboxes=inputs[6:]
    return prob_loss(label_and_bboxes,branch_and_decodedbranch)
def bbox_loss_lambda_func(inputs):
    branch_and_decodedbranch=inputs[0:6]
    label_and_bboxes=inputs[6:]
    return bbox_loss(label_and_bboxes,branch_and_decodedbranch)  
def giou_loss_lambda_func(inputs):
    branch_and_decodedbranch=inputs[0:6]
    label_and_bboxes=inputs[6:]
    return giou_loss(label_and_bboxes,branch_and_decodedbranch)  
ConfidenceLoss=Lambda(function=conf_loss_lambda_func,name="conf_loss")
ProbabilityLoss=Lambda(function=prob_loss_lambda_func,name="prob_loss")
BboxLoss=Lambda(function=bbox_loss_lambda_func,name="bbox_loss")
GiouLoss=Lambda(function=giou_loss_lambda_func,name="giou_loss")
# class ConfidenceLoss(Layer):
#     '''
#     output conf loss  
#     input  [branch_1,decoded_branch_1,  branch_2,decoded_branch_2,branch_3,decoded_branch_3,label_1,bboxes_1,label_2,bboxes_2,label_3,bboxes_3]]
#     '''
#     def __init__(self,**kwargs):
#         super(ConfidenceLoss,self).__init__(**kwargs)

#     def call(self,inputs):
#         branch_and_decodedbranch=inputs[0:6]
#         label_and_bboxes=inputs[6:]
#         return self._loss(branch_and_decodedbranch,label_and_bboxes)

#     def compute_output_shape(self,input_shape):
#         return []
#     def _loss(self,branch_and_decodedbranch,label_and_bboxes):
#         return conf_loss(label_and_bboxes,branch_and_decodedbranch)

# class ProbabilityLoss(Layer):
#     '''
#     output prob loss  
#     input  [branch_1,decoded_branch_1,  branch_2,decoded_branch_2,branch_3,decoded_branch_3,label_1,bboxes_1,label_2,bboxes_2,label_3,bboxes_3]]
#     '''
#     def __init__(self,**kwargs):
#         super(ProbabilityLoss,self).__init__(**kwargs)

#     def call(self,inputs):
#         branch_and_decodedbranch=inputs[0:6]
#         label_and_bboxes=inputs[6:]
#         return self._loss(branch_and_decodedbranch,label_and_bboxes)

#     def compute_output_shape(self,input_shape):
#         return []
#     def _loss(self,branch_and_decodedbranch,label_and_bboxes):
#         return prob_loss(label_and_bboxes,branch_and_decodedbranch)
# class BboxLoss(Layer):
#     '''
#     output bbox loss  
#     input  [branch_1,decoded_branch_1,  branch_2,decoded_branch_2,branch_3,decoded_branch_3,label_1,bboxes_1,label_2,bboxes_2,label_3,bboxes_3]]
#     '''
#     def __init__(self,**kwargs):
#         super(BboxLoss,self).__init__(**kwargs)

#     def call(self,inputs):
#         branch_and_decodedbranch=inputs[0:6]
#         label_and_bboxes=inputs[6:]
#         return self._loss(branch_and_decodedbranch,label_and_bboxes)

#     def compute_output_shape(self,input_shape):
#         return []
#     def _loss(self,branch_and_decodedbranch,label_and_bboxes):
#         return bbox_loss(label_and_bboxes,branch_and_decodedbranch)
        