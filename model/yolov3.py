from .darknet import Darknet53,DBL,res_unit
from keras.models import Model
from keras.layers import Input,Concatenate,Add,Reshape
import tensorflow as tf
from .layers import Upsample,Decode,ConfidenceLoss,ProbabilityLoss,BboxLoss,MeanIou,GiouLoss
from .utils import get_anchors
from . import model_config
import numpy as np
from keras.optimizers import Adam

def yolov3(inputs,class_num):
    branch1,branch2,outputs=Darknet53(inputs)

    ########large object branch#########
    outputs=DBL(outputs,521,(1,1))
    outputs=DBL(outputs,1024,(3,3))
    outputs=DBL(outputs,521,(1,1))
    outputs=DBL(outputs,1024,(3,3))
    outputs=DBL(outputs,521,(1,1))

    large_branch=DBL(outputs,1024,(3,3))
    large_branch=DBL(large_branch,3*(class_num+5),(1,1),activate=False,bn=False)#shape = [None, 13, 13, 255]
    
    ########medium object branch########
    outputs=DBL(outputs,256,(1,1))
    outputs=Upsample(2,2,method=1)(outputs)#使用的是最近邻插值方法，不需要学习，减少了网络参数
    outputs=Concatenate(axis=-1)([outputs,branch2])

    outputs=DBL(outputs,256,(1,1))
    outputs=DBL(outputs,512,(3,3))
    outputs=DBL(outputs,256,(1,1))
    outputs=DBL(outputs,512,(3,3))
    outputs=DBL(outputs,256,(1,1))

    medium_branch=DBL(outputs,512,(3,3))
    medium_branch=DBL(medium_branch,3*(class_num+5),(1,1),activate=False,bn=False)#shape = [None, 26, 26, 255]

    ########small object branch########
    outputs=DBL(outputs,128,(1,1))
    outputs=Upsample(2,2,1)(outputs)
    outputs=Concatenate(axis=-1)([outputs,branch1])

    outputs=DBL(outputs,128,(1,1))
    outputs=DBL(outputs,256,(3,3))
    outputs=DBL(outputs,128,(1,1))
    outputs=DBL(outputs,256,(3,3))
    outputs=DBL(outputs,128,(1,1))

    small_branch=DBL(outputs,256,(3,3))
    small_branch=DBL(small_branch,3*(class_num+5),(1,1),activate=False,bn=False)#shape = [None, 52, 52, 255]

    return [small_branch,medium_branch,large_branch]


def YOLOv3_model(mode='train',optimizer="adam"):
    input_image=Input(shape=model_config.input_shape,name="input_image")
    slabel=Input(shape=(52,52,3,5+model_config.class_num),name="slabel")
    sbboxes=Input(shape=(model_config.max_bbox_per_scale,4),name="sbboxes")
    mlabel=Input(shape=(26,26,3,5+model_config.class_num),name="mlabel")
    mbboxes=Input(shape=(model_config.max_bbox_per_scale,4),name="mbboxes")
    llabel=Input(shape=(13,13,3,5+model_config.class_num),name="llabel")
    lbboxes=Input(shape=(model_config.max_bbox_per_scale,4),name="lbboxes")

    all_branch=yolov3(input_image,model_config.class_num)

    
    anchors=get_anchors("./anchors_size.txt")
    # anchors=np.array(model_config.anchors_size,dtype=np.float32)

    if mode=='train':
        branch_and_decodedbranch_tensor=[]
        for i,branch in enumerate(all_branch):
            pred_tensor=Decode(model_config.class_num,stride=model_config.strides[i],anchor_size=anchors[i],anchor_num=model_config.anchor_num_per_grid)(branch)
            branch_and_decodedbranch_tensor.append(branch)
            branch_and_decodedbranch_tensor.append(pred_tensor)
        target=[slabel,sbboxes,mlabel,mbboxes,llabel,lbboxes]
        branch_and_decodedbranch_tensor.extend(target)
        matched_tensor=branch_and_decodedbranch_tensor

        ######output losses ,metrics##############################
        conf_loss=ConfidenceLoss(matched_tensor)
        prob_loss=ProbabilityLoss(matched_tensor)
        # bbox_loss=BboxLoss(matched_tensor)
        giou_loss=GiouLoss(matched_tensor)
        m_iou=MeanIou(matched_tensor)

        model=Model(input=[input_image,slabel,sbboxes,mlabel,mbboxes,llabel,lbboxes],
                        output=[conf_loss,prob_loss,giou_loss])
        
        ######添加losses################################
        model._losses=[]
        model._per_input_losses={}
        for loss_name in ["conf_loss","prob_loss","giou_loss"]:
            layer = model.get_layer(loss_name)
            if layer.output in model.losses:
                continue
            model.add_loss(layer.output)
        model.compile(optimizer=Adam())
        #######添加metric#########################################
        model.metrics_names.extend(["m_iou","conf_loss","prob_loss","giou_loss","lr"])
        model.metrics_tensors.extend([m_iou,conf_loss,prob_loss,giou_loss,model.optimizer.lr])
        
        ####################################################
        
        
        # def mIOU(y_true,y_pred):
        #     return y_pred
        # model.compile(optimizer='adam',loss=[lambda y_true,y_pred: y_pred]*4,
        #                 loss_weights=[1.0,0.5,0.5,0.],
        #                 metrics={"m_iou":mIOU})
        return model
    elif mode=='inference':
        decodedbranch_tensor=[]
        for i,branch in enumerate(all_branch):
            pred_tensor=Decode(model_config.class_num,stride=model_config.strides[i],anchor_size=anchors[i],anchor_num=model_config.anchor_num_per_grid)(branch)
            decodedbranch_tensor.append(Reshape(target_shape=(-1,5+model_config.class_num))(pred_tensor))
        all_decoded_pred_tensor=Concatenate(axis=1,name="all_pred")(decodedbranch_tensor)
        
        model=Model(input=input_image,output=all_decoded_pred_tensor)
        return model
