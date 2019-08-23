import tensorflow as tf
from . import model_config
from .metrics import bboxes_iou


def conf_loss(y_true,y_pred):
    '''
    y_pred: [branch_1,decoded_branch_1,  branch_2,decoded_branch_2,branch_3,decoded_branch_3]
    y_true: [label_1,bboxes_1,label_2,bboxes_2,label_3,bboxes_3]
    label_xywh    = label[:, :, :, :, 0:4]  
    respond_bbox  = label[:, :, :, :, 4:5]#有物体置1  
    label_prob    = label[:, :, :, :, 5:]  
    bboxes 所有真实 boxes 
    label 为每个anchor分配的target 
    '''
    def _conf_loss(branch,decoded_branch,label,bboxes):
        branch=tf.reshape(branch,tf.shape(decoded_branch))
        raw_conf,pred_conf=branch[...,4:5],decoded_branch[...,4:5]
        respond_bbox  = label[..., 4:5]
        # return tf.reduce_mean(tf.reduce_sum(tf.pow(respond_bbox-pred_conf,2),axis=[1,2,3,4]))#SSE
        '''focal loss'''
        pred_xywh=decoded_branch[...,0:4]
        iou=bboxes_iou(pred_xywh[:,:,:,:,tf.newaxis,:],bboxes[:,tf.newaxis,tf.newaxis,tf.newaxis,:,:])
        max_iou=tf.expand_dims(tf.reduce_max(iou,axis=-1),axis=-1)
        respond_bg=(1.0-respond_bbox)*tf.cast(max_iou<model_config.negtive_iou_threshold,tf.float32)
        
        conf_focal=tf.pow(respond_bbox-pred_conf,2)#alpha=2
        # focal_loss=conf_focal*(
        #     respond_bbox*tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox,logits=raw_conf)
        #     +
        #     respond_bg*tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox,logits=raw_conf)
        # )
        focal_loss=respond_bbox*tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox,logits=raw_conf)+respond_bg*tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox,logits=raw_conf)
        return tf.reduce_mean(tf.reduce_sum(focal_loss,axis=[1,2,3,4]))
    conf_loss_total=0
    for i in range(3):
        branch,decoded_branch=y_pred[i*2],y_pred[i*2+1]
        # label,bboxes=y_true[i]
        label,bboxes=y_true[i*2],y_true[i*2+1]
        conf_loss_total+=_conf_loss(branch,decoded_branch,label,bboxes)
    return conf_loss_total
    
def prob_loss(y_true,y_pred):
    def _prob_loss(branch,decoded_branch,label,bboxes):
        # input_shape=tf.shape(branch)
        # batch_size,out_size=input_shape[0],input_shape[1]
        # branch=tf.reshape(branch,(batch_size,out_size,out_size,model_config.anchor_num_per_grid,5+model_config.class_num))
        branch=tf.reshape(branch,tf.shape(decoded_branch))
        raw_prob,pred_prob=branch[...,5:],decoded_branch[...,5:]
        respond_bbox  = label[..., 4:5]
        label_prob    = label[..., 5:] 
        return tf.reduce_mean(tf.reduce_sum(respond_bbox*tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=raw_prob),axis=[1,2,3,4]))#交叉熵
        # return tf.reduce_mean(tf.reduce_sum(0.5*respond_bbox*tf.pow(label_prob-pred_prob,2),axis=[1,2,3,4]))#差的平方和
    prob_loss_total=0
    for i in range(3):
        branch,decoded_branch=y_pred[i*2],y_pred[i*2+1]
        label,bboxes=y_true[i*2],y_true[i*2+1]
        prob_loss_total+=_prob_loss(branch,decoded_branch,label,bboxes)
    return prob_loss_total

def bbox_loss(y_true,y_pred):
    def _bbox_loss(branch,decoded_branch,label,bboxes):
        pred_xywh=decoded_branch[...,0:4]
        label_xywh=label[...,0:4]
        respond_bbox  = label[..., 4:5]
        input_size=tf.constant(model_config.input_shape[1],dtype=tf.float32)
        bbox_loss_scale=2.0-label_xywh[...,2:3]*label_xywh[...,3:4]/(input_size**2)
        sse_loss=0.5*tf.pow((label_xywh[...,0:4]-pred_xywh[...,0:4])/input_size,2)
        return tf.reduce_mean(tf.reduce_sum(respond_bbox*bbox_loss_scale*sse_loss,axis=[1,2,3,4]))
    bbox_loss_total=0
    for i in range(3):
        branch,decoded_branch=y_pred[i*2],y_pred[i*2+1]
        label,bboxes=y_true[i*2],y_true[i*2+1]
        bbox_loss_total+=_bbox_loss(branch,decoded_branch,label,bboxes)
    return bbox_loss_total


def bbox_giou(boxes1, boxes2):

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou


def giou_loss(y_true,y_pred):
    def _giou_loss(branch,decoded_branch,label,bboxes):
        pred_xywh=decoded_branch[...,0:4]
        label_xywh=label[...,0:4]
        respond_bbox  = label[..., 4:5]
        giou=tf.expand_dims(bbox_giou(pred_xywh,label_xywh),axis=-1)
        input_size=tf.constant(model_config.input_shape[1],dtype=tf.float32)
        bbox_loss_scale=2.0-label_xywh[...,2:3]*label_xywh[...,3:4]/(input_size**2)
        return tf.reduce_mean(tf.reduce_sum(respond_bbox*(1.0-giou),axis=[1,2,3,4]))#仅仅giou loss
        # return tf.reduce_mean(tf.reduce_sum(respond_bbox*bbox_loss_scale*(1.0-giou),axis=[1,2,3,4]))#乘上(2-truth_x*truth_y),truth_x和truth_y都是归一化后
    giou_loss_total=0
    for i in range(3):
        branch,decoded_branch=y_pred[i*2],y_pred[i*2+1]
        label,bboxes=y_true[i*2],y_true[i*2+1]
        giou_loss_total+=_giou_loss(branch,decoded_branch,label,bboxes)
    return giou_loss_total