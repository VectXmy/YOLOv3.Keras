import tensorflow as tf

def bboxes_iou(boxes1, boxes2):
    '''
    bbox (cx,cy,w,h)
    '''
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area

def m_iou(y_true,y_pred):
    def _iou(branch,decoded_branch,label,bboxes):
        #bboxes shape (batch_size,max_bbox_per_scale,4)-->(batch_size,1,1,1,max_bbox_per_scale,4)
        #pred_xywh shape (batch_size,out_size,out_size,3,4)-->(batch_size,out_size,out_size,3,1,4)
        pred_xywh=decoded_branch[...,0:4]
        # input_shape=tf.shape(pred_xywh)
        # batch_size,out_size,anchor_num=input_shape[0],input_shape[1],input_shape[3]
        # pred_bboxes=tf.reshape(pred_xywh,(batch_size,out_size*out_size*anchor_num,4))
        # iou=bbox_iou(pred_bboxes,bboxes)
        iou=bboxes_iou(pred_xywh[:, :, :, :, tf.newaxis, :], bboxes[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :])
        return tf.reduce_mean(tf.reduce_sum(iou,axis=[1,2,3,4]))
        # return tf.reduce_mean(tf.reduce_mean(iou,axis=-1))
    iou_total=0
    for i in range(3):
        branch,decoded_branch=y_pred[i*2],y_pred[i*2+1]
        label,bboxes=y_true[i*2],y_true[i*2+1]
        iou_total+=_iou(branch,decoded_branch,label,bboxes)
    m_iou=iou_total
    return m_iou