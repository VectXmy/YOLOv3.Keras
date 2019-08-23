strides=[8,16,32]
input_shape=[416,416,3]
class_num=5
input_size=[416]
max_bbox_per_scale=150
positive_iou_threshold=0.7
negtive_iou_threshold=0.5
#anchor_size=[[],[],[]]
anchor_num_per_grid=3#改成由上面计算得到

epochs=40
warmup_epochs=3