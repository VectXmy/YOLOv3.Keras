from . import model_config
from keras.utils.data_utils import Sequence
import cv2
import tensorflow as tf
import numpy as np
import os
from . import utils

class YoloDataLoader(Sequence):
    '''
    annotations format  
    <image_path> <x1,y1,x2,y2,class_index> <x1,y1,x2,y2,class_index> <x1,y1,x2,y2,class_index>....
    bboxes 最后一位是class_index
    '''  
    def __init__(self,anno_path='./data/train.txt',batchsize=4,max_bbox_per_scale=model_config.max_bbox_per_scale,shuffle=True,aug=True):
        self.anno_path=anno_path
        self.batchsize=batchsize
        self.aug=aug
        self.input_size=np.array(model_config.input_size)
        self.strides=np.array(model_config.strides)
        self.annos=self.load_annotation(self.anno_path)
        self.out_size=self.input_size//self.strides
        self.anchor_num=model_config.anchor_num_per_grid
        self.max_bbox_per_scale=max_bbox_per_scale
        self.shuffle=shuffle
        self.class_num=model_config.class_num
        self.anchor_size=utils.get_anchors("./anchors_size.txt")
        self.iou_thr=model_config.positive_iou_threshold
    def __len__(self):
        return int(np.ceil(len(self.annos) / float(self.batchsize)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.annos)

    def __getitem__(self,index):
        ###############用于被填充的batch######################
        batch_anno=self.annos[index * self.batchsize:(index + 1) * self.batchsize]
        batch_image=np.zeros((self.batchsize,self.input_size[0],self.input_size[0],3),dtype=np.float32)
        batch_slabel=np.zeros((self.batchsize,self.out_size[0],self.out_size[0],self.anchor_num,5+self.class_num),dtype=np.float32)
        batch_mlabel=np.zeros((self.batchsize,self.out_size[1],self.out_size[1],self.anchor_num,5+self.class_num),dtype=np.float32)
        batch_llabel=np.zeros((self.batchsize,self.out_size[2],self.out_size[2],self.anchor_num,5+self.class_num),dtype=np.float32)
        batch_sbboxes=np.zeros((self.batchsize,self.max_bbox_per_scale,4),dtype=np.float32)
        batch_mbboxes=np.zeros((self.batchsize,self.max_bbox_per_scale,4),dtype=np.float32)
        batch_lbboxes=np.zeros((self.batchsize,self.max_bbox_per_scale,4),dtype=np.float32)

        for batch_index,anno in enumerate(batch_anno): #处理batch里每行anno，填充到batch里
            image,bboxes=self.parse_anno(anno)
            if self.aug:
                #do some augmentation
                pass
            image,bboxes=self.preprocess_anno(image,[self.input_size[0]]*2,bboxes)

            slabel, mlabel, llabel, sbboxes, mbboxes, lbboxes = self.generate_target(bboxes)
            batch_image[batch_index,...]=image
            batch_slabel[batch_index,...]=slabel
            batch_mlabel[batch_index,...]=mlabel
            batch_llabel[batch_index,...]=llabel
            batch_sbboxes[batch_index,...]=sbboxes
            batch_mbboxes[batch_index,...]=mbboxes
            batch_lbboxes[batch_index,...]=lbboxes
        inputs=[batch_image,batch_slabel,batch_sbboxes,batch_mlabel,batch_mbboxes,batch_llabel,batch_lbboxes]
        outputs=[]
        # inputs={"input_image":batch_image,"slable":batch_slabel,"mlabel":batch_mlabel,"llabel":batch_llabel,"sbboxes":batch_sbboxes,"mbboxes":batch_mbboxes,"lbboxes":batch_lbboxes}
        # outputs={}

        return inputs,outputs
    def load_annotation(self,anno_path):
        '''
        annotations format
        <image_path> <x1,y1,x2,y2,class_index> <x1,y1,x2,y2,class_index> <x1,y1,x2,y2,class_index>....
        '''
        with open(anno_path,'r') as f:
            lines=f.readlines()
            annos=[line.strip() for line in lines if len(line.strip().split()[1:])!=0]
        np.random.shuffle(annos)
        return annos
    def parse_anno(self,anno):
        data=anno.split()
        image_path=data[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        image=cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        try:
            bboxes=np.array([list(map(int, box.split(','))) for box in data[1:]])
        except:
            bboxes=np.array([list(map(int,list(map(float, box.split(','))))) for box in data[1:]])
        return image,bboxes
    def preprocess_anno(self,image,input_ksize,bboxes):
        ih, iw    = input_ksize
        h,  w, _  = image.shape

        scale = min(iw/w, ih/h)
        nw, nh  = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
        image_paded = image_paded / 255.

        if bboxes is None:
            return image_paded

        else:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale + dw
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale + dh
            return image_paded, bboxes
    def generate_target(self,bboxes):
        all_label=[np.zeros((self.out_size[i],self.out_size[i],self.anchor_num,5+self.class_num)) for i in range(3)]
        all_bboxes=[np.zeros((self.max_bbox_per_scale,4)) for _ in range(3)]
        bboxes_count=np.zeros((3,),dtype=np.int32)
        for bbox in bboxes:
            
            bbox_class_index_sparse=bbox[4]
            bbox_class_index_onehot=np.zeros(self.class_num,dtype=np.float32)
            bbox_class_index_onehot[bbox_class_index_sparse]=1.0
            ###label smooth
            # uniform_distribution=np.full(self.class_num,1.0/self.class_num)
            # deta=0.01
            # bbox_class_index_onehot=bbox_class_index_onehot*(1-deta)+deta*uniform_distribution
            ###############
            orig_bbox_xywh=bbox[0:4]
            #(x1,y1,x2,y2)--->(c1,c2,w,h)
            orig_bbox_xywh=np.concatenate([(orig_bbox_xywh[2:]+orig_bbox_xywh[:2])*0.5,
                                            orig_bbox_xywh[2:]-orig_bbox_xywh[:2]],axis=-1)
            #(1,4)/(3,1)-->(3,4)广播机制(1,4)-->(3,4) (3,1)-->(3,4)
            fmap_bbox_xywh=1.0*orig_bbox_xywh[np.newaxis,:]/self.strides[:,np.newaxis]#转换到fmap上的坐标进行求iou
            
            iou=[]
            exist_positive=False
            for i in range(3):#分别与三个scale的输出做iou,只跟每个scale的对应位置的三个anchor做iou
                anchor_xywh=np.zeros((self.anchor_num,4))#scaled_bbox_xywh对应的anchor
                anchor_xywh[:,0:2]=np.floor(fmap_bbox_xywh[i,0:2]).astype(np.int32)+0.5#因为是中心点所以加0.5?
                anchor_xywh[:,2:4]=self.anchor_size[i]

                per_fmap_bbox_xywh=fmap_bbox_xywh[i]

                scaled_iou=utils.bbox_iou(per_fmap_bbox_xywh[np.newaxis,:],anchor_xywh)
                iou.append(scaled_iou)
                iou_mask=scaled_iou>self.iou_thr

                if np.any(iou_mask):
                    xind,yind=np.floor(fmap_bbox_xywh[i,0:2]).astype(np.int32)
                    # all_label[i][yind,xind,iou_mask,:]=0
                    all_label[i][yind,xind,iou_mask,0:4]=orig_bbox_xywh
                    all_label[i][yind,xind,iou_mask,4:5]=1.0
                    all_label[i][yind,xind,iou_mask,5:]=bbox_class_index_onehot
                    if not exist_positive:
                        exist_positive=True
            
                if bboxes_count[i]<self.max_bbox_per_scale:
                    all_bboxes[i][bboxes_count[i],0:4]=orig_bbox_xywh
                    bboxes_count[i]+=1
            if not exist_positive:
                best_anchor_ind=np.argmax(np.array(iou).reshape(-1),axis=-1)
                best_scale=int(best_anchor_ind/self.anchor_num)#第几个scale和第几个anchor
                best_anchor=int(best_anchor_ind%self.anchor_num)
                xind,yind=np.floor(fmap_bbox_xywh[best_scale,0:2]).astype(np.int32)

                all_label[best_scale][yind,xind,best_anchor,0:4]=orig_bbox_xywh
                all_label[best_scale][yind,xind,best_anchor,4:5]=1.0
                all_label[best_scale][yind,xind,best_anchor,5:]=bbox_class_index_onehot

                if bboxes_count[best_scale]<self.max_bbox_per_scale:
                    all_bboxes[best_scale][bboxes_count[best_anchor],0:4]=orig_bbox_xywh
                    bboxes_count[best_scale]+=1
        return all_label[0],all_label[1],all_label[2],all_bboxes[0],all_bboxes[1],all_bboxes[2]








        



