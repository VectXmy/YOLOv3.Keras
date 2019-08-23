from model.yolov3 import YOLOv3_model
from model.dataloader import YoloDataLoader
from keras.callbacks import EarlyStopping,TensorBoard,ModelCheckpoint
from model import model_config
from model.callbacks import CosineDecayLR
from keras.models import load_model
from model import layers

if __name__=="__main__":
    import numpy as np
    import tensorflow as tf
    import keras.backend as K
    print(K.learning_phase())
    K.set_learning_phase(1)
    print(K.learning_phase())


    train_dataloader=YoloDataLoader(anno_path="./data/train.txt",batchsize=6)
    val_dataloader=YoloDataLoader(anno_path="./data/val.txt",batchsize=6)
    model=YOLOv3_model(mode='train')
    # model.load_weights("./checkpoints/safe_helmet_weights.01_0-4.88.hdf5",by_name=True)
    # model.summary()
    # print(type(train_dataloader[0]),type(train_dataloader[0][0]))
    # for i in train_dataloader[0][0]:
    #     print(i.shape,type(i))
    callbacks=[]
    callbacks.append(CosineDecayLR(train_dataloader,init_lr=1e-4,warmup_epochs=model_config.warmup_epochs,epochs=model_config.epochs))
    callbacks.append(TensorBoard(log_dir="./logs/0822",update_freq="batch"))
    callbacks.append(EarlyStopping(monitor="val_loss",patience=5))
    callbacks.append(ModelCheckpoint("./checkpoints/0822/safe_helmet_weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                                monitor="val_loss",save_best_only=False,save_weights_only=True))

    model.fit_generator(generator=train_dataloader,validation_data=val_dataloader,
                            epochs=model_config.epochs,verbose=1,use_multiprocessing=True,
                            callbacks=callbacks)

    
    
    