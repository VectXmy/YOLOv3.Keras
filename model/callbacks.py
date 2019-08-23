from keras.callbacks import Callback
from keras import backend as K
import numpy as np

class CosineDecayLR(Callback):
    def __init__(self,train_dataloader,init_lr=1e-3,end_lr=1e-6,warmup_epochs=5,epochs=30):
        super(CosineDecayLR,self).__init__()
        self.init_lr=init_lr
        self.end_lr=end_lr
        self.warmup_epochs=warmup_epochs
        self.epochs=epochs
        self.steps_per_epoch=len(train_dataloader)
        self.warmup_steps=self.warmup_epochs*self.steps_per_epoch
        self.total_steps=self.epochs*self.steps_per_epoch
        self.global_steps=1

    def on_batch_end(self,batch,logs=None):
        self.global_steps+=1
        if self.global_steps<self.warmup_steps:
            lr=self.global_steps/self.warmup_steps*self.init_lr
        else:
            lr=self.end_lr+0.5*(self.init_lr-self.end_lr)*(
                (1+np.cos((self.global_steps-self.warmup_steps)/(self.total_steps-self.warmup_steps)*np.pi))
            )

        # self.model.optimizer.lr=lr
        K.set_value(self.model.optimizer.lr,lr)
        
    def on_train_begin(self,logs=None):
        try:
            lr=self.global_steps/self.warmup_steps*self.init_lr
        except:
            lr=self.global_steps*self.init_lr
        K.set_value(self.model.optimizer.lr,lr)

    