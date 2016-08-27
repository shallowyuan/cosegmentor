# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a cosegment R-CNN network.

MaskLossLayer implements a Caffe Python layer.
"""

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg


class MaskLossLayer(caffe.Layer):
    """SEG R-CNN data layer used for training."""

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        #self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape((1))
        self._name_to_top_map['loss'] = idx
        idx += 1
        self.count=0
        self.output=open(cfg.TRAIN_SEGLOSS_OUTPUT,'w')
        self.losscount=0
        self.smoothloss=0.
        self.losscache=np.ones((100,1),dtype='float32')*-1
                
        print 'MaskRegLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""

        predict=bottom[0].data.squeeze()
        pweights=bottom[1].data.squeeze()
        labels =np.where(pweights>0)[0]
        gt     =bottom[2].data.squeeze()
        assert gt.shape == predict.shape
        if self.count==0:
            #print gt,'\n-------------------',pweights,'\n----------------------',predict,'\n---------------'
            self.count+=1
        #print np.mean(np.log(1+np.exp(-np.multiply(gt,predict))),axis=1),'\n',pweights

        predict=predict[labels,...]
        pweights=pweights[labels]
        gt=gt[labels,...]

        loss= np.average(np.mean(np.log(1+np.exp(-np.multiply(gt,predict))),axis=1),weights=pweights)
        # output loss
        if self.losscache[self.count]<0:
            self.losscache[self.count]=loss
            self.smoothloss=(self.count*self.smoothloss+loss)/(self.count+1)
        else:
            self.smoothloss+=(loss-self.losscache[self.count])/100
            self.losscache[self.count]=loss
        self.count=(self.count+1)%100
        if self.count==0:
            self.output.write('{}\n'%self.smoothloss)


    def backward(self, top, propagate_down, bottom):
        """This layer does  propagate gradients."""
        if propagate_down[0]:
            gradients=np.zeros(bottom[0].shape,dtype=np.float32)
            weights=bottom[1].data.squeeze()
            labels =np.where(weights>0)[0]
            gt     =bottom[2].data[labels,...]
            predict=bottom[0].data[labels,...]
            weights=weights[labels]
            assert gt.shape == predict.shape
            pweights=np.tile(weights,[predict.shape[1],1,predict.shape[2],predict.shape[3]]).transpose([3,0,1,2])
            pweights=pweights/np.sum(pweights)



            gradients[labels,...]=-pweights*np.exp(-gt*predict)*gt/(1+np.exp(-gt*predict))
            #print gradients.squeeze(),'\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            bottom[0].diff[...]=gradients*top[0].diff.flat[0]


    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
