import unittest
import numpy as np

import caffe
from caffe.proto import caffe_pb2
from gradient_check_util import GradientChecker

class TestGradientChecker(unittest.TestCase):

    def setUp(self):
        shape_data = (128, 25, 1, 1)
        shape_weights=(128,1, 1, 1)
        pdata = caffe.Blob(shape_data)
        mask = caffe.Blob(shape_data)

        weights= caffe.Blob(shape_weights)
        self.rng = np.random.RandomState(356)
        pdata.data[...] = self.rng.randn(*shape_data)
        mask.data[...] = self.rng.randn(*shape_data)
        weights.data[...] = self.rng.randn(*shape_weights)
        mask.data[...]= np.absolute(mask.data)>0.5
        mask.data[...]=2*mask.data[...]-1
        weights.data[...]=np.absolute(weights.data)>0.5
        #print pdata.data.squeeze(), mask.data.squeeze(), weights.data.squeeze()

        self.bottom = [pdata,  weights, mask]
        self.top = [caffe.Blob([])]


    def test_mask_loss(self):
        lp = caffe_pb2.LayerParameter()
        lp.type = "Python"
        lp.python_param.module='mask_reg_layer.layer'
        lp.python_param.layer='MaskLossLayer'
        layer = caffe.create_layer(lp)

        checker = GradientChecker(1e-2, 1e-1)
        checker.check_gradient_exhaustive(
            layer, self.bottom, self.top, check_bottom=[0])


if __name__ == '__main__':
    unittest.main()