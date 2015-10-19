import numpy as np
from pylearn2.utils import serial
from pylearn2.models.mlp import MLP
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.format.target_format import convert_to_one_hot
from pylearn2.space import Conv2DSpace
from pylearn2.models.mlp import ConvRectifiedLinear
from pylearn2.models.mlp import Softmax
from pylearn2.costs.cost import SumOfCosts
from pylearn2.costs.cost import MethodCost
from pylearn2.costs.mlp import WeightDecay
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.training_algorithms.learning_rule import Momentum
from pylearn2.termination_criteria import And
from pylearn2.train import Train
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.training_algorithms.learning_rule import MomentumAdjustor
from pylearn2.termination_criteria import MonitorBased
from pylearn2.termination_criteria import EpochCounter
filepath = '/veu4/usuaris18/pierre/data/MNIST/mnist.pkl'
train_set,valid_set,test_set=serial.load(filepath)
train_set_x , train_set_y = train_set
valid_set_x , valid_set_y = valid_set
test_set_x , test_set_y = test_set
train_set_y = convert_to_one_hot(integer_vector=train_set_y,max_labels=10)
valid_set_y = convert_to_one_hot(integer_vector=valid_set_y,max_labels=10)
test_set_y = convert_to_one_hot(integer_vector=test_set_y,max_labels=10)
print np.shape(test_set_y)

CNN_model = MLP(batch_size = 100,
                  input_space = Conv2DSpace(shape = [28,28],num_channels=1),
                  layers = [ConvRectifiedLinear(layer_name='h2',
                     output_channels=64,
                     irange= .05,
                     kernel_shape=[5, 5],
                     pool_shape=[4, 4],
                     pool_stride=[2, 2],
                     max_kernel_norm=1.9365),
                        ConvRectifiedLinear(layer_name='h3',
                     output_channels=64,
                     irange= .05,
                     kernel_shape=[5, 5],
                     pool_shape=[4, 4],
                     pool_stride=[2, 2],
                     max_kernel_norm=1.9365),
                        Softmax(max_col_norm=1.9365,
                     layer_name='y',
                     n_classes=10,
                     istdev=.05)]
)

cost_function = SumOfCosts(costs = [MethodCost(method='cost_from_X'),WeightDecay(coeffs=[.00005,.00005,.00005])])

trainer = SGD(batch_size=100,
        learning_rate=.01,
        learning_rule=Momentum(init_momentum=0.5),
        monitoring_dataset= {'valid':DenseDesignMatrix(X=valid_set_x,y=valid_set_y),'test':DenseDesignMatrix(X=test_set_x,y=test_set_y)},
        cost=cost_function,
        termination_criterion =And(criteria=[MonitorBased(channel_name="valid_y_misclass",prop_decrease=0.50,N=10),EpochCounter(max_epochs=500)])
)
trained_model=Train(dataset=DenseDesignMatrix(X=train_set_x,y=train_set_y),model=CNN_model,
                        algorithm=trainer,
                        extensions=[MonitorBasedSaveBest(channel_name='valid_y_misclass',
                                        save_path="./convolutional_network_best.pkl"),
                                MomentumAdjustor(start=1,
                                                saturate=10,
                                                final_momentum=.99)],
                        save_path='/CNN/MNIST_CNN.pkl',
                        save_freq=1
)
trained_model.main_loop()