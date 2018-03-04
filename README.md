# CapsNet
This project implements the most important parts from the paper **Dynamic Routing Between Capsules** by Sara Sabour,
Nicholas Frosst and Geoffrey E. Hinton (2017). (https://arxiv.org/abs/1710.09829)
Additionally you can find a report with an in-depth explanation of capsule neworks and our results under *report.pdf*.

## Usage
First get a local copy of this repository on your machine with `git clone https://github.com/lukeeffenberger/caps-net.git`.

### Training and Testing
You can simply train the network with `python capsnet_mnist.py -m train`. The best model (lowest validation loss), which
is found during the training process will be stored in `tmp/model_weights/`. The summaries for TensorBoard will be stored in
`tmp/summaries`. If you want to train the pretrained model with the weights stored in `model_weights/` use
`python capsnet_mnist.py -m train_on`.

To have a look at the summaries in TensorBoard use `tensorboard --logdir=tmp/summaries`.

For testing the model stored in `model_weights/` use `python capsnet_mnist.py -m test`. The test error will simply
be printed in the console.

### What the dimension of the capsules represent
To investigate the "meaning" of the dimensions of different capsules you can make use of our Jupyter Notebook
`dimension_representation.ipynb`. It loads the weights stored in `model_weights/`. You can open it with 
`jupyter notebook dimension_representation.ipynb`.

There are two differenten functions in there:

- `random_digit_all_dimensions()`: This will just get a random sample for a random digit from the training data and perturb all dimensions, each at a time, and plot all the resulting images. This can be used to broadly explore what the reconstruction network does.
- `specific_digit_specific_dimension(digit, dimension)`: For a specified digit (0-9) this function will get a random sample for this digit from the training data and only plot the perturbation results for the specific dimension. This can be used to
affirmate findings from the broad exploration before.

### Using the layer modules
We implemented the different layers in classes for easily constructing new capsule networks. These clases can be found in the 
folder `layers/`.

You  can import a layer for example with `from layers.capslayer import CapsLayer`. This layer can then easily be integrated in the data flow graph, e.g.:
```
output = CapsLayer(
              count2 = 5,
              dim2 = 4,
              rout_iter = 2
         )(input)
```
For more detailed explanation which parameters the different classes take, you can check the documentation.
