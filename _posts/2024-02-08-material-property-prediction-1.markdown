---
layout: post
title: "Material Property Prediction Part 1: A Naive Approach"
date: 2024-02-08 15:24:17 -0000
categories: deep-learning
---
 <script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      processEscapes: true
    }
  });
</script>
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>


Given its 3D coordinates & atomic numbers of its constituent atoms, can you predict the atomization energy of a molecule? In this first attempt I'm going to try a naive approach by training a fully connected neueral network on a supervised task.

### The QM7 dataset

The QM7 dataset [[1]](http://quantum-machine.org/datasets/) consists of 7165 molecules stable organic molecules that have up to 23 constituent atoms. It's a subset of the larger GDB-13 dataset [[2]](https://gdb.unibe.ch/downloads/) which consists of ~1 billion molecules - it's nice to start with a computationally easier problem.

Each molecule has two fixed lengths matrixes describing it even if it consists of fewer than 23 atoms:
1. `23 x 3` matrix containing information about its atom positions
2. `23 x 1` vector containing the atomic numbers of the atoms in their respective positions.

For e.g. a sample entry in the dataset looks like this:

```Python
R = [[ 1.8897262 ,  0.        ,  0.        ],
    [ 4.6184907 ,  0.        ,  0.        ],
    [ 6.867567  ,  0.        ,  0.        ],
    [ 9.596332  ,  0.        ,  0.        ],
    [10.316582  ,  2.7353597 ,  0.15233083],
    [10.638704  ,  4.0255136 ,  2.673301  ],
    [12.828274  ,  3.4122975 ,  1.1119337 ],
    [ 1.1453441 ,  1.9292403 ,  0.02254443],
    [ 1.1450795 , -0.9839615 ,  1.6594819 ],
    [ 1.1450039 , -0.94497645, -1.6819507 ],
    [10.345476  , -0.8279646 , -1.7417984 ],
    [10.345608  , -1.0861768 ,  1.5949667 ],
    [ 9.660261  ,  3.832837  , -1.4499112 ],
    [10.418041  ,  2.9208362 ,  4.3819723 ],
    [10.137946  ,  6.002734  ,  2.8299594 ],
    [13.497936  ,  4.9887447 ,  0.21958618],
    [ 0.        ,  0.        ,  0.        ],
    [ 0.        ,  0.        ,  0.        ],
    [ 0.        ,  0.        ,  0.        ],
    [ 0.        ,  0.        ,  0.        ],
    [ 0.        ,  0.        ,  0.        ],
    [ 0.        ,  0.        ,  0.        ],
    [ 0.        ,  0.        ,  0.        ]]

Z = [6., 6., 6., 6., 6., 6., 7., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
    0., 0., 0., 0., 0., 0.]
 ```

Similar to handling images as input to computer vision models, we flatten the `R` and `Z` to be vectors with dim `92 x 1` for each molecule (92 = 23 rows x 4 columns). The flattening works because each entry in the input vector will always correspond to the same entry in either `R` or `Z`.

Our model's output will be a scalar quantity representing the atomization energy of the molecule. **TODO**: write about mean + std stuff

### Model Implementation

To make things _even_ more interesting I'll be coding the neueral network from scratch using just numpy. My hope was that this will give me a deeper and more satisfying understanding of how gradients flow.

Let's start with how we want the network to look like:

```Python
layers = [
        input_layer,
        LinearLayer(dim_in=input_layer.num_out, dim_out=400),
        Tanh(),
        LinearLayer(400, 100),
        Tanh(),
        LinearLayer(100, 1),
        output_layer,
    ]
```

This seems reasonable. We're projecting the input of length 92 into 400 dimensions, then 100, then to a final 1 dimension. Hopefully in between the connections the neural network will be able to learn an ok representation. Since the network is so small I don't think choosing tanh will be much of a penalty over relu.

So what do the individual layers' classes look like? Here's an abstract class that contains the potential methods we would need for a layer.

```Python
class Module:
    @abstractmethod
    def forward(self, X):
        """Forward pass to compute the output of the module."""
        pass

    @abstractmethod
    def backward(self, DY):
        """Backward pass to compute gradients with respect to module parameters."""
        pass

    @abstractmethod
    def update(self, lr):
        """Update the parameters of the module."""
        pass

    # @abstractmethod
    # def average(self, nn, a):
    #     """Average the parameters of the module with another module."""
    #     pass  
```

Every layer needs way to propagate inferences with `forward()` and a way to backpropagate gradients `backward()`. We also need to a way to `update()` the weights and biases of a given layer. TODO: Finally, we also need to `average()`. With this template we'll be able to implement the `Input`, `Linear`, `Output` layers and their `Sequential` container.

**1. Input**

The easiest layer is perhaps the input. It's given a tuple of matrixes `(R, Z)` and all we need to do is flatten them and also calculate the layer's output dimension. It may look like we're doing duplicate computation while initializing and when forward() is called but in `__init__()` we need to calcualte the output dimension so that the next layer can initialize properly - and in `forward()` we also need to flatten. However `inp` and `x` can be of different shape as `inp` is whatever we initialize the nn with and `X` is probably a minibatch during training / inference. 

We start by inherting the `Module` we just created:

```Python
class Input(Module):
    def __init__(self, inp):
        """Initialize the input layer of the MLP
        Args:
            inp (tuple): tuple of numpy arrays (R, Z) of shape (nbatch, natoms, 3) and (nbatch, natoms)
        """
        R, Z = inp
        sample_in = np.concatenate([R, np.expand_dims(Z, -1)], axis=-1)
        self.num_out = sample_in.shape[-2] * sample_in.shape[-1]

    def forward(self, X):
        """Given input inp, perform a forward pass through layer"""
        R, Z = X
        rz = np.concatenate([R, np.expand_dims(Z, -1)], axis=-1)
```

**2. Linear**

The linear layer is where it gets a bit more interesting. Now we need a way to pass information both forward and backward as well as update the weights. We start by randomly initializing the weights and biases. TODO: Adjusting the learning rate by $\frac{1}{\sqrt{m}}$

```Python
class Linear(Module):
    """Linear layer with weights W and bias B"""

    def __init__(self, m, n):
        """Initialize the weights as randomly sampled from normal distribution and then scaled
            by 1/sqrt(m). Initialize the biases as zeros.

        Args:
            m (int): number of input features
            n (int): number of output features
        """
        # Adjust the learning rate by 1/sqrt(m) to account for the scaling of the weights
        self.lr = 1 / m**0.5
        self.W = np.random.normal(0, 1 / m**0.5, [m, n]).astype("float32")
        self.B = np.zeros([n]).astype("float32")

    def forward(self, X):
        """Perform forward pass through linear layer"""
        self.X = X
        self.output = np.matmul(X, self.W) + self.B
        return self.output

    def backward(self, dY):
        """Perform backward pass through linear layer"""
        # for weights chain rule is dL / dW = (dL / dY) * (dY / dW)
        # Y = XW + B so dY / dW = X
        # dL / dY = dY
        self.dW = self.X.T @ dY

        # for biases chain rule is dL / dB = (dL / dY) * (dY / dB)
        # Y = XW + B so dY / dB = 1
        # dL / dY = dY
        # sum over all samples in batch
        self.dB = np.sum(dY, axis=0)

        # dL / dY = dY by definition
        if dY.ndim == 1:
            dY = dY.reshape(-1, 1)

        self.grad = dY @ self.W.T

        return self.grad

    def update(self, lr):
        """Update the weights and biases of linear layer"""
        W_update = lr * self.lr * self.dW
        if W_update.ndim == 1:
            W_update = W_update.reshape(-1, 1)

        self.W -= W_update
        self.B -= lr * self.lr * self.dB

    def average(self, nn, a):
        """Average the weights and biases of linear layer with another linear layer"""
        self.W = a * nn.W + (1 - a) * self.W
        self.B = a * nn.B + (1 - a) * self.B
```




#### Resources:

- [[1] QM7 Dataset](http://quantum-machine.org/datasets/)
- [2 GDB-13](https://gdb.unibe.ch/downloads/)