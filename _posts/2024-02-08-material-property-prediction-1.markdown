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


Given its 3D coordinates & atomic numbers of its constituent atoms, can you predict the atomization energy of a molecule? In this first attempt I'm going to try a naive approach by training a fully connected neueral network on a supervised task. To make things _even_ more interesting I'll be writing the neueral network from scratch using just numpy. My hope was this will give me a deeper and more satisfying understanding of how information flows.

### The QM7 dataset

The QM7 dataset [[1]](http://quantum-machine.org/datasets/) consists of 7165 molecules stable organic molecules that have up to 23 constituent atoms. It's a subset of the larger GDB-13 dataset [[2]](https://gdb.unibe.ch/downloads/) which consists of ~1 billion molecules - it's nice to start with a computationally easier problem.

Each molecule has two fixed lengths matrixes describing it even if it consists of fewer than 23 atoms:
1. `23 x 3` matrix `R` containing information about its atom positions
2. `23 x 1` vector `Z` containing the atomic numbers of the atoms in their respective positions.

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
        """Backward pass to compute gradients with respect to module 
        parameters."""
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

The easiest layer is perhaps the input. It's given a tuple of matrixes `(R, Z)` and all we need to do is flatten it and also calculate the layer's output dimension. It may look like we're doing duplicate computation while initializing and when forward() is called but in `__init__()` we need to calcualte the output dimension so that the next layer can initialize properly - and in `forward()` we also need to flatten. However `inp` and `x` can be of different shape as `inp` is whatever we initialize the nn with and `X` is probably a minibatch during training / inference. 

```Python
class Input(Module):
    def __init__(self, inp):
        """Initialize the input layer of the MLP
        Args:
            inp (tuple): tuple of numpy arrays (R, Z) of shape 
                         (nbatch, natoms, 3) and (nbatch, natoms)
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

The linear layer is where it gets a bit more interesting. Now we need a way to pass information both forward and backward as well as update the parameters. This class is long so lets break it up. We start by randomly initializing the weights and biases. TODO: Adjusting the learning rate by $\frac{1}{\sqrt{m}}$

```Python
class Linear(Module):
    """Linear layer with weights W and bias B"""

    def __init__(self, m, n):
        """Initialize the weights as randomly sampled from normal distribution 
        and scale lr by 1/sqrt(m). Initialize the biases as zeros.

        Args:
            m (int): number of input features
            n (int): number of output features
        """
        # adjust lr by 1/sqrt(m) to account for scaling of weights
        self.lr = 1 / m**0.5
        self.W = np.random.normal(0, 1 / m**0.5, [m, n]).astype("float32")
        self.B = np.zeros([n]).astype("float32")

```

What does the output to the linear layer mean? It means that we're trying to find the output $Y$ in $Y = WX + B$ where $W$ are the weights of the layer and $B$ its biases. Note that since $W$ and $B$ are matrixes we have to matrix multiply them. We store `self.X` because we use it again in the backward pass. Seems easy enough.
```Python
    def forward(self, X):
        """Perform forward pass through linear layer"""
        self.X = X
        self.output = np.matmul(X, self.W) + self.B
        return self.output
```

Ok `backward()` is slightly tricky. The big idea is that somehow we want to measure how much each of the weights and biases in the layer affect the final loss. This will eventually help us in adjusting the parameters and pointing the model in a better direction at the end of the epoch. 

For the weights: in mathematical terms we can say that we want to find $\frac{dL}{dW}$ where $L$ represents the loss - or how bad the model was compared to the ground truth. With a bit of chain rule we can decompose this into other derivatives we might know:

\\[\frac{dL}{dW} = \frac{dL}{dY} \times \frac{dY}{dW}\\]

a-ha! We know that since we're in a linear layer and $Y = MX + B$  then 

\\[\frac{dY}{dW} = X\\]

Well that's convenient - since we stored `self.X` in the forward pass we can reuse it here. Finally, we're given $\frac{dL}{dY}$ as a input to the function `dL` so we can write `self.dW = self.X.T @ dY`. 

Now for the biases: now we want to find $\frac{dL}{dB}$. Similarly with chain rule:
\\[\frac{dL}{dB} = \frac{dL}{dY} \times \frac{dY}{dB}\\]

and 

\\[\frac{dY}{dW} = 1\\]

huh. So `self.dB = dY`? Generally, yeah, but not so quick: beacuse we'll likely be using a minibatch during training we'll actually want to use `self.dB = np.sum(dY, axis=0)`.

Finally for calculating the gradient, we want to find an expression for how much the loss changes with respect to the input to this layer. At first this might seem a little weird but we must remember that the inputs we're talking about to _this_ layer are the outputs of the _previous_ layer. In other words, we need a way to propagate the gradient backwards so that the previous layers can also use it.

\\[\frac{dL}{dX} = \frac{dL}{dY} \times \frac{dY}{dX}\\]

We know `dY` is given to us and $\frac{dY}{dX} = W$ so the expression is `self.grad = dY @ self.W.T`. 
 
An important note about `update()` here is that we want to adjust the amount we update the weights and biases by the learning rate (sometimes referred to as $\alpha$). The usual analogy people use is that the gradient dictates which direction we want to go in and the learning rate tells us how big of a step to take. For more intuition about backpropagation I'd recommend checking out Andrej Karpathy's excellent video [[3]](https://www.youtube.com/watch?v=VMj-3S1tku0).

```Python
    def backward(self, dY):
        """Perform backward pass through linear layer"""
        self.dW = self.X.T @ dY

        # sum over all samples in batch
        self.dB = np.sum(dY, axis=0)

        # dL / dX = dL / dY @ dY / dX
        self.grad = dY @ self.W.T

        return self.grad

    def update(self, lr):
        """Update the weights and biases of linear layer"""
        self.W -= lr * self.lr * self.dW
        self.B -= lr * self.lr * self.dB

    # def average(self, nn, a):
    #     """Average the weights and biases of linear layer with another 
    #         linear layer"""
    #     self.W = a * nn.W + (1 - a) * self.W
    # self.B = a * nn.B + (1 - a) * self.B
```

**3. Tanh**

This layer is really straightforward. We need a nonlinear activiation function to learn nonlinear functions. So the `forward()` method is just using calling the tanh function like so `self.Y = np.tanh(X)`. Then for `backward()` we want to differentiate the function and lucky for us someone has already found it to be $1 - tanh^2$. We stored the output of `forward()` as `self.Y` which we can reuse in our backward to express the derivative as $1 - Y^2$. Then to propagate the gradient that was given to us from the previous layer we have $(1 - Y^2) \times dY$.

```Python
class Tanh(Module):
    def forward(self, X):
        self.Y = np.tanh(X)
        return self.Y

    def backward(self, dY):
        self.grad = (1 - self.Y**2) * dY
        return self.grad
```

**4. Output**

**5. Sequential**

The `Sequential` class is our container for neatly packaging all of the network's sublayers. Its `forward()` and `backward()` methods exists as orchestrators for how the information flows in the network. The only real difference between their implementations is that in `backward()` we have to remember to iterate backwards through the layers. The way we've implemented `update()` is analgous to `torch.optim.Optimizer.step()` in PyTorch and we would call it once we've found the gradients and want to then update all our paramters.

```Python
class Sequential(Module):
    def __init__(self, modules):
        self.modules = modules

    def forward(self, X):
        """Given input X, perform a full forward pass through the MLP"""
        # iterate through layers and call forward() on output of each previous layer
        for module in self.modules:
            X = module.forward(X)

        return X

    def backward(self, dY):
        """Perform a full backward pass through the MLP.
        dY is gradient of the loss w.r.t the final output"""
        for module in reversed(self.modules):
            dY = module.backward(dY)

        return dY

    def update(self, lr):
        for m in self.modules:
            m.update(lr)

    # def average(self, nn, a):
    #     for m, n in zip(self.modules, nn.modules):
    #         m.average(n, a)
```

And there you have it! That's all the layers you'll need to train a dense network. Pretty amazing how little code there is right. 


#### Resources:

- [[1] QM7 Dataset](http://quantum-machine.org/datasets/)
- [[2] GDB-13](https://gdb.unibe.ch/downloads/)
- [[3] Andrej Karpathy's Intro to NNs and Backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)