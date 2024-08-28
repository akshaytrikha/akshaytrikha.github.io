---
layout: post
title: "Solving Darcy Flow with Neural PDEs"
date: 2024-03-11 13:11:17 -0000
categories: deep-learning
---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML">
</script>

**Background**

How do we force a neural network to only predict functions that satisfy some physical contraint? Let's take the example of 2D darcy flow, a partial differential equation (PDE) that describes the flow of a fluid through a porous medium. Darcy flow states:

<!-- $$ -\nabla \cdot \left( \nu(x,y) \nabla u(x,y) \right) = f(x,y) \quad \quad \quad \quad x,y \in (0,1) \quad (1)$$ -->

<!-- With boundary condition: -->

<!-- $$ u(x,y) = 0 \quad \quad \quad \quad (x,y) \in \partial(0,1)^2 $$ -->

\begin{align}
    -\nabla \cdot \left( \nu(x,y) \nabla u(x,y) \right) &= f(x,y) & x,y &\in (0,1) & \tag{1} \newline
\end{align}

With boundary condition 

\begin{align}
    u(x,y) &= 0 & (x,y) &\in \partial(0,1)^2 \tag{2}
\end{align}

Where 

\begin{align}
    \nabla &= \text{divergence operator} \newline
    \nu(x,y) &= \text{diffusion coefficient} \newline
    u(x,y) &= \text{flow pressure} \newline
    f(x,y) &= \text{forcing function} = 1 \quad \forall x,y
\end{align}

**Data**

Data is available courtesy of the [Anima AI + Science Lab](http://tensorlab.cms.caltech.edu/users/anima/) Lab at CalTech as part of their work [[3]](https://arxiv.org/pdf/2010.08895). 

We'll be given as input to our model a tensor with shape (4000 samples, x, y, $$\nu$$) and asked to predict $$K$$, the set of basis functions 

For training data we will use the url [[4]](https://drive.google.com/file/d/16kmLCmuPe_q_wpphEXtgJswFtxHE6Plf/view?usp=sharing)  and for validation the url [[5]](https://drive.google.com/file/d/1sQBpormuajXlf2h3YiMnL9qVzzFoSwBB/view?usp=sharing).


----------------------

Ok great, we have a differential equation and our goal is to train a neural network that guesses solutions to this. In our heads, we imagine our solutions could be expressed as a linear combination of basis functions and their corresponding weights 

$$u = \Sigma_i k_i w_i \tag{3}$$


Now we see a problem - we need to optimize two things with this approach: we need to predict optimal basis functions in addition to optimal weights for our final prediction to be good. This type of problem is called bi-level optimization and we're gonna solve it with wizardry (or at least I think so). So in this ideal world our neural network is going to predict some basis functions and then we're gonna stack a traditional linear solver at the end to give us the basis functions' weights. 


<figure>
    <br>
    <div style="text-align: center;">
        <img src="{{site.url}}/assets/darcy-flow/architecture.jpeg" alt="Darcy Flow Neural ODE Architecture"/>
        <figcaption>Fig 1. Architecture</figcaption>
    </div>
    <br>
</figure>

**How to Backpropagate?**

In order for our model to learn anything we're going to need to pass the gradients backwards from the linear solver layer to the ResNet. In this dream world we could simply update our parameters like so:

$$\theta^{l+1} = \theta^l - \alpha \cdot \frac{DL}{D\theta} \tag{4}$$

But how do we know $$\frac{DL}{D\theta}$$ if we have this weird linear solver layer in our system? i.e. what does it mean to calculate the gradient of the loss with respect to the parameters of the linear solver layer - even more so when we have how many N steps the solver will take to converge?

But this is weird ... we have no way of knowing how many steps the 

<!-- **Optimality Condition** -->


<!-- \mathcal{F}(u)  -->



<!-- We're going to implmenet a differentiable optimization layer (I know, crazy) that will allow us to  -->

#### References:

- [[1] Efficient and Modular Implicit Differentiation](https://arxiv.org/pdf/2105.15183)
- [[2] Learning Differentiable Solvers For Systems With Hard Constraints](https://arxiv.org/pdf/2207.08675)
- [[3] Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/pdf/2010.08895)
- [[4] Darcy Flow Training Data](https://drive.google.com/file/d/16kmLCmuPe_q_wpphEXtgJswFtxHE6Plf/view?usp=sharing) 
- [[5] Darcy Flow Validation Data](https://drive.google.com/file/d/1sQBpormuajXlf2h3YiMnL9qVzzFoSwBB/view?usp=sharing) 