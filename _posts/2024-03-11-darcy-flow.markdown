---
layout: post
title: "Solving Darcy Flow with Neural PDEs"
date: 2024-03-11 13:11:17 -0000
categories: deep-learning
---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML">
</script>

How do we force a neural network to only predict functions that satisfy some physical contraint? Let's take the example of 2D darcy flow, a partial differential equation (PDE) that describes the flow of a fluid through a porous medium. Darcy flow states:

$$ -\nabla \cdot \left( \nu(x,y) \nabla u(x,y) \right) = f(x,y) = \quad x,y \in (0,1) $$

With boundary condition:

$$ u(x,y) = 0 \quad (x,y) \in \partial(0,1)^2 $$

Ok great, we have a differential equation and our goal is to train a neural network that guesses solutions to this. In our heads, we imagine our solutions to be of the form $$u = \sigma_i k_i w_i$$ which is a product of basis functions $k$ and their corresponding weights $w$. Now we see a problem - we need to optimize two things with this approach: we need to predict optimal basis functions in addition to optimal weights for our final prediction to be optimal. This type of problem is called bi-level optimization and we're gonna solve it with wizardry (or at least I think so).


<!-- \mathcal{F}(u)  -->



We're going to implmenet a differentiable optimization layer (I know, crazy) that will allow us to 