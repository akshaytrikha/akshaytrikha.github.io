---
layout: post
title: "Electrical Anomaly Detection Synesthesia"
date: 2024-07-28 21:53:17 -0000
categories: deep-learning
---

When you manufacture something you typically want to test it works before shipping it off to your customers and calling it a day. At a battery company, you would typically cycle (charge and discharge) the battery a few times to determine if it was up to par. However, sometimes your electrical test equipment itself fails! To help out with this my scientist friends asked me to train an anomaly detection model for the electrical signals to determine if it was the tester's hardware or software that failed rather than the battery.

There were 3 approaches I thought about when training this model:
1. Use some sort of Recurrent Neural Network (RNN), 1-D CNN, or Long Short-Term Memory (LSTM) model
2. Train an unsupervised autoencoder and then finetune it with supervised labels

Option 2 sounds very cool in theory but takes a lot of effort in practice. I tried various approaches with option 1 but somehow couldn't get a model to converge. Until I had a somewhat dumb, and inefficient idea.

### Broadcasting 1-D signals to 2-D images

Most of the models I train are for vision tasks so I'm biased to thinking about 2-D and 3-D data. My plan was to imagine that a 1000 datapoint 1-D signal was the first row of an image and then copy that row 1000 times to end up with a 1000x1000 pixel image. 



Since I had 3 input channels (voltage, current, and charge) I made this into an RGB image.



Looks kind of whack right? I thought so too, until I trained a model and it worked. To reiterate, I was being lazy so I wanted to choose a model that was pretrained and lightweight. I wanted something lightweight that could be deployed on the edge so I settled on the small version of [MobileNetV3](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html#torchvision.models.mobilenet_v3_small). 


### Why Did This Work?