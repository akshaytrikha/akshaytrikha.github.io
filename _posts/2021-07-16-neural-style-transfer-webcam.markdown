---
layout: post
title: "Neural Style Transfer Webcam"
date: 2021-07-16 12:50:00 -0000
categories: deep-learning
thumbnail: /assets/thumbnails/style-transfer.jpeg
tldr: "stylize anything"
---

### Overview

I wanted to learn how to build and deploy an ML system in the real world and thought this would be a fun place to start. You can find my project's code [here](https://github.com/akshaytrikha/style-transfer).

<figure>
    <br>
    <div style="text-align: center;">
        <img src="{{site.url}}/assets/style-transfer/style-transfer.gif" alt="style transfer demo gif" style="width: 100%"/>
        <figcaption style="text-align: center">
            Try the demo yourself <a href="https://akshaytrikha.github.io/style-transfer/" target="_blank" rel="noopener noreferrer">here</a>
        </figcaption>
    </div>
    <br>
</figure>

<!-- <figure>
    <br>

        <img src="{{site.url}}/assets/mask-finding/fast_mask_finding.png" alt="ML defect detection"/>
        <figcaption>Fig 1. Mask Finding</figcaption>

    <br>
</figure> -->

Using your webcam as input, this project generates stylized images in 400ms intervals. It uses two pretrained Tensorflow.js neural networks, sourced from Reiichiro Nakano's [arbitrary-image-stylization-tfjs](https://github.com/reiinakano/arbitrary-image-stylization-tfjs) repo.

1. The first network is used to learn the style of a given image and generate a style representation.
2. The second network is then used for style transfer, or using the style representation to generate a stylized output image.

The original models were actually regular tf models, but Reiichiro distilled them into smaller networks so they would run faster. For a more detailed breakdown of how the networks work, check out his blog [post](https://magenta.tensorflow.org/blog/2018/12/20/style-transfer-js/).

### Building the React App

I used Nicholas Renotte's [ReactComputerVisonTemplate](https://github.com/nicknochnack/ReactComputerVisionTemplate) repo as a template for interfacing with a webcam and rendering output back on screen. The big idea is to use the `react-webcam` component to take screenshots, generate their style representations, and then display their stylized images. In pseudo-ish code the entire app is basically:

```JavaScript
import Webcam from "react-webcam";

const predict = async () => {
    // First wait for models to load
    await loadModels();
    // init style image and generate style representation
    await initStyleImage();

    // every 400ms
    setInterval(() => {
        captureScreenshot();
        generateStylizedImage();
    }, 400);
}
```

### Lessons learned

1. A very fun fact about using TensorFlow.js is that your browser is locally running inference, and all your data is kept on your device.

2. You have to do a lot of things asynchronously e.g. fetching the models, loading the models, running inference. Timing all of this was how I decided to introduce the delay of 400ms.

3. tfjs is slightly different to using tf in python in small ways. You have to use methods like `tf.ready().then(() => {})` that returns a promise when tfjs is ready (if the WebGL backend is ready).

4. I used the html `canvas` element to display images and would use `document.getElementById("element-name").src = ...` directly to update an element. I standardized the display sizes in px to make it easier to deal with.

5. To make them persistent I stored the model weights in the repo itself so that they're accessible with e.g. `transferModel = await tf.loadGraphModel(process.env.PUBLIC_URL + '/models/style-prediction/model.json')`

#### Resources:

- <https://github.com/reiinakano/arbitrary-image-stylization-tfjs>
- <https://magenta.tensorflow.org/blog/2018/12/20/style-transfer-js/>
- <https://styletransfer.art>
