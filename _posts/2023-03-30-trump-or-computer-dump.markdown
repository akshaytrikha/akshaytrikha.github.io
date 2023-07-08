---
layout: post
title:  "Trump or Computer Dump?"
date:   2023-04-23 21:53:17 -0000
categories: deep-learning
---

<script
	type="module"
	src="https://gradio.s3-us-west-2.amazonaws.com/3.27.0/gradio.js"
></script>

<gradio-app src="https://akshaytrikha-gpt2-trump.hf.space"></gradio-app>

\\
In early 2019 OpenAI published some [research](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) showing that large language models, when trained on colossal amounts of text, begin to behave in unpredictably intelligently on tasks they weren't originally trained on. Humorously, part the original dataset for the model was scraped from Reddit - which isn't always known to be the home of constructive conversation and accurate information so it's even more surprising how good their model was!

A year after that, while taking a natural language processing course I thought it would be fun to finetune GPT-2 on a corpus of then President Trump's tweets seeing as he has such a distinct style and voice. I obtained his tweets from the super convenient [Trump Twitter Archive](https://www.thetrumparchive.com/) which contained ~56,500 of his tweets from 2009 until he was kicked off the platform on January 8th, 2021. The project took me around 3 weeks if I remember correctly and this weekend as I was dog sitting I remembered that project and got excited about trying to recreate it with the latest tools available, namely HuggingFace. It was awesome to have a benchmark to compare against to measure and truly appreciate the efficiency and ergonomics of HuggingFace's transformers library. I was able to replicate the finetuning through a short [notebook](https://github.com/akshaytrikha/GPT-2-Trump/blob/master/huggingface_trump.ipynb) + host it using gradio on a HuggingFace space for you to play with in just a couple of days.