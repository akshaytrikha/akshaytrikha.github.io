---
layout: post
title: "Cursor for Contracts"
date: 2025-10-09 13:11:17 -0000
categories: Research
thumbnail: /assets/thumbnails/contract-ui.jpeg
tldr: "Created a contract editor with inline clause LLM suggestions"
---

I recently negotiated my first contract, and had no idea what I was doing. Having used Cursor so much for programming I wished there was something equivalent for editing contracts. Something that would let me easily concede or be more aggressive in certain clauses using legalese. 

I had an itch to scratch so I built a demo for this sort of UI I would like to see more of in document editing - inspired by the essay [AI Horseless Carriages](https://koomen.dev/essays/horseless-carriages/). 

Running locally it uses [gemma3:12b](https://deepmind.google/models/gemma/gemma-3/) with [Ollama](https://github.com/ollama/ollama) amazingly quick even on my M3 mac, but on this web demo it's using gemini-2.0-flash-lite.

<!-- TODO: if not mobile render this: -->
<!-- <style>
#root {
  min-height: auto !important;
}
#root .app-container {
  min-height: auto !important;
}
</style> -->

<!-- <link rel="stylesheet" href="{{site.url}}/assets/contract-ui/build/static/css/main.0c014cc3.css">
<div id="root"></div>
<script>
  // Set the contract path for Jekyll deployment before 
  window.CONTRACT_BASE_PATH = '{{site.url}}/assets/contract-ui/build/contracts/';
</script>
<script src="{{site.url}}/assets/contract-ui/build/static/js/main.8d8eb0f6.js"></script> -->

<!-- // TODO: if mobile render this -->
<!-- <figure style="display: flex; justify-content: center;">
    <div style="text-align: center;">
      <object type="image/svg+xml" data="{{site.url}}/assets/contract-ui/contract-ui-demo.gif"></object>
    </div>
    <br>
</figure> -->

<style>
@media (max-width: 768px) {
  .desktop-only { display: none; }
}
@media (min-width: 769px) {
  .mobile-only { display: none; }
}
#root {
  min-height: auto !important;
}
#root .app-container {
  min-height: auto !important;
}
</style>

<div class="desktop-only">
  <link rel="stylesheet" href="{{site.url}}/assets/contract-ui/build/static/css/main.0c014cc3.css">
  <div id="root"></div>
  <script>
    window.CONTRACT_BASE_PATH = '{{site.url}}/assets/contract-ui/build/contracts/';
  </script>
  <script src="{{site.url}}/assets/contract-ui/build/static/js/main.940ed3a5.js"></script>
</div>

<div class="mobile-only">
  Come back on a desktop to see the interactive demo.
  <figure style="display: flex; justify-content: center;">
    <div style="text-align: center;">
      <img src="{{site.url}}/assets/contract-ui/contract-ui-demo.gif" alt="Contract UI Demo">
    </div>
  </figure>
</div>


<details markdown="1">
<summary style="cursor: pointer; font-weight: bold; margin-bottom: 20px;">How did I serve this demo?</summary>

I wanted the app to run completely locally since contracts can be sensitive, but didn't want visitors to this page to have to fetch 1GB worth of a language model. I settled on using Gemini via a [Vercel Function](https://vercel.com/docs/functions) proxy server which has my API key on it. I also used an [Upstash Redis KV Store](https://upstash.com/docs/redis/overall/getstarted) that limits inferences based on IP address to prevent abuse.

This was surprisingly accessible with the help of language models, and most of all *free* given what I expect the traffic to this site to be. Crazy!

<figure style="display: flex; justify-content: center;">
    <div style="text-align: center;">
      <object type="image/svg+xml" data="{{site.url}}/assets/contract-ui/contract-ui.svg" style="width: 450px; height: auto;"></object>
    </div>
    <br>
</figure>
</details>




#### References:

- [1] [AI Horseless Carriages](https://koomen.dev/essays/horseless-carriages/)
- [2] [Gemma3](https://deepmind.google/models/gemma/gemma-3/)
- [3] [Ollama](https://github.com/ollama/ollama)
- [4] [Vercel Function](https://vercel.com/docs/functions)
- [5] [Upstash Redis KV Store](https://upstash.com/docs/redis/overall/getstarted)