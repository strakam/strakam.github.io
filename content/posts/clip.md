---
author: "Matej Straka"
title: "CLIP: Contrastive Language-Image Pretraining"
date: "2024-04-19"
description: "A short introduction to CLIP, the model behind current State-of-the-Art Computer Vision models."
ShowToc: false
ShowBreadCrumbs: false
---

## Motivation
Ever since Deep Learning started as a field, we have been observing an increased performance of models coming
from using bigger datasets and more compute. One such compute and data demanding architecture, Transformer [9],
is used in many State-of-the-Art methods. The architecture properties and recent scaling laws [3] indicate that
leveraging large amounts of data should lead to increased performance.

However, the problem with datasets such as ImageNet is that they grow slowly and have a weak annotation 
(usually just one word). On the other hand, websites such as Wikipedia or Instagram, contain bilions of images that
contain some richer annotation in form of captions. CLIP leverages these large 'annotated' datasets to train large
vision and language transformers in a fashion from which interesting properties emerge.

### Shared Latent Space
The first interesting property is that CLIP doesn't just learn arbitrary representations of text and images,
but it embeds them into the same latent space. This basically means that when we get vector representation from CLIP
for text and for image, we can compute the similarity between using cosine similarity. This enables us to use CLIP
for text-image retrieval or text conditioning in image generation or image segmentation.

### Zero-Shot Transfer
The second crucial property of CLIP is that it unlocks zero-shot transfer. In practice, instead of encoding image classes
as one-hot vectors, we can encode them simply as 'strings'. Then, since CLIP text encoder can encode any text, it can
therefore encode any label we want. This allows us to use CLIP on our custom labels without any transfer learning.
See code down below.

## Method


## Main results

## Roll Your Own Classifier!

## Conclusion


