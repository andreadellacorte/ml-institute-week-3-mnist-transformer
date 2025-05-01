# Task

This week, we are building a model that takes an image containing a sequence of numbers and decodes it into a corresponding list of digits.

# Dataset

MNIST

# References

## Repos

- [besarthoxhaj/attention](https://github.com/besarthoxhaj/attention)

## Docs

- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe)

## Videos
- [ ] [Visual Guide to Transformer Neural Networks - (Episode 1) Position Embeddings](https://www.youtube.com/watch?v=dichIcUZfOw)
- [x] [Transformers (how LLMs work) explained visually | DL5](https://www.youtube.com/watch?v=wjZofJX0v4M&t=203s)
- [ ] [Vision Transformer made easy: a short tutorial](https://www.youtube.com/watch?si=iIYQixHB01MZEOpf)
- [x] [Attention in transformers, step-by-step | DL6](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- [ ] [Implementing Vision Transformer (ViT) from Scratch](https://medium.com/data-science/implementing-vision-transformer-vit-from-scratch-3e192c6155f0)
- [ ] [Implement and Train ViT From Scratch for Image Recognition - PyTorch](https://www.youtube.com/watch?v=Vonyoz6Yt9c)
- [ ] [ChatGPT video](https://chatgpt.com/share/680f573d-7ebc-8007-b31d-238be748b172)
- [ ] [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Paper Explained)](https://www.youtube.com/watch?v=TrdevFK_am4)

### Prior Study

- [ ] [Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

# TODOs

## Work

- [x] Implement unfold layer
- [x] Implement projection layer
- [x] Implement classifier layer
- [x] Encoder simplified architecture
- [x] Evaluate performance single digit
- [ ] Encoder complete architecture
  - [x] Rescale
  - [x] Masking
  - [x] Multi-head
  - [ ] Residuals / Norm / MLP
- [ ] Evaluate performance single digit
- [ ] Multi-digit dataset prep ([example](https://github.com/YuriiOks/mlx-w3-mnist-transformer/blob/feat/phase3-sequence-gen/src/mnist_transformer/dataset.py))
- [ ] Multi-digit dataset to decoder integration
- [ ] Encoder / Decoder integration
- [ ] Evaluate performance

## Study

- [ ] Watch videos on transformers; understand e2e architecture
- [ ] Check https://github.com/DanielBryars/MLX3-VisionTransformers for learnings

## Experiments

- [ ] Compare results of hugging face vs pytorch dataset MNIST
  - [x] Integrate multi-data download
- [x] Use sequential module to refactor independent models into one
- [x] Replace my unfold function with nn.unfold (thanks again @DanielBryars for showing me)

## Tooling

- [x] wandb integration
- [x] wandb sweep
- [ ] learn python typing (e.g., `from typing import Tuple` + functions signatures `-> Tuple`)

## Follow ups

- [ ] [Scattered digits](https://github.com/guillaumeboniface/mnist_transformer/blob/3d1349b55d2590a7c319330cf97c432ba8c80b63/dataset.py#L53)
- [ ] [Augmented Digits](https://github.com/YuriiOks/mlx-w3-mnist-transformer/blob/feat/phase3-sequence-gen/src/mnist_transformer/dataset.py)