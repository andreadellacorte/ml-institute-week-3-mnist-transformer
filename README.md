# Task

This week we are creating a model that, provided an image with a set of numbers, decodes them into a list of

# Dataset

MNIST

# References

## Repos

- [https://github.com/besarthoxhaj/attention](besarthoxhaj/attention)

## Docs

- [https://karpathy.github.io/2019/04/25/recipe/](A Recipe for Training Neural Networks)

## Videos
- [ ] [https://www.youtube.com/watch?v=dichIcUZfOw](Visual Guide to Transformer Neural Networks - (Episode 1) Position Embeddings)
- [X] [https://www.youtube.com/watch?v=wjZofJX0v4M&t=203s](Transformers (how LLMs work) explained visually | DL5)
- [ ] [https://www.youtube.com/watch?si=iIYQixHB01MZEOpf](Vision Transformer made easy: a short tutorial)
- [X] [https://www.youtube.com/watch?v=eMlx5fFNoYc](Attention in transformers, step-by-step | DL6)
- [ ] [https://medium.com/data-science/implementing-vision-transformer-vit-from-scratch-3e192c6155f0](Implementing Vision Transformer (ViT) from Scratch)
- [ ] [https://www.youtube.com/watch?v=Vonyoz6Yt9c](Implement and Train ViT From Scratch for Image Recognition - PyTorch)
- [ ] [https://chatgpt.com/share/680f573d-7ebc-8007-b31d-238be748b172](ChatGPT video)
- [ ] [https://www.youtube.com/watch?v=TrdevFK_am4](An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Paper Explained))

Prior study

- [ ] [https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ](Neural Networks: Zero to Hero)

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
- [ ] Multi-digit dataset prep ([https://github.com/YuriiOks/mlx-w3-mnist-transformer/blob/feat/phase3-sequence-gen/src/mnist_transformer/dataset.py](example))
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