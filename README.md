# Task

This week we are creating a model that, provided an image with a set of numbers, decodes them into a list of

# Dataset

MNIST

# Reference Videos

- [ ] [https://www.youtube.com/watch?v=dichIcUZfOw](Visual Guide to Transformer Neural Networks - (Episode 1) Position Embeddings)
- [ ] [https://www.youtube.com/watch?v=wjZofJX0v4M&t=203s](Transformers (how LLMs work) explained visually | DL5)
- [ ] [https://www.youtube.com/watch?si=iIYQixHB01MZEOpf](Vision Transformer made easy: a short tutorial)
- [ ] [https://www.youtube.com/watch?v=eMlx5fFNoYc](Attention in transformers, step-by-step | DL6)
- [ ] [https://medium.com/data-science/implementing-vision-transformer-vit-from-scratch-3e192c6155f0](Implementing Vision Transformer (ViT) from Scratch)
- [ ] [https://www.youtube.com/watch?v=Vonyoz6Yt9c](Implement and Train ViT From Scratch for Image Recognition - PyTorch)

# TODOs

## Work

- [x] Implement unfold layer
- [x] Implement projection layer
- [x] Implement classifier layer
- [x] Encoder simplified architecture
- [x] Evaluate performance single digit
- [ ] Encoder complete architecture
- [ ] Evaluate performance single digit
- [ ] Multi-digit dataset prep
- [ ] Multi-digit dataset to decoder integration
- [ ] Encoder / Decoder integration
- [ ] Evaluate performance

## Study
- [ ] Watch videos on transformers; understand e2e architecture
- [ ] Check https://github.com/DanielBryars/MLX3-VisionTransformers for learnings

## Experiments
- [ ] Compare results of hugging face vs pytorch dataset MNIST
- [ ] Use sequential module to refactor independent models into one
- [ ] Replace my unfold function with nn.unfold (thanks again <@1337087693401751697> for showing me)

## Tooling

- [x] wandb integration
- [x] wandb sweep

## Follow ups
- [ ] [Scattered digits](https://github.com/guillaumeboniface/mnist_transformer/blob/3d1349b55d2590a7c319330cf97c432ba8c80b63/dataset.py#L53)