# Transfer learning

Uses machine learning to classify images. The model is fine-tuned on the CIFAR-10 dataset. It uses the pre-trained model ResNet-18.

Uses pytorch and torchvision.

## Training the model

```bash
python3 src/train_model.py
```

## Running the model

```bash
python3 src/run_model.py test_image_ship.png
```

## ResNet-18:

ResNet, short for "Residual Networks", is a family of deep neural architectures known for their deep layers, yet they maintain impressive efficiency and performance. The unique feature of ResNet architectures is their "skip connections" or "residual connections" that bypass one or more layers.

ResNet-18 is one of the smaller variants in the ResNet family, containing 18 layers, specifically:

Initial convolutional and pooling layers.
Four "blocks" with two convolutional layers each.
A final average pooling and fully connected layer leading to the output.
The idea behind the residual connections is to address the "vanishing gradient" problem that can occur in deep neural networks. As the network gets deeper, gradients during back-propagation can become extremely small, essentially causing the network to "forget" and not learn effectively. Residual connections allow gradients to bypass layers, making it easier to train deeper networks. This mechanism improves the training process and generalization.

In practice, ResNet architectures, including ResNet-18, have shown state-of-the-art performance on various image classification tasks, especially when pre-trained on large datasets like ImageNet.
