import torch.nn as nn

class BaselineCNN(nn.Module):
    """
    A simple baseline Convolutional Neural Network (CNN) for image classification.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        relu1 (nn.ReLU): ReLU activation function following the first convolutional layer.
        pool1 (nn.MaxPool2d): Max pooling layer following the first ReLU activation.
        conv2 (nn.Conv2d): The second convolutional layer.
        relu2 (nn.ReLU): ReLU activation function following the second convolutional layer.
        pool2 (nn.MaxPool2d): Max pooling layer following the second ReLU activation.
        conv3 (nn.Conv2d): The third convolutional layer.
        relu3 (nn.ReLU): ReLU activation function following the third convolutional layer.
        pool3 (nn.MaxPool2d): Max pooling layer following the third ReLU activation.
        fc_sq (nn.Sequential): A sequence of fully connected layers and dropout layers.

    Parameters:
        drop_rate (float): The dropout rate for regularization.
        num_classes (int): The number of output classes for classification.
    """
    def __init__(self, drop_rate, num_classes):
        super(BaselineCNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the number of features after the convolutional blocks
        num_features = 64 * 28 * 28
        # Fully connected layers
        self.fc_sq = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(256, num_classes)
                    )

    def forward(self, x):
        """
        Defines the forward pass of the CNN.

        Parameters:
            x (torch.Tensor): The input tensor containing the image data.

        Returns:
            torch.Tensor: The output tensor after passing through the CNN.
        """
        # Apply the first convolutional block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Apply the second convolutional block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Apply the third convolutional block
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        # Apply the fully connected layers
        x = self.fc_sq(x)
        return x
