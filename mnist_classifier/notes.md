# PROJECT 1 - MNIST CLASSIFIER 

## Env Set Up Instructions:
```bash
conda create --name vision_ai
conda install pip
pip install torch torchvision matplotlib scikit-learn
```

## Build Custom DataSet Class
Let's say you have:
```bash
/data/
├── train/
│   ├── cat/
│   │   ├── img1.jpg
│   │   └── ...
│   ├── dog/
│       ├── img2.jpg
│       └── ...
```

#### Step 1: Import the tools
```python
import os
from PIL import Image
from torch.utils.data import Dataset
```
#### Step 2: Defining a Custom DataSet
```python
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Step 1: Enumerate subfolders
        classes = sorted(os.listdir(root_dir))
        for idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name

            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")  # or "L" for grayscale

        if self.transform:
            image = self.transform(image)

        return image, label
```

#### Step 3: Use It
```python 
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = CustomImageDataset(root_dir="./data/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

```
#### BONUS: Test It
```python 
for images, labels in train_loader:
    print(images.shape)  # torch.Size([32, 3, 128, 128])
    print(labels)
    break

```

#### Variations to Make / Try Later:
Use Case | Changes Needed
Images + CSV | Read file paths + labels from a .csv
Multi-label | Change labels[idx] to a vector (e.g., torch.tensor([0,1,0]))
Segmentation | Load both image + mask per sample
3D Medical Images | Use nibabel or SimpleITK instead of PIL


## Creating Custom Transforms Pipeline
Transforms refers to the operations you will / may need to do, on the images before feeding them into your model. Below we will see an example of a basic one and then how we can add some more transforms to it. The last thing I will look at is how to create custom things to put into the transforms logic.

Here are some typical transforms:
- Convert formats (e.g., PIL → Tensor)
- Normalize pixel values
- Augment data (e.g., flips, crops, rotations)
- Standardize size, channels, etc.

#### Basic Transform Template:
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),              # PIL → Tensor & scale to [0.0, 1.0]
    transforms.Normalize((0.5,), (0.5,))  # Mean and std for normalization
])
```

#### Adding More Complexity to the Transform Template
```python
transforms.Compose([
    transforms.Resize((32, 32)),        # Resize image to 32x32
    transforms.RandomHorizontalFlip(),  # 50% chance to flip horizontally
    transforms.RandomRotation(10),      # Rotate randomly within ±10 degrees
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize channels
])
```

>_*NOTE:*_ The Normalize((mean,), (std,)) must match the number of channels: MNIST is grayscale → 1 channel → use (0.5,). RGB → use 3 values: (0.5, 0.5, 0.5).

#### Custom Transform Class & Using it in Compose
```python
class AddGaussianNoise:
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean



transform = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.1)
])
```

## Customizing DataLoaders and DataSets

#### What is a DataLoader?
A DataLoader is a class an object that wraps around your Data Set and gives you specific and common fuinctionality related to working with the data set, like:
- Mini-batches
- Shuffling 
- Parralell Loading (via `num_workers`)
- Custom Sampling

#### Manual DataLoader set up
```python
from torch.utils.data import Dataset
from PIL import Image
import os

class MyImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")  # or "RGB"
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
```
#### DataLoader usage
```python
from torch.utils.data import DataLoader

train_dataset = MyImageDataset(image_paths, labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

#### Advanced Topics for Later: Custom Sampling & Collation
- Weighted sampling: Useful for imbalanced datasets.
- Custom collate_fn: Lets you modify how batches are created.

---
## The Model
#### Defining the Building Blocks
Here we will create the model by coding up the layers.

```python
def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(64 * 7 * 7, 128)
    self.fc2 = nn.Linear(128, 10)
```

```python
self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
```
Conv2D: A filter (like a small scanner) that slides across the image.

1 → input channel (grayscale = 1 channel)

32 → output channels (number of filters to learn)

3 → filter size (3x3 patch)

padding=1 → keeps image size the same (28x28)

🧠 Imagine 32 little 3×3 windows scanning across your image and learning to detect basic things like edges or lines.

```python
self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
```
Now we go deeper: input is 32 channels, output is 64 filters.

These filters now look at the features from the first layer, not raw pixels.

📚 Analogy: First filters learn edges, this layer learns shapes from edges.

```python
self.pool = nn.MaxPool2d(2, 2)
```
Downsamples the image by taking the max value in a 2×2 block.

Cuts height and width in half.

🧊 Think of this as summarizing regions of the image while keeping the strongest features.

```python
self.fc1 = nn.Linear(64 * 7 * 7, 128)
```
Fully connected layer.

By this point, image has been shrunk to 7x7 in size and has 64 channels.

We flatten this 3D blob to a 1D vector of size 64 × 7 × 7 = 3136.

This is like going from features → neurons.

```python
self.fc2 = nn.Linear(128, 10)
```
Final layer → outputs 10 values, one for each digit class (0 to 9).

#### The Forward Function
```python
def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 64 * 7 * 7)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
```
🔹 Original Image
Shape: [1, 28, 28]
(1 grayscale channel, 28x28 pixels)

🧱 Step 1: conv1
```python
x = self.conv1(x)
```

Input: [1, 28, 28]

Operation: nn.Conv2d(1, 32, 3, padding=1)

Output: [32, 28, 28]

Applies 32 filters to 1 input channel → results in 32-channel output

🔌 Step 2: ReLU + MaxPool
```python
x = self.pool(F.relu(x))
```

Input: [32, 28, 28]

Output after pooling: [32, 14, 14]

🧱 Step 3: conv2
```python
x = self.conv2(x)
```

Input: [32, 14, 14] ✅

Operation: nn.Conv2d(32, 64, 3, padding=1)

Output: [64, 14, 14]

We're applying 64 new filters to the 32-channel "image". Each new filter learns to combine patterns across all 32 input channels → results in 64 new channels.

🔌 Step 4: ReLU + MaxPool
```python
x = self.pool(F.relu(x))
```

Input: [64, 14, 14]

Output after pooling: [64, 7, 7]

✅ Final Summary Table (Cleaned Up)

Step	Operation	Input Shape	Output Shape	Description
1	conv1	[1, 28, 28]	[32, 28, 28]	Apply 32 filters to 1 channel
2	ReLU + Pool	[32, 28, 28]	[32, 14, 14]	Downsample
3	conv2	[32, 14, 14]	[64, 14, 14]	Apply 64 filters to 32 input channels
4	ReLU + Pool	[64, 14, 14]	[64, 7, 7]	Downsample
5	Flatten	[64, 7, 7]	[3136]	Prepare for dense layers
6	fc1	[3136]	[128]	Dense layer
7	fc2	[128]	[10]	Output layer for 10 classes
🧠 Quick Visual Analogy (text-style)
scss

```bash
[1x28x28 image]
  ↓ Conv2D (32 filters)
[32x28x28 feature maps]
  ↓ Pool
[32x14x14]
  ↓ Conv2D (64 filters)
[64x14x14]
  ↓ Pool
[64x7x7]
  ↓ Flatten
[1D vector: 3136 values]
  ↓ Fully Connected (128)
  ↓ Fully Connected (10)
[Class scores]
```

🔑 Key Concept: Filters vs Channels
You define Conv2d(in_channels, out_channels, ...).

out_channels = number of filters you're learning.

Each filter is applied to the entire depth of the input and outputs 1 channel.

So Conv2d(32, 64, ...) means:

“Hey PyTorch, learn 64 filters that each scan across all 32 input channels.”

## Training Loop
```python
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):  # 5 epochs to start
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete. Loss: {loss.item():.4f}")
```
```python
model = SimpleCNN()
```
Here we instantiate the model we defined earlier.

```python
criterion = nn.CrossEntropyLoss()
```
Defines your loss function.

CrossEntropyLoss is good for classification problems (like predicting a digit 0-9).

It measures: "How far are my predictions from the true labels?"

❌ Wrong predictions → high loss
✅ Correct predictions → low loss

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
Defines your optimizer.

Adam is a popular method for tweaking the model’s parameters (weights) intelligently.

It will use the gradients computed during loss.backward() to adjust the parameters.

lr=0.001 means "small careful updates" — learning rate.










Inside the Batch Training
```python
optimizer.zero_grad()
```

Clear out old gradients from the previous batch.

Otherwise gradients accumulate and mess things up.

```python
output = model(images)
```

Run the images through your model.

output is now raw scores for each of the 10 classes (before softmax).

Example:
one image might produce something like [2.1, 0.3, -1.5, ..., 0.8] → score for each class.

```python
loss = criterion(output, labels)
```

Compare the model’s prediction (output) with the true labels (labels).

loss is a single number: how bad was this batch of predictions?

```python
loss.backward()
```

PyTorch automatically computes the gradients for each model parameter.

Gradient = "how much would changing this parameter make the loss go up or down?"

```python
optimizer.step()
```

Apply the gradients.

This nudges the model’s filters and weights slightly to reduce the loss next time.

📚 Analogy: Like gently steering a boat toward the correct direction based on current error.

```python
print(f"Epoch {epoch+1} complete. Loss: {loss.item():.4f}")
```

After each full epoch (one pass over the data), print the final loss from the last batch.

loss.item() extracts the raw float number from the tensor.

📈 Visual High-Level Flow
```bash
1. Create model
2. Define loss function (how wrong we are)
3. Define optimizer (how we fix mistakes)

For each epoch:
    For each batch of images:
        1. Zero out old gradient info
        2. Predict outputs
        3. Calculate how wrong (loss)
        4. Compute gradients (backprop)
        5. Update weights using optimizer
    Print loss after the epoch
```

🧠 Final Summary (Even Simpler)

Step	What Happens	Why It Matters
model(images)	Make a guess	Predict labels from input
loss = criterion(output, labels)	Measure how wrong	Get a score for how bad our guess was
loss.backward()	Figure out "who to blame"	Compute how to change each weight
optimizer.step()	Make tiny adjustments	Update the model to be a little smarter