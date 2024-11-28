# sign-recog
A model based on data collected in classroom.
### Preprocessing pipeline:
This is the step by step process to get a single signature from
the image of the sheet:
1) Load the images and converted to grayscale and binary
using OpenCV.
2) A binary threshold is applied, followed by dilation to
make the signature strokes bold.
3) Find the largest contour which is the complete grid.
4) The grid warped to a standard perspective to normalize
the orientation.
5) Applied erosion on such that only vertical and horizontal
lines remain
6) Map these lines on the original binary image to remove
the grid lines from the image
7) Hough transform is used to detect the lines and create
lines on the grid on its own as the previous lines are
removed
8) Individual signature cells are cropped and resized to a
standard dimension.

### CNN Architecture
- Data Augmentation (Random Rotation, Random Zoom, and Color Inversion)
- Rescaling layer (1/255)
- 4 Convolutional layers (16, 32, 64, 64 filters) with ReLU activation and MaxPooling
- Dropout layer (0.2)
- Flatten layer
- Dense layer (128 units) with ReLU activation
- Batch Normalization
- Output Dense layer with softmax activation

### Training Process 
The model was trained with the following parameters:
- Batch size: Default (32)
- Epochs: 50 for my CNN (with early stopping)
- Learning rate: 1e-3 for custom CNN
- Early stopping: Patience of 10 epochs, min delta of 0.001

Resulting in 84 percent accuracy
