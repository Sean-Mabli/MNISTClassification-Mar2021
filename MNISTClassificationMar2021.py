import numpy as np
import mnist

# Set The # Of Filters In The Convolutional Layer
ConvolutionLayer1AndMaxPooling1Length = 4

# Set Weights To Random Values
ConvolutionLayer1Weights = np.random.uniform(-0.25, 0.25, (ConvolutionLayer1AndMaxPooling1Length, 3, 3))
WeightsInputToOutput = np.zeros((676, 10))
OutputBiases = np.zeros(10)

# Set The Stride And Learning Rates
Stride = 2
HiddenLayerLearningRate = 0.1
ConvolutionalLearningRate = 0.005

# Equations
def Sigmoid(Input):
  return 1 / (1 + np.exp(-Input))
def DerivativeOfSigmoid(Input):
  return Input * (1 - Input)

def StableSoftMax(Input):
  return np.exp(Input - max(Input)) / sum(np.exp(Input - max(Input)))
def DerivativeOfStableSoftMax(Input):
  Output = np.zeros(Input.size)
  for i in range(Output.size):
    Output[i] = (np.exp(Input[i]) * (sum(np.exp(Input)) - np.exp(Input[i])))/(sum(np.exp(Input))) ** 2
  return Output

def ReLU(Input):
  Output = np.maximum(0, Input)
  return Output
def DerivativeOfReLU(Input):
  Output = np.zeros(Input.size)
  for i in range(Input.size):
    Output[i] = 0 if (Input[i] <= 0) else 1
  return Output

# Forward Propagation Functions
def Convolution(InputImage, Filter):
  OutputWidth = InputImage[0].size
  OutputHeight = int(InputImage.size / InputImage[0].size)
  NumberOfFilters = int(Filter.size / Filter[0][0].size / int(Filter[0].size / Filter[0][0].size))
  OutputWidth -= 2
  OutputHeight -= 2
  OutputArray = np.zeros((NumberOfFilters, OutputHeight, OutputWidth))
  for FilterNumber in range(NumberOfFilters):
    for i in range(OutputHeight):
      for j in range(OutputWidth):
        OutputArray[FilterNumber, i, j] = np.sum(np.multiply(InputImage[i : i + 3, j : j + 3], Filter[FilterNumber, :, :]))
  return OutputArray

def MaxPooling(InputArray, Stride):
  OutputWidth = int(InputArray[0][0].size / 2)
  OutputHeight = int(InputArray[0].size / InputArray[0][0].size / 2)
  NumberOfFilters = int(InputArray.size / (OutputWidth * 2) / (OutputHeight * 2))
  OutputArray = np.zeros((NumberOfFilters, OutputHeight, OutputWidth))
  for FilterNumber in range(NumberOfFilters):
    for i in range(OutputHeight):
      for j in range(OutputWidth):
        OutputArray[FilterNumber, i, j] = max(InputArray[FilterNumber, i * Stride, j * Stride], InputArray[FilterNumber, i * Stride + 1, j * Stride], InputArray[FilterNumber, i * Stride, j * Stride + 1], InputArray[FilterNumber, i * Stride + 1, j * Stride + 1])
  return OutputArray

def HiddenLayer(InputLayer, Weights, OutputLength, Bias, Activation):
  Output = np.zeros(OutputLength)
  for i in range(OutputLength):
    Output[i] = np.sum(Weights[:, i] * InputLayer[:])
  Output += Bias
  
  # Apply Activation Function
  if(Activation == "ReLU"):
    Output = ReLU(Output)
  if(Activation == "Sigmoid"):
    Output = Sigmoid(Output)
  if(Activation == "StableSoftMax"):
    Output = StableSoftMax(Output)
  return Output

# Back Propagation Functions
def HiddenLayerBackpropagation(CurrentLayer, FollowingLayer, FollowingLayerError, WeightCurrentLayerToFollowingLayer, FollowingLayerBiases, Activation, LearningRate):
  # Calculate Gradients
  FollowingLayerGradient = np.zeros(FollowingLayer.size)
  if(Activation == "ReLU"):
    FollowingLayerGradient = np.multiply(DerivativeOfReLU(FollowingLayer), FollowingLayerError) * LearningRate
  if(Activation == "Sigmoid"):
    FollowingLayerGradient = np.multiply(DerivativeOfSigmoid(FollowingLayer), FollowingLayerError) * LearningRate
  if(Activation == "StableSoftMax"):
    FollowingLayerGradient = np.multiply(DerivativeOfStableSoftMax(FollowingLayer), FollowingLayerError) # * LearningRate
      
  # Calculate Current Layer Error
  CurrentLayerError = np.zeros(CurrentLayer.size)
  for i in range(CurrentLayer.size):
      CurrentLayerError[i] = sum(np.transpose(WeightCurrentLayerToFollowingLayer)[:, i] * FollowingLayerError[:])
      
  # Apply Deltas To The Weights And Biases
  FollowingLayerBiases += FollowingLayerGradient
  WeightCurrentLayerToFollowingLayer += np.outer(np.transpose(CurrentLayer), FollowingLayerGradient)
  return CurrentLayerError

def MaxPoolingBackpropagation(PreviousConvolutionalLayer, CurrentMaxPoolingLayer, CurrentMaxPoolingLayerError):
  OutputWidth = PreviousConvolutionalLayer[0][0].size
  OutputHeight = int(PreviousConvolutionalLayer[0].size / PreviousConvolutionalLayer[0][0].size)
  NumberOfFilters = int(PreviousConvolutionalLayer.size / OutputWidth / OutputHeight)
  PreviousConvolutionalLayerError = np.zeros(PreviousConvolutionalLayer.shape)
  for FilterNumber in range(NumberOfFilters):
    for i in range(OutputHeight):
      for j in range(OutputWidth):
        if(CurrentMaxPoolingLayer[FilterNumber, int(i / 2), int(j / 2)] == PreviousConvolutionalLayer[FilterNumber, i, j]):
          PreviousConvolutionalLayerError[FilterNumber, i, j] = CurrentMaxPoolingLayerError[FilterNumber, int(i / 2), int(j / 2)] 
        else:
          PreviousConvolutionalLayerError[FilterNumber, i, j] = 0
  return PreviousConvolutionalLayerError

def ConvolutionBackpropagation(InputImage, ConvolutionError, Filter, LearningRate):
  ImageWidth = InputImage[0].size
  ImageHeight = int(InputImage.size / InputImage[0].size)
  NumberOfFilters = int(Filter.size / Filter[0][0].size / int(Filter[0].size / Filter[0][0].size))
  FilterGradients = np.zeros((NumberOfFilters, 3, 3))
  for FilterNumber in range(NumberOfFilters):
    for i in range(ImageHeight - 2):
      for j in range(ImageWidth - 2):
        FilterGradients[FilterNumber, :, :] += InputImage[i : (i + 3), j : (j + 3)] * ConvolutionError[FilterNumber, i, j]
  Filter += FilterGradients * LearningRate

def TestNetwork():
  NumberCorrect = 0
  Generation = 0
  for i in range(TestImageLoaded):
    InputImage = (TestImages[i] / 255) - 0.5

    ConvolutionLayer1 = Convolution(InputImage, ConvolutionLayer1Weights)
    MaxPooling1 = MaxPooling(ConvolutionLayer1, 2)
    Input = MaxPooling1.flatten()
    Output = HiddenLayer(Input, WeightsInputToOutput, 10, OutputBiases, "StableSoftMax")
    
    NumberCorrect += 1 if np.argmax(Output) == TestLabels[i] else 0
    Generation += 1
  return NumberCorrect / Generation

# Load MNIST Training And Testing Images
TrainingImageLoaded = 1000
TestImageLoaded = 1000
TrainingImages = mnist.train_images()[0:TrainingImageLoaded]
TrainingLabels = mnist.train_labels()[0:TrainingImageLoaded]
TestImages = mnist.test_images()[0:TestImageLoaded]
TestLabels = mnist.test_labels()[0:TestImageLoaded]

for Generation in range(1000):
  # Set Input
  Random = np.random.randint(0, TrainingImageLoaded)
  InputImage = (TrainingImages[Random] / 255) - 0.5
  RealOutput = np.zeros(10)
  RealOutput[TrainingLabels[Random]] = 1
  
  # Forward Propagation
  ConvolutionLayer1 = Convolution(InputImage, ConvolutionLayer1Weights)
  MaxPooling1 = MaxPooling(ConvolutionLayer1, Stride)
  Input = MaxPooling1.flatten()
  Output = HiddenLayer(Input, WeightsInputToOutput, 10, OutputBiases, "StableSoftMax")

  # Back Propagation
  OutputError = RealOutput - Output
  InputError = HiddenLayerBackpropagation(Input, Output, OutputError, WeightsInputToOutput, OutputBiases, "StableSoftMax", HiddenLayerLearningRate) 
  MaxPooling1Error = InputError.reshape(MaxPooling1.shape)
  
  ConvolutionalError = MaxPoolingBackpropagation(ConvolutionLayer1, MaxPooling1, MaxPooling1Error)
  ConvolutionBackpropagation(InputImage, ConvolutionalError, ConvolutionLayer1Weights, ConvolutionalLearningRate)
  
print(TestNetwork())