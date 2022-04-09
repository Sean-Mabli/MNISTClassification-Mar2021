import numpy as np
from emnist import extract_training_samples, extract_test_samples
import aiinpy as ai
import random

intrain, outtrain = extract_training_samples('digits')
intrainreal = np.zeros((5000, 54, 40))
for i in range(5000):
  randomone = random.randint(0, 26)
  randomtwo = random.randint(0, 12)
  intrainreal[i, randomone : randomone + 28, randomtwo : randomtwo + 28] = (intrain[i] / 255) - 0.5
outtrainreal = np.zeros((5000, 10))
for i in range(5000):
  outtrainreal[i, outtrain[i]] = 1

intest, outtest = extract_test_samples('digits')
intestreal = np.zeros((1000, 54, 40))
for i in range(1000):
  randomone = random.randint(0, 26)
  randomtwo = random.randint(0, 12)
  intestreal[i, randomone : randomone + 28, randomtwo : randomtwo + 28] = (intest[i] / 255) - 0.5
outtestreal = np.zeros((1000, 10))
for i in range(1000):
  outtestreal[i, outtest[i]] = 1

model = ai.model((54, 40), 10, [
  ai.conv(inshape=(54, 40), filtershape=(4, 3, 3), learningrate=0.01, activation=ai.relu()),
  ai.pool(stride=(2, 2), filtershape=(2, 2), opperation='Max'),
  ai.nn(outshape=10, activation=ai.stablesoftmax(), learningrate=0.1, weightsinit=(0, 0))
])

model.train((intrain, outtrainreal), 5000)
print(model.test((intest, outtestreal)))