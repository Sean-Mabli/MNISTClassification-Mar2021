import numpy as np
from emnist import extract_training_samples, extract_test_samples
import aiinpy as ai
import random
import wandb

wandb.init(project="cnn-nonsquare")
config = wandb.config
config.filters = 4
config.filtersize = 3
config.convlr = 0.01
config.nnlr = 0.1
config.filterstwo = 4
config.filtersizetwo = 3
config.convlrtwo = 0.01
config.gen = 10000
config.trainsize = 5000
config.testsize = 1000

intrain, outtrain = extract_training_samples('digits')
intrainreal = np.zeros((config.trainsize, 54, 40))
for i in range(config.trainsize):
  randomone = random.randint(0, 26)
  randomtwo = random.randint(0, 12)
  intrainreal[i, randomone : randomone + 28, randomtwo : randomtwo + 28] = (intrain[i] / 255) - 0.5
outtrainreal = np.zeros((config.trainsize, 10))
for i in range(config.trainsize):
  outtrainreal[i, outtrain[i]] = 1

intest, outtest = extract_test_samples('digits')
intestreal = np.zeros((config.testsize, 54, 40))
for i in range(config.testsize):
  randomone = random.randint(0, 26)
  randomtwo = random.randint(0, 12)
  intestreal[i, randomone : randomone + 28, randomtwo : randomtwo + 28] = (intest[i] / 255) - 0.5
outtestreal = np.zeros((config.testsize, 10))
for i in range(config.testsize):
  outtestreal[i, outtest[i]] = 1

model = ai.model((54, 40), 10, [
  ai.conv(inshape=(54, 40), filtershape=(config.filters, config.filtersize, config.filtersize), learningrate=config.convlr, activation=ai.relu()),
  ai.pool(stride=(2, 2), filtershape=(2, 2), opperation='Max'),
  ai.conv(inshape=(4, 26, 19), filtershape=(config.filterstwo, config.filtersizetwo, config.filtersizetwo), learningrate=config.convlrtwo, activation=ai.relu()),
  ai.pool(stride=(2, 2), filtershape=(2, 2), opperation='Max'),
  ai.nn(outshape=10, activation=ai.stablesoftmax(), learningrate=config.nnlr, weightsinit=(0, 0))
])

model.train((intrainreal, outtrainreal), config.gen)
wandb.log({"accuracy": model.test((intestreal, outtestreal))})