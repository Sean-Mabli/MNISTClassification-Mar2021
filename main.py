import numpy as np
from emnist import extract_training_samples, extract_test_samples
import aiinpy as ai
import random
import wandb
import pickle

wandb.init(project="cnn-nonsquare")
config = wandb.config
config.filters = 13
config.filtersize = 7
config.convlr = 0.007553907690039779
config.nnlr = 0.33755062899073013
config.gen = 17867
config.trainsize = 15991
config.testsize = 4650

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
  ai.nn(outshape=10, activation=ai.stablesoftmax(), learningrate=config.nnlr, weightsinit=(0, 0))
])

model.train((intrainreal, outtrainreal), config.gen)
wandb.log({"accuracy": model.test((intestreal, outtestreal))})

pickle.dump(model.model[0].filter, open("convfilter.p", "w"))
pickle.dump(model.model[2].bias, open("convbias.p", "w"))
pickle.dump(model.model[2].weights, open("nnweights.p", "w"))
pickle.dump(model.model[2].biases, open("nnbiases.p", "w"))