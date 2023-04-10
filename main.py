from network import *

net = network()
net.train()

# net.save()

net.load()
net.generate('a photo of a happy woman with long hair')