import nets
import numpy as np

x = np.zeros((1,512,512,3))
y = nets.resnet(x, version='resnet50', filters=64, weight_decay=1e-5)

print(y.shape)
