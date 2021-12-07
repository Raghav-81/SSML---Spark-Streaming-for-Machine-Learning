import matplotlib.pyplot as plt
import numpy as np

# Do file handeling here
ctr = 0
with open('graph_sgd.txt', 'r') as f:
	content = f.read()
	colist = content.split('\n')
	for i in colist:
		if i:
			ctr += 1
models = []
for i in range(ctr):
	model = list(map(float, input().split()))
	models.append(model)
	
tmodel = np.transpose(models)

idx = 0
for y in tmodel:
	if idx == 1:
		idx += 1
		continue
	plt.plot([i for i in range(1, len(y)+1)], y)
	plt.xlabel('Batch No.')
	if idx == 0:
		plt.ylabel('Precision')
	if idx == 2:
		plt.ylabel('F1 Score')
	if idx == 3:
		plt.ylabel('Accuracy')
	plt.show()
	idx += 1
