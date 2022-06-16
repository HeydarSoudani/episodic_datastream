import torch
from sklearn.mixture import GaussianMixture
import numpy as np

def pseudo_labeler(data, n_component=2, ratio=1.0):
	"""
	Compute ...
	Args:
		data: list of (sample, label, feature)
	Returns:
		---
	""" 
	p_data = []
	samples = torch.stack([item[0] for item in data])
	labels = torch.tensor([item[1] for item in data])
	features = torch.stack([item[2] for item in data])
	seen_labels = torch.unique(labels)

	gmm = GaussianMixture(n_components=n_component,  random_state=0)
	gmm.fit(features)
	gmm_predict = gmm.predict(features)

	for i in range(n_component):
		component_idx = np.where(gmm_predict == i)[0].astype(int)
		n = component_idx.shape[0]
		component_idx_ratio = \
			component_idx[np.random.choice(
				range(n), size=int(n*ratio), replace=False
			)]

		plabel = torch.argmax(torch.bincount(labels[component_idx_ratio])).item()
		component_samples = samples[component_idx]
		component_features = features[component_idx]
		p_data.extend([
			(component_samples[i], plabel, component_features[i])
			for i in range(n)
		])

	return p_data




