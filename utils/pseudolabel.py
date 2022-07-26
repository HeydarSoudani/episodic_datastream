import torch
from sklearn.mixture import GaussianMixture
import numpy as np
import random



def pseudo_labeler(data, n_component=2, ratio=1.0):
  """
  Compute ...
  Args:
    data: list of (sample, label, feature)
  Returns:
    ---
  """ 
  print(n_component)
  print(ratio)
  p_data = []
  samples = torch.stack([item[0] for item in data])
  labels = torch.tensor([item[1] for item in data])
  features = torch.squeeze(torch.stack([item[2] for item in data]))

  gmm = GaussianMixture(n_components=n_component, random_state=0)
  gmm.fit(features.detach().cpu().numpy())
  gmm_predict = gmm.predict(features.detach().cpu().numpy())
  
  for i in range(n_component):
    component_idx = np.where(gmm_predict == i)[0].astype(int)
    component_samples = samples[component_idx]
    component_features = features[component_idx]
    component_labels = labels[component_idx]

    n = component_idx.shape[0]
    component_idx_ratio = random.sample(list(component_idx), int(n*ratio))
      # component_idx[np.random.choice(
      #   range(n), size=int(n*ratio), replace=False
      # )]
    plabel = torch.argmax(torch.bincount(labels[component_idx_ratio])).item()
		
    p_data.extend([
      (
        component_samples[j],
        component_labels[j] if j in component_idx_ratio else plabel,
        component_features[j])
      for j in range(n)
    ])

  return p_data


