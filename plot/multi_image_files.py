import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():

  ### == prequential dataset ==================
  labels = ['Rotated-MNIST', 'Permuted-MNIST', 'Rotated-FashionMNIST', 'Permuted-FashionMNIST']
  fig, axs = plt.subplots(nrows=1, ncols=4, constrained_layout=True, figsize=(16, 5))
  for idx, label in enumerate(labels):
    image = mpimg.imread('photos/{}.png'.format(label))
    axs[idx].imshow(image, aspect="auto")    

    axs[idx].set_title(label, c='royalblue')
    axs[idx].set_xticklabels(())
    axs[idx].set_yticklabels(())
    axs[idx].set_xticks(())
    axs[idx].set_yticks(())
    axs[idx].set_aspect('equal')
    axs[idx].spines['top'].set_visible(False)
    axs[idx].spines['right'].set_visible(False)
    axs[idx].spines['bottom'].set_visible(False)
    axs[idx].spines['left'].set_visible(False)

  # plt.axis("tight")
  plt.subplots_adjust(wspace=0, hspace=0)
  plt.show()

  ### == OWR feature space ====================
  # row_labels = ['CPE', 'Prototypical']
  # col_labels = ['task1', 'task2', 'task3', 'task4', 'task5']
  # fig, axs = plt.subplots(nrows=2, ncols=5, constrained_layout=True, figsize=(14, 4.5))
  # for ax, row in zip(axs[:,0], row_labels):
  #   ax.set_ylabel(row, rotation=90, ha='center')

  # for ax, col in zip(axs[0], col_labels):
  #   ax.set_title(col)
  
  # for row_idx, row_name in enumerate(row_labels):
  #   for col_idx, col_name in enumerate(col_labels):
  #     image = mpimg.imread('photos/{}_{}.png'.format(row_name, col_name))
  #     axs[row_idx][col_idx].imshow(image, aspect="auto")
      
  #     axs[row_idx][col_idx].set_xticklabels(())
  #     axs[row_idx][col_idx].set_yticklabels(())
  #     axs[row_idx][col_idx].set_xticks(())
  #     axs[row_idx][col_idx].set_yticks(())
  #     axs[row_idx][col_idx].set_aspect('equal')
  #     axs[row_idx][col_idx].spines['top'].set_visible(False)
  #     axs[row_idx][col_idx].spines['right'].set_visible(False)
  #     axs[row_idx][col_idx].spines['bottom'].set_visible(False)
  #     axs[row_idx][col_idx].spines['left'].set_visible(False)

  # # plt.axis("tight")
  # plt.subplots_adjust(wspace=0, hspace=0)
  # plt.show()
      
  


if __name__ == '__main__':
  main()
