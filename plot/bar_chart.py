import numpy as np
import matplotlib.pyplot as plt



def main():

  # 0.2k, 0.5k, 1k, 2k
  our_data   = [79.37, 00.00, 00.00, 00.00]
  cope_data  = [00.00, 30.00, 00.00, 00.00]
  icarl_data = [00.00, 30.00, 40.00, 00.00]
  mir_data   = [00.00, 30.00, 40.00, 00.00]

  X = np.array(4)
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  
  ax.bar(X - 0.3, our_data, color = 'b', width = 0.2, align='center')
  ax.bar(X - 0.1, cope_data, color = 'g', width = 0.2, align='center')
  ax.bar(X + 0.1, icarl_data, color = 'r', width = 0.2, align='center')
  ax.bar(X + 0.3, mir_data,  color = 'y', width = 0.2, align='center')

  ax.set_xticklabels(['0.2k', '0.5k', '1k', '2k'])
  ax.set_xlabel('Memory size')
  ax.set_ylabel('Accuracy (%)')

  plt.show()

if __name__ == '__main__':
  main()