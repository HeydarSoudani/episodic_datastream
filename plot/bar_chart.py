import numpy as np
import matplotlib.pyplot as plt



def main():

  # Now for fmnist

  # 0.2k, 0.5k, 1k, 2k
  our_data   = [79.37, 83.29, 10.00, 86.13] #[00.000±0.000, 00.000±0.000, 00.000±0.000, 00.000±0.000]
  our_ce_data= [10.00, 10.00, 10.00, 10.00] #[00.000±0.000, 00.000±0.000, 00.000±0.000, 00.000±0.000]
  cope_data  = [75.91, 79.14, 81.83, 83.78] #[75.917±0.653, 79.143±4.987, 81.837±2.786, 83.780±2.204]
  icarl_data = [49.21, 51.61, 50.38, 51.93] #[49.213±1.739, 51.613±2.696, 50.380±2.098, 50.933±1.401]
  mir_data   = [10.00, 10.00, 10.00, 10.00] #[00.000±0.000, 00.000±0.000, 00.000±0.000, 00.000±0.000]

  N = 4
  ind = np.arange(N)  # the x locations for the groups
  width = 0.2
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  
  ax.bar(ind - 1.5*width, our_data, color='teal', width=width, align='center')
  ax.bar(ind - 0.5*width, cope_data, color='orchid', width=width, align='center')
  ax.bar(ind + 0.5*width, icarl_data, color='dodgerblue', width=width, align='center')
  ax.bar(ind + 1.5*width, mir_data,  color='salmon', width=width, align='center')

  ax.set_xticks(ind)
  # ax.set_xticks(ind+width)
  ax.set_xticklabels(['0.2k', '0.5k', '1k', '2k'])
  ax.set_xlabel('Memory size')
  ax.set_ylabel('Accuracy (%)')

  plt.show()

if __name__ == '__main__':
  main()