import numpy as np
import matplotlib.pyplot as plt



def main():

  # OUR, CoPE, iCaRL, MIR
  data = [
    [10.00, 00.00, 00.00, 00.00], #0.2k
    [00.00, 30.00, 00.00, 00.00], #0.5k
    [00.00, 70.00, 40.00, 60.00],  #1k
    [00.00, 00.00, 00.00, 50.00],  #2k
  ]
  X = np.array(4)
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
  ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
  ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)
  
  ax.set_xticklabels(['0.2k', '0.5k', '1k', '2k'])
  ax.set_xlabel('Memory size')
  ax.set_ylabel('Accuracy (%)')

  plt.show()

if __name__ == '__main__':
  main()