import numpy as np
import matplotlib.pyplot as plt



def main():

  # For fmnist: (0.2k, 0.5k, 1k, 2k)
  our_data   = [80.69, 84.54, 85.43, 86.12] #[80.696±0.673, 84.540±0.092, 85.430±0.145, 86.120±0.876]
  our_err    = [0.67, 0.09, 0.14, 0.87]

  our_ce_data= [81.07, 84.53, 86.24, 87.07] #[81.073±1.335, 84.536±0.584, 86.243±0.440, 87.070±0.433]
  our_ce_err = [1.33, 0.58, 0.44, 0.43]
  
  cope_data  = [75.91, 79.14, 81.83, 83.78] #[75.917±0.653, 79.143±4.987, 81.837±2.786, 83.780±2.204]
  cope_err   = [0.65, 4.98, 2.78, 2.20]
  
  icarl_data = [49.21, 51.61, 50.38, 50.93] #[49.213±1.739, 51.613±2.696, 50.380±2.098, 50.933±1.401]
  icarl_err  = [1.73, 2.69, 2.09, 1.40]
  
  rvr_data   = [10.00, 10.00, 10.00, 10.00] #[00.000±0.000, 79.723±3.010, 83.627±1.220, 83.247±0.540]
  rvr_err    = [0.10, 3.01, 1.22, 0.54]
  
  gem_data   = [10.00, 10.00, 10.00, 10.00] #[00.000±0.000, 00.000±0.000, 00.000±0.000, 00.000±0.000] blueviolet
  gem_err    = [0.10, 0.10, 0.10, 0.10]
  
  gss_data   = [10.00, 10.00, 10.00, 10.00] #[00.000±0.000, 00.000±0.000, 00.000±0.000, 00.000±0.000]
  gss_err    = [0.10, 0.10, 0.10, 0.10]


  N = 4
  ind = np.arange(N)  # the x locations for the groups
  width = 0.1
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  
  ax.bar(
    ind - 2.0*width,
    our_data,
    yerr=our_err,
    color='forestgreen',
    width=width,
    align='center',
    label='OUR: 84.19±2.09'
  )
  ax.bar(
    ind - 1.0*width,
    our_ce_data,
    yerr=our_ce_err,
    color='gold',
    width=width,
    align='center',
    label='OUR-CE: 84.73±2.30'
  )
  ax.bar(
    ind - 0.0*width,
    cope_data,
    yerr=cope_err,
    color='orchid',
    width=width,
    align='center',
    label='CoPE: 80.16±2.95'
  )
  ax.bar(
    ind + 1.0*width,
    icarl_data,
    yerr=icarl_err,
    color='dodgerblue',
    width=width,
    align='center',
    label='iCaRL: 50.53±0.87'
  )
  ax.bar(
    ind + 2.0*width,
    rvr_data,
    yerr=rvr_err,
    color='salmon',
    width=width,
    align='center',
    label='reservoir: 00.00±0.00'
  )

  ax.set_xticks(ind)
  # ax.set_xticks(ind+width)
  ax.set_xticklabels(['0.2k', '0.5k', '1k', '2k'])
  ax.set_xlabel('Memory size')
  ax.set_ylabel('Accuracy (%)')

  ax.legend()
  ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0, 1.2, 0.2),
          fancybox=True, shadow=False, ncol=3, fontsize=8)

  plt.show()

if __name__ == '__main__':
  main()