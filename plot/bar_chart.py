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
  
  rvr_data   = [72.61, 79.72, 83.62, 83.24] #[72.613±1.161, 79.723±3.010, 83.627±1.220, 83.247±0.540]
  rvr_err    = [1.16, 3.01, 1.22, 0.54]
  
  gem_data   = [39.69, 35.46, 30.00, 30.00] #[39.693±14.881, 35.463±0.638, 00.000±0.000, 00.000±0.000] blueviolet
  gem_err    = [14.88, 0.63, 0.10, 0.10]
  
  gss_data   = [10.00, 10.00, 10.00, 10.00] #[00.000±0.000, 00.000±0.000, 00.000±0.000, 00.000±0.000]
  gss_err    = [0.10, 0.10, 0.10, 0.10]


  N = 4
  ind = np.arange(N)  # the x locations for the groups
  width = 0.1
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  
  ax.bar(
    ind - 2.5*width,
    our_data,
    yerr=our_err,
    color='royalblue', #cornflowerblue
    width=width,
    align='center',
    label='OUR: 84.19±2.09'
  )
  ax.bar(
    ind - 1.5*width,
    our_ce_data,
    yerr=our_ce_err,
    color='hotpink',
    width=width,
    align='center',
    label='OUR-CE: 84.73±2.30'
  )
  ax.bar(
    ind - 0.5*width,
    cope_data,
    yerr=cope_err,
    color='blueviolet',
    width=width,
    align='center',
    label='CoPE: 80.16±2.95'
  )
  ax.bar(
    ind + 0.5*width,
    icarl_data,
    yerr=icarl_err,
    color='gold',
    width=width,
    align='center',
    label='iCaRL: 50.53±0.87'
  )
  ax.bar(
    ind + 1.5*width,
    rvr_data,
    yerr=rvr_err,
    color='darkorange',
    width=width,
    align='center',
    label='reservoir: 79.79±4.41'
  )
  ax.bar(
    ind + 2.5*width,
    gem_data,
    yerr=gem_err,
    color='limegreen',
    width=width,
    align='center',
    label='GEM: 00.00±0.00'
  )


  ax.set_xticks(ind)
  # ax.set_xticks(ind+width)
  ax.set_xticklabels(['0.2k', '0.5k', '1k', '2k'])
  ax.set_xlabel('Memory size')
  ax.set_ylabel('Accuracy (%)')
  ax.set_ylim([20, 100])

  ax.legend()
  ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0, 1.2, 0.2),
          fancybox=True, shadow=False, ncol=3, fontsize=8)

  plt.show()

if __name__ == '__main__':
  main()