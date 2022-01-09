from hausdorff import hausdorff_distance


def hausdorff_dist():

  test_data = read_csv(
    './data/{}_stream_novel.csv'.format(args.dataset),
    sep=',').values 

  


  hausdorff_distance(features_source, features_target, distance="cosine")