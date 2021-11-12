
from trainers import train
# from detectors.reptile_detector import replite_detector
from detectors.pt_detector import pt_detector

def init_learn(model, train_data, base_labels, args, device):

  ### == Train Model =================
  train(model, train_data, args, device)
  # batch_train(model, train_data, args, device)

  ## == Save Novel detector =========
  if args.which_model == 'best':
    model.load(args.best_model_path)
  # _ = replite_detector(model, train_data, args, device)
  _ = pt_detector(model, train_data, base_labels, args, device)



if __name__ == '__main__':
  pass