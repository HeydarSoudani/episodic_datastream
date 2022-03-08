import torch
from torch.utils.data import DataLoader
import os
import time
import numpy as np
from pandas import read_csv

from trainers.episodic_train import train
from trainers.episodic_train import train as episodic_train
from trainers.batch_train import train as batch_train

from detectors.pt_detector import detector_preparation
from datasets.dataset import SimpleDataset
from utils.preparation import transforms_preparation
from evaluation import in_stream_evaluation


def stream_learn(model,
                 learner,
                 memory,
                 detector,
                 args, device):
    print('================================ Stream Learning ================================')
    
    # == time lists ============================
    retrainin_times = []
    detector_times = []
    memory_times = []
    eval_times = []

    # == Set retrain params ====================
    args.epochs = args.retrain_epochs
    args.meta_iteration = args.retrain_meta_iteration

    # == Data ==================================
    stream_data = read_csv(
        os.path.join(args.data_path, args.stream_file),
        sep=',', header=None).values

    # == Classes start points ===================
    f = open('output.txt', 'w')

    labels = stream_data[:, -1]
    label_set = set(labels)
    for label in label_set:
      start_point = np.where(labels == label)[0][0]
      print('Class {} starts at {}'.format(label, start_point))
      f.write("[Class %5d], Start point: %5d \n" % (label, start_point))

    if args.use_transform:
        _, test_transform = transforms_preparation()
        stream_dataset = SimpleDataset(
            stream_data, args, transforms=test_transform)
    else:
        stream_dataset = SimpleDataset(stream_data, args)
    dataloader = DataLoader(dataset=stream_dataset,
                            batch_size=1, shuffle=False)

    # == Stream ================================
    unknown_buffer = []
    known_buffer = {i: [] for i in detector._known_labels}
    detection_results = []
    last_idx = 0

    eval_start_time = time.time()

    for i, data in enumerate(dataloader):
        
        if i == 10000:
            break

        model.eval()
        with torch.no_grad():
            sample, label = data
            sample, label = sample.to(device), label.to(device)
            _, feature = model.forward(sample)
            real_novelty = label.item() not in detector._known_labels
            detected_novelty, predicted_label, prob = detector(
                feature, learner.prototypes)
            detection_results.append(
                (label.item(), predicted_label, real_novelty, detected_novelty))

            sample = torch.squeeze(sample, 0)  # [1, 28, 28]
            if detected_novelty:
                unknown_buffer.append((sample, label))
            else:
                known_buffer[predicted_label].append((sample, label))

            if (i+1) % 100 == 0:
                print("[stream %5d]: %d, %2d, %7.4f, %5s, %5s, %d" %
                      (i+1, label, predicted_label, prob, real_novelty, detected_novelty, len(unknown_buffer)))

        if (i+1) % args.known_retrain_interval == 0 \
                or len(unknown_buffer) == args.buffer_size:
            print('=== Retraining... =================')

            # == Preparing buffer ==================
            if (i+1) % args.known_retrain_interval == 0:
                buffer = []
                for label, data in known_buffer.items():
                    n = len(data)
                    if n > args.known_per_class:
                        idxs = np.random.choice(
                            range(n), size=args.known_per_class, replace=False)
                        buffer.extend([data[i] for i in idxs])
                    else:
                        buffer.extend(data)

            elif len(unknown_buffer) == args.buffer_size:
                buffer = unknown_buffer

            # == 1) evaluation ======================
            sample_num = i-last_idx

            CwCA, M_new, F_new, cm, acc_per_class = in_stream_evaluation(
                detection_results, detector._known_labels)
            eval_times.append(time.time() - eval_start_time)

            print("[On %5d samples]: %7.4f, %7.4f, %7.4f" %
                  (sample_num, CwCA, M_new, F_new))
            print("confusion matrix: \n%s\n" % cm)
            print("acc per class: %s\n" % acc_per_class)

            f.write("[In sample %2d], [On %5d samples]: %7.4f, %7.4f, %7.4f \n" %
                    (i, sample_num, CwCA, M_new, F_new))
            f.write("acc per class: %s\n" % acc_per_class)

            # == 2) Preparing retrain data ==========
            mem_start_time = time.time()
            new_train_data = memory.select(buffer, return_data=True)
            memory_times.append(time.time() - mem_start_time)
            print('Retrain data number: {}'.format(new_train_data.shape[0]))
            print('===========================')

            # == 3) Retraining Model ================
            retrain_start_time = time.time()
            if args.algorithm == 'batch':
                batch_train(
                    model,
                    learner,
                    new_train_data,
                    args, device)
            else:
                episodic_train(
                    model,
                    learner,
                    new_train_data,
                    args, device)
            retrainin_times.append(time.time() - retrain_start_time)

            # == 4) Recalculating Detector ==========
            print("Calculating detector ...")
            detector_start_time = time.time()
            _, new_known_labels, intra_distances\
                = detector_preparation(model,
                                       learner.prototypes,
                                       new_train_data,
                                       args, device)
            detector.threshold_calculation(intra_distances,
                                           new_known_labels,
                                           args.std_coefficient)
            detector_times.append(time.time() - detector_start_time)
            print("Detector Threshold: {}".format(detector.thresholds))
            detector.save(args.detector_path)
            print("Detector has been saved in {}.".format(args.detector_path))

            # == 5) Update parameters ===============
            known_labels = list(known_buffer.keys())
            labels_diff = list(set(new_known_labels)-set(known_labels))
            for label in labels_diff:
              print('Class {} detected at {}'.format(label, i))
              f.write("[Class %2d], Detected point: %5d \n" % (label, i))

            if len(unknown_buffer) == args.buffer_size:
              if len(labels_diff) != 0:
                for label in labels_diff:
                  known_buffer[label] = []
              unknown_buffer.clear()
            if (i+1) % args.known_retrain_interval == 0:
              known_buffer = {i: [] for i in detector._known_labels}

            # == Set parameters =====
            detection_results.clear()
            last_idx = i

            print('=== Streaming... =================')
            eval_start_time = time.time()
            # time.sleep(1.5)

    # == Last evaluation ========================
    sample_num = i-last_idx
    CwCA, M_new, F_new, cm, acc_per_class = in_stream_evaluation(
        detection_results, detector._known_labels)
    print("[On %5d samples]: %7.4f, %7.4f, %7.4f" %
          (sample_num, CwCA, M_new, F_new))
    print("confusion matrix: \n%s" % cm)
    print("acc per class: %s\n" % acc_per_class)
    eval_times.append(time.time() - eval_start_time)
    f.write("[In sample %5d], [On %5d samples]: %7.4f, %7.4f, %7.4f \n" %
            (i, sample_num, CwCA, M_new, F_new))
    f.write("acc per class: %s\n" % acc_per_class)
    f.close()

    ## == Print time
    all_time = sum(retrainin_times)+sum(detector_times)+sum(memory_times)+sum(eval_times)
    print(all_time)
    print(sum(retrainin_times))
    print(sum(detector_times))
    print(sum(memory_times))
    print(sum(eval_times))
    print("Time: %7.4f, %7.4f, %7.4f, %7.4f, %7.4f"%
        (sum(retrainin_times), sum(detector_times), sum(memory_times), sum(eval_times), all_time ))
     
    # print(retrainin_times)
    # print(detector_times)
    # print(memory_times)
    # print(eval_times)
    # print('Retrainin time is: {:.4f}s'.format(sum(retrainin_times)))
    # print('Detector time is: {:.4f}s'.format(sum(detector_times)))
    # print('Memory time is: {:.4f}s'.format(sum(memory_times)))
    # print('Eval time is: {:.4f}s'.format(sum(eval_times)))
    # print('All stream time is: {:.4f}s'.format(sum(retrainin_times)+sum(detector_times)+sum(memory_times)+sum(eval_times)))
