import argparse

import numpy as np


def parse_args():
    p = argparse.ArgumentParser("Evaluate top-k error from predicted result")
    p.add_argument(
        'ground_truth', type=str,
        help='Groud truth text file. Each line should contain integer '
        'value which correspond with the ground truth label index.')
    p.add_argument(
        'prediction', type=str,
        help='Prediction text file. Each line should contain integer '
        'values which correspond with the predicted label indices in '
        'decending order of confidence. Only top-5 will be used.'
    )
    return p.parse_args()


def run(ground_truth, prediction):
    gt = np.loadtxt(ground_truth, delimiter=' ')
    assert gt.ndim == 1 or gt.shape[1] == 1
    gt = gt.reshape(-1, 1)

    pred = np.loadtxt(prediction, delimiter=' ',
                      usecols=list(range(5)))
    assert pred.shape[1] >= 5
    pred = pred[:, :5]

    correct = gt == pred
    accuracy = np.cumsum(np.sum(correct, axis=0) / len(correct))
    print('Top-1 error rate: %f%%' % float((1. - accuracy[0]) * 100.))
    print('Top-5 error rate: %f%%' % float((1. - accuracy[4]) * 100.))


if __name__ == '__main__':
    args = parse_args()
    run(args.ground_truth, args.prediction)
