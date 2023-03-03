import pickle
from argparse import ArgumentParser, Namespace
import numpy as np
from math import floor, ceil
from pprint import pprint
import matplotlib.pyplot as plt

MODEL_NAME = 'our_model.pickle'
TIMEFRAMES = [ f'{x:02}:00' for x in range(24) ]

def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('timeframe', nargs='+', choices=TIMEFRAMES)

    return parser.parse_args()

def main() -> None:
    args = get_args()

    inputs = np.array(
        list(
            map(
                lambda x: TIMEFRAMES.index(x),
                args.timeframe
            )
        )
    ).reshape(-1, 1)

    #inputs = np.linspace(0,24,100).reshape(-1, 1)

    model = pickle.load(open(MODEL_NAME, "rb"))

    res = model.predict(inputs).flatten()

    plt.plot(res)
    plt.show()

    return [ pred for pred in res ]

if __name__ == '__main__':
    res = main()
    pprint(res)