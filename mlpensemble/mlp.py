import timeit
from logging import DEBUG, basicConfig, getLogger

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor

logger = getLogger(__name__)
logger.setLevel(DEBUG)
basicConfig(
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Objective:
    def __init__(self, dataset_x, dataset_y):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y  # .reshape(len(dataset_y), 1)
        self.mlp_max_iter = 530000
        self.mlp_n_layers = [1, 10]
        self.mlp_n_neurons = [4, 64]
        self.mlp_warm_start = [True, False]
        self.mlp_activation = ["identity", "logistic", "tanh", "relu"]
        self.test_size = 0.4
        self.seconds_history = []
        self.scores_history = []
        self.models = {}
        if len(set(dataset_y)) > 2:
            # if len(set(np.array(self.dataset_y)[:, 0])) > 2:
            self.is_regressor = True
        else:
            self.is_regressor = False
        self.max_num_store_models = 5

    def generate_params(self, trial):
        params = {}
        n_layers = trial.suggest_int(
            "n_layers", self.mlp_n_layers[0], self.mlp_n_layers[1]
        )
        layers = []
        for i in range(n_layers):
            layers.append(
                trial.suggest_int(str(i), self.mlp_n_neurons[0], self.mlp_n_neurons[1])
            )

        params["hidden_layer_sizes"] = set(layers)
        params["max_iter"] = self.mlp_max_iter
        params["early_stopping"] = True
        params["warm_start"] = trial.suggest_categorical(
            "mlp_warm_start", self.mlp_warm_start
        )
        params["activation"] = trial.suggest_categorical(
            "mlp_activation", self.mlp_activation
        )
        return params

    def __call__(self, trial):
        x_train, x_test, y_train, y_test = train_test_split(
            self.dataset_x, self.dataset_y, test_size=self.test_size
        )
        params = self.generate_params(trial)
        # logger.debug(self.dataset_y)
        # logger.debug(set(list(self.dataset_y.flatten())))
        if self.is_regressor:
            model = MLPRegressor(**params)
        else:
            model = MLPClassifier(**params)

        # logger.debug(model)
        # logger.debug((x_train.shape, y_train.shape))
        # model.fit(x_train, y_train)
        seconds = timeit.timeit(lambda: model.fit(x_train, y_train), number=1)
        self.seconds_history.append(seconds)
        score = model.score(x_test, y_test)
        logger.debug((trial.number, score, model))
        self.scores_history.append(score)
        self.models[trial.number] = model

        for index, rank in enumerate(rankdata(-np.array(self.scores_history))):
            if rank > self.max_num_store_models:
                self.models[index] = None

        return score
