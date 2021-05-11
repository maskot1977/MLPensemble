import timeit
from logging import DEBUG, basicConfig, getLogger

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor

basicConfig(
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = getLogger(__name__)
logger.setLevel(DEBUG)


class Objective:
    def __init__(self, mlp_objective):
        self.mlp_objective = mlp_objective
        self.mlp_max_iter = 530000
        self.mlp_n_layers = [1, 5]
        self.mlp_n_neurons = [10, 64]
        self.mlp_warm_start = [True, False]
        self.mlp_activation = ["identity", "logistic", "tanh", "relu"]
        self.mlp_learning_rate = ["constant", "invscaling", "adaptive"]
        self.test_size = 0.4
        self.seconds_history = []
        self.scores_history = []
        self.best_score = None
        self.best_model = None

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

        params["hidden_layer_sizes"] = layers
        params["max_iter"] = self.mlp_max_iter
        params["early_stopping"] = True
        params["warm_start"] = trial.suggest_categorical(
            "mlp_warm_start", self.mlp_warm_start
        )
        params["activation"] = trial.suggest_categorical(
            "mlp_activation", self.mlp_activation
        )
        params["learning_rate"] = trial.suggest_categorical(
            "mlp_learning_rate", self.mlp_learning_rate
        )
        return params

    def __call__(self, trial):
        x_train, x_test, y_train, y_test = train_test_split(
            self.mlp_objective.dataset_x,
            self.mlp_objective.dataset_y,
            test_size=self.mlp_objective.test_size,
        )
        params = self.generate_params(trial)

        estimators = []
        key = ""
        for index, model in self.mlp_objective.models.items():
            if model is not None:
                in_out = trial.suggest_int("model_" + str(index), 0, 1)
                key += str(in_out)
                if in_out == 1:
                    estimators.append(("model_" + str(index), model))

        if len(estimators) == 0:
            return 0 - 530000

        if self.mlp_objective.is_regressor:
            final_estimator = MLPRegressor(**params)
        else:
            final_estimator = MLPClassifier(**params)

        model = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
        )
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        if self.best_score is None or self.best_score < score:
            self.best_score = score
            self.best_model = model

        return score
