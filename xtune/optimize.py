# -*- coding:utf-8 -*-
# Author: xiansong@nj.iscas.ac.cn

import numpy as np
from sklearn.utils import check_random_state

from .forest import RandomForestRegressor
from .acquisition import Acquisition
from .history import History
from .sampling import RandomSearch
from .util import convert_configurations_to_array


class KernelTune:
    def __init__(self, configspace, n_initial_points=10, random_state=None):
        self.configspace = configspace
        self.random_state = check_random_state(random_state)
        self.configspace.seed(self.random_state.randint(100000000))
        self.space_size = int(self.configspace.estimate_size())

        self.history = History(self.configspace)
        self.initial_configurations = self.configspace.sample_configuration(n_initial_points)
        self.n_initial_points = len(self.initial_configurations)

        self.model = None
        self.time_model = None
        self.acquisition = None
        self.optimizer = None
        self.run_status = 0
        self.build_optimizer()

    def build_optimizer(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        self.time_model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        self.acquisition = Acquisition(self.model, self.time_model, xi=0.001,  best_perf=0.0)
        self.optimizer = RandomSearch(configspace=self.configspace, acq_func=self.acquisition,
                                      random_state=self.random_state, n_sample=min(self.space_size, 30000))

    def update_observation(self, observation):
        if not self.history.in_history(observation.configuration):
            return self.history.update_observation(observation)

    def get_configuration(self):
        n_evaluated = len(self.history.configurations)
        if n_evaluated == 0:
            return self.configspace.get_default_configuration()

        if n_evaluated < self.n_initial_points:
            return self.initial_configurations[n_evaluated]

        self.train(self.history)
        self.acquisition.update(model=self.model,
                                time_model=self.time_model,
                                best_perf=self.history.best_perf,
                                n_evaluated=n_evaluated)

        configurations = self.optimizer.consort()
        for configuration in configurations:
            if configuration not in self.history.configurations:
                return configuration

        return self.configspace.sample_configuration()

    def train(self, history):
        configurations = convert_configurations_to_array(history.configurations)
        performances = np.array(history.performances)
        self.model.fit(configurations, performances)
        elapsedtimes = np.array(history.runtimes)
        self.time_model.fit(configurations, elapsedtimes)

    def optimized_result(self):
        result = list(self.history.optimized_result[0])
        result[0] = result[0].get_dictionary()
        return result