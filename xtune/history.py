# -*- coding:utf-8 -*-
# Author: xiansong@nj.iscas.ac.cn

import json
import numpy as np
from .util import SUCCEED


class Observation(object):
    def __init__(self, configuration, performance=None, runtime=None, status=SUCCEED):
        self.status = status
        self.configuration = configuration
        self.performance = performance
        self.runtime = runtime

    def __repr__(self):
        info = {
            "configuration": self.configuration.get_dictionary(),
            "performance": self.performance,
            "runtime": self.runtime if self.runtime else None,
            "status": self.status,
        }

        return "\n" + json.dumps(info, indent=4)


class History(object):
    def __init__(self, configspace):
        self.configspace = configspace
        self._configurations = []
        self._performances = []
        self._runtimes = []
        self._trial_states = []
        self._failed_indexs = []
        self._best_perf = np.inf
        self._optimized_record = []

    def __len__(self):
        return len(self.configurations)

    def update_observation(self, observated):
        self._performances.append(observated.performance)
        self._configurations.append(observated.configuration)
        self._trial_states.append(observated.status)
        self._runtimes.append(observated.runtime)
        if observated.status == SUCCEED:
            self._update_observation(observated)
        else:
            self._failed_indexs.append(len(self.performances) - 1)

    def _update_observation(self, observated):
        performance = observated.performance
        configuration = observated.configuration
        runtime = observated.runtime
        if len(self._optimized_record) > 0:
            if performance < self._best_perf:
                self._best_perf = performance
                self._optimized_record = [(configuration, performance, runtime)]
            elif performance == self._best_perf:
                self._optimized_record.append((configuration, performance, runtime))
        else:
            self._best_perf = performance
            self._optimized_record.append((configuration, performance, runtime))

    def in_history(self, configuration):
        if configuration in self._configurations:
            return True
        return False

    @property
    def performances(self):
        return self._performances

    @property
    def configurations(self):
        return self._configurations

    @property
    def runtimes(self):
        return self._runtimes

    @property
    def optimized_result(self):
        return self._optimized_record

    @property
    def best_perf(self):
        return self._best_perf
