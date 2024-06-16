# -*- coding: utf-8 -*-
import random
from xtune.history import Observation
from xtune.optimize import KernelTune
from benchmark import objective, space
from xtune.util import FAILED, SUCCEED, MAXPERF, MAXTIME


def demo():
    cs = space()
    max_runs = int(cs.estimate_size())
    random_state = random.randint(100000, 10000000)
    print(f"MaxRun: {max_runs}")
    print(f"RandomState: {random_state}")
    advisor = KernelTune(configspace=cs, n_initial_points=4, random_state=random_state)
    for i in range(max_runs):
        config = advisor.get_configuration()
        try:
            performance, runtime = objective(config)
            observation = Observation(configuration=config,
                                      performance=performance,
                                      runtime=runtime,
                                      status=SUCCEED)
        except Exception as e:
            observation = Observation(configuration=config, performance=MAXPERF, runtime=MAXTIME, status=FAILED)

        advisor.update_observation(observation)
        print(f"Observation {i+1}: {observation}")
        result = advisor.optimized_result()
        print(f"OptimizedResult: {result}")


if __name__ == "__main__":
    demo()






