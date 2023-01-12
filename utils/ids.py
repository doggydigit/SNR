from typing import Literal


OptimizerID = Literal['SGD', 'Adam']
SGD: OptimizerID = 'SGD'
ADAM: OptimizerID = 'Adam'
OPTIMIZERS = [SGD, ADAM]

SimulationType = Literal[
    'mono', 'gonogo', 'gonogo_avabs', 'td_salience', 'td_salience_naive'
]
REGRESSION: SimulationType = 'mono'
GONOGO: SimulationType = 'gonogo'
GONOGO_AVABS: SimulationType = 'gonogo_avabs'
TD_SALIENCE: SimulationType = 'td_salience'
TD_SALIENCE_NAIVE: SimulationType = 'td_salience_naive'
LOSS_SIMULATIONS: list[SimulationType] = [REGRESSION, TD_SALIENCE, TD_SALIENCE_NAIVE]
PERF_SIMULATIONS: list[SimulationType] = [GONOGO, GONOGO_AVABS]
PRIMARY_SIMULATIONS: list[SimulationType] = [REGRESSION, GONOGO, GONOGO_AVABS, TD_SALIENCE_NAIVE]
SECONDARY_SIMULATIONS: list[SimulationType] = [TD_SALIENCE]
SIMULATION_TYPES: list[SimulationType] = [REGRESSION, GONOGO, GONOGO_AVABS, TD_SALIENCE, TD_SALIENCE_NAIVE]
