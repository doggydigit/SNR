from typing import List, Literal

SNKey = Literal['n_noise', 'n_signal', 'noise_std', 'signal_amplitude', 'n_noise_td', 'n_signal_td']
NUMBER_NOISE: SNKey = 'n_noise'
NUMBER_SIGNAL: SNKey = 'n_signal'
NOISE_STD: SNKey = 'noise_std'
SIGNAL_AMPLITUDE: SNKey = 'signal_amplitude'
NUMBER_NOISE_TD: SNKey = 'n_noise_td'
NUMBER_SIGNAL_TD: SNKey = 'n_signal_td'

NOISE_KEYS: List[SNKey] = [NUMBER_NOISE, NUMBER_SIGNAL, NUMBER_NOISE_TD, NUMBER_SIGNAL_TD]

TrainKey = Literal['lr', 'optimizer', 'lr_td']
LEARNING_RATE: TrainKey = 'lr'
OPTIMIZER: TrainKey = 'optimizer'
LEARNING_RATE_TD: TrainKey = 'lr_td'

SimulationKey = Literal['simulation', 'seed']
SIMULATION: SimulationKey = 'simulation'
SEED: SimulationKey = 'seed'

OutcomeKey = Literal['loss', 'performance']
LOSS: OutcomeKey = 'loss'
PERFORMANCE: OutcomeKey = 'performance'

Key = SNKey | TrainKey | SimulationKey | OutcomeKey
