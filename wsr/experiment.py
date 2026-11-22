"""Current WSR-v0.1 entry point: frozen official Mamba2-130M and real text.

Prior synthetic experiments remain available explicitly as wsr.legacy_experiment.
"""
from .pretrained_experiment import CASES, main
from .legacy_experiment import loss_at  # Compatibility for archived diagnostic tests.

CONFIGS=CASES

if __name__=='__main__':main()
