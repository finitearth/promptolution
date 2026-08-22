"""Module for config-driven experiments through Hydra.

The entry point is :func:`promptolution.experiment.launch.launch`. It is deliberately not
re-exported here: the module is named ``launch`` as well, so binding the function to that name in
the package would shadow the module and ``import promptolution.experiment.launch`` would hand back
the function instead.
"""
