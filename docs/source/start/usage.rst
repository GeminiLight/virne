
Basic Usage
===========


Minimal Example
---------------


.. code:: python

    # get config
    config = get_config()

    # generate p_net and v_net dataset
    Generator.generate_dataset(config)

    # create scenario with Env and Solver
    scenario = create_scenario(config)

    # use Solver in Env
    scenario.run()
