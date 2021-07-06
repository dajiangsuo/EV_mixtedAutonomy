"""Evaluates the baseline performance of figureeight without RL control.

Baseline is human acceleration and intersection behavior.
"""

import numpy as np
from flow.core.experiment import Experiment
from flow.core.params import InitialConfig
from flow.core.params import SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
from flow.controllers import IDMController
from flow.controllers import ContinuousRouter
from flow.benchmarks.figureeight0 import flow_params


def figure_eight_baseline(num_runs, render=True):
    """Run script for all figure eight baselines.

    Parameters
    ----------
        num_runs : int
            number of rollouts the performance of the environment is evaluated
            over
        render : bool, optional
            specifies whether to use the gui during execution

    Returns
    -------
        Experiment
            class needed to run simulations
    """
    exp_tag = flow_params['exp_tag']
    sim_params = flow_params['sim']
    env_params = flow_params['env']
    net_params = flow_params['net']
    initial_config = flow_params.get('initial', InitialConfig())
    traffic_lights = flow_params.get('tls', TrafficLightParams())

    # modify the rendering to match what is requested
    sim_params.render = render

    # set the evaluation flag to True
    env_params.evaluate = True

    # we want no autonomous vehicles in the simulation
    vehicles = VehicleParams()
    vehicles.add(veh_id='human',
                 acceleration_controller=(IDMController, {'noise': 0.2}),
                 routing_controller=(ContinuousRouter, {}),
                 car_following_params=SumoCarFollowingParams(
                     speed_mode='obey_safe_speed',
                 ),
                 num_vehicles=14)

    # import the network class
    network_class = flow_params['network']

    # create the network object
    network = network_class(
        name=exp_tag,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights
    )

    # import the environment class
    env_class = flow_params['env_name']

    # create the environment object
    env = env_class(env_params, sim_params, network)

    exp = Experiment(env)

    results = exp.run(num_runs, env_params.horizon)
    avg_speed = np.mean(results['mean_returns'])

    return avg_speed


if __name__ == '__main__':
    runs = 2  # number of simulations to average over
    res = figure_eight_baseline(num_runs=runs)

    print('---------')
    print('The average speed across {} runs is {}'.format(runs, res))
