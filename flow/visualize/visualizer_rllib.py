"""Visualizer for rllib experiments.

Attributes
----------
EXAMPLE_USAGE : str
    Example call to the function, which is
    ::

        python ./visualizer_rllib.py /tmp/ray/result_dir 1

parser : ArgumentParser
    Command-line argument parser
"""

import argparse
from datetime import datetime
import gym
import numpy as np
import os
import sys
import time
from copy import deepcopy
from collections import defaultdict

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
from flow.utils.rllib import get_flow_params
from flow.utils.rllib import get_rllib_config
from flow.utils.rllib import get_rllib_pkl


EXAMPLE_USAGE = """
example usage:
    python ./visualizer_rllib.py /ray_results/experiment_dir/result_dir 1

Here the arguments are:
1 - the path to the simulation results
2 - the number of the checkpoint
"""


def visualizer_rllib(args):
    """Visualizer for RLlib experiments.

    This function takes args (see function create_parser below for
    more detailed information on what information can be fed to this
    visualizer), and renders the experiment associated with it.
    """
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    config = get_rllib_config(result_dir)

    # check if we have a multiagent environment but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policies', None):
        multiagent = True
        pkl = get_rllib_pkl(result_dir)
        config['multiagent'] = pkl['multiagent']
    else:
        multiagent = False

    # Run on only one cpu for rendering purposes
    config['num_workers'] = 0

    flow_params = get_flow_params(config)

    # hack for old pkl files
    # TODO(ev) remove eventually
    sim_params = flow_params['sim']
    setattr(sim_params, 'num_clients', 1)

    # Determine agent and checkpoint
    config_run = config['env_config']['run'] if 'run' in config['env_config'] \
        else None
    if args.run and config_run:
        if args.run != config_run:
            print('visualizer_rllib.py: error: run argument '
                  + '\'{}\' passed in '.format(args.run)
                  + 'differs from the one stored in params.json '
                  + '\'{}\''.format(config_run))
            sys.exit(1)
    if args.run:
        agent_cls = get_agent_class(args.run)
    elif config_run:
        agent_cls = get_agent_class(config_run)
    else:
        print('visualizer_rllib.py: error: could not find flow parameter '
              '\'run\' in params.json, '
              'add argument --run to provide the algorithm or model used '
              'to train the results\n e.g. '
              'python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO')
        sys.exit(1)

    sim_params.restart_instance = True

    # specify emission file path
    dir_path = os.path.dirname(os.path.realpath(__file__))

    emission_path = '{0}/test_time_rollout/'.format(dir_path)
    sim_params.emission_path = emission_path if args.gen_emission else None

    # pick your rendering mode
    if args.render_mode == 'sumo_web3d':
        sim_params.num_clients = 2
        sim_params.render = False
    elif args.render_mode == 'drgb':
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
    elif args.render_mode == 'sumo_gui':
        sim_params.render = False # this will be set to true after creating agent and gym
        print('NOTE: With render mode {}, an extra instance of the SUMO GUI '
              'will display before the GUI for visualizing the result. Click '
              'the green Play arrow to continue.'.format(args.render_mode))
    elif args.render_mode == 'no_render':
        sim_params.render = False
    if args.save_render:
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
        sim_params.save_render = True

    # Create and register a gym+rllib env
    create_env, env_name = make_create_env(params=flow_params, version=0)
    register_env(env_name, create_env)

    # Start the environment with the gui turned on and a path for the
    # emission file
    env_params = flow_params['env']
    env_params.restart_instance = False
    if args.evaluate:
        env_params.evaluate = True

    # lower the horizon if testing
    if args.horizon:
        config['horizon'] = args.horizon
        env_params.horizon = args.horizon

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    agent.restore(checkpoint)

    if hasattr(agent, "local_evaluator") and \
            os.environ.get("TEST_FLAG") != 'True':
        env = agent.local_evaluator.env
    else:
        env = gym.make(env_name)

    if args.render_mode == 'sumo_gui':
        env.sim_params.render = True # set to true after initializing agent and env

    # if restart_instance, don't restart here because env.reset will restart later
    if not sim_params.restart_instance:
        env.restart_simulation(sim_params=sim_params)

    use_lstm = config['model'].get('use_lstm', False)
    if use_lstm:
        state_size = config['model']['lstm_cell_size']
        lstm_state = [np.zeros(state_size), np.zeros(state_size)]
        if multiagent:
            lstm_state = {key: deepcopy(lstm_state) for key in config['multiagent']['policies'].keys()}

    rewards = []
    if multiagent:
        rewards = defaultdict(list)
        policy_map_fn = config['multiagent']['policy_mapping_fn'].func

    # collect travel time of Jordan and EV
    

    # Simulate and collect metrics
    final_outflows = []
    final_inflows = []
    mean_speed = []
    std_speed = []
    for i in range(args.num_rollouts):
        obs = env.reset()
        kv = env.k.vehicle
        rollout_speeds = []
        rollout_reward = 0
        if multiagent:
            rollout_reward = defaultdict(int)
        for _ in range(env_params.horizon):
            rollout_speeds.append(np.mean(kv.get_speed(kv.get_ids())))
            if multiagent:
                action = {}
                for agent_id in obs.keys():
                    if use_lstm:
                        action[agent_id], obs[agent_id], logits = agent.compute_action(
                            obs[agent_id],
                            obs=lstm_state[agent_id],
                            policy_id=policy_map_fn(agent_id)
                        )
                    else:
                        action[agent_id] = agent.compute_action(
                            obs[agent_id],
                            policy_id=policy_map_fn(agent_id)
                        )
            else:
                action = agent.compute_action(obs)
            obs, reward, done, _ = env.step(action)
            if multiagent:
                done = done['__all__']
                for agent_id, agent_reward in reward.items():
                    rollout_reward[policy_map_fn(agent_id)] += agent_reward
            else:
                rollout_reward += reward
            
            if done:
                break

        if multiagent:
            for agent_id, reward in rollout_reward.items():
                rewards[agent_id].append(reward)
                print('rollout %s, agent %s reward: %.5g' % (i, agent_id, reward))
        else:
            rewards.append(rollout_reward)
            print('rollout %s, reward: %.5g' % (i, rollout_reward))
        mean_speed.append(np.nanmean(rollout_speeds))
        std_speed.append(np.nanstd(rollout_speeds))
        # Compute rate of inflow / outflow in the last 500 steps
        final_outflows.append(kv.get_outflow_rate(500))
        final_inflows.append(kv.get_inflow_rate(500))

    print('\n==== Summary of results: mean (std) [rollout1, rollout2, ...] ====')
    mean, std = np.mean, np.std
    if multiagent:
        for agent_id, agent_rewards in rewards.items():
            print('agent %s rewards: %.4g (%.4g) %s' % (
                agent_id, mean(agent_rewards), std(agent_rewards), agent_rewards))
    else:
        print('rewards: %.4g (%.4g) %s' % (
                mean(rewards), std(rewards), rewards))

    print('mean speeds (m/s): %.4g (%.4g) %s' % (
        mean(mean_speed), std(mean_speed), mean_speed))
    print('std speeds: %.4g (%.4g) %s' % (
        mean(std_speed), std(std_speed), std_speed))

    print('inflows (veh/hr): %.4g (%.4g) %s' % (
        mean(final_inflows), std(final_inflows), final_inflows))
    print('outflows (veh/hr): %.4g (%.4g) %s' % (
        mean(final_outflows), std(final_outflows), final_outflows))

    # Compute throughput efficiency in the last 500 sec of the
    throughput = [o / i for o, i in zip(final_outflows, final_inflows)]
    print('throughput efficiency: %.4g (%.4g) %s' % (
        mean(throughput), std(throughput), throughput))

    # terminate the environment
    env.terminate()

    # if prompted, convert the emission file into a csv file
    if args.gen_emission:
        time.sleep(0.1)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        emission_filename = '{0}-emission.xml'.format(env.network.name)

        emission_path = \
            '{0}/test_time_rollout/{1}'.format(dir_path, emission_filename)

        # convert the emission file into a csv file
        emission_to_csv(emission_path)

        # print the location of the emission csv file
        emission_path_csv = emission_path[:-4] + ".csv"
        print("\nGenerated emission file at " + emission_path_csv)

        # delete the .xml version of the emission file
        os.remove(emission_path)

    # if we wanted to save the render, here we create the movie
    if args.save_render:
        dirs = os.listdir(os.path.expanduser('~')+'/flow_rendering')
        # Ignore hidden files
        dirs = [d for d in dirs if d[0] != '.']
        dirs.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d-%H%M%S"))
        recent_dir = dirs[-1]
        # create the movie
        movie_dir = os.path.expanduser('~') + '/flow_rendering/' + recent_dir
        save_dir = os.path.expanduser('~') + '/flow_movies'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        os_cmd = "cd " + movie_dir + " && ffmpeg -i frame_%06d.png"
        os_cmd += " -pix_fmt yuv420p " + dirs[-1] + ".mp4"
        os_cmd += "&& cp " + dirs[-1] + ".mp4 " + save_dir + "/"
        os.system(os_cmd)


def create_parser():
    """Create the parser to capture CLI arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='[Flow] Evaluates a reinforcement learning agent '
                    'given a checkpoint.',
        epilog=EXAMPLE_USAGE)

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional input parameters
    parser.add_argument(
        '--run',
        type=str,
        help='The algorithm or model to train. This may refer to '
             'the name of a built-on algorithm (e.g. RLLib\'s DQN '
             'or PPO), or a user-defined trainable function or '
             'class registered in the tune registry. '
             'Required for results trained with flow-0.2.0 and before.')
    parser.add_argument(
        '--num_rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--gen_emission',
        action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Specifies whether to use the \'evaluate\' reward '
             'for the environment.')
    parser.add_argument(
        '--render_mode',
        type=str,
        default='sumo_gui',
        help='Pick the render mode. Options include sumo_web3d, '
             'rgbd and sumo_gui')
    parser.add_argument(
        '--save_render',
        action='store_true',
        help='Saves a rendered video to a file. NOTE: Overrides render_mode '
             'with pyglet rendering.')
    parser.add_argument(
        '--horizon',
        type=int,
        help='Specifies the horizon.')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=1, local_mode=True)
    visualizer_rllib(args)
