"""Environment for training the acceleration behavior of vehicles in a ring."""

from flow.core import rewards
from flow.envs.base import Env
from gym.spaces.box import Box
from flow.controllers.car_following_models import SimCarFollowingController
from flow.controllers import JordanControllerMulti
from flow.core.params import InFlows, NetParams, SumoCarFollowingParams, TrafficLightParams, VehicleParams
from random import randint
from flow.envs.base import Env
import collections
import logging
import os

import numpy as np

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    'max_accel': 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    'max_decel': 3,
    # desired velocity for all vehicles in the network, in m/s
    'target_velocity': 10,
    # specifies whether vehicles are to be sorted by position during a
    # simulation step. If set to True, the environment parameter
    # self.sorted_ids will return a list of all vehicles sorted in accordance
    # with the environment
    'sort_vehicles': False
}


class AccelEnv(Env):
    """Fully observed acceleration environment.

    This environment used to train autonomous vehicles to improve traffic flows
    when acceleration actions are permitted by the rl agent.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s
    * sort_vehicles: specifies whether vehicles are to be sorted by position
      during a simulation step. If set to True, the environment parameter
      self.sorted_ids will return a list of all vehicles sorted in accordance
      with the environment

    States
        The state consists of the velocities and absolute position of all
        vehicles in the network. This assumes a constant number of vehicles.

    Actions
        Actions are a list of acceleration for each rl vehicles, bounded by the
        maximum accelerations and decelerations specified in EnvParams.

    Rewards
        The reward function is the two-norm of the distance of the speed of the
        vehicles in the network from the "target_velocity" term. For a
        description of the reward, see: flow.core.rewards.desired_speed

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.

    Attributes
    ----------
    prev_pos : dict
        dictionary keeping track of each veh_id's previous position
    absolute_position : dict
        dictionary keeping track of each veh_id's absolute position
    obs_var_labels : list of str
        referenced in the visualizer. Tells the visualizer which
        metrics to track
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))

        # variables used to sort vehicles by their initial position plus
        # distance traveled
        self.prev_pos = dict()
        self.absolute_position = dict()

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(self.initial_vehicles.num_rl_vehicles, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ['Velocity', 'Absolute_pos']
        return Box(
            low=0,
            high=1,
            shape=(2 * self.initial_vehicles.num_vehicles, ),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        sorted_rl_ids = [
            veh_id for veh_id in self.sorted_ids
            if veh_id in self.k.vehicle.get_rl_ids()
        ]
        self.k.vehicle.apply_acceleration(sorted_rl_ids, rl_actions)

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        else:
            return rewards.desired_velocity(self, fail=kwargs['fail'])

    def get_state(self):
        """See class definition."""
        speed = [self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed()
                 for veh_id in self.sorted_ids]
        pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.network.length()
               for veh_id in self.sorted_ids]

        return np.array(speed + pos)

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes, and
        update the sorting of vehicles using the self.sorted_ids variable.
        """
        # specify observed vehicles
        if self.k.vehicle.num_rl_vehicles > 0:
            for veh_id in self.k.vehicle.get_human_ids():
                self.k.vehicle.set_observed(veh_id)

        # update the "absolute_position" variable
        for veh_id in self.k.vehicle.get_ids():
            this_pos = self.k.vehicle.get_x_by_id(veh_id)

            if this_pos == -1001:
                # in case the vehicle isn't in the network
                self.absolute_position[veh_id] = -1001
            else:
                change = this_pos - self.prev_pos.get(veh_id, this_pos)
                self.absolute_position[veh_id] = \
                    (self.absolute_position.get(veh_id, this_pos) + change) \
                    % self.k.network.length()
                self.prev_pos[veh_id] = this_pos

    @property
    def sorted_ids(self):
        """Sort the vehicle ids of vehicles in the network by position.

        This environment does this by sorting vehicles by their absolute
        position, defined as their initial position plus distance traveled.

        Returns
        -------
        list of str
            a list of all vehicle IDs sorted by position
        """
        if self.env_params.additional_params['sort_vehicles']:
            return sorted(self.k.vehicle.get_ids(), key=self._get_abs_position)
        else:
            return self.k.vehicle.get_ids()

    def _get_abs_position(self, veh_id):
        """Return the absolute position of a vehicle."""
        return self.absolute_position.get(veh_id, -1001)

    def reset(self):
        """See parent class.

        This also includes updating the initial absolute position and previous
        position.
        """
        FLOW_RATE = 1000
        V_MAX_EV = 35
        V_MAX_CARS = 15
        V_ENTER = 10
        # 175m length edges ensures V2X communication is achievable.
        INNER_LENGTH = 175 #300
        LONG_LENGTH = 175 #100
        SHORT_LENGTH = 175 #300
        N_ROWS = 2
        N_COLUMNS = 1
        NUM_CARS_LEFT = 1
        NUM_CARS_RIGHT = 1
        NUM_CARS_TOP = 1
        NUM_CARS_BOT = 1
        for _ in range(100):
            try:
                # introduce new inflows within the pre-defined inflow range
                inflow = InFlows()
                edge_humans_enter = 'right0_0'

                # Adding human driven vehicles.
                inflow.add(
                    veh_type='idm',
                    edge=edge_humans_enter,
                    vehs_per_hour=FLOW_RATE,
                    depart_lane='free',
                    depart_speed=V_ENTER)

                edge_EV_enter = 'right0_0'
                edge_RL_enter = 'right0_0'

                
                # randomize the EV and RL arrivals  
                RL_entrance_time = randint(30,150)
                EV_entrance_time = RL_entrance_time + randint(1,30)
                

                #print(RL_entrance_time)
                #print(EV_entrance_time)

                inflow.add(
                    veh_type='jordan',
                    edge=edge_RL_enter,
                    vehs_per_hour=FLOW_RATE,
                    depart_lane= 1, 
                    depart_speed=V_ENTER,
                    begin=RL_entrance_time,
                    number = 1,
                    name = 'jordan')
    
                inflow.add(
                    veh_type='emergency',
                    edge=edge_EV_enter,
                    vehs_per_hour=FLOW_RATE,
                    depart_lane= 0,
                    depart_speed=V_MAX_EV,
                    begin=EV_entrance_time,
                    number = 1,
                    name = 'emergency')
                

                grid_array = {
                    "short_length": SHORT_LENGTH,
                    "inner_length": INNER_LENGTH,
                    "long_length": LONG_LENGTH,
                    "row_num": N_ROWS,
                    "col_num": N_COLUMNS,
                    "cars_left": NUM_CARS_LEFT,
                    "cars_right": NUM_CARS_RIGHT,
                    "cars_top": NUM_CARS_TOP,
                    "cars_bot": NUM_CARS_BOT
                }

                additional_net_params = {
                    'speed_limit': V_MAX_EV,
                    'grid_array': grid_array,
                    'horizontal_lanes': 2,
                    'vertical_lanes': 2,
                    'traffic_lights': True,
                    "random_start": False
                }

                net_params = NetParams(
                        inflows=inflow,
                        additional_params=additional_net_params)

                vehicles = VehicleParams()
                vehicles.add(
                    veh_id='idm',
                    acceleration_controller=(SimCarFollowingController, {}),
                    car_following_params=SumoCarFollowingParams(
                    min_gap=2.5,
                    accel=3.0,
                    decel=7.5,  # avoid collisions at emergency stops
                    max_speed=V_MAX_CARS,
                    speed_mode="all_checks",
                    ),
                    num_vehicles=0)

                vehicles.add(
                    veh_id="jordan",
                    acceleration_controller=(JordanControllerMulti, {}),
                    car_following_params=SumoCarFollowingParams(
                    min_gap=2.5,
                    accel=3.0,
                    decel=7.5,  # avoid collisions at emergency stops
                    max_speed=V_MAX_CARS,
                    speed_mode="all_checks",
                    ),
                num_vehicles=0)

                vehicles.add(
                    veh_id='emergency',
                    acceleration_controller=(SimCarFollowingController, {}),
                    car_following_params=SumoCarFollowingParams(
                        min_gap=1.0,
                        accel=5.0,
                        decel=7.5,  # avoid collisions at emergency stops
                        max_speed=V_MAX_EV,
                    ),
                    num_vehicles=0)

                #  Add traffic lights
                tl_logic = TrafficLightParams(baseline=False)
                phases = [{
                    "duration": "31",
                    "minDur": "8",
                    "maxDur": "45",
                    "state": "GGrrGGrr"
                }, {
                    "duration": "6",
                    "minDur": "3",
                    "maxDur": "6",
                    "state":"yyrryyrr"
                }, {
                    "duration": "31",
                    "minDur": "8",
                    "maxDur": "45",
                    "state":"rrGGrrGG"
                }, {
                    "duration": "6",
                    "minDur": "3",
                    "maxDur": "6",
                    "state":"rryyrryy"
                }]
                tl_logic.add("center0", phases=phases, programID=1)
                tl_logic.add("center1", phases=phases, programID=1)
                # recreate the network object
                self.network = self.network.__class__(
                        name=self.network.orig_name,
                        vehicles=vehicles,
                        net_params=net_params,
                        initial_config=self.initial_config,
                        traffic_lights=tl_logic)

            except Exception as e:
                print('error on reset ', e)

        obs = super().reset()

        #for veh_id in self.k.vehicle.get_ids():
        #    self.absolute_position[veh_id] = self.k.vehicle.get_x_by_id(veh_id)
        #    self.prev_pos[veh_id] = self.k.vehicle.get_x_by_id(veh_id)

        return obs
