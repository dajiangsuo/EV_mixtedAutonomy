"""Environments for networks with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m grid.
"""

from copy import deepcopy
import numpy as np
import re

from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces import Tuple

from flow.core import rewards
from flow.envs.base import Env

from flow.controllers.rlcontroller import RLController
from flow.controllers import SimCarFollowingController, GridRouter
from flow.core.params import SumoCarFollowingParams, VehicleParams, InFlows

ADDITIONAL_ENV_PARAMS = {
    # minimum switch time for each traffic light (in seconds)
    "switch_time": 2.0,
    # whether the traffic lights should be actuated by sumo or RL
    # options are "controlled" and "actuated"
    "tl_type": "controlled",
    # determines whether the action space is meant to be discrete or continuous
    "discrete": False,
    # whether the inflow should be reset on each rollout
    "reset_inflow": False,
    # base inflow per edge (veh/hr)
    "inflow_base": 300,
    # the range of edge inflows is the base inflow +/- the inflow_delta
    "inflow_delta": 200,
    # fraction of inflows which are AVs
    "fraction_av": 0.1,
    # speed at which vehicles enter the network
    "speed_enter": 30,
}

ADDITIONAL_PO_ENV_PARAMS = {
    # num of vehicles the agent can observe on each incoming edge
    "num_observed": 2,
    # velocity to use in reward functions
    "target_velocity": 30,
}


class TrafficLightGridEnv(Env):
    """Environment used to train traffic lights.

    Required from env_params:
    * switch_time: minimum time a light must be constant before
      it switches (in seconds).
      Earlier RL commands are ignored.
    * tl_type: whether the traffic lights should be actuated by sumo or RL,
      options are respectively "actuated" and "controlled"
    * discrete: determines whether the action space is meant to be discrete or
      continuous
    States
        An observation is the distance of each vehicle to its intersection, a
        number uniquely identifying which edge the vehicle is on, and the speed
        of the vehicle.
    Actions
        The action space consist of a list of float variables ranging from 0-1
        specifying whether a traffic light is supposed to switch or not. The
        actions are sent to the traffic light in the grid from left to right
        and then top to bottom.
    Rewards
        The reward is the negative per vehicle delay minus a penalty for
        switching traffic lights
    Termination
        A rollout is terminated once the time horizon is reached.
    Additional
        Vehicles are rerouted to the start of their original routes once they
        reach the end of the network in order to ensure a constant number of
        vehicles.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):

        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        self.flow_rate = env_params.additional_params["inflow_base"]
        self.num_lanes = network.net_params.additional_params[
            "horizontal_lanes"]
        self.grid_array = network.net_params.additional_params["grid_array"]
        self.rows = self.grid_array["row_num"]
        self.cols = self.grid_array["col_num"]
        # self.num_observed = self.grid_array.get("num_observed", 3)
        self.num_traffic_lights = self.rows * self.cols
        self.tl_type = env_params.additional_params.get('tl_type')

        super().__init__(env_params, sim_params, network, simulator)

        # Saving env variables for plotting
        self.steps = env_params.horizon
        self.obs_var_labels = {
            'edges': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'velocities': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'positions': np.zeros((self.steps, self.k.vehicle.num_vehicles))
        }

        # Keeps track of the last time the traffic lights in an intersection
        # were allowed to change (the last time the lights were allowed to
        # change from a red-green state to a red-yellow state.)
        self.last_change = np.zeros((self.rows * self.cols, 1))
        # Keeps track of the direction of the intersection (the direction that
        # is currently being allowed to flow. 0 indicates flow from top to
        # bottom, and 1 indicates flow from left to right.)
        self.direction = np.zeros((self.rows * self.cols, 1))
        # Value of 1 indicates that the intersection is in a red-yellow state.
        # value 0 indicates that the intersection is in a red-green state.
        self.currently_yellow = np.zeros((self.rows * self.cols, 1))

        # when this hits min_switch_time we change from yellow to red
        # the second column indicates the direction that is currently being
        # allowed to flow. 0 is flowing top to bottom, 1 is left to right
        # For third column, 0 signifies yellow and 1 green or red
        self.min_switch_time = env_params.additional_params["switch_time"]

        if self.tl_type != "actuated":
            for i in range(self.rows * self.cols):
                self.k.traffic_light.set_state(
                    node_id='center' + str(i), state="GrGr")
                self.currently_yellow[i] = 0

        # # Additional Information for Plotting
        # self.edge_mapping = {"top": [], "bot": [], "right": [], "left": []}
        # for i, veh_id in enumerate(self.k.vehicle.get_ids()):
        #     edge = self.k.vehicle.get_edge(veh_id)
        #     for key in self.edge_mapping:
        #         if key in edge:
        #             self.edge_mapping[key].append(i)
        #             break

        # check whether the action space is meant to be discrete or continuous
        self.discrete = env_params.additional_params.get("discrete", False)

    @property
    def action_space(self):
        """See class definition."""
        if self.discrete:
            return Discrete(2 ** self.num_traffic_lights)
        else:
            return Box(
                low=-1,
                high=1,
                shape=(self.num_traffic_lights,),
                dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        speed = Box(
            low=0,
            high=1,
            shape=(self.initial_vehicles.num_vehicles,),
            dtype=np.float32)
        dist_to_intersec = Box(
            low=0.,
            high=np.inf,
            shape=(self.initial_vehicles.num_vehicles,),
            dtype=np.float32)
        edge_num = Box(
            low=0.,
            high=1,
            shape=(self.initial_vehicles.num_vehicles,),
            dtype=np.float32)
        traffic_lights = Box(
            low=0.,
            high=1,
            shape=(3 * self.rows * self.cols,),
            dtype=np.float32)
        return Tuple((speed, dist_to_intersec, edge_num, traffic_lights))

    def get_state(self):
        """See class definition."""
        # compute the normalizers
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"],
                       grid_array["long_length"],
                       grid_array["inner_length"])

        # get the state arrays
        speeds = [
            self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed()
            for veh_id in self.k.vehicle.get_ids()
        ]
        dist_to_intersec = [
            self.get_distance_to_intersection(veh_id) / max_dist
            for veh_id in self.k.vehicle.get_ids()
        ]
        edges = [
            self._convert_edge(self.k.vehicle.get_edge(veh_id)) /
            (self.k.network.network.num_edges - 1)
            for veh_id in self.k.vehicle.get_ids()
        ]

        state = [
            speeds, dist_to_intersec, edges,
            self.last_change.flatten().tolist(),
            self.direction.flatten().tolist(),
            self.currently_yellow.flatten().tolist()
        ]
        return np.array(state)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # check if the action space is discrete
        if self.discrete:
            # convert single value to list of 0's and 1's
            rl_mask = [int(x) for x in list('{0:0b}'.format(rl_actions))]
            rl_mask = [0] * (self.num_traffic_lights - len(rl_mask)) + rl_mask
        else:
            # convert values less than 0.0 to zero and above to 1. 0's indicate
            # that we should not switch the direction
            rl_mask = rl_actions > 0.0

        for i, action in enumerate(rl_mask):
            if self.currently_yellow[i] == 1:  # currently yellow
                self.last_change[i] += self.sim_step
                # Check if our timer has exceeded the yellow phase, meaning it
                # should switch to red
                if self.last_change[i] >= self.min_switch_time:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i),
                            state="GrGr")
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i),
                            state='rGrG')
                    self.currently_yellow[i] = 0
            else:
                if action:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i),
                            state='yryr')
                    else:
                        self.k.traffic_light.set_state(
                            node_id='center{}'.format(i),
                            state='ryry')
                    self.last_change[i] = 0.0
                    self.direction[i] = not self.direction[i]
                    self.currently_yellow[i] = 1

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        return - rewards.min_delay_unscaled(self) \
            - rewards.boolean_action_penalty(rl_actions >= 0.5, gain=1.0)

    def reset(self):
        """Reset the environment with a new inflow rate.
        The diverse set of inflows are used to generate a policy that is more
        robust with respect to the inflow rate. The inflow rate is update by
        creating a new network similar to the previous one, but with a new
        Inflow object with a rate within the additional environment parameter
        "inflow_delta", which offsets the inflow_base, with +/- inflow_delta.
        **WARNING**: The inflows assume there are vehicles of type
        "followerstopper" and "human" within the VehicleParams object.
        """
        add_params = self.env_params.additional_params
        if add_params.get("reset_inflow"):
            inflow_base = add_params.get("inflow_base")
            inflow_delta = add_params.get("inflow_delta")
            self.flow_rate = np.random.uniform(inflow_base - inflow_delta,
                                               inflow_base + inflow_delta)

            fraction_av = add_params.get("fraction_av")
            speed_enter = add_params.get("speed_enter")

            # FIXME(cathywu) Code is repeated in several places. Consolidate.
            # inflows of vehicles are place on all outer edges (listed here)
            outer_edges = []
            outer_edges += ["left{}_{}".format(self.rows, i) for i in range(
                self.cols)]
            outer_edges += ["bot{}_0".format(i) for i in range(self.rows)]

            # We try this for 100 trials in case unexpected errors during
            # instantiation.
            for _ in range(100):
                try:
                    # introduce new inflows within the pre-defined inflow range
                    inflow = InFlows()
                    for edge in outer_edges:
                        inflow.add(
                            veh_type="human",
                            edge=edge,
                            vehs_per_hour=self.flow_rate * (1-fraction_av),
                            depart_lane="free",
                            depart_speed=speed_enter)
                        inflow.add(
                            veh_type="followerstopper",
                            edge=edge,
                            vehs_per_hour=self.flow_rate * fraction_av,
                            depart_lane="free",
                            depart_speed=speed_enter)

                    # all other network parameters should match the previous
                    # environment (we only want to change the inflow)
                    net_params = deepcopy(self.net_params)
                    net_params.inflows = inflow

                    # we place a sufficient number of vehicles to ensure they
                    # confirm with the total number specified above. We also
                    # use a "right_of_way" speed mode to support traffic
                    # light compliance
                    vehicles = VehicleParams()
                    vehicles.add(
                        veh_id="human",
                        acceleration_controller=(SimCarFollowingController, {}),
                        car_following_params=SumoCarFollowingParams(
                            min_gap=2.5,
                            max_speed=speed_enter,
                            decel=7.5,  # avoid collisions at emergency stops
                            speed_mode="right_of_way",
                        ),
                        routing_controller=(GridRouter, {}))
                    vehicles.add(
                        veh_id="followerstopper",
                        acceleration_controller=(RLController, {}),
                        car_following_params=SumoCarFollowingParams(
                            speed_mode=9,
                        ),
                        routing_controller=(GridRouter, {})
                    )

                    # recreate the network object
                    self.network = self.network.__class__(
                        name=self.network.orig_name,
                        vehicles=vehicles,
                        net_params=net_params,
                        initial_config=self.initial_config)

                    observation = super().reset()

                    # reset the timer to zero
                    self.time_counter = 0

                    return observation

                except Exception as e:
                    print('error on reset ', e)

        # perform the generic reset function
        observation = super().reset()

        # reset the timer to zero
        self.time_counter = 0

        return observation

    # ===============================
    # ============ UTILS ============
    # ===============================

    def get_distance_to_intersection(self, veh_ids):
        """Determine the distance from a vehicle to its next intersection.
        Parameters
        ----------
        veh_ids : str or str list
            vehicle(s) identifier(s)
        Returns
        -------
        float (or float list)
            distance to closest intersection
        """
        if isinstance(veh_ids, list):
            return [self.get_distance_to_intersection(veh_id)
                    for veh_id in veh_ids]
        return self.find_intersection_dist(veh_ids)

    def find_intersection_dist(self, veh_id):
        """Return distance from intersection.
        Return the distance from the vehicle's current position to the position
        of the node it is heading toward.
        """
        edge_id = self.k.vehicle.get_edge(veh_id)
        # FIXME this might not be the best way of handling this
        if edge_id == "":
            return -10
        if 'center' in edge_id:
            return 0
        edge_len = self.k.network.edge_length(edge_id)
        relative_pos = self.k.vehicle.get_position(veh_id)
        dist = edge_len - relative_pos
        return dist

    def _convert_edge(self, edges):
        """Convert the string edge to a number.
        Start at the bottom left vertical edge and going right and then up, so
        the bottom left vertical edge is zero, the right edge beside it  is 1.
        The numbers are assigned along the lowest column, then the lowest row,
        then the second lowest column, etc. Left goes before right, top goes
        before bottom.
        The values are zero indexed.
        Parameters
        ----------
        edges : list of str or str
            name of the edge(s)
        Returns
        -------
        list of int or int
            a number uniquely identifying each edge
        """
        if isinstance(edges, list):
            return [self._split_edge(edge) for edge in edges]
        else:
            return self._split_edge(edges)

    def _split_edge(self, edge):
        """Act as utility function for convert_edge."""
        if edge:
            if edge[0] == ":":  # center
                center_index = int(edge.split("center")[1][0])
                base = ((self.cols + 1) * self.rows * 2) \
                    + ((self.rows + 1) * self.cols * 2)
                return base + center_index + 1
            else:
                pattern = re.compile(r"[a-zA-Z]+")
                edge_type = pattern.match(edge).group()
                edge = edge.split(edge_type)[1].split('_')
                row_index, col_index = [int(x) for x in edge]
                if edge_type in ['bot', 'top']:
                    rows_below = 2 * (self.cols + 1) * row_index
                    cols_below = 2 * (self.cols * (row_index + 1))
                    edge_num = rows_below + cols_below + 2 * col_index + 1
                    return edge_num if edge_type == 'bot' else edge_num + 1
                if edge_type in ['left', 'right']:
                    rows_below = 2 * (self.cols + 1) * row_index
                    cols_below = 2 * (self.cols * row_index)
                    edge_num = rows_below + cols_below + 2 * col_index + 1
                    return edge_num if edge_type == 'left' else edge_num + 1
        else:
            return 0

    def _get_relative_node(self, agent_id, direction):
        """Yield node number of traffic light agent in a given direction.
        For example, the nodes in a grid with 2 rows and 3 columns are
        indexed as follows:
            |     |     |
        --- 3 --- 4 --- 5 ---
            |     |     |
        --- 0 --- 1 --- 2 ---
            |     |     |
        See flow.network.grid for more information.
        Example of function usage:
        - Seeking the "top" direction to ":center0" would return 3.
        - Seeking the "bottom" direction to ":center0" would return -1.
        :param agent_id: agent id of the form ":center#"
        :param direction: top, bottom, left, right
        :return: node number
        """
        ID_IDX = 1
        agent_id_num = int(agent_id.split("center")[ID_IDX])
        if direction == "top":
            node = agent_id_num + self.cols
            if node >= self.cols * self.rows:
                node = -1
        elif direction == "bottom":
            node = agent_id_num - self.cols
            if node < 0:
                node = -1
        elif direction == "left":
            if agent_id_num % self.cols == 0:
                node = -1
            else:
                node = agent_id_num - 1
        elif direction == "right":
            if agent_id_num % self.cols == self.cols - 1:
                node = -1
            else:
                node = agent_id_num + 1
        else:
            raise NotImplementedError

        return node

    def additional_command(self):
        """See parent class.
        Used to insert vehicles that are on the exit edge and place them
        back on their entrance edge.
        """
        for veh_id in self.k.vehicle.get_ids():
            self._reroute_if_final_edge(veh_id)

    def _reroute_if_final_edge(self, veh_id):
        """Reroute vehicle associated with veh_id.
        Checks if an edge is the final edge. If it is return the route it
        should start off at.
        """
        edge = self.k.vehicle.get_edge(veh_id)
        if edge == "":
            return
        if edge[0] == ":":  # center edge
            return
        pattern = re.compile(r"[a-zA-Z]+")
        edge_type = pattern.match(edge).group()
        edge = edge.split(edge_type)[1].split('_')
        row_index, col_index = [int(x) for x in edge]

        # find the route that we're going to place the vehicle on if we are
        # going to remove it
        route_id = None
        if edge_type == 'bot' and col_index == self.cols:
            route_id = "bot{}_0".format(row_index)
        elif edge_type == 'top' and col_index == 0:
            route_id = "top{}_{}".format(row_index, self.cols)
        elif edge_type == 'left' and row_index == 0:
            route_id = "left{}_{}".format(self.rows, col_index)
        elif edge_type == 'right' and row_index == self.rows:
            route_id = "right0_{}".format(col_index)

        if route_id is not None:
            type_id = self.k.vehicle.get_type(veh_id)
            lane_index = self.k.vehicle.get_lane(veh_id)
            # remove the vehicle
            self.k.vehicle.remove(veh_id)
            # reintroduce it at the start of the network
            self.k.vehicle.add(
                veh_id=veh_id,
                edge=route_id,
                type_id=str(type_id),
                lane=str(lane_index),
                pos="0",
                speed="max")

    def get_closest_to_intersection(self, edges, num_closest, padding=False):
        """Return the IDs of the vehicles that are closest to an intersection.
        For each edge in edges, return the IDs (veh_id) of the num_closest
        vehicles in edge that are closest to an intersection (the intersection
        they are heading towards).
        This function performs no check on whether or not edges are going
        towards an intersection or not, it just gets the vehicles that are
        closest to the end of their edges.
        If there are less than num_closest vehicles on an edge, the function
        performs padding by adding empty strings "" instead of vehicle ids if
        the padding parameter is set to True.
        Parameters
        ----------
        edges : str | str list
            ID of an edge or list of edge IDs.
        num_closest : int (> 0)
            Number of vehicles to consider on each edge.
        padding : bool (default False)
            If there are less than num_closest vehicles on an edge, perform
            padding by adding empty strings "" instead of vehicle ids if the
            padding parameter is set to True (note: leaving padding to False
            while passing a list of several edges as parameter can lead to
            information loss since you will not know which edge, if any,
            contains less than num_closest vehicles).
        Usage
        -----
        For example, consider the following network, composed of 4 edges
        whose ids are "edge0", "edge1", "edge2" and "edge3", the numbers
        being vehicles all headed towards intersection x. The ID of the vehicle
        with number n is "veh{n}" (edge "veh0", "veh1"...).
                            edge1
                            |   |
                            | 7 |
                            | 8 |
               -------------|   |-------------
        edge0    1 2 3 4 5 6  x                 edge2
               -------------|   |-------------
                            | 9 |
                            | 10|
                            | 11|
                            edge3
        And consider the following example calls on the previous network:
        >>> get_closest_to_intersection("edge0", 4)
        ["veh6", "veh5", "veh4", "veh3"]
        >>> get_closest_to_intersection("edge0", 8)
        ["veh6", "veh5", "veh4", "veh3", "veh2", "veh1"]
        >>> get_closest_to_intersection("edge0", 8, padding=True)
        ["veh6", "veh5", "veh4", "veh3", "veh2", "veh1", "", ""]
        >>> get_closest_to_intersection(["edge0", "edge1", "edge2", "edge3"],
                                         3, padding=True)
        ["veh6", "veh5", "veh4", "veh8", "veh7", "", "", "", "", "veh9",
         "veh10", "veh11"]
        Returns
        -------
        str list
            If n is the number of edges given as parameters, then the returned
            list contains n * num_closest vehicle IDs.
        Raises
        ------
        ValueError
            if num_closest <= 0
        """
        if num_closest <= 0:
            raise ValueError("Function get_closest_to_intersection called with"
                             "parameter num_closest={}, but num_closest should"
                             "be positive".format(num_closest))

        if isinstance(edges, list):
            ids = [self.get_closest_to_intersection(edge, num_closest)
                   for edge in edges]
            # flatten the list and return it
            return [veh_id for sublist in ids for veh_id in sublist]

        # get the ids of all the vehicles on the edge 'edges' ordered by
        # increasing distance to end of edge (intersection)
        veh_ids_ordered = sorted(self.k.vehicle.get_ids_by_edge(edges),
                                 key=self.get_distance_to_intersection)

        # return the ids of the num_closest vehicles closest to the
        # intersection, potentially with ""-padding.
        pad_lst = [""] * (num_closest - len(veh_ids_ordered))
        return veh_ids_ordered[:num_closest] + (pad_lst if padding else [])


class TrafficLightGridPOEnv(TrafficLightGridEnv):
    """Environment used to train traffic lights.

    Required from env_params:
    * switch_time: minimum switch time for each traffic light (in seconds).
      Earlier RL commands are ignored.
    * num_observed: number of vehicles nearest each intersection that is
      observed in the state space; defaults to 2
    States
        An observation is the number of observed vehicles in each intersection
        closest to the traffic lights, a number uniquely identifying which
        edge the vehicle is on, and the speed of the vehicle.
    Actions
        The action space consist of a list of float variables ranging from 0-1
        specifying whether a traffic light is supposed to switch or not. The
        actions are sent to the traffic light in the grid from left to right
        and then top to bottom.
    Rewards
        The reward is the delay of each vehicle minus a penalty for switching
        traffic lights
    Termination
        A rollout is terminated once the time horizon is reached.
    Additional
        Vehicles are rerouted to the start of their original routes once they
        reach the end of the network in order to ensure a constant number of
        vehicles.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        for p in ADDITIONAL_PO_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter "{}" not supplied'.format(p))

        # number of vehicles nearest each intersection that is observed in the
        # state space; defaults to 2
        self.num_observed = env_params.additional_params.get("num_observed", 2)

        # used during visualization
        self.observed_ids = []

    @property
    def observation_space(self):
        """State space that is partially observed.
        Velocities, distance to intersections, edge number (for nearby
        vehicles) from each direction, edge information, and traffic light
        state.
        """
        tl_box = Box(
            low=0.,
            high=1,
            shape=(3 * 4 * self.num_observed * self.num_traffic_lights +
                   2 * len(self.k.network.get_edge_list()) +
                   3 * self.num_traffic_lights,),
            dtype=np.float32)
        return tl_box

    def get_state(self):
        """See parent class.
        Returns self.num_observed number of vehicles closest to each traffic
        light and for each vehicle its velocity, distance to intersection,
        edge_number traffic light state. This is partially observed
        """
        speeds = []
        dist_to_intersec = []
        edge_number = []
        max_speed = max(
            self.k.network.speed_limit(edge)
            for edge in self.k.network.get_edge_list())
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])
        all_observed_ids = []

        for _, edges in self.network.node_mapping:
            for edge in edges:
                observed_ids = \
                    self.get_closest_to_intersection(edge, self.num_observed)
                all_observed_ids += observed_ids

                # check which edges we have so we can always pad in the right
                # positions
                speeds += [
                    self.k.vehicle.get_speed(veh_id) / max_speed
                    for veh_id in observed_ids
                ]
                dist_to_intersec += [
                    (self.k.network.edge_length(
                        self.k.vehicle.get_edge(veh_id)) -
                        self.k.vehicle.get_position(veh_id)) / max_dist
                    for veh_id in observed_ids
                ]
                edge_number += \
                    [self._convert_edge(self.k.vehicle.get_edge(veh_id)) /
                     (self.k.network.network.num_edges - 1)
                     for veh_id in observed_ids]

                if len(observed_ids) < self.num_observed:
                    diff = self.num_observed - len(observed_ids)
                    speeds += [0] * diff
                    dist_to_intersec += [0] * diff
                    edge_number += [0] * diff

        # now add in the density and average velocity on the edges
        density = []
        velocity_avg = []
        for edge in self.k.network.get_edge_list():
            ids = self.k.vehicle.get_ids_by_edge(edge)
            if len(ids) > 0:
                # TODO(cathywu) Why is there a 5 here?
                density += [5 * len(ids) / self.k.network.edge_length(edge)]
                velocity_avg += [np.mean(
                    [self.k.vehicle.get_speed(veh_id) for veh_id in
                     ids]) / max_speed]
            else:
                density += [0]
                velocity_avg += [0]
        self.observed_ids = all_observed_ids
        return np.array(
            np.concatenate([
                speeds, dist_to_intersec, edge_number, density, velocity_avg,
                self.last_change.flatten().tolist(),
                self.direction.flatten().tolist(),
                self.currently_yellow.flatten().tolist()
            ]))

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        if self.env_params.evaluate:
            return - rewards.min_delay_unscaled(self)
        else:
            return (- rewards.min_delay_unscaled(self) +
                    rewards.penalize_standstill(self, gain=0.2))

    def additional_command(self):
        """See class definition."""
        # specify observed vehicles
        [self.k.vehicle.set_observed(veh_id) for veh_id in self.observed_ids]


class TrafficLightGridTestEnv(TrafficLightGridEnv):
    """
    Class for use in testing.
    This class overrides RL methods of green wave so we can test
    construction without needing to specify RL methods
    """

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        pass

    def compute_reward(self, rl_actions, **kwargs):
        """No return, for testing purposes."""
        return 0
