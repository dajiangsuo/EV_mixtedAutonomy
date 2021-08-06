"""Environment for training the acceleration behavior of vehicles in a ring."""
import traci
from flow.core import rewards
from flow.envs.base import Env

from gym.spaces.box import Box
import collections
import re

import numpy as np

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    'max_accel': 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    'max_decel': 3,
    # desired velocity for all vehicles in the network, in m/s
    'target_velocity': 30,
    # specifies whether vehicles are to be sorted by position during a
    # simulation step. If set to True, the environment parameter
    # self.sorted_ids will return a list of all vehicles sorted in accordance
    # with the environment
    'sort_vehicles': False,
    "num_rl": 1,


}

MAX_LANES = 2

class AccelEnv_forEV(Env):
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
        #self.prev_pos = dict()
        #self.absolute_position = dict()
        # maximum number of controlled vehicles
        self.grid_array = network.net_params.additional_params["grid_array"]
        self.num_rl = env_params.additional_params["num_rl"]

        # queue of rl vehicles waiting to be controlled
        self.rl_queue = collections.deque()

        # names of the rl vehicles controlled at any step
        self.rl_veh = []

        # used for visualization: the vehicles behind and after RL vehicles
        # (ie the observed vehicles) will have a different color
        self.leader = []
        self.follower = []
        self.rows = self.grid_array["row_num"]
        self.cols = self.grid_array["col_num"]
        # self.num_observed = self.grid_array.get("num_observed", 3)
        self.num_traffic_lights = self.rows * self.cols
        self.tl_type = env_params.additional_params.get('tl_type')

        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(self.num_rl, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        
        """The observation consists of the speeds and bumper-to-bumper headways of
        the vehicles immediately preceding and following autonomous vehicle, the speed
        and position (i.e., distance to intersection) of the emergency vehicle, 
        as well as the position (i.e., distance to intersection) ego speed of the autonomous 
        vehicles."""
        #self.obs_var_labels = ['Velocity', 'Absolute_pos']
        """return Box(
            low=0,
            high=1,
            shape=(2 * self.initial_vehicles.num_vehicles, ),
            dtype=np.float32)"""

        return Box(low=-1, high=1, shape=(10 * self.num_rl, ), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        for i, rl_id in enumerate(self.rl_veh):
            # ignore rl vehicles outside the network
            if rl_id not in self.k.vehicle.get_rl_ids():
                continue
            self.k.vehicle.apply_acceleration(rl_id, rl_actions[i])

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        """Note: the id for emergency vehicle will be emergencyVel0"""
        """Refer to flow.core.params for how veh_ids are generated"""
        """Be careful, I must ensure emergency vehicle is the first one to be added"""
        """One thing hasn't fully understand is that how to make sure ev will always get
        back to the original point: Asn-routing_controller=(GridRouter, {}),"""
        """if self.env_params.evaluate:
            #return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
            return self.k.vehicle.get_speed(self.k.vehicle.get_ids())
        else:
            return rewards.desired_velocity(self, fail=kwargs['fail'])"""
        #return np.mean(self.k.vehicle.get_speed(self.k.vehicle.get_ids()))
        #return self.k.vehicle.get_speed('emergency_00.0')
        # get the position of rl. If there is no rl vehicle in the network, learn nothing
        if len(self.rl_veh) == 0:
            return 0  # learn nothing if there is no　RL

        rl_id = self.rl_veh[0]
        rl_pos =  self.k.vehicle.get_position(rl_id)
        rl_spd = self.k.vehicle.get_speed(rl_id)
        edge_num_rl = self.k.vehicle.get_edge(rl_id)


        ev_string = 'emergency'
        ev_speed = -1001
        

        for veh_id in self.k.vehicle.get_ids():
            if ev_string in veh_id:
                ev_speed = self.k.vehicle.get_speed(veh_id)
                ev_id = veh_id
                break

        
        if ev_speed == -1001:
            #print("ev_speed = -1001")
            #ev_speed = 0
            #return 0
            return rl_spd

        ev_pos =  self.k.vehicle.get_position(ev_id)
        if ev_pos == -1001:
            #return 0
            return rl_spd

        

        #  if rl is leaving the intersection, drive as fast as possible
        if edge_num_rl != 'right0_0':
            return rl_spd

        edge_num_ev = self.k.vehicle.get_edge(ev_id)
        # check if ev is approaching the intersection rather than leaving
        if edge_num_ev != 'right0_0':
            #return 0.5*ev_speed+0.5*avg_spd_edge
            return rl_spd

        
        veh_ids = self.k.vehicle.get_ids_by_edge('right0_0')
        veh_left_list = [veh for veh in veh_ids if self.k.vehicle.get_lane(veh)==1]
        avg_spd_edge = (sum(self.k.vehicle.get_speed(veh_left_list)) /
                     len(veh_left_list))

        ev_lane = self.k.vehicle.get_lane(ev_id)
        # if ev has switched to the left lane, then RL should drive as fast as possible
        if ev_lane == 1: 
            #return 0.5*ev_speed + 0.5*avg_spd_edge
            #return avg_spd_edge
            return rl_spd
        
        veh_left_num = 0
        veh_right_num = 0
        veh_left_list = []



        # Get the average speed of the lane where the RL vehicle locates
        # per edge data (average speed, density
        #avg_spd_edge = 0
        #edge = self.k.vehicle.get_edge(self.rl_veh[0])
        
        
        for veh_id in veh_ids:
            lane_pos = self.k.vehicle.get_position(veh_id)
            lane_idx = self.k.vehicle.get_lane(veh_id)
            if lane_idx == 0:
                if lane_pos > ev_pos:
                    veh_right_num +=1
            else:
                if lane_pos > rl_pos:
                    veh_left_num +=1
            
        #  use the average speed of vehicles in the same (left) lane as the RL as the negative reward
        #  The goal is to have the RL block the traffic behind such that EV can swtich to the left lane
        #  to travel at higher speed in case of congestions in the right lane
        #if ev_speed != -1001 and ev_speed > 0.1 and len(veh_ids) > 0:
        if veh_right_num > veh_left_num:
            #rewards = 0.8*ev_speed - 0.2*avg_spd_edge
            #rewards = -avg_spd_edge
            rewards = -rl_spd

            #density = len(veh_ids) / self.k.network.edge_length(edge)
        else:
            #avg_spd_edge = 0
            #rewards = 0.5*ev_speed + 0.5*avg_spd_edge
            #rewards = avg_spd_edge
            rewards = rl_spd
        

        return rewards

        #ev_speed = self.k.vehicle.get_speed('emergency_00.0')
        #return ev_speed


    def get_state(self):
        """See class definition."""
        # compute the normalizers
        
        """
        speed = [self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed()
                 for veh_id in self.sorted_ids]
        pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.network.length()
               for veh_id in self.sorted_ids]

        return np.array(speed + pos)
        """
        self.leader = []
        self.follower = []
        veh_left_num = 0
        veh_right_num = 0
        lane_changing_flag = 1
        lane_index_ev = 1

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = 2*max(self.grid_array["short_length"],
                       self.grid_array["long_length"],
                       self.grid_array["inner_length"])


        #max_length = self.k.network.length()


        observation = [0 for _ in range(10 * self.num_rl)]
        for i, rl_id in enumerate(self.rl_veh):
            # rl vehicle data (absolute position, speed, lane index)
            # get the edge and convert it to a number
            
            edge_num = self.k.vehicle.get_edge(rl_id)
            if edge_num is None or edge_num == '' or edge_num[0] == ':':
                #lane_group = -1
                this_position = 0

            elif edge_num == 'right0_0':
                #edge_num = int(edge_num) / 2 # we only need to consider two edges
                #lane_group = 1
                #lane_pos = traci.vehicle.getLanePosition(rl_id)
                lane_pos =  self.k.vehicle.get_position(rl_id)
                rl_pos = lane_pos

                lane_pos = 500 - lane_pos
                if lane_pos < 10:
                    lane_cell = 1
                elif lane_pos < 20:
                    lane_cell = 2
                elif lane_pos < 30:
                    lane_cell = 3
                elif lane_pos < 40:
                    lane_cell = 4
                elif lane_pos < 50:
                    lane_cell = 5
                elif lane_pos < 60:
                    lane_cell = 6 
                elif lane_pos < 100:
                    lane_cell = 7
                elif lane_pos < 160:
                    lane_cell = 8
                elif lane_pos < 250:
                    lane_cell = 9
                elif lane_pos <= 500:
                    lane_cell = 10

                #lane_id = traci.vehicle.getLaneID(rl_id)
                #lane_id = self.k.vehicle.get_lane(rl_id)
                lane_index = self.k.vehicle.get_lane(rl_id)
                if lane_index == 0: #'right0_0_0':
                    this_position = lane_cell/20.0
                else:
                    this_position = (lane_cell+10.0)/20.0


            else:
                #lane_group = -1
                this_position = 0

            

            this_speed = self.k.vehicle.get_speed(rl_id)
            if this_speed == -1001:
                print("rl speed is:-1001")
                this_speed = 0
            #this_lane = self.k.vehicle.get_lane(veh_id) / MAX_LANES


            lead_id = self.k.vehicle.get_leader(rl_id)
            follower = self.k.vehicle.get_follower(rl_id)

            if lead_id in ["", None] or self.k.vehicle.get_speed(lead_id) == -1001 \
                or self.k.vehicle.get_x_by_id(lead_id) == -1001 \
                or self.k.vehicle.get_x_by_id(rl_id) == -1001 \
                or self.k.vehicle.get_length(rl_id) == -1001:
                # in case leader is not visible
                lead_speed = max_speed
                lead_head = max_length
            else:
                self.leader.append(lead_id)
                lead_speed = self.k.vehicle.get_speed(lead_id)
                #lead_head = self.k.vehicle.get_x_by_id(lead_id) \
                #    - self.k.vehicle.get_x_by_id(rl_id) \
                #    - self.k.vehicle.get_length(rl_id)

                lead_head = self.k.vehicle.get_headway(rl_id)

            if follower in ["", None]:
                # in case follower is not visible
                follow_speed = 0
                follow_head = max_length
            else:
                self.follower.append(follower)
                follow_speed = self.k.vehicle.get_speed(follower)
                follow_head = self.k.vehicle.get_headway(follower)

            ev_string = 'emergency'
            ev_speed = -1001
            ev_id = -1001

            # find the id of EV
            for veh_id in self.k.vehicle.get_ids():

                if ev_string in veh_id:
                    ev_id = veh_id
                    ev_speed = self.k.vehicle.get_speed(veh_id)
                    print("ev id is:",veh_id)
                    break

            # get ev position as a cell number. If ev_id == -1001, return 0 as its position
            if ev_id == -1001:
                ev_position = 0
            else:
                edge_num_ev = self.k.vehicle.get_edge(ev_id)
                if edge_num_ev is None or edge_num_ev == '' or edge_num_ev[0] == ':':
                #lane_group = -1
                    ev_position = 0

                elif edge_num_ev == 'right0_0':
                #edge_num = int(edge_num) / 2 # we only need to consider two edges
                #lane_group = 1
                    lane_pos = self.k.vehicle.get_position(ev_id)
                    ev_pos = lane_pos
                    lane_pos = 500 - lane_pos
                    if lane_pos < 10:
                        lane_cell = 1
                    elif lane_pos < 20:
                        lane_cell = 2
                    elif lane_pos < 30:
                        lane_cell = 3
                    elif lane_pos < 40:
                        lane_cell = 4
                    elif lane_pos < 50:
                        lane_cell = 5
                    elif lane_pos < 60:
                        lane_cell = 6 
                    elif lane_pos < 100:
                        lane_cell = 7
                    elif lane_pos < 160:
                        lane_cell = 8
                    elif lane_pos < 250:
                        lane_cell = 9
                    elif lane_pos <= 500:
                        lane_cell = 10

                #lane_id = traci.vehicle.getLaneID(rl_id)
                    lane_index_ev = self.k.vehicle.get_lane(ev_id)
                #print("lane indx is:",lane_index)
                    if lane_index_ev == 0: #'right0_0_0':
                        ev_position = lane_cell/20.0
                    else:
                        ev_position = (lane_cell+10.0)/20.0


                else:
                #lane_group = -1
                    ev_position = 0


            if ev_speed == -1001:
                #print("ev speed is:-1001")
                ev_speed = 0
                    
            if this_position != 0 and ev_position != 0 and edge_num == 'right0_0' and edge_num_ev == 'right0_0' and lane_index_ev == 0:
                # only in this way we should car about the density of vehicles in two lanes
                edge = 'right0_0'
                veh_ids = self.k.vehicle.get_ids_by_edge(edge)
                for veh_id in veh_ids:
                    lane_pos = self.k.vehicle.get_position(veh_id)
                    lane_idx = self.k.vehicle.get_lane(veh_id)
                    if lane_idx == 0:
                        if lane_pos > ev_pos:
                            veh_right_num +=1
                    else:
                        if lane_pos > rl_pos:
                            veh_left_num +=1

                if veh_right_num > veh_left_num:
                    lane_changing_flag = 0
        


            




            observation[10 * i + 0] = this_speed / max_speed
            observation[10 * i + 1] = this_position

            observation[10 * i + 2] = (lead_speed - this_speed) / max_speed
            observation[10 * i + 3] = lead_head / max_length

            observation[10 * i + 4] = (this_speed - follow_speed) / max_speed
            observation[10 * i + 5] = follow_head / max_length

            observation[10 * i + 6] = ev_speed / max_speed
            observation[10 * i + 7] = ev_position
            observation[10 * i + 8] = lane_index_ev
            observation[10 * i + 9] = lane_changing_flag

        return observation

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes, and
        update the sorting of vehicles using the self.sorted_ids variable.
        
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
                self.prev_pos[veh_id] = this_pos"""
        
        # add rl vehicles that just entered the network into the rl queue
        super().additional_command()

        for veh_id in self.k.vehicle.get_rl_ids():
            if veh_id not in list(self.rl_queue) + self.rl_veh:
                self.rl_queue.append(veh_id)

        # remove rl vehicles that exited the network
        for veh_id in list(self.rl_queue):
            if veh_id not in self.k.vehicle.get_rl_ids():
                self.rl_queue.remove(veh_id)
        for veh_id in self.rl_veh:
            if veh_id not in self.k.vehicle.get_rl_ids():
                self.rl_veh.remove(veh_id)

        # fil up rl_veh until they are enough controlled vehicles
        while len(self.rl_queue) > 0 and len(self.rl_veh) < self.num_rl:
            rl_id = self.rl_queue.popleft()
            self.rl_veh.append(rl_id)

        # specify observed vehicles
        for veh_id in self.leader + self.follower:
            self.k.vehicle.set_observed(veh_id)
        
        
        """Used to insert vehicles that are on the exit edge and place them
        back on their entrance edge.
        """

        #super().additional_command()
        ev_string = 'emergency'
        rl_string = 'rl'
        for veh_id in self.k.vehicle.get_ids():
            if ev_string in veh_id or rl_string in veh_id:
                self._reroute_if_final_edge(veh_id)

    def _reroute_if_final_edge(self, veh_id):
        """Reroute vehicle associated with veh_id.

        Checks if an edge is the final edge. If it is return the route it
        should start off at.
        """
        ev_string = 'emergency'
        rl_string = 'rl'

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
            self.k.simulation.simulation_step()
            #self.k.vehicle.update(reset=False)
            #self.kernel_api.vehicle.saveState()
            #self.kernel_api.vehicle.loadState()
            if ev_string in veh_id:
                self.k.vehicle.add(
                    veh_id=veh_id,
                    edge=route_id,
                    type_id=str(type_id),
                    lane=str(0),
                    #lane="random",
                    pos="0",
                    speed="max")
            else:
                self.k.vehicle.add(
                    veh_id=veh_id,
                    edge=route_id,
                    type_id=str(type_id),
                    #lane=str(lane_index),
                    lane=str(1),
                    #lane="random",
                    pos="0",
                    speed="max")

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
        obs = super().reset()

        """
        for veh_id in self.k.vehicle.get_ids():
            self.absolute_position[veh_id] = self.k.vehicle.get_x_by_id(veh_id)
            self.prev_pos[veh_id] = self.k.vehicle.get_x_by_id(veh_id)"""
        self.leader = []
        self.follower = []
        return obs
