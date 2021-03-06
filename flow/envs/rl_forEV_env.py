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

        self.rl_learn_flag = False

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

        return Box(low=-1, high=1, shape=(8 * self.num_rl, ), dtype=np.float32)

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

        # 08-17-2021 updates: re-define rewards to make it more 'abstract'
        # reward = 
        #          (1-alpha)*ev_spd - alpha*rl_spd      if: rl_pos > ev_pos (i.e., rl in front of EV) AND ev_lane == 0 (ev hasn't switched lane)
        #          rl_ev                                otherwise (rl should drive as fast as possible)

        # learn nothing if there is no???RL
        if len(self.rl_veh) == 0:
            print("no rl in the network")
            return 0  

        
        rl_id = self.rl_veh[0]
        rl_pos =  self.k.vehicle.get_x_by_id(rl_id)
        rl_spd = self.k.vehicle.get_speed(rl_id)
        if rl_pos == -1001 or rl_spd == -1001:
            print("rl is invalid")
            return 0
        #edge_num_rl = self.k.vehicle.get_edge(rl_id)


        ev_string = 'emergency'
        ev_id = -1001
        
        for veh_id in self.k.vehicle.get_ids():
            if ev_string in veh_id:
                ev_id = veh_id
                break

        # learn nothing if there is no???ev
        if ev_id == -1001:
            print("no ev found")
            return 0

        ev_pos =  self.k.vehicle.get_x_by_id(ev_id)
        ev_spd = self.k.vehicle.get_speed(ev_id)
        ev_lane = self.k.vehicle.get_lane(ev_id)

        if ev_pos == -1001 or ev_spd == -1001 or ev_lane == -1001:
            print("error when getting ev states")
            return 0
            #return rl_spd

        # normalize ev_spd & rl_spd
        max_speed = self.k.network.max_speed()
        rl_spd_norm = rl_spd / max_speed
        ev_spd_norm = ev_spd / max_speed
        alpha = 0.9

        if rl_pos > ev_pos:
            flag_rl_pos = 0
        else:
            flag_rl_pos =1


        if abs(rl_spd) < 0.1 and abs(ev_spd) < 0.1:
            self.rl_learn_flag = True

        # reward function
        if flag_rl_pos == 0 and ev_lane == 0 and self.rl_learn_flag == True:        
            reward =  (1-alpha)*ev_spd_norm - alpha*rl_spd_norm

        else:
            reward = rl_spd_norm
        
        

        return reward

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
        #veh_left_num = 0
        #veh_right_num = 0
        #lane_changing_flag = 1
        #ev_lane = 1

        # normalizing constants
        max_speed = self.k.network.max_speed()
        max_length = 2*max(self.grid_array["short_length"],
                       self.grid_array["long_length"],
                       self.grid_array["inner_length"])


        #max_length = self.k.network.length()


        observation = [0 for _ in range(8 * self.num_rl)]
        for i, rl_id in enumerate(self.rl_veh):
            # rl vehicle data (absolute position, speed, lane index)
            # get the edge and convert it to a number

            # get states related to RL
            if rl_id in ["", None] or self.k.vehicle.get_speed(rl_id) == -1001 \
                or self.k.vehicle.get_x_by_id(rl_id) == -1001:
                rl_spd = 0
            else:
                rl_spd = self.k.vehicle.get_speed(rl_id)
                rl_pos = self.k.vehicle.get_x_by_id(rl_id)

            # get states related to ev
            ev_string = 'emergency'
            ev_id = -1001
        
            for veh_id in self.k.vehicle.get_ids():
                if ev_string in veh_id:
                #ev_speed = self.k.vehicle.get_speed(veh_id)
                    #ev_pos =  self.k.vehicle.get_x_by_id(ev_id)
                    #ev_spd = self.k.vehicle.get_speed(ev_id)
                    #ev_lane = self.k.vehicle.get_lane(ev_id)
                    ev_id = veh_id
                    break

        
            if ev_id == -1001 or self.k.vehicle.get_x_by_id(ev_id) == -1001 \
                or self.k.vehicle.get_speed(ev_id) == -1001 or self.k.vehicle.get_lane(ev_id) == -1001:
                ev_spd = 0
                ev_lane = 1
            else:
                ev_spd = self.k.vehicle.get_speed(ev_id)
                ev_lane = self.k.vehicle.get_lane(ev_id)

            # check the position of rl relative to ev and set the flag_rl_pos
            if self.k.vehicle.get_x_by_id(rl_id) == -1001 or self.k.vehicle.get_x_by_id(ev_id)== -1001:
                flag_rl_pos = 1
            elif self.k.vehicle.get_x_by_id(rl_id) > self.k.vehicle.get_x_by_id(ev_id):
                flag_rl_pos = 0
            else:
                flag_rl_pos =1  

            
            

           


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

    

            observation[8 * i + 0] = rl_spd / max_speed
            

            observation[8 * i + 1] = (lead_speed - rl_spd) / max_speed
            observation[8 * i + 2] = lead_head / max_length

            observation[8 * i + 3] = (rl_spd - follow_speed) / max_speed
            observation[8 * i + 4] = follow_head / max_length

            observation[8 * i + 5] = ev_spd / max_speed
            
            observation[8 * i + 6] = ev_lane
            observation[8 * i + 7] = flag_rl_pos

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
        """
        ev_string = 'emergency'
        rl_string = 'rl'
        for veh_id in self.k.vehicle.get_ids():
            if ev_string in veh_id or rl_string in veh_id:
                self._reroute_if_final_edge(veh_id)"""

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
        self.rl_learn_flag = False

        """
        for veh_id in self.k.vehicle.get_ids():
            self.absolute_position[veh_id] = self.k.vehicle.get_x_by_id(veh_id)
            self.prev_pos[veh_id] = self.k.vehicle.get_x_by_id(veh_id)"""
        self.leader = []
        self.follower = []
        return obs
