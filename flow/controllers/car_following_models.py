"""
Contains several custom car-following control models.

These controllers can be used to modify the acceleration behavior of vehicles
in Flow to match various prominent car-following models that can be calibrated.

Each controller includes the function ``get_accel(self, env) -> acc`` which,
using the current state of the world and existing parameters, uses the control
model to return a vehicle acceleration.
"""
import math
import numpy as np

from flow.controllers.base_controller import BaseController


class CFMController(BaseController):
    """CFM controller.

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : SumoCarFollowingParams
        see parent class
    k_d : float
        headway gain (default: 1)
    k_v : float
        gain on difference between lead velocity and current (default: 1)
    k_c : float
        gain on difference from desired velocity to current (default: 1)
    d_des : float
        desired headway (default: 1)
    v_des : float
        desired velocity (default: 8)
    time_delay : float, optional
        time delay (default: 0.0)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 car_following_params,
                 k_d=1,
                 k_v=1,
                 k_c=1,
                 d_des=1,
                 v_des=8,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None):
        """Instantiate a CFM controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)

        self.veh_id = veh_id
        self.k_d = k_d
        self.k_v = k_v
        self.k_c = k_c
        self.d_des = d_des
        self.v_des = v_des

    def get_accel(self, env):
        """See parent class."""
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        if not lead_id:  # no car ahead
            return self.max_accel

        lead_vel = env.k.vehicle.get_speed(lead_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)

        d_l = env.k.vehicle.get_headway(self.veh_id)

        return self.k_d*(d_l - self.d_des) + self.k_v*(lead_vel - this_vel) + \
            self.k_c*(self.v_des - this_vel)


class BCMController(BaseController):
    """Bilateral car-following model controller.

    This model looks ahead and behind when computing its acceleration.

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.params.SumoCarFollowingParams
        see parent class
    k_d : float
        gain on distances to lead/following cars (default: 1)
    k_v : float
        gain on vehicle velocity differences (default: 1)
    k_c : float
        gain on difference from desired velocity to current (default: 1)
    d_des : float
        desired headway (default: 1)
    v_des : float
        desired velocity (default: 8)
    time_delay : float
        time delay (default: 0.5)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 car_following_params,
                 k_d=1,
                 k_v=1,
                 k_c=1,
                 d_des=1,
                 v_des=8,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None):
        """Instantiate a Bilateral car-following model controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)

        self.veh_id = veh_id
        self.k_d = k_d
        self.k_v = k_v
        self.k_c = k_c
        self.d_des = d_des
        self.v_des = v_des

    def get_accel(self, env):
        """See parent class.

        From the paper:
        There would also be additional control rules that take
        into account minimum safe separation, relative speeds,
        speed limits, weather and lighting conditions, traffic density
        and traffic advisories
        """
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        if not lead_id:  # no car ahead
            return self.max_accel

        lead_vel = env.k.vehicle.get_speed(lead_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)

        trail_id = env.k.vehicle.get_follower(self.veh_id)
        trail_vel = env.k.vehicle.get_speed(trail_id)

        headway = env.k.vehicle.get_headway(self.veh_id)
        footway = env.k.vehicle.get_headway(trail_id)

        return self.k_d * (headway - footway) + \
            self.k_v * ((lead_vel - this_vel) - (this_vel - trail_vel)) + \
            self.k_c * (self.v_des - this_vel)


class LACController(BaseController):
    """Linear Adaptive Cruise Control.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.params.SumoCarFollowingParams
        see parent class
    k_1 : float
        design parameter (default: 0.8)
    k_2 : float
        design parameter (default: 0.9)
    h : float
        desired time gap  (default: 1.0)
    tau : float
        lag time between control input u and real acceleration a (default:0.1)
    time_delay : float
        time delay (default: 0.5)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 car_following_params,
                 k_1=0.3,
                 k_2=0.4,
                 h=1,
                 tau=0.1,
                 a=0,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None):
        """Instantiate a Linear Adaptive Cruise controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)

        self.veh_id = veh_id
        self.k_1 = k_1
        self.k_2 = k_2
        self.h = h
        self.tau = tau
        self.a = a

    def get_accel(self, env):
        """See parent class."""
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        lead_vel = env.k.vehicle.get_speed(lead_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)
        headway = env.k.vehicle.get_headway(self.veh_id)
        L = env.k.vehicle.get_length(self.veh_id)
        ex = headway - L - self.h * this_vel
        ev = lead_vel - this_vel
        u = self.k_1*ex + self.k_2*ev
        a_dot = -(self.a/self.tau) + (u/self.tau)
        self.a = a_dot*env.sim_step + self.a

        return self.a


class OVMController(BaseController):
    """Optimal Vehicle Model controller.

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.params.SumoCarFollowingParams
        see parent class
    alpha : float
        gain on desired velocity to current velocity difference
        (default: 0.6)
    beta : float
        gain on lead car velocity and self velocity difference
        (default: 0.9)
    h_st : float
        headway for stopping (default: 5)
    h_go : float
        headway for full speed (default: 35)
    v_max : float
        max velocity (default: 30)
    time_delay : float
        time delay (default: 0.5)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 car_following_params,
                 alpha=1,
                 beta=1,
                 h_st=2,
                 h_go=15,
                 v_max=30,
                 time_delay=0,
                 noise=0,
                 fail_safe=None):
        """Instantiate an Optimal Vehicle Model controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)
        self.veh_id = veh_id
        self.v_max = v_max
        self.alpha = alpha
        self.beta = beta
        self.h_st = h_st
        self.h_go = h_go

    def get_accel(self, env):
        """See parent class."""
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        if not lead_id:  # no car ahead
            return self.max_accel

        lead_vel = env.k.vehicle.get_speed(lead_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)
        h_dot = lead_vel - this_vel

        # V function here - input: h, output : Vh
        if h <= self.h_st:
            v_h = 0
        elif self.h_st < h < self.h_go:
            v_h = self.v_max / 2 * (1 - math.cos(math.pi * (h - self.h_st) /
                                                 (self.h_go - self.h_st)))
        else:
            v_h = self.v_max

        return self.alpha * (v_h - this_vel) + self.beta * h_dot


class LinearOVM(BaseController):
    """Linear OVM controller.

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.params.SumoCarFollowingParams
        see parent class
    v_max : float
        max velocity (default: 30)
    adaptation : float
        adaptation constant (default: 0.65)
    h_st : float
        headway for stopping (default: 5)
    time_delay : float
        time delay (default: 0.5)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 car_following_params,
                 v_max=30,
                 adaptation=0.65,
                 h_st=5,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None):
        """Instantiate a Linear OVM controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)
        self.veh_id = veh_id
        # 4.8*1.85 for case I, 3.8*1.85 for case II, per Nakayama
        self.v_max = v_max
        # TAU in Traffic Flow Dynamics textbook
        self.adaptation = adaptation
        self.h_st = h_st

    def get_accel(self, env):
        """See parent class."""
        this_vel = env.k.vehicle.get_speed(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        # V function here - input: h, output : Vh
        alpha = 1.689  # the average value from Nakayama paper
        if h < self.h_st:
            v_h = 0
        elif self.h_st <= h <= self.h_st + self.v_max / alpha:
            v_h = alpha * (h - self.h_st)
        else:
            v_h = self.v_max

        return (v_h - this_vel) / self.adaptation


class IDMController(BaseController):
    """Intelligent Driver Model (IDM) controller.

    For more information on this controller, see:
    Treiber, Martin, Ansgar Hennecke, and Dirk Helbing. "Congested traffic
    states in empirical observations and microscopic simulations." Physical
    review E 62.2 (2000): 1805.

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.param.SumoCarFollowingParams
        see parent class
    v0 : float
        desirable velocity, in m/s (default: 30)
    T : float
        safe time headway, in s (default: 1)
    a : float
        max acceleration, in m/s2 (default: 1)
    b : float
        comfortable deceleration, in m/s2 (default: 1.5)
    delta : float
        acceleration exponent (default: 4)
    s0 : float
        linear jam distance, in m (default: 2)
    dt : float
        timestep, in s (default: 0.1)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 v0=30,
                 T=1,
                 a=1,
                 b=1.5,
                 delta=4,
                 s0=2,
                 time_delay=0.0,
                 dt=0.1,
                 noise=0,
                 fail_safe=None,
                 car_following_params=None):
        """Instantiate an IDM controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        self.dt = dt

    def get_accel(self, env):
        """See parent class."""
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        # in order to deal with ZeroDivisionError
        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':  # no car ahead
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) /
                (2 * np.sqrt(self.a * self.b)))

        return self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)


class SimCarFollowingController(BaseController):
    """Controller whose actions are purely defined by the simulator.

    Note that methods for implementing noise and failsafes through
    BaseController, are not available here. However, similar methods are
    available through sumo when initializing the parameters of the vehicle.

    Usage: See BaseController for usage example.
    """

    def get_accel(self, env):
        """See parent class."""
        return None


class JordanController(BaseController):
    """The controller for splitting platoon near intersection to assistant the manuver (i.e., lane change) of emergency vehicle.

    Implamented by Dajiang Suo based on the work by Jordan et al., see:
    Jordan et al. "Path Clearance for Emergency Vehicles Through the Use of Vehicle-to-Vehicle Communication." Transportation
    Research Board (2013)

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.param.SumoCarFollowingParams
        see parent class
    w : float
        the shockwave speed (for queue discharge)
    v_ev: float
        the desired speed of EV
    v_N: float
        the normal speed other vehicles (except RL and EV) travel
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe 
        +++++++++++++++++++++++++++++++++++++
        `he vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None,
                 car_following_params=None,
                 w=25,
                 v_ev=30,
                 v_N=15):
        """Instantiate an IDM controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)
        self.w = w
        self.v_ev = v_ev
        self.v_N = v_N
        self.Jordan_stop_flag = False
        # caculate x_L, the optimal position for platoon splitting


    def get_accel(self, env):
        """control the acceleration & deceleration of RL."""
        # if the RL does not locate at the edge of right0_0, then we should judge wether the RL should decelerate or drive based on the krauss model


        # Depending on whether RL locates at a position (x_a) further away or closer to the intersection compared to the optimal splitting
        # point (x_L), we can devide the accel/decel strategy into two scenarios:
        # scenario 1. x_a > x_L
        # scenario 2. x_a < x_L

        # get the current position and spd of the ego vehicle 

        pos_abs =  env.k.vehicle.get_x_by_id(self.veh_id)
        edge_num_rl = env.k.vehicle.get_edge(self.veh_id)
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        v_lead = env.k.vehicle.get_speed(lead_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        # deriving v_safe and v_desired based on the krauss model
        # v_safe = v_lead + (minGap-v_lead*tao)/(v/decel+tao)
        #minGap=2.5
        curr_headway = h
        decel=7.5
        tao = 1 # human reaction time
        max_spd = 30
        v_safe  = v_lead + (curr_headway-v_lead*tao)/(v/decel+tao)
        v_desired = min(v_safe,v+1.5,max_spd)
        

        # get ev status
        ev_string = 'emergency'
        ev_id = -1001
        
        for veh_id in env.k.vehicle.get_ids():
            if ev_string in veh_id:
                ev_id = veh_id
                break

        # if no ev, then follows what krauss model specifies
        if ev_id == -1001:
            print("no EV")
            #print("no ev found")
            if lead_id is None or lead_id == '':
                #print("no EV: drive at max")
                return (max_spd-v)/env.sim_step
            else:
                #print("no EV: drive at desired spd:",v_desired)
                #print("current spd of Jordan veh:",v)
                return (v_desired-v)/env.sim_step


        # when ev exists in the network, we would need to determine if both ev and rl are in the bottom edge and ev is further away from the stop line
        #ev_pos =  env.k.vehicle.get_position(ev_id)
        #d = 500 - ev_pos # ev_from_inter



        ev_spd = env.k.vehicle.get_speed(ev_id)
        edge_num_ev = env.k.vehicle.get_edge(ev_id)
        ev_lane = env.k.vehicle.get_lane(ev_id)

        if self.Jordan_stop_flag == True:
            if ev_lane ==1:
                self.Jordan_stop_flag = False
            return (0-v)/env.sim_step

        # if both ev and RL stopped in a queue and if both locates at the bottom edge (i.e., right0_0)
        # note: if the goal of the CAV is to assist the EV to cross two intersections, how will this influence its acceleration strategies?


        if abs(v) < 0.5 and abs(ev_spd) < 0.5 and edge_num_ev == 'right0_0' and edge_num_rl == 'right0_0':
            print("meet Jordan conditions")
            ev_pos =  env.k.vehicle.get_position(ev_id)
            d = 500 - ev_pos # ev_from_inter
            pos_edge = env.k.vehicle.get_position(self.veh_id)
            x_a = 500 - pos_edge

            # derive the optimal position x_L
            x_L = d*((1/self.w+1/self.v_N)/(1/self.w+2/self.v_N-1/self.v_ev))
            print("d the pos of ev from inters:",d)
            print("x_a the actual pos of Jordan veh",x_a)
            print("x_L the optimal pos of platoon splitting",x_L)

            if x_a <= x_L:
                print("Jordan: vehicle stop")
                self.Jordan_stop_flag = True
                return (0-v)/env.sim_step # rl can should stop to wait for the ev to switch the lane
            else:
                print("Jordan: vehicle accel")
                t_s = d/self.w + (d-x_L)/self.v_ev
                t_a = x_a/self.w
                spd_jordan =  (x_a-x_L)/(t_s-t_a)
                return (spd_jordan - v)/env.sim_step

        # else if the CAV locates in the first intersection downstream, is its best driving strategy to wait and stay stationary until the EV switches to the left lane?


        
        #print("with EV: drive at desired spd:",v_desired)
        #print("current spd of Jordan veh:",v)
        #print("ev veh spd:",ev_spd)
        #print("Jordan veh edge:",edge_num_rl)
        #print("ev veh edge:",edge_num_ev)
        #print("Jordan conditions not met")
        return (v_desired-v)/env.sim_step

class JordanControllerMulti(BaseController):
    """The controller for splitting platoon near intersection to assistant the manuver (i.e., lane change) of emergency vehicle.

    Implamented by Dajiang Suo based on the work by Jordan et al., see:
    Jordan et al. "Path Clearance for Emergency Vehicles Through the Use of Vehicle-to-Vehicle Communication." Transportation
    Research Board (2013)

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.param.SumoCarFollowingParams
        see parent class
    w : float
        the shockwave speed (for queue discharge)
    v_ev: float
        the desired speed of EV
    v_N: float
        the normal speed other vehicles (except RL and EV) travel
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe 
        +++++++++++++++++++++++++++++++++++++
        `he vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 veh_id,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None,
                 car_following_params=None,
                 w=15,
                 v_ev=30,
                 v_N=15,
                 edge1_name = 'right0_0',
                 edge1_len = 175,
                 center_name = 'center0_4',
                 center_len = 40,
                 edge2_name = 'right1_0',
                 edge2_len = 175):
        """Instantiate an IDM controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)
        self.w = w
        self.v_ev = v_ev
        self.v_N = v_N
        self.edge1_name = edge1_name
        self.edge1_len = edge1_len
        self.center_name = center_name
        self.center_len = center_len
        self.edge2_name = edge2_name
        self.edge2_len = edge2_len
        self.Jordan_stop_flag = False
        self.Jordan_accel_flag = False
        # caculate x_L, the optimal position for platoon splitting


    

    def get_accel(self, env):
        """control the acceleration & deceleration of RL."""
        # if the RL does not locate at the edge of right0_0, then we should judge wether the RL should decelerate or drive based on the krauss model


        # Depending on whether RL locates at a position (x_a) further away or closer to the intersection compared to the optimal splitting
        # point (x_L), we can devide the accel/decel strategy into two scenarios:
        # scenario 1. x_a > x_L
        # scenario 2. x_a < x_L


        #  apply lane changing to the EV at everytime steps
        max_length = 175

        veh_ids = env.k.vehicle.get_ids()
        for veh in veh_ids:

            is_rl_behind = False
            own_lane = 0
            next_lane = 0
            own_leader = ""
            next_lane_leader = ""
            own_leader_dist = 10000
            next_lane_leader_dist = 10000
            ev_id = ""
            rl_id = ""

            can_break = False
            special_case = False

            if veh.startswith("emergency"):
                ev_id = veh
                lane = env.k.vehicle.get_lane(veh)

                if lane == 1:
                    # already in the right lane, no chnage needed
                    break

                pos = max_length - env.k.vehicle.get_position(veh)
                edge = env.k.vehicle.get_edge(veh)
                
                if edge == "right1_0":
                    ev_pos_from_dest = max_length + pos
                elif edge == "right0_0":
                    ev_pos_from_dest = 2 * max_length + pos
                elif edge == ":center0_4":
                    ev_pos_from_dest = max_length + pos
                else:
                    # EV is in the outgoing right2_0 road
                    ev_pos_from_dest = pos 

                for sub_veh in veh_ids:

                    sub_egde = env.k.vehicle.get_edge(sub_veh)
                    sub_lane = env.k.vehicle.get_lane(sub_veh)

                    # get the vehicle in the same lane as EV but infront of EV
                    if sub_egde == edge:
                        sub_pos = max_length - env.k.vehicle.get_position(sub_veh)
                        if edge == "right1_0":
                            sub_pos_from_dest = max_length + sub_pos
                        elif edge == "right0_0":
                            sub_pos_from_dest = 2 * max_length + sub_pos
                        elif edge == ":center0_4":
                            sub_pos_from_dest = max_length + sub_pos
                        else:
                            sub_pos_from_dest = sub_pos 

                        if sub_pos_from_dest < ev_pos_from_dest:
                            if sub_lane != lane:
                                next_lane += 1
                                # update the possible candidate to be the next lane immediate leader of EV
                                if ev_pos_from_dest - sub_pos_from_dest < next_lane_leader_dist:
                                    next_lane_leader = sub_veh
                                    next_lane_leader_dist = ev_pos_from_dest - sub_pos_from_dest
                            else:
                                # update the possible candidate to be the own lane immediate leader of EV
                                own_lane += 1
                                if ev_pos_from_dest - sub_pos_from_dest < own_leader_dist:
                                    own_leader = sub_veh
                                    own_leader_dist = ev_pos_from_dest - sub_pos_from_dest
                    
                    # we only need to do a EV lane change when the RL vehicle is behind once the lane changing is performed
                    if sub_veh.startswith("jordan"):
                        rl_id = sub_veh
                        rl_pos = max_length - env.k.vehicle.get_position(sub_veh)
                        rl_edge = env.k.vehicle.get_edge(sub_veh)

                        if rl_edge == "right1_0":
                            rl_pos_from_dest = max_length + rl_pos
                        elif rl_edge == "right0_0":
                            rl_pos_from_dest = 2 * max_length + rl_pos
                        elif rl_edge == ":center0_4":
                            rl_pos_from_dest = max_length + rl_pos
                        else:
                            rl_pos_from_dest = rl_pos 

                        if rl_pos_from_dest < ev_pos_from_dest:
                            # can not do any lane changes now since the RL vehicle is in front
                            can_break = True
                            break

                        # leading vehicle of the RL vehicle
                        lead_id = env.k.vehicle.get_leader(sub_veh)
                        print("lead id:" + str(lead_id))
                        if lead_id in ["", None]:
                            special_case = True  
                        else:
                            lead_pos = max_length - env.k.vehicle.get_position(lead_id)
                            lead_edge = env.k.vehicle.get_edge(lead_id)
                            if lead_edge == "right1_0":
                                lead_pos_from_dest = max_length + lead_pos
                            elif lead_edge == "right0_0":
                                lead_pos_from_dest = 2 * max_length + lead_pos
                            elif lead_edge == ":center0_4":
                                lead_pos_from_dest = max_length + lead_pos
                            else:
                                lead_pos_from_dest = lead_pos 

                            if lead_pos_from_dest < ev_pos_from_dest:
                                is_rl_behind = True
                            else:
                                is_rl_behind = False

                if can_break:
                    print("-----C100")
                    env.k.vehicle.apply_lane_change(ev_id, 0) # no change
                    break
                      
                left = 1
                no_change = 0
                own_speed = env.k.vehicle.get_speed(own_leader)
                next_speed = env.k.vehicle.get_speed(next_lane_leader)
                ev_edge = env.k.vehicle.get_edge(ev_id)
                rl_edge = env.k.vehicle.get_edge(rl_id)
                print("is RL behind: " + str(is_rl_behind))
                if ev_edge == "right2_0" and rl_edge == "right0_0":
                    # no lane changing if the two vehicles are far apart
                    print("-----C101")
                    env.k.vehicle.apply_lane_change(ev_id, no_change)
                    break

                if own_speed < 0 or next_speed < 0:
                    if special_case:
                        if own_lane >= next_lane:
                            if own_lane == 1:
                                print("C1")
                                # only changing to left lane is allowed
                                env.k.vehicle.apply_lane_change(ev_id, no_change)
                            else:
                                print("C2")
                                env.k.vehicle.apply_lane_change(ev_id, left)
                        else:
                            print("C3")
                            env.k.vehicle.apply_lane_change(ev_id, no_change)
                    elif own_lane >= next_lane and is_rl_behind:
                        if own_lane == 1:
                            # only changing to left lane is allowed
                            print("C4")
                            env.k.vehicle.apply_lane_change(ev_id, no_change)
                        else:
                            print("C5")
                            env.k.vehicle.apply_lane_change(ev_id, left)
                    else:
                        print("C6")
                        env.k.vehicle.apply_lane_change(ev_id, no_change)
                elif own_lane >= next_lane and own_speed <= next_speed: 
                    if special_case:
                        if own_lane == 1:
                            # only changing to left lane is allowed
                            print("C7")
                            env.k.vehicle.apply_lane_change(ev_id, no_change)
                        else:
                            print("C8")
                            env.k.vehicle.apply_lane_change(ev_id, left)
                    elif is_rl_behind:
                        if own_lane == 1:
                            # only changing to left lane is allowed
                            print("C9")
                            env.k.vehicle.apply_lane_change(ev_id, no_change)
                        else:
                            print("C10")
                            env.k.vehicle.apply_lane_change(ev_id, left)
                    else:
                        print("C11")
                        env.k.vehicle.apply_lane_change(ev_id, no_change)
                break

        # get the current position and spd of the ego vehicle 
        #self.apply_lane_change()

        pos_abs =  env.k.vehicle.get_x_by_id(self.veh_id)
        edge_num_rl = env.k.vehicle.get_edge(self.veh_id)
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        v_lead = env.k.vehicle.get_speed(lead_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        # deriving v_safe and v_desired based on the krauss model
        # v_safe = v_lead + (minGap-v_lead*tao)/(v/decel+tao)
        #minGap=2.5
        curr_headway = h
        decel=7.5
        tao = 1 # human reaction time
        max_spd = 30
        v_safe  = v_lead + (curr_headway-v_lead*tao)/(v/decel+tao)
        v_desired = min(v_safe,v+1.5,max_spd)
        

        # get ev status
        ev_string = 'emergency'
        ev_id = -1001
        
        for veh_id in env.k.vehicle.get_ids():
            if ev_string in veh_id:
                ev_id = veh_id
                break

        # if no ev, then follows what krauss model specifies
        if ev_id == -1001:
            print("no EV")
            #print("no ev found")
            if lead_id is None or lead_id == '':
                #print("no EV: drive at max")
                return (max_spd-v)/env.sim_step
            else:
                #print("no EV: drive at desired spd:",v_desired)
                #print("current spd of Jordan veh:",v)
                return (v_desired-v)/env.sim_step


        # when ev exists in the network, we would need to determine if both ev and rl are in the bottom edge and ev is further away from the stop line
        #ev_pos =  env.k.vehicle.get_position(ev_id)
        #d = 500 - ev_pos # ev_from_inter



        ev_spd = env.k.vehicle.get_speed(ev_id)
        edge_num_ev = env.k.vehicle.get_edge(ev_id)
        ev_lane = env.k.vehicle.get_lane(ev_id)
        z = self.edge2_len + self.center_len
        ev_pos =  env.k.vehicle.get_position(ev_id)

        if self.Jordan_stop_flag == True:
            if ev_lane ==1:
                self.Jordan_stop_flag = False
                return (v_desired-v)/env.sim_step
            return (0-v)/env.sim_step

        if self.Jordan_accel_flag == True:
            if ev_lane == 1:
                self.Jordan_accel_flag = False
                return (v_desired-v)/env.sim_step
            return (self.Jordan_accel_speed - v)/env.sim_step

        # if both ev and RL stopped in a queue and if both locates at the bottom edge (i.e., right0_0)
        # note: if the goal of the CAV is to assist the EV to cross two intersections, how will this influence its acceleration strategies?


        if abs(v) < 0.5 and abs(ev_spd) < 0.5 and edge_num_ev == 'right0_0' and edge_num_rl == 'right0_0':
            print("meet Jordan conditions near the 1st intersection")
            d = self.edge1_len - ev_pos # ev_from_intersection, note that the length of the road segment now is set to 300
            pos_edge = env.k.vehicle.get_position(self.veh_id)
            x_a = self.edge1_len - pos_edge

            # derive the optimal position x_L
            #x_L = d*((1/self.w+1/self.v_N)/(1/self.w+2/self.v_N-1/self.v_ev))
            x_L_nominator = d*(1/self.w+1/self.v_N)+z*(1/self.v_ev-self.v_N)
            x_L_denominator = 1/self.w+2/self.v_N-1/self.v_ev
            x_L = x_L_nominator/x_L_denominator
            #print("d the pos of ev from inters:",d)
            #print("x_a the actual pos of Jordan veh",x_a)
            #print("x_L the optimal pos of platoon splitting",x_L)

            # I need to consider two scenarios where x_L > 0 and x_L < 0
            if x_L < 0: # optimal splitting point is at the position downstream of the first intersection
                # if CAV is the first vehicle in the queue
                lead_veh_id = env.k.vehicle.get_leader(self.veh_id)
                edge_num_leader = env.k.vehicle.get_edge(lead_veh_id)
                if edge_num_leader == edge2_name:
                    # determine if the queue length in the second intersection is less than a certain threshold, the CAV stand still to support the early lane-changing by 
                    # the ev, rather than travel at the Jordan speed.
                    leader_pos = env.k.vehicle.get_position(lead_veh_id)
                    leader_pos = self.edge2_len - leader_pos
                    if (1/self.w+1/self.v_N)*leader_pos <= d/self.w + (d+z)/self.v_ev:
                        #set Jordan stop flag to true
                        self.Jordan_stop_flag = True
                        return (0-v)/env.sim_step
                    else:
                        # ev should still switch at the optimal point, how to determine the jordan speed in this case?
                        self.Jordan_accel_flag = True
                        t_a = x_a/self.w
                        t_s = d/self.w + (d-x_L)/self.v_ev
                        self.Jordan_accel_speed = (x_a-x_L)/(t_s-t_a)
                        return (self.Jordan_accel_speed - v)/env.sim_step

            else:
                if x_a <= x_L:
                    print("Jordan: vehicle stop")
                    self.Jordan_stop_flag = True
                    return (0-v)/env.sim_step # rl can should stop to wait for the ev to switch the lane
                else:
                    print("Jordan: vehicle accel")
                    self.Jordan_accel_flag = True

                    t_s = d/self.w + (d-x_L)/self.v_ev
                    t_a = x_a/self.w
                    spd_jordan =  (x_a-x_L)/(t_s-t_a)
                    self.Jordan_accel_speed = spd_jordan
                    return (spd_jordan - v)/env.sim_step


        elif abs(v) < 0.5 and abs(ev_spd) < 0.5 and edge_num_ev == 'right1_0' and edge_num_rl == 'right1_0':
            print("meet Jordan conditions near the 2nd intersection")
            ev_pos =  env.k.vehicle.get_position(ev_id)
            d = self.edge2_len - ev_pos # ev_from_inter
            pos_edge = env.k.vehicle.get_position(self.veh_id)
            x_a = self.edge2_len - pos_edge

            # derive the optimal position x_L
            x_L = d*((1/self.w+1/self.v_N)/(1/self.w+2/self.v_N-1/self.v_ev))
            print("d the pos of ev from inters:",d)
            print("x_a the actual pos of Jordan veh",x_a)
            print("x_L the optimal pos of platoon splitting",x_L)

            if x_a <= x_L:
                print("Jordan: vehicle stop")
                self.Jordan_stop_flag = True
                return (0-v)/env.sim_step # rl can should stop to wait for the ev to switch the lane
            else:
                print("Jordan: vehicle accel")
                self.Jordan_accel_flag = True
                t_s = d/self.w + (d-x_L)/self.v_ev
                t_a = x_a/self.w
                spd_jordan =  (x_a-x_L)/(t_s-t_a)
                self.Jordan_accel_speed = spd_jordan
                return (spd_jordan - v)/env.sim_step

        # else if the CAV locates in the first intersection downstream, its best driving strategy to wait and stay stationary until the EV switches to the left lane?
        # Ans: if the queue length in the second road segment is at another one 
        elif abs(v) < 0.5 and abs(ev_spd) < 0.5 and edge_num_ev == self.edge1_name and edge_num_rl == self.edge2_name:
            # if av locates in a position that allows queue discharge before ev reaches the stop line of the second intersection, av should travel as usual
            follower_veh_id = env.k.vehicle.get_follower(self.veh_id)
            edge_num_follower = env.k.vehicle.get_edge(follower_veh_id)
            av_pos = env.k.vehicle.get_position(self.veh_id)
            av_pos = self.edge2_len - av_pos
            d = self.edge1_len - ev_pos

            if edge_num_follower == self.edge1_name and (1/self.w+1/self.v_N)*av_pos <= d/self.w + (d+z)/self.v_ev:
                return (v_desired-v)/env.sim_step

            else:
                self.Jordan_stop_flag = True
                return (0-v)/env.sim_step

            
                # if the head_veh locates in a position higher than the x_L, then travel more aggresively
        
        #print("with EV: drive at desired spd:",v_desired)
        #print("current spd of Jordan veh:",v)
        #print("ev veh spd:",ev_spd)
        #print("Jordan veh edge:",edge_num_rl)
        #print("ev veh edge:",edge_num_ev)
        #print("Jordan conditions not met")
        return (v_desired-v)/env.sim_step