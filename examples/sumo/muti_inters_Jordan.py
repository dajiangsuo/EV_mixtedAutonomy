"""Traffic Light Grid example."""
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams, SumoLaneChangeParams
from flow.controllers import SimCarFollowingController, GridRouter, IDMController, RLController, SimLaneChangeController,JordanController,JordanControllerMulti
from flow.envs import TrafficLightGridPOEnv
from flow.envs.rl_forEV_env import ADDITIONAL_ENV_PARAMS 
from flow.envs.ring.accel import AccelEnv
from flow.networks import TrafficLightGridNetwork
from flow.core.experiment import Experiment

USE_INFLOWS = True
# Que: this should be syncronized with RL expriment
# Que: How many vehicles should be added in one episode???
HORIZON = 600
# number of rollouts per training iteration
N_ROLLOUTS = 1
# number of parallel workers
N_CPUS = 3

EXP_NUM = 0


# inflow rate at the highway
FLOW_RATE = 1000
# percent of autonomous vehicles
RL_PENETRATION = [0.1, 0.25, 0.33][EXP_NUM]
# num_rl term (see ADDITIONAL_ENV_PARAMs)
NUM_RL = [1, 13, 17][EXP_NUM]

V_MAX_EV = 35
V_MAX_CARS = 15
V_ENTER = 10

# what each element really means?
INNER_LENGTH = 175 #300
LONG_LENGTH = 175 #100
SHORT_LENGTH = 175 #300
# adding one more row
N_ROWS = 2
# how I may get a sense of 
N_COLUMNS = 1
NUM_CARS_LEFT = 1
NUM_CARS_RIGHT = 1
NUM_CARS_TOP = 1
NUM_CARS_BOT = 1
tot_cars = (NUM_CARS_LEFT + NUM_CARS_RIGHT) * N_COLUMNS \
           + (NUM_CARS_BOT + NUM_CARS_TOP) * N_ROWS

def gen_edges(col_num, row_num):
    """Generate the names of the outer edges in the traffic light grid network.

    Parameters
    ----------
    col_num : int
        number of columns in the traffic light grid
    row_num : int
        number of rows in the traffic light grid

    Returns
    -------
    list of str
        names of all the outer edges
    """
    edges = []
    for i in range(col_num):
        edges += ['left' + str(row_num) + '_' + str(i)]
        edges += ['right' + '0' + '_' + str(i)]

    # build the left and then the right edges
    for i in range(row_num):
        edges += ['bot' + str(i) + '_' + '0']
        edges += ['top' + str(i) + '_' + str(col_num)]

    return edges


def get_inflow_params(col_num, row_num, additional_net_params):
    """Define the network and initial params in the presence of inflows.

    Parameters
    ----------
    col_num : int
        number of columns in the traffic light grid
    row_num : int
        number of rows in the traffic light grid
    additional_net_params : dict
        network-specific parameters that are unique to the traffic light grid

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the network
    """
    initial = InitialConfig(
        spacing='custom', lanes_distribution=float('inf'), shuffle=True)

    inflow = InFlows()
    edge_humans_enter = 'right0_0'
    outer_edges = gen_edges(col_num, row_num)

    
    #Adding human driven vehicles.
    inflow.add(
        veh_type='idm',
        edge=edge_humans_enter,
        vehs_per_hour=FLOW_RATE,
        depart_lane='free',
        depart_speed=V_ENTER)
    """
    for i in range(len(outer_edges)):
        inflow.add(
            veh_type='idm',
            edge=outer_edges[i],
            #probability=0.25,
            vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE,
            departLane='random',
            departSpeed=10)
            #number =3,
            #color = 'white')
    """
    
    

    # adding a single emergency vehicle with the red color
    # Que. 1: how to make sure the new EV won't be added until the previous
    #      One leaves?
    # Que. 2: What should be the speed of EV? Better to find a reference
    # Que. 3: which lane should the EV enter? How to set it?
    # Que. 4: which mode of departure we should choose for EVs?


    #  Adding RL agent
    #  Que. 1: which mode of departure we should choose for RL?
    #  Que. 2: the speed of the RL.
    edge_EV_enter = 'right0_0'
    
    

    inflow.add(
        veh_type='jordan',
        edge=edge_EV_enter,
        #probability=0.25,
        vehs_per_hour=FLOW_RATE,
        departLane= 1, #"free",
        departSpeed=V_ENTER,
        begin=130,
        number = 1,
        name = 'jordan')
    
    inflow.add(
        veh_type='emergency',
        edge=edge_EV_enter,
        #probability=0.25,
        vehs_per_hour=FLOW_RATE,
        departLane= 0, #"free",
        departSpeed=V_MAX_EV,
        begin=155,
        number = 1,
        name = 'emergency')
        #color = 'green')


    
    
        
    
    net = NetParams(
        inflows=inflow,
        additional_params=additional_net_params)

    return initial, net

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

additional_env_params = {
        'target_velocity': 50,
        'switch_time': 3.0,
        'num_observed': 2,
        'discrete': False,
        'tl_type': 'controlled'
    }

additional_net_params = {
    'speed_limit': 35,
    'grid_array': grid_array,
    'horizontal_lanes': 2,
    'vertical_lanes': 2,
    'traffic_lights': True,
    "random_start": False
}

vehicles = VehicleParams()
vehicles.add(
    veh_id='idm',
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        minGap=2.5,
        accel=3.0,
        decel=7.5,  # avoid collisions at emergency stops
        max_speed=V_MAX_CARS,
        speed_mode="all_checks",
    ),
    #routing_controller=(GridRouter, {}),
    #color = 'white',
    num_vehicles=0)
    #num_vehicles=0)
    #num_vehicles=1)

vehicles.add(
    veh_id="jordan",
    acceleration_controller=(JordanControllerMulti, {}),
    #lane_change_controller=(SimLaneChangeController, {}),
    car_following_params=SumoCarFollowingParams(
        minGap=2.5,
        accel=3.0,
        decel=7.5,  # avoid collisions at emergency stops
        max_speed=V_MAX_CARS,
        speed_mode="all_checks",
    ),
    #routing_controller=(GridRouter, {}),
    #lane_change_params=SumoLaneChangeParams(
    #    lane_change_mode="strategic",
    #),
    #color = 'green',
    num_vehicles=0)

#  Que 1.: what I should see for car_following_params?
vehicles.add(
    veh_id='emergency',
    acceleration_controller=(SimCarFollowingController, {}),
    lane_change_controller=(SimLaneChangeController, {}),
    car_following_params=SumoCarFollowingParams(
        minGap=1.0,
        accel=5.0,
        decel=7.5,  # avoid collisions at emergency stops
        max_speed=V_MAX_EV,
        #speed_mode="all_checks",
    ),
    #lane_change_params=SumoLaneChangeParams(
    #    lane_change_mode=1621, # according to this tutorial, 1621 means having all lang
    #),
    #routing_controller=(GridRouter, {}),
    #color = 'red',
    num_vehicles=0)

#  Add RL controlled vehicles. the settings for car_following_params seems 
#  to be arbitrary.
#  Que 1. Should I add lane change controller?
#  Que 1+. More fundamental question: is lane change a good or a bad behavior
#          if changing lane can reduce the travel time of EVs?
"""
vehicles.add(
    veh_id='rl',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode="obey_safe_speed",
        decel=1.5,
    ),
    num_vehicles=1)
"""



#  Add traffic lights
tl_logic = TrafficLightParams(baseline=False)
phases = [{
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    #"state": "GrGrGrGrGrGr"
    "state": "GgrrGgrr"
}, {
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    #"state": "yryryryryryr"
    "state":"yyrryyrr"
}, {
    "duration": "31",
    "minDur": "8",
    "maxDur": "45",
    #"state": "rGrGrGrGrGrG"
    "state":"rrGgrrGg"
}, {
    "duration": "6",
    "minDur": "3",
    "maxDur": "6",
    #"state": "ryryryryryry"
    "state":"rryyrryy"
}]
tl_logic.add("center0", phases=phases, programID=1)
tl_logic.add("center1", phases=phases, programID=1)

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS.copy())

if USE_INFLOWS:
    initial_config, net_params = get_inflow_params(
        col_num=N_COLUMNS,
        row_num=N_ROWS,
        additional_net_params=additional_net_params)

#sim_params = SumoParams(sim_step=0.1, render=True)
sim_params = SumoParams(sim_step=0.5, render=False,restart_instance=True)


network = TrafficLightGridNetwork(
        name="grid-intersection",
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=tl_logic)

env = AccelEnv(env_params, sim_params, network)

exp =  Experiment(env)

exp.run(1,600)