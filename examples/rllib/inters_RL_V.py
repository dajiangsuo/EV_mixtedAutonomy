"""Traffic Light Grid example."""
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams, SumoLaneChangeParams
from flow.controllers import SimCarFollowingController, GridRouter, IDMController, RLController, SimLaneChangeController
from flow.envs import TrafficLightGridPOEnv
from flow.envs.rl_forEV_env import ADDITIONAL_ENV_PARAMS 
from flow.envs import AccelEnv_forEV
from flow.networks import TrafficLightGridNetwork

USE_INFLOWS = True
# time horizon of a single rollout
HORIZON = 200
# number of rollouts per training iteration
N_ROLLOUTS = 1
# number of parallel workers
N_CPUS = 3

EXP_NUM = 0


# inflow rate at the highway
FLOW_RATE = 400
# percent of autonomous vehicles
RL_PENETRATION = [0.1, 0.25, 0.33][EXP_NUM]
# num_rl term (see ADDITIONAL_ENV_PARAMs)
NUM_RL = [1, 13, 17][EXP_NUM]

V_ENTER = 15
INNER_LENGTH = 300
LONG_LENGTH = 100
SHORT_LENGTH = 300
N_ROWS = 1
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
    outer_edges = gen_edges(col_num, row_num)

    
    #Adding human driven vehicles.
    
    for i in range(len(outer_edges)):
        inflow.add(
            veh_type='idm',
            edge=outer_edges[i],
            #probability=0.25,
            vehs_per_hour=(1 - RL_PENETRATION) * FLOW_RATE,
            departLane='free',
            departSpeed=10)
            #number =3,
            #color = 'white')
            
    
    

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
        veh_type='rl',
        edge=edge_EV_enter,
        #probability=0.25,
        vehs_per_hour=RL_PENETRATION * FLOW_RATE,
        departLane= 1, #"free",
        departSpeed=10,
        number = 1,
        name = 'rl')
    
    
    inflow.add(
        veh_type='emergency',
        edge=edge_EV_enter,
        #probability=0.25,
        vehs_per_hour=RL_PENETRATION * FLOW_RATE,
        departLane= 0, #"free",
        departSpeed=10,
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
        decel=7.5,  # avoid collisions at emergency stops
        max_speed=V_ENTER,
        speed_mode="all_checks",
    ),
    #routing_controller=(GridRouter, {}),
    #color = 'white',
    num_vehicles=tot_cars)
    #num_vehicles=0)
    #num_vehicles=1)

vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    #lane_change_controller=(SimLaneChangeController, {}),
    car_following_params=SumoCarFollowingParams(
        minGap=2.5,
        decel=7.5,  # avoid collisions at emergency stops
        max_speed=V_ENTER,
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
        minGap=2.5,
        decel=7.5,  # avoid collisions at emergency stops
        max_speed=30,
        #speed_mode="all_checks",
    ),
    lane_change_params=SumoLaneChangeParams(
        lane_change_mode="strategic",
    ),
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

if USE_INFLOWS:
    initial_config, net_params = get_inflow_params(
        col_num=N_COLUMNS,
        row_num=N_ROWS,
        additional_net_params=additional_net_params)

flow_params = dict(
    # name of the experiment
    exp_tag='traffic_light_grid',

    # name of the flow environment the experiment is running on
    #env_name=TrafficLightGridPOEnv,
    env_name=AccelEnv_forEV,

    # name of the network class the experiment is running on
    network=TrafficLightGridNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=1,
        render=False,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        #additional_params=additional_env_params,
        #sims_per_step=5,
        #warmup_steps=0,
        additional_params=ADDITIONAL_ENV_PARAMS.copy(),
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component). This is
    # filled in by the setup_exps method below.
    net=net_params,

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig). This is filled in by the
    # setup_exps method below.
    initial=initial_config,


    # traffic lights to be introduced to specific nodes (see
    # flow.core.params.TrafficLightParams)
    tls=tl_logic,
)

import json

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

# number of parallel workers
N_CPUS = 2
# number of rollouts per training iteration
N_ROLLOUTS = 1

ray.shutdown()
ray.init(num_cpus=N_CPUS+1,object_store_memory=500 * 1024 * 1024)

# The algorithm or model to train. This may refer to "
#      "the name of a built-on algorithm (e.g. RLLib's DQN "
#      "or PPO), or a user-defined trainable function or "
#      "class registered in the tune registry.")
alg_run = "PPO"

agent_cls = get_agent_class(alg_run)
config = agent_cls._default_config.copy()
config["num_workers"] = N_CPUS - 1  # number of parallel workers
config["train_batch_size"] = HORIZON * N_ROLLOUTS  # batch size
config["gamma"] = 0.999  # discount rate
config["model"].update({"fcnet_hiddens": [16, 16]})  # size of hidden layers in network
config["use_gae"] = True  # using generalized advantage estimation
config["lambda"] = 0.97  
config["sgd_minibatch_size"] = min(16 * 1024, config["train_batch_size"])  # stochastic gradient descent
config["kl_target"] = 0.02  # target KL divergence
config["num_sgd_iter"] = 10  # number of SGD iterations
config["horizon"] = HORIZON  # rollout horizon

# save the flow params for replay
flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                       indent=4)  # generating a string version of flow_params
config['env_config']['flow_params'] = flow_json  # adding the flow_params to config dict
config['env_config']['run'] = alg_run

# Call the utility function make_create_env to be able to 
# register the Flow env for this experiment
create_env, gym_name = make_create_env(params=flow_params, version=0)

# Register as rllib env with Gym
register_env(gym_name, create_env)

trials = run_experiments({
    flow_params["exp_tag"]: {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": 20,  # number of iterations between checkpoints
        "checkpoint_at_end": True,  # generate a checkpoint at the end
        "max_failures": 999,
        "stop": {  # stopping conditions
            "training_iteration": 300,  # number of iterations to stop after
        },
    },
})

ray.shutdown()


