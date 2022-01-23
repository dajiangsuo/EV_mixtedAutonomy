"""Contains an experiment class for running simulations."""

import logging
import datetime
import numpy as np
import time
import os
from random import randint

from flow.core.util import emission_to_csv


class Experiment:
    """
    Class for systematically running simulations in any supported simulator.

    This class acts as a runner for a network and environment. In order to use
    it to run an network and environment in the absence of a method specifying
    the actions of RL agents in the network, type the following:

        >>> from flow.envs import Env
        >>> env = Env(...)
        >>> exp = Experiment(env)  # for some env and network
        >>> exp.run(num_runs=1, num_steps=1000)

    If you wish to specify the actions of RL agents in the network, this may be
    done as follows:

        >>> rl_actions = lambda state: 0  # replace with something appropriate
        >>> exp.run(num_runs=1, num_steps=1000, rl_actions=rl_actions)

    Finally, if you would like to like to plot and visualize your results, this
    class can generate csv files from emission files produced by sumo. These
    files will contain the speeds, positions, edges, etc... of every vehicle
    in the network at every time step.

    In order to ensure that the simulator constructs an emission file, set the
    ``emission_path`` attribute in ``SimParams`` to some path.

        >>> from flow.core.params import SimParams
        >>> sim_params = SimParams(emission_path="./data")

    Once you have included this in your environment, run your Experiment object
    as follows:

        >>> exp.run(num_runs=1, num_steps=1000, convert_to_csv=True)

    After the experiment is complete, look at the "./data" directory. There
    will be two files, one with the suffix .xml and another with the suffix
    .csv. The latter should be easily interpretable from any csv reader (e.g.
    Excel), and can be parsed using tools such as numpy and pandas.

    Attributes
    ----------
    env : flow.envs.Env
        the environment object the simulator will run
    """

    def __init__(self, env):
        """Instantiate Experiment."""
        self.env = env

        logging.info("Starting experiment {} at {}".format(
            env.network.name, str(datetime.datetime.utcnow())))

        logging.info("Initializing environment.")

    # this version of run function is only for generating results (ev travel) time for the Jordan controller case
    # For normal use comment this one and uncomment the original version below.
    """
    def run(self, num_runs, num_steps, rl_actions=None, output_to_terminal=True, convert_to_csv=False):
        Run the given network for a set number of runs and steps per run.

        Parameters
        ----------
        num_runs : int
            number of runs the experiment should perform
        num_steps : int
            number of steps to be performs in each run of the experiment
        rl_actions : method, optional
            maps states to actions to be performed by the RL agents (if
            there are any)
        convert_to_csv : bool
            Specifies whether to convert the emission file created by sumo
            into a csv file

        Returns
        -------
        info_dict : dict
            contains returns, average speed per step
        
        # raise an error if convert_to_csv is set to True but no emission
        # file will be generated, to avoid getting an error at the end of the
        # simulation
        # set TEST_RUN to True only if you want to collect travel times of EV and RL 
        TEST_RUN = True
        

        if convert_to_csv and self.env.sim_params.emission_path is None:
            raise ValueError(
                'The experiment was run with convert_to_csv set '
                'to True, but no emission file will be generated. If you wish '
                'to generate an emission file, you should set the parameter '
                'emission_path in the simulation parameters (SumoParams or '
                'AimsunParams) to the path of the folder where emissions '
                'output should be generated. If you do not wish to generate '
                'emissions, set the convert_to_csv parameter to False.')

        info_dict = {}
        if rl_actions is None:
            def rl_actions(*_):
                return None


        # collecting experiment results, ret = return
        # reward
        overall_return_all_runs = []
        mean_return_all_runs = []
        per_step_return_all_runs = []

        # speed
        per_step_speed_all_runs = []
        mean_speed_over_all_runs= []
        std_speed_over_all_runs = []

        # throughput
        inflow_over_all_runs = []
        outflow_over_all_runs = []

        # start collecting ev and jordan travel times
        if TEST_RUN:
            f = open("travel_time_data.csv", "a")
            for rl_enter in range(30,151):
                for ev_enter in range(rl_enter+1, rl_enter+30):
                    rl_travel_time = 0
                    ev_travel_time = 0 
                    state = self.env.reset()
                    
                    for _ in range(num_steps):
                        vehicles = self.env.unwrapped.k.vehicle
                        veh_ids = vehicles.get_ids()

                        for id in veh_ids:
                            if id.startswith("jordan"):
                                rl_edge = vehicles.get_edge(id)
                                # only measure travel time till it reaches the start of second intersection
                                if not rl_edge.startswith(":center1"):
                                    rl_travel_time += 0.5 # since we simulate in 0.5 steps
                            elif id.startswith("emergency"):
                                ev_edge = vehicles.get_edge(id)
                                # only measure travel time till it reaches the start of second intersection
                                if not ev_edge.startswith(":center1"):
                                    ev_travel_time += 0.5 # since we simulate in 0.5 steps

                        
                        state, reward, done, _ = self.env.step(rl_actions(state))
                        #state, reward, done, _ = env.step(action)
                        
                        
                        if done:
                            break
                
                    print("Jordan Enter: " + str(rl_enter) + " EV Enter: " + str(ev_enter) \
                        + " Jordan Time: " + str(rl_travel_time) + " EV Time: " + str(ev_travel_time))
                    write_string = str(rl_enter) + "," + str(ev_enter) + "," + str(rl_travel_time) + "," + str(ev_travel_time) + "\n"
                    f.write(write_string)
            f.close()
            exit(0)
        # for each run
        for i in range(num_runs):
            logging.info("Run #" + str(i+1))
            state = self.env.reset()

            # reward
            overall_return_one_run = 0
            per_step_return_one_run = []

            # speed
            per_step_speed_one_run = np.zeros(num_steps)

            # for each step
            for j in range(num_steps):

                # get the states, rewards, etc
                state, reward, done, _ = self.env.step(rl_actions(state))

                # store the returns
                overall_return_one_run += reward
                per_step_return_one_run.append(reward)

                # store the averaged speed of all vehicles at this step
                per_step_speed_one_run[j] = np.mean(
                    self.env.k.vehicle.get_speed(self.env.k.vehicle.get_ids()))

                if done:
                    break

            # reward
            overall_return_all_runs.append(overall_return_one_run)
            mean_return_all_runs.append(np.mean(per_step_return_one_run))
            per_step_return_all_runs.append(per_step_return_one_run)

            # speed
            per_step_speed_all_runs.append(per_step_speed_one_run)
            mean_speed_over_all_runs.append(np.mean(per_step_speed_one_run))
            std_speed_over_all_runs.append(np.std(per_step_speed_one_run))

            # get the outflows and inflows for the past 500 seconds, if the simulation is less than
            # 500 seconds then this will get all inflows (the number of vehicles entering the network)
            # and outflows (the number of vehicles leaving the network)
            inflow_over_all_runs.append(self.env.k.vehicle.get_inflow_rate(int(500)))
            outflow_over_all_runs.append(self.env.k.vehicle.get_outflow_rate(int(500)))

            # compute the throughput efficiency
            if np.all(np.array(inflow_over_all_runs) > 1e-5):
                throughput_over_all_runs = [
                    x / y for x, y in zip(outflow_over_all_runs, inflow_over_all_runs)]
            else:
                throughput_over_all_runs = [0] * len(inflow_over_all_runs)

        info_dict["overall_return_all_runs"] = overall_return_all_runs
        info_dict["mean_return_all_runs"] = mean_return_all_runs
        info_dict["per_step_return_all_runs"] = per_step_return_all_runs
        info_dict["per_step_speed_all_runs"] = per_step_speed_all_runs
        info_dict["mean_ret_all"] = np.mean(overall_return_all_runs)
        info_dict["std_ret_all"] = np.std(overall_return_all_runs)

        info_dict["mean_inflows"] = np.mean(inflow_over_all_runs)
        info_dict["mean_outflows"] = np.mean(outflow_over_all_runs)

        info_dict["max_spd_all"]  = np.max(mean_speed_over_all_runs)
        info_dict["min_spd_all"]  = np.min(mean_speed_over_all_runs)
        info_dict["mean_spd_all"] = np.mean(mean_speed_over_all_runs)
        info_dict["std_spd_all"]  = np.std(mean_speed_over_all_runs)
        info_dict["max_tpt_all"]  = np.max(throughput_over_all_runs)
        info_dict["min_tpt_all"]  = np.min(throughput_over_all_runs)
        info_dict["mean_tpt_all"] = np.mean(throughput_over_all_runs)
        info_dict["std_tpt_all"]  = np.std(throughput_over_all_runs)


        if output_to_terminal:
            print("Round {0} -- Return: {1}".format(i+1, overall_return_one_run))
            print("Return: {} (avg), {} (std)".format(
                info_dict["mean_ret_all"], info_dict["std_ret_all"]))

            print("Speed (m/s): {} (avg), {} (std)".format(
                info_dict["mean_spd_all"], info_dict["std_spd_all"]))

            print("Throughput (veh/hr): {} (avg), {} (std)".format(
                info_dict["mean_tpt_all"], info_dict["std_tpt_all"]))

        self.env.terminate()

        if convert_to_csv:
            # wait a short period of time to ensure the xml file is readable
            time.sleep(0.1)

            # collect the location of the emission file
            dir_path = self.env.sim_params.emission_path
            emission_filename = "{0}-emission.xml".format(self.env.network.name)
            emission_path = os.path.join(dir_path, emission_filename)

            # convert the emission file into a csv
            emission_to_csv(emission_path)

            # Delete the .xml version of the emission file.
            os.remove(emission_path)

        return info_dict
        """

    #Note: uncomment this for normal experiment
    
    def run(self, num_runs, num_steps, rl_actions=None, output_to_terminal=True, convert_to_csv=False):
        """
        Run the given network for a set number of runs and steps per run.

        Parameters
        ----------
        num_runs : int
            number of runs the experiment should perform
        num_steps : int
            number of steps to be performs in each run of the experiment
        rl_actions : method, optional
            maps states to actions to be performed by the RL agents (if
            there are any)
        convert_to_csv : bool
            Specifies whether to convert the emission file created by sumo
            into a csv file

        Returns
        -------
        info_dict : dict
            contains returns, average speed per step
        """
        
        # raise an error if convert_to_csv is set to True but no emission
        # file will be generated, to avoid getting an error at the end of the
        # simulation
        if convert_to_csv and self.env.sim_params.emission_path is None:
            raise ValueError(
                'The experiment was run with convert_to_csv set '
                'to True, but no emission file will be generated. If you wish '
                'to generate an emission file, you should set the parameter '
                'emission_path in the simulation parameters (SumoParams or '
                'AimsunParams) to the path of the folder where emissions '
                'output should be generated. If you do not wish to generate '
                'emissions, set the convert_to_csv parameter to False.')

        info_dict = {}
        if rl_actions is None:
            def rl_actions(*_):
                return None

        # collecting experiment results, ret = return
        # reward
        overall_return_all_runs = []
        mean_return_all_runs = []
        per_step_return_all_runs = []

        # speed
        per_step_speed_all_runs = []
        mean_speed_over_all_runs= []
        std_speed_over_all_runs = []

        # throughput
        inflow_over_all_runs = []
        outflow_over_all_runs = []

        # for each run
        for i in range(num_runs):
            logging.info("Run #" + str(i+1))
            state = self.env.reset()

            # reward
            overall_return_one_run = 0
            per_step_return_one_run = []

            # speed
            per_step_speed_one_run = np.zeros(num_steps)

            # for each step
            for j in range(num_steps):

                # get the states, rewards, etc
                state, reward, done, _ = self.env.step(rl_actions(state))

                # store the returns
                overall_return_one_run += reward
                per_step_return_one_run.append(reward)

                # store the averaged speed of all vehicles at this step
                per_step_speed_one_run[j] = np.mean(
                    self.env.k.vehicle.get_speed(self.env.k.vehicle.get_ids()))

                if done:
                    break

            # reward
            overall_return_all_runs.append(overall_return_one_run)
            mean_return_all_runs.append(np.mean(per_step_return_one_run))
            per_step_return_all_runs.append(per_step_return_one_run)

            # speed
            per_step_speed_all_runs.append(per_step_speed_one_run)
            mean_speed_over_all_runs.append(np.mean(per_step_speed_one_run))
            std_speed_over_all_runs.append(np.std(per_step_speed_one_run))

            # get the outflows and inflows for the past 500 seconds, if the simulation is less than
            # 500 seconds then this will get all inflows (the number of vehicles entering the network)
            # and outflows (the number of vehicles leaving the network)
            inflow_over_all_runs.append(self.env.k.vehicle.get_inflow_rate(int(500)))
            outflow_over_all_runs.append(self.env.k.vehicle.get_outflow_rate(int(500)))

            # compute the throughput efficiency
            if np.all(np.array(inflow_over_all_runs) > 1e-5):
                throughput_over_all_runs = [
                    x / y for x, y in zip(outflow_over_all_runs, inflow_over_all_runs)]
            else:
                throughput_over_all_runs = [0] * len(inflow_over_all_runs)

        info_dict["overall_return_all_runs"] = overall_return_all_runs
        info_dict["mean_return_all_runs"] = mean_return_all_runs
        info_dict["per_step_return_all_runs"] = per_step_return_all_runs
        info_dict["per_step_speed_all_runs"] = per_step_speed_all_runs
        info_dict["mean_ret_all"] = np.mean(overall_return_all_runs)
        info_dict["std_ret_all"] = np.std(overall_return_all_runs)

        info_dict["mean_inflows"] = np.mean(inflow_over_all_runs)
        info_dict["mean_outflows"] = np.mean(outflow_over_all_runs)

        info_dict["max_spd_all"]  = np.max(mean_speed_over_all_runs)
        info_dict["min_spd_all"]  = np.min(mean_speed_over_all_runs)
        info_dict["mean_spd_all"] = np.mean(mean_speed_over_all_runs)
        info_dict["std_spd_all"]  = np.std(mean_speed_over_all_runs)
        info_dict["max_tpt_all"]  = np.max(throughput_over_all_runs)
        info_dict["min_tpt_all"]  = np.min(throughput_over_all_runs)
        info_dict["mean_tpt_all"] = np.mean(throughput_over_all_runs)
        info_dict["std_tpt_all"]  = np.std(throughput_over_all_runs)


        if output_to_terminal:
            print("Round {0} -- Return: {1}".format(i+1, overall_return_one_run))
            print("Return: {} (avg), {} (std)".format(
                info_dict["mean_ret_all"], info_dict["std_ret_all"]))

            print("Speed (m/s): {} (avg), {} (std)".format(
                info_dict["mean_spd_all"], info_dict["std_spd_all"]))

            print("Throughput (veh/hr): {} (avg), {} (std)".format(
                info_dict["mean_tpt_all"], info_dict["std_tpt_all"]))

        self.env.terminate()

        if convert_to_csv:
            # wait a short period of time to ensure the xml file is readable
            time.sleep(0.1)

            # collect the location of the emission file
            dir_path = self.env.sim_params.emission_path
            emission_filename = "{0}-emission.xml".format(self.env.network.name)
            emission_path = os.path.join(dir_path, emission_filename)

            # convert the emission file into a csv
            emission_to_csv(emission_path)

            # Delete the .xml version of the emission file.
            os.remove(emission_path)

        return info_dict
    

    """
    def run(self, num_runs, num_steps, rl_actions=None, output_to_terminal=True, convert_to_csv=False):
        
        
        # raise an error if convert_to_csv is set to True but no emission
        # file will be generated, to avoid getting an error at the end of the
        # simulation
        # set TEST_RUN to True only if you want to collect travel times of EV and RL 
        TEST_RUN = True
        

        if convert_to_csv and self.env.sim_params.emission_path is None:
            raise ValueError(
                'The experiment was run with convert_to_csv set '
                'to True, but no emission file will be generated. If you wish '
                'to generate an emission file, you should set the parameter '
                'emission_path in the simulation parameters (SumoParams or '
                'AimsunParams) to the path of the folder where emissions '
                'output should be generated. If you do not wish to generate '
                'emissions, set the convert_to_csv parameter to False.')

        info_dict = {}
        if rl_actions is None:
            def rl_actions(*_):
                return None


        # collecting experiment results, ret = return
        # reward
        overall_return_all_runs = []
        mean_return_all_runs = []
        per_step_return_all_runs = []

        # speed
        per_step_speed_all_runs = []
        mean_speed_over_all_runs= []
        std_speed_over_all_runs = []

        # throughput
        inflow_over_all_runs = []
        outflow_over_all_runs = []

        # start collecting ev and jordan travel times
        if TEST_RUN:
            f11 = open("one-one.csv", "a")
            f22 = open("two-two.csv", "a")
            f21 = open("two-one.csv", "a")
            rl_one_ev_one_queues = []
            rl_two_ev_one_queues = []
            rl_two_ev_two_queues = []
            for ev_enter in range(1000):
                rl_enter = randint(37,112)
                ev_enter = rl_enter + randint(1,30)

                rl_travel_time = 0
                ev_travel_time = 0

                rl_queue_length = -1
                ev_queue_length = -1

                rl_one = 0
                rl_two = 0
                ev_one = 0
                ev_two = 0

                state = self.env.reset(TEST_RUN, ev_enter, rl_enter)
                #state = env.reset(TEST_RUN, ev_enter, rl_enter)
                    
                for _ in range(num_steps):
                    vehicles = self.env.unwrapped.k.vehicle
                    veh_ids = vehicles.get_ids()

                    for id in veh_ids:
                        if id.startswith("jordan"):
                            rl_edge = vehicles.get_edge(id)
                            rl_lane = vehicles.get_lane(id)
                            rl_edge_lane = rl_edge + "_" + str(rl_lane)
                            rl_speed = vehicles.get_speed(id)
                            rl_lead = vehicles.get_leader(id)
                            rl_lead_edge = vehicles.get_edge(rl_lead)
                            rl_lead_lane = vehicles.get_lane(rl_lead)
                            rl_lead_edge_lane = rl_lead_edge + "_" + str(rl_lead_lane)

                            if rl_edge_lane == "right0_0_1":
                                if rl_lead_edge_lane != "right0_0_1" and rl_speed < 0.3 and not rl_one:
                                    # rl vehicle is the first in the queue of the first intersection
                                    rl_queue_length = 0
                                    rl_one = 1
                                elif rl_lead_edge_lane == "right0_0_1" and rl_speed < 0.3 and not rl_one:
                                    if not rl_one:
                                        for sub_id in veh_ids:
                                            if vehicles.get_edge(sub_id) == "right0_0" and vehicles.get_lane(sub_id) == 1 and  vehicles.get_speed(sub_id) < 0.3:
                                            # rl vehicle is not the first in the queue of the first intersection
                                                rl_queue_length += 1
                                                rl_one = 1
                            elif rl_edge_lane == "right1_0_1":
                                if rl_lead_edge_lane != "right1_0_1" and rl_speed < 0.3 and not rl_two:
                                    # rl vehicle is the first in the queue of the second intersection
                                    rl_queue_length = 0
                                    rl_two = 1
                                    
                                elif rl_lead_edge_lane == "right1_0_1" and rl_speed < 0.3:
                                    if not rl_two:
                                        for sub_id in veh_ids:
                                            if vehicles.get_edge(sub_id) == "right1_0" and vehicles.get_lane(sub_id) == 1 and vehicles.get_speed(sub_id) < 0.3:
                                            # rl vehicle is not the first in the queue of the second intersection
                                                rl_queue_length += 1
                                                rl_two = 1
                                                
                        # only measure travel time till it reaches the start of second intersection
                            if not rl_edge.startswith(":center1"):
                                rl_travel_time += 0.5 # since we simulate in 0.5 steps
                            


                        elif id.startswith("emergency"):
                            ev_edge = vehicles.get_edge(id)
                            ev_lane = vehicles.get_lane(id)
                            ev_edge_lane = ev_edge + "_" + str(ev_lane)
                            ev_speed = vehicles.get_speed(id)
                            ev_lead = vehicles.get_leader(id)
                            ev_lead_edge = vehicles.get_edge(ev_lead)
                            ev_lead_lane = vehicles.get_lane(ev_lead)
                            ev_lead_edge_lane = ev_lead_edge + "_" + str(ev_lead_lane)

                            if ev_edge_lane == "right0_0_0":
                                if ev_lead_edge_lane != "right0_0_0" and ev_speed < 0.3 and not ev_one:
                                    # EMS vehicle is the first in the queue of the first intersection
                                    ev_queue_length = 0
                                    ev_one = 1
                                
                                elif ev_lead_edge_lane == "right0_0_0" and ev_speed < 0.3:
                                    if not ev_one:
                                        for sub_id in veh_ids:
                                            if vehicles.get_edge(sub_id) == "right0_0" and vehicles.get_lane(sub_id) == 0 and vehicles.get_speed(sub_id) < 0.3:
                                                # rl vehicle is not the first in the queue of the first intersection
                                                ev_queue_length += 1
                                                ev_one = 1

                            elif ev_edge_lane == "right1_0_0":
                                if ev_lead_edge_lane != "right1_0_0" and ev_speed < 0.3 and not ev_two:
                                    # rl vehicle is the first in the queue of the second intersection
                                    ev_queue_length = 0
                                    ev_two = 1
                                elif ev_lead_edge_lane == "right1_0_0" and ev_speed < 0.3:
                                    if not ev_two:
                                        for sub_id in veh_ids:
                                            if vehicles.get_edge(sub_id) == "right1_0" and vehicles.get_lane(sub_id) == 0 and vehicles.get_speed(sub_id) < 0.3:
                                                # rl vehicle is not the first in the queue of the second intersection
                                                ev_queue_length += 1
                                                ev_two = 1

                            # only measure travel time till it reaches the start of second intersection
                            if not ev_edge.startswith(":center1"):
                                ev_travel_time += 0.5 # since we simulate in 0.5 steps
                                

                        
                    state, reward, done, _ = self.env.step(rl_actions(state))
                    #state, reward, done, _ = env.step(action)
                        
                        
                    if done:
                        break

                if (rl_one or rl_two) and (ev_one or ev_two):
                    if rl_one and ev_one:
                        if (rl_queue_length,ev_queue_length) not in rl_one_ev_one_queues:
                            rl_one_ev_one_queues.append((rl_queue_length,ev_queue_length))
                            print(rl_one_ev_one_queues)
                            print("rl_one_ev_one_queues: " + str(len(rl_one_ev_one_queues)))
                            print("RL Enter: " + str(rl_enter) + " EV Enter: " + str(ev_enter) + " RL queue: " + str(rl_queue_length) + " EV queue: " + str(ev_queue_length) \
                            + " RL one: " + str(rl_one) + " RL two: " + str(rl_two) + " EV one: " + str(ev_one)  + " EV two: " + str(ev_two)\
                            +   " RL Time: " + str(rl_travel_time) + " EV Time: " + str(ev_travel_time))
                            write_string = str(rl_enter) + "," + str(ev_enter) + "," + str(rl_queue_length) + "," + str(ev_queue_length) + "," + str(rl_one) + "," + str(rl_two) + "," \
                            + str(ev_one) + "," + str(ev_two) + "," + str(rl_travel_time) + "," + str(ev_travel_time) + "\n"
                            f11.write(write_string)

                    elif rl_two and ev_two:
                        if (rl_queue_length,ev_queue_length) not in rl_two_ev_two_queues:
                            rl_two_ev_two_queues.append((rl_queue_length,ev_queue_length))
                            print(rl_two_ev_two_queues)
                            print("rl_two_ev_two_queues: " + str(len(rl_two_ev_two_queues)))
                            print("RL Enter: " + str(rl_enter) + " EV Enter: " + str(ev_enter) + " RL queue: " + str(rl_queue_length) + " EV queue: " + str(ev_queue_length) \
                            + " RL one: " + str(rl_one) + " RL two: " + str(rl_two) + " EV one: " + str(ev_one)  + " EV two: " + str(ev_two)\
                            +   " RL Time: " + str(rl_travel_time) + " EV Time: " + str(ev_travel_time))
                            write_string = str(rl_enter) + "," + str(ev_enter) + "," + str(rl_queue_length) + "," + str(ev_queue_length) + "," + str(rl_one) + "," + str(rl_two) + "," \
                            + str(ev_one) + "," + str(ev_two) + "," + str(rl_travel_time) + "," + str(ev_travel_time) + "\n"
                            f22.write(write_string)

                    elif rl_two and ev_one:
                        if (rl_queue_length,ev_queue_length) not in rl_two_ev_one_queues:
                            rl_two_ev_one_queues.append((rl_queue_length,ev_queue_length))
                            print(rl_two_ev_one_queues)
                            print("rl_two_ev_one_queues: " + str(len(rl_two_ev_one_queues)))
                            print("RL Enter: " + str(rl_enter) + " EV Enter: " + str(ev_enter) + " RL queue: " + str(rl_queue_length) + " EV queue: " + str(ev_queue_length) \
                            + " RL one: " + str(rl_one) + " RL two: " + str(rl_two) + " EV one: " + str(ev_one)  + " EV two: " + str(ev_two)\
                            +   " RL Time: " + str(rl_travel_time) + " EV Time: " + str(ev_travel_time))
                            write_string = str(rl_enter) + "," + str(ev_enter) + "," + str(rl_queue_length) + "," + str(ev_queue_length) + "," + str(rl_one) + "," + str(rl_two) + "," \
                            + str(ev_one) + "," + str(ev_two) + "," + str(rl_travel_time) + "," + str(ev_travel_time) + "\n"
                            f21.write(write_string)



            f11.close()
            f22.close()
            f21.close()
            exit(0)        
            
        # for each run
        for i in range(num_runs):
            logging.info("Run #" + str(i+1))
            state = self.env.reset()

            # reward
            overall_return_one_run = 0
            per_step_return_one_run = []

            # speed
            per_step_speed_one_run = np.zeros(num_steps)

            # for each step
            for j in range(num_steps):

                # get the states, rewards, etc
                state, reward, done, _ = self.env.step(rl_actions(state))

                # store the returns
                overall_return_one_run += reward
                per_step_return_one_run.append(reward)

                # store the averaged speed of all vehicles at this step
                per_step_speed_one_run[j] = np.mean(
                    self.env.k.vehicle.get_speed(self.env.k.vehicle.get_ids()))

                if done:
                    break

            # reward
            overall_return_all_runs.append(overall_return_one_run)
            mean_return_all_runs.append(np.mean(per_step_return_one_run))
            per_step_return_all_runs.append(per_step_return_one_run)

            # speed
            per_step_speed_all_runs.append(per_step_speed_one_run)
            mean_speed_over_all_runs.append(np.mean(per_step_speed_one_run))
            std_speed_over_all_runs.append(np.std(per_step_speed_one_run))

            # get the outflows and inflows for the past 500 seconds, if the simulation is less than
            # 500 seconds then this will get all inflows (the number of vehicles entering the network)
            # and outflows (the number of vehicles leaving the network)
            inflow_over_all_runs.append(self.env.k.vehicle.get_inflow_rate(int(500)))
            outflow_over_all_runs.append(self.env.k.vehicle.get_outflow_rate(int(500)))

            # compute the throughput efficiency
            if np.all(np.array(inflow_over_all_runs) > 1e-5):
                throughput_over_all_runs = [
                    x / y for x, y in zip(outflow_over_all_runs, inflow_over_all_runs)]
            else:
                throughput_over_all_runs = [0] * len(inflow_over_all_runs)

        info_dict["overall_return_all_runs"] = overall_return_all_runs
        info_dict["mean_return_all_runs"] = mean_return_all_runs
        info_dict["per_step_return_all_runs"] = per_step_return_all_runs
        info_dict["per_step_speed_all_runs"] = per_step_speed_all_runs
        info_dict["mean_ret_all"] = np.mean(overall_return_all_runs)
        info_dict["std_ret_all"] = np.std(overall_return_all_runs)

        info_dict["mean_inflows"] = np.mean(inflow_over_all_runs)
        info_dict["mean_outflows"] = np.mean(outflow_over_all_runs)

        info_dict["max_spd_all"]  = np.max(mean_speed_over_all_runs)
        info_dict["min_spd_all"]  = np.min(mean_speed_over_all_runs)
        info_dict["mean_spd_all"] = np.mean(mean_speed_over_all_runs)
        info_dict["std_spd_all"]  = np.std(mean_speed_over_all_runs)
        info_dict["max_tpt_all"]  = np.max(throughput_over_all_runs)
        info_dict["min_tpt_all"]  = np.min(throughput_over_all_runs)
        info_dict["mean_tpt_all"] = np.mean(throughput_over_all_runs)
        info_dict["std_tpt_all"]  = np.std(throughput_over_all_runs)


        if output_to_terminal:
            print("Round {0} -- Return: {1}".format(i+1, overall_return_one_run))
            print("Return: {} (avg), {} (std)".format(
                info_dict["mean_ret_all"], info_dict["std_ret_all"]))

            print("Speed (m/s): {} (avg), {} (std)".format(
                info_dict["mean_spd_all"], info_dict["std_spd_all"]))

            print("Throughput (veh/hr): {} (avg), {} (std)".format(
                info_dict["mean_tpt_all"], info_dict["std_tpt_all"]))

        self.env.terminate()

        if convert_to_csv:
            # wait a short period of time to ensure the xml file is readable
            time.sleep(0.1)

            # collect the location of the emission file
            dir_path = self.env.sim_params.emission_path
            emission_filename = "{0}-emission.xml".format(self.env.network.name)
            emission_path = os.path.join(dir_path, emission_filename)

            # convert the emission file into a csv
            emission_to_csv(emission_path)

            # Delete the .xml version of the emission file.
            os.remove(emission_path)

        return info_dict"""