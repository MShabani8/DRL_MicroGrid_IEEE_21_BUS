import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gym
import gym.spaces as spaces
from components import *

# days range
DEFAULT_DAY0=0
DEFAULT_DAYN=1

# Balancing market prices
DEFAULT_DOWN_REG = np.genfromtxt("down_regulation.csv", delimiter=',', skip_header=1, usecols=[-1]) / 10
DEFAULT_UP_REG = np.genfromtxt("up_regulation.csv", delimiter=',', skip_header=1, usecols=[-1]) / 10
DEFAULT_TRANSFER_PRICE_IMPORT = 0.97
DEFAULT_TRANSFER_PRICE_EXPORT = 0.09

# Length of one episode
DEFAULT_ITERATIONS = 24

# Price responsive loads
DEFAULT_NUM_LOADS = 1875
DEFAULT_BASE_LOAD = np.array(
    [.4, .3,.2,.2,.2,.2,.3,.5,.6,.6,.5,.5,.5,.4,.4,.6,.8,1.4,1.2,.9,.8,.6,.5,.4])

DEFAULT_NUM_HEAT = 725
MIN_HEAT_LOAD = np.array(
    [.4, .6, .7, 1, 1.2, 1.4, 2.8, 6, .1, 1, .6, .1, .1, .1, .1, .3, .5, 4.8, 4.2, 3.9, 3.5, 3.2, .1, .3])

DEFAULT_MARKET_PRICE = 5.48
DEFAULT_PRICE_TIERS = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])

DEFAULT_NUM_CHPS = 3
DEFAULT_NUM_DGS = 3
DEFAULT_NUM_WTS = 2

# Battery characteristics (kwh)
DEFAULT_BAT_CAPACITY=2500 #multiplate battery should be implemented
DEFAULT_MAX_CHARGE=1250
DEFAULT_MAX_DISCHARGE=1250


MAX_R = 100

# Rendering lists
SOCS_RENDER = []
LOADS_RENDER = []
HEAT_RENDER = []
BATTERY_RENDER = []
CHP_POWER_RENDER = []
HEAT_POWER_RENDER = []
DG_POWER_RENDER = []
WT_POWER_RENDER = []
PRICE_RENDER = []
ENERGY_BOUGHT_RENDER = []
GRID_PRICES_BUY_RENDER = []
GRID_PRICES_SELL_RENDER = []
ENERGY_GENERATED_RENDER = []
TOTAL_CONSUMPTION_RENDER=[]
TOTAL_ENERGY_RENDER = []

MMBTU = 2

NUM_BOILER_ACTIONS = 5
NUM_CHP_ACTIONS = 5
NUM_DG_ACTIONS = 5
NUM_PRICE_ACTIONS = 5

ACTIONS = [[i, j, k, l, p, t] for i in range(NUM_CHP_ACTIONS) for j in range(NUM_BOILER_ACTIONS) for k in range(NUM_DG_ACTIONS) for l in range(NUM_PRICE_ACTIONS) for p in range(2) for t in range(2)]


class MicroGridEnv(gym.Env):
    def __init__(self,**kwargs):

        # parameters (we have to define it through kwargs because
        # of how Gym works...)
        self.iterations = kwargs.get("iterations", DEFAULT_ITERATIONS)
        self.num_loads = kwargs.get("num_loads", DEFAULT_NUM_LOADS)
        self.num_chps = kwargs.get("num_loads", DEFAULT_NUM_CHPS)
        self.num_dgs = kwargs.get("num_loads", DEFAULT_NUM_DGS)
        self.num_wts = kwargs.get("num_loads", DEFAULT_NUM_WTS)
        self.typical_load = kwargs.get("base_load", DEFAULT_BASE_LOAD)
        self.min_heat = kwargs.get("min_heat", MIN_HEAT_LOAD)
        self.market_price = kwargs.get("normal_price", DEFAULT_MARKET_PRICE)
        self.price_tiers = kwargs.get("price_tiers", DEFAULT_PRICE_TIERS)
        self.day0 = kwargs.get("day0", DEFAULT_DAY0)
        self.dayn = kwargs.get("dayn", self.day0+1)
        self.down_reg = kwargs.get("down_reg", DEFAULT_DOWN_REG)
        self.up_reg = kwargs.get("up_reg", DEFAULT_UP_REG)
        self.imp_fees = kwargs.get("imp_fees", DEFAULT_TRANSFER_PRICE_IMPORT)
        self.exp_fees = kwargs.get("exp_fees", DEFAULT_TRANSFER_PRICE_EXPORT)
        self.bat_capacity = kwargs.get("battery_capacity", DEFAULT_BAT_CAPACITY)
        self.max_discharge = kwargs.get("max_discharge", DEFAULT_MAX_DISCHARGE)
        self.max_charge = kwargs.get("max_charge", DEFAULT_MAX_CHARGE)

        # The current day: pick randomly
        # self.day = random.randint(self.day0, self.dayn-1)
        self.day = self.day0
        self.time_step = 0

        self.loads_parameters = []

        self.grid = Grid(down_reg=self.down_reg,up_reg=self.up_reg, exp_fees=self.exp_fees, imp_fees=self.imp_fees)
        self.battery = Battery(capacity=self.bat_capacity, useD=0.9, dissipation=0.001, rateC=0.9, maxDD=self.max_discharge, chargeE=self.max_charge)

        self.loads = [self._create_load(*self._create_load_parameters()) for _ in range(self.num_loads)]
        self.chps = [self._create_chp(*self._create_chp_parameters()) for _ in range(self.num_chps)]
        self.dgs = [self._create_dg(*self._create_dg_parameters()) for _ in range(self.num_dgs)]
        self.wts = [self._create_wind(*self._create_wind_parameters()) for _ in range(self.num_wts)]

        self.action_space_sep = spaces.Box(low=0, high=1, dtype=np.float32,
                                       shape=(24,))
        self.action_space = spaces.Discrete(2500)

        # Observations: A vector of loads +battery soc+ power generation + price + temperature + time of day
        self.observation_space = spaces.Box(low=-100, high=100, dtype=np.float32,
                                            shape=(7 + self.num_chps + self.num_dgs + self.num_wts,))

    def _create_load_parameters(self):
        """
        Initialize one load randomly,
        and return it.
        """
        # Hardcoded initialization values to create
        # bunch of different loads

        price_sensitivity = random.normalvariate(0.4, 0.3)
        max_v_load = random.normalvariate(0.4, 0.01)
        patience= int(random.normalvariate(10,6))
        return [price_sensitivity, max_v_load,patience]

    def _create_chp_parameters(self):
        
        efficiency = 0.8
        chp_production = 2.0
        fuel_cost = MMBTU # dollar per 1000 kg/h
        externalities = 2 * 10**(-3)
        emission_factor = 0.2
        max_thermal_power = 2500 # gigajouls
        max_boiler_fuel = 2000 # Range: 500 kg/h to 1500 kg/h
        max_chp_fuel = 2000 # Range: 800 kg/h to 1800 kg/h

        return [efficiency, externalities, chp_production,
                 fuel_cost, emission_factor, max_thermal_power, max_boiler_fuel, max_chp_fuel]

    def _create_dg_parameters(self):
        
        efficiency = 0.8
        externalities = 2 * 10**(-3)
        emission_factor = 0.2
        a_coefficient = 2.4 * 10**(-5)
        b_coefficient = 4.7 * 10**(-5)
        c_coefficient = 2.9 * 10**(-5)
        min_power_output = 50 # kwh
        max_power_output = 500 # kwh
        start_up_cost = 45
        max_start_up_count = 100
        max_ramp_up = 200 # unknown
        max_ramp_down = 300 # unknown
        
        return [efficiency, externalities,
                         emission_factor, max_ramp_up, max_ramp_down,
                         start_up_cost,a_coefficient, b_coefficient,
                         c_coefficient, min_power_output, max_power_output,
                         max_start_up_count]

    def _create_wind_parameters(self):
        
        max_electric_power = 500
        operation_maintance_cost = 0.01
        cut_in = 3.0
        cut_out = 25.0
        nominal = 9.0

        return [max_electric_power, operation_maintance_cost, cut_in, cut_out, nominal]

    def _create_load(self, price_sensitivity, max_v_load,patience):
        load = Load(price_sensitivity, base_load=self.typical_load, max_v_load=max_v_load, patience=patience)
        return load
    
    def _create_chp(self,efficiency, externalities, chp_production,
                 fuel_cost, emission_factor, max_thermal_power, max_boiler_fuel, max_chp_fuel):
        
        chp = CHP(efficiency, externalities, chp_production,
                 fuel_cost, emission_factor, max_thermal_power, max_boiler_fuel, max_chp_fuel)
        return chp
    
    def _create_dg(self, efficiency, externalities,
                         emission_factor, max_ramp_up, max_ramp_down,
                         start_up_cost,a_coefficient, b_coefficient,
                         c_coefficient, min_power_output, max_power_output,
                         max_start_up_count):
        
        dg = DieselGenerator(efficiency, externalities,
                         emission_factor, max_ramp_up, max_ramp_down,
                         start_up_cost,a_coefficient, b_coefficient,
                         c_coefficient, min_power_output, max_power_output,
                         max_start_up_count)
        return dg
    
    def _create_wind(self, max_electric_power, operation_maintance_cost, cut_in, cut_out, nominal):
        wt = WT(max_electric_power, operation_maintance_cost, 
                cut_in, cut_out, nominal, wind_data = WIND_SPEED_DATA)
        return wt

    def _build_state(self):
        """
        Return current state representation as one vector.
        Returns:
            state: 1D state vector, containing Loads, current battery soc, current power generation,
                   current temperature, current price and current time (hour) of day
        """

        # Scaling between 0 and 1
        # Minimum soc is -1

        loads = self.typical_load[(self.time_step) % 24]
        loads = (loads - min(self.typical_load)) / (max(self.typical_load) - min(self.typical_load))
        
        heats = self.min_heat[(self.time_step) % 24]
        heats = (heats - min(self.min_heat)) / (max(self.min_heat) - min(self.min_heat))
                        
        wind_speeds = np.array([wt.get_wind_speed(self.day*self.iterations+self.time_step) for wt in self.wts])
        
        power_gen_dgs = np.array([dg.current_power_gen() / (dg.max_power_output)  for dg in self.dgs])
              
        power_gen_chps = np.array([chp.current_power_gen() / (chp.max_power_output())  for chp in self.chps])
        
        price = self.grid.buy_prices[self.day*self.iterations+self.time_step]
        price = (price -
                 np.average(self.grid.buy_prices[self.day*self.iterations:self.day*self.iterations+self.iterations])) \
                / np.std(self.grid.buy_prices[self.day*self.iterations:self.day*self.iterations+self.iterations])

        price_grid_sell = self.grid.sell_prices[self.day*self.iterations+self.time_step]
        price_grid_sell = (price_grid_sell -
                 np.average(self.grid.sell_prices[self.day*self.iterations:self.day*self.iterations + self.iterations])) \
                / np.std(self.grid.sell_prices[self.day*self.iterations:self.day*self.iterations+self.iterations])

        high_price = min(self.high_price/4,1)

        time_step = (self.time_step)/(self.iterations-1)

        state = np.concatenate(([loads, heats, high_price, time_step, self.battery.SoC,
                         price, price_grid_sell], power_gen_chps, power_gen_dgs, wind_speeds))
        return state  

    def step(self, action):
        """
        Arguments:
            action: A list.

        Returns:
            state: Current state
            reward: How much reward was obtained on last action
            terminal: Boolean on if the game ended (maximum number of iterations)
            info: None (not used here)
        """
        if type(action) is not list:
            action = ACTIONS[action]

        self.grid.set_time(self.day*self.iterations + self.time_step)
        
        reward = 0
        max_chps_fuel = sum(chp.max_chp_fuel for chp in self.chps)
        max_boilers_fuel = sum(chp.max_boiler_fuel for chp in self.chps)
        max_dgs_power = sum(dg.max_power_output for dg in self.dgs)

        price_action = action[3]
        chp_action = action[0] * max_chps_fuel / NUM_CHP_ACTIONS
        boiler_action = action[1] * max_boilers_fuel / NUM_CHP_ACTIONS
        dg_action = action[2] * max_dgs_power / NUM_DG_ACTIONS
        
        
        self.high_price += price_action - 2
        if self.high_price > 4:
            price_action = 2
            self.high_price = 4

        energy_deficiency_action = action[4]
        energy_excess_action = action[5]
        
        # Get the energy generated by the DER
        available_energy = 0
        available_heat = 0
        total_heats = DEFAULT_NUM_HEAT * self.min_heat[self.time_step%24]
        for chp in self.chps:
            available_energy += chp.power_generated(chp_action)
            available_heat += chp.heat_generated(chp_action, boiler_action)
            if(chp_action > chp.max_chp_fuel):
                chp_action -= chp.max_chp_fuel
            else:
                chp_action = 0
            if(boiler_action > chp.max_boiler_fuel):
                boiler_action -= chp.max_boiler_fuel
            else:
                boiler_action = 0     
            
        for dg in self.dgs:
            if(dg_action < dg.power_generated(dg_action) * 2): 
                available_energy += dg.power_generated(dg_action)
                break
            else:
                available_energy += dg.power_generated(dg_action)
                dg_action = dg_action - dg.power_generated(dg_action)
                
        for wt in self.wts:
            available_energy += wt.power_generated(self.day*self.iterations + self.time_step)
        
        # print(self.dgs[0].power_generated(power = dg_action))
        
        cost = sum(chp.calculate_total_cost_boiler() for chp in self.chps) + sum(chp.calculate_total_cost_chp() for chp in self.chps) + sum(dg.calculate_total_cost() for dg in self.dgs)
        reward-= cost 

        for load in self.loads:
            load.react(price_tier=price_action, time_day=self.time_step%24)

        total_loads = sum([l.load() for l in self.loads])
        # print("Total loads",total_loads)
        # We fulfilled the load with the available energy.
        available_heat -= total_heats   
        available_energy -= total_loads
        
        if(available_heat < 0):
            reward -= available_heat * available_heat / (10 ** 4)
        else:
            reward += total_loads / (10 ** 2)    
        # Constraint of charging too high prices
        
        # We calculate the return based on the sale price.
        self.sale_price = self.price_tiers[price_action] + self.market_price

        # Division by 100 to transform from cents to dollars
        reward += total_loads * (self.sale_price) / 100

        # print("Available energy:", available_energy)
        if available_energy > 0:
            if energy_excess_action:
                available_energy = self.battery.charge(available_energy)
            reward -= available_energy * available_energy / (10 ** 5)
            self.energy_bought = 0

        else:
            if energy_deficiency_action:
                available_energy += self.battery.supply(-available_energy)
                # print("after energy was taken from battery", available_energy)
            self.energy_bought = -available_energy
            reward -= available_energy / 100
            reward += self.grid.buy(self.energy_bought) / 100
            self.energy_sold = 0

        # Proceed to next timestep.
        self.time_step += 1
        # Build up the representation of the current state (in the next timestep)
        state = self._build_state()


        terminal = self.time_step == self.iterations

        return state, reward/MAX_R , terminal

    def reset(self, day0, dayn, day=None):

        if day == None:
            self.day= random.randint(day0,dayn)
        else:
            self.day = day
        print("Day:", self.day)
        self.time_step = 0

        self.high_price = 0

        return self._build_state()

    def reset_all(self,day=None):

        if day == None:
            self.day= self.day0
        else:
            self.day = day
        print("Day:", self.day)
        self.time_step = 0
        self.battery.reset()
        self.high_price = 0
        self.loads.clear()
        self.loads = [self._create_load(*self._create_load_parameters()) for _ in range(self.num_loads)]


        return self._build_state()

    def render(self,name=''):
        LOADS_RENDER.append([l.load() for l in self.loads])
        HEAT_RENDER.append(DEFAULT_NUM_HEAT * self.min_heat[self.time_step%24])
        PRICE_RENDER.append(self.sale_price)
        BATTERY_RENDER.append(self.battery.SoC)
        CHP_POWER_RENDER.append(sum(chp.power_output for chp in self.chps))
        DG_POWER_RENDER.append(sum(dg.power_output for dg in self.dgs))
        WT_POWER_RENDER.append(sum(wt.power_output for wt in self.wts))
        TOTAL_ENERGY_RENDER.append(sum(wt.power_output for wt in self.wts) + sum(dg.power_output for dg in self.dgs) + sum(chp.power_output for chp in self.chps))
        HEAT_POWER_RENDER.append(sum(chp.heat_output for chp in self.chps))
        ENERGY_BOUGHT_RENDER.append(self.energy_bought)
        GRID_PRICES_BUY_RENDER.append(self.grid.buy_prices[self.day * self.iterations + self.time_step-1])
        GRID_PRICES_SELL_RENDER.append(self.grid.sell_prices[self.day * self.iterations + self.time_step-1])
        TOTAL_CONSUMPTION_RENDER.append(np.sum([l.load() for l in self.loads]))
        if self.time_step==self.iterations:
            fig=plt.figure()
            ax = plt.axes()
            ax.set_facecolor("silver")
            ax.yaxis.grid(True)
            plt.plot(PRICE_RENDER,color='k')
            plt.title("SALE PRICES")
            plt.xlabel("Time (h)")
            plt.ylabel("€ cents")
            plt.show()
            
            ax = plt.axes()
            ax.set_facecolor("silver")
            ax.set_xlabel("Time (h)")
            ax.yaxis.grid(True)
            plt.plot(np.array(BATTERY_RENDER),color='k')
            plt.title("ESS SOC")
            plt.xlabel("Time (h)")
            # ax4.set_ylabel("BATTERY SOC")
            plt.show()
            
            
            ax = plt.axes()
            ax.set_facecolor("silver")
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("kWh")
            ax.yaxis.grid(True)
            plt.plot(np.array(TOTAL_CONSUMPTION_RENDER), color='k')
            plt.title("Electric Demand")
            plt.xlabel("Time (h)")
            plt.ylabel("kW")
            plt.show()
            #
            #
            #
            ax = plt.axes()
            ax.set_facecolor("silver")
            ax.yaxis.grid(True)
            plt.plot(np.array(HEAT_RENDER),color='k')
            plt.title("Heat Demand")
            plt.xlabel("Time (h)")
            plt.ylabel("kW")
            plt.show()
            #
            #
            #
            ax = plt.axes()
            ax.set_facecolor("silver")
            ax.yaxis.grid(True)
            plt.plot(np.array(HEAT_POWER_RENDER),color='k')
            plt.title("CHP Heat GENERATED")
            plt.xlabel("Time (h)")
            plt.ylabel("kW")
            plt.show()
            #
            #
            #            
            ax = plt.axes()
            ax.set_facecolor("silver")
            ax.set_xlabel("Time (h)")
            ax.yaxis.grid(True)
            plt.plot(np.array(self.typical_load), color='k')
            plt.title("Expected Individual basic load (L_b)")
            plt.xlabel("Time (h)")
            plt.ylabel("kWh")
            plt.show()
            
            ax = plt.axes()
            ax.set_facecolor("silver")
            ax.set_ylabel("kW")
            ax.set_xlabel("Time (h)")
            ax.yaxis.grid(True)
            plt.boxplot(np.array(LOADS_RENDER).T)
            plt.title("Hourly residential loads")
            plt.xlabel("Time (h)")
            plt.show()
            #
            #
            #
            ax = plt.axes()
            ax.set_facecolor("silver")
            ax.yaxis.grid(True)
            plt.plot(np.array(CHP_POWER_RENDER),color='k')
            plt.title("CHP ENERGY GENERATED")
            plt.xlabel("Time (h)")
            plt.ylabel("kW")
            plt.show()
            #
            #
            #
            ax = plt.axes()
            ax.set_facecolor("silver")
            ax.yaxis.grid(True)
            plt.plot(np.array(DG_POWER_RENDER),color='k')
            plt.title("DG ENERGY GENERATED")
            plt.xlabel("Time (h)")
            plt.ylabel("kW")
            plt.show()
            #
            #
            #
            ax = plt.axes()
            ax.set_facecolor("silver")
            ax.yaxis.grid(True)
            plt.plot(np.array(WT_POWER_RENDER),color='k')
            plt.title("WT ENERGY GENERATED")
            plt.xlabel("Time (h)")
            plt.ylabel("kW")
            plt.show()
            #
            #
            #
            ax = plt.axes()
            ax.set_facecolor("silver")
            ax.yaxis.grid(True)
            plt.plot(np.array(TOTAL_ENERGY_RENDER),color='k')
            plt.title("TOTAL ENERGY GENERATED")
            plt.xlabel("Time (h)")
            plt.ylabel("kW")
            plt.show()
            #
            #
            #

            # np.save(name + 'Cost' + str(self.day) + '.npy', self.grid.total_cost(np.array(GRID_PRICES_RENDER),np.array(ENERGY_BOUGHT_RENDER)))
            SOCS_RENDER.clear()
            LOADS_RENDER.clear()
            PRICE_RENDER.clear()
            BATTERY_RENDER.clear()
            CHP_POWER_RENDER.clear()
            HEAT_RENDER.clear()
            DG_POWER_RENDER.clear()
            WT_POWER_RENDER.clear()
            GRID_PRICES_BUY_RENDER.clear()
            GRID_PRICES_SELL_RENDER.clear()
            ENERGY_BOUGHT_RENDER.clear()
            ENERGY_GENERATED_RENDER.clear()
            TOTAL_CONSUMPTION_RENDER.clear()
            TOTAL_ENERGY_RENDER.clear()
            HEAT_POWER_RENDER.clear()

    def close(self):
        """
        Nothing to be done here, but has to be defined
        """
        return

    def seedy(self, s):
        """
        Set the random seed for consistent experiments
        """
        random.seed(s)
        np.random.seed(s)


if __name__ == '__main__':
    env = MicroGridEnv()
    env.seedy(1)
    # Save the rewards in a list
    rewards = []
    # reset the environment to the initial state
    state = env.reset(day0 = 0, dayn= 1)
    # Call render to prepare the visualization
    i = 0
    # Interact with the environment (here we choose random actions) until the terminal state is reached
    while True:
        i+= 1
        i= i%5
        action=[i,5,5,1,1,1]

        print(action)

        state, reward, terminal = env.step(list(action))
        env.render()
        print(reward)
        rewards.append(reward)
        if terminal:
            break
    print("Total Reward:", sum(rewards))


    states = np.array(rewards)
    plt.plot(rewards)
    plt.title("rewards")
    plt.xlabel("Time")
    plt.ylabel("rewards")
    plt.show()

