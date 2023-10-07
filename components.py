import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gym
import gym.spaces as spaces
import math

WIND_SPEED_DATA = np.genfromtxt("wind_speed.csv", delimiter=',', skip_header=1, usecols=[0]) 


class Battery:
    # Simulates the battery system of the microGrid
    def __init__(self, capacity, useD, dissipation, rateC, maxDD, chargeE):
        self.capacity = capacity  # full charge battery capacity
        self.useD = useD  # useful discharge coefficient
        self.dissipation = dissipation  # dissipation coefficient of the battery
        self.rateC = rateC  # charging rate
        self.maxDD = maxDD  # maximum power that the battery can deliver per timestep
        self.chargeE = chargeE  # max Energy given to the battery to charge
        self.RC = 0  # remaining capacity


    def charge(self, E):
        empty = self.capacity - self.RC
        if empty <= 0:
            return E
        else:
            self.RC += self.rateC * min(E,self.chargeE)
            leftover = self.RC - self.capacity + max(E-self.chargeE,0)
            self.RC = min(self.capacity, self.RC)
            return max(leftover, 0)

    def supply(self, E):
        remaining = self.RC
        self.RC -= min(E, remaining,self.maxDD)
        self.RC = max(self.RC, 0)
        return min(E, remaining,self.maxDD) * self.useD

    def dissipate(self):
        self.RC = self.RC * math.exp(- self.dissipation)

    @property
    def SoC(self):
        return self.RC / self.capacity

    def reset(self):
        self.RC=0

class CHP:
    def __init__(self, efficiency, externalities, chp_production,
                 fuel_cost, emission_factor, max_thermal_power, max_boiler_fuel, max_chp_fuel):
        
        self.max_boiler_fuel = max_boiler_fuel
        self.max_chp_fuel = max_chp_fuel

        self.boiler_heat = 0
        self.chp_heat = 0
        self.consumed_fuel_chp = 0
        self.consumed_fuel_boiler = 0
        
        self.efficiency = efficiency
        self.chp_production = chp_production
        self.fuel_cost = fuel_cost
        self.externalities = externalities
        self.emission_factor = emission_factor
        self.max_thermal_power = max_thermal_power
        self.power_output = 0
        self.heat_output = 0

    def max_power_output(self):
        return (self.max_chp_fuel) / (1 + self.chp_production)
    
    def calculate_fuel_cost_chp(self):
        return (self.consumed_fuel_chp/1000) * self.fuel_cost

    def calculate_emission_cost_chp(self):
        return (self.consumed_fuel_chp) * self.emission_factor * self.externalities

    def calculate_fuel_cost_boiler(self):
        return (self.consumed_fuel_boiler/1000) * self.fuel_cost

    def calculate_emission_cost_boiler(self):
        return (self.consumed_fuel_boiler) * self.emission_factor * self.externalities

    def calculate_total_cost_chp(self):
        return (self.calculate_fuel_cost_chp() + self.calculate_emission_cost_chp())
    
    def calculate_total_cost_boiler(self):
        return (self.calculate_fuel_cost_boiler() + self.calculate_emission_cost_boiler())
    
    def heat_generated(self, consumed_fuel_chp, consumed_fuel_boiler):
        self.boiler_heat = consumed_fuel_boiler * self.efficiency 
        self.chp_heat = (consumed_fuel_chp * self.chp_production) / (1 + self.chp_production)
        self.heat_output = self.boiler_heat + self.chp_heat

        if self.heat_output >= self.max_thermal_power:
            self.heat_output = self.max_thermal_power

        return self.heat_output

    def power_generated(self, consumed_fuel_chp):
        self.power_output = (consumed_fuel_chp) / (1 + self.chp_production)
        self.consumed_fuel_chp = consumed_fuel_chp
            
        return self.power_output
    
    def fuel_chp(self):
        return self.chp_fuel
    
    def fuel_boiler(self):
        return self.boiler_fuel
    
    def current_power_gen(self):
        return self.power_output
    
    # def next_step_heat(self):

    def reset(self):
        self.boiler_fuel = 0
        self.chp_fuel = 0
        self.boiler_heat = 0
        self.chp_heat = 0

class DieselGenerator:
    def __init__(self, efficiency, externalities, emission_factor,
                 max_ramp_up, max_ramp_down, start_up_cost,a_coefficient,
                 b_coefficient, c_coefficient, min_power_output,max_power_output,
                 max_start_up_count):
        # Initialize generator parameters
        self.max_ramp_up = max_ramp_up
        self.max_ramp_down = max_ramp_down
        self.start_up_cost = start_up_cost
        self.start_up = False
        self.efficiency = efficiency
        self.externalities = externalities
        self.emission_factor = emission_factor
        self.a_coefficient = a_coefficient
        self.b_coefficient = b_coefficient
        self.c_coefficient = c_coefficient
        self.min_power_output = min_power_output
        self.max_power_output = max_power_output
        self.max_start_up_count = max_start_up_count
        self.start_up_count = 0
        self.prev_power_output = 0
        self.power_output = 0
    
    def calculate_fuel_cost(self):
        return (self.a_coefficient * (self.power_output ** 2) + 
                self.b_coefficient * self.power_output + self.c_coefficient) 
    
    def calculate_emission_cost(self):
        return self.power_output * self.emission_factor * self.externalities
    
    def calculate_total_cost(self):
        return self.calculate_fuel_cost() + self.calculate_emission_cost() + self.calculate_start_up_penalty()

    def calculate_start_up_penalty(self):
        if self.start_up:
            self.start_up = False
            self.start_up_count += 1
            return self.start_up_cost
        else:
            return 0

    def power_generated(self, power):
        
        if(power > self.prev_power_output):
            if(self.prev_power_output == 0):
                    self.start_up = True
            if(power - self.prev_power_output <= self.max_ramp_up):
                ramp_up_condition = power - self.prev_power_output
            else:
                ramp_up_condition = self.max_ramp_up

            self.power_output = min(self.power_output + ramp_up_condition, self.max_power_output)
            
        else:
            ramp_down_condition = self.prev_power_output - power <= self.max_ramp_down 
            if ramp_down_condition:
                self.power_output = self.power_output - ramp_down_condition
                if(self.power_output < self.min_power_output):
                    self.power_output = 0
            else:
                self.power_output = self.power_output - self.max_ramp_down
                if(self.power_output < self.min_power_output):
                    self.power_output = 0           
        
        self.prev_power_output = self.power_output
            
        return self.power_output

    def current_power_gen(self):
        return self.power_output
    
    def reset(self):
        self.power_output = 0
        self.start_up = False
        
class WT:
    def __init__(self, max_electric_power, operation_maintance_cost, 
                 cut_in, cut_out, nominal, wind_data):
        
        self.wind_data = wind_data
        self.max_electric_power = max_electric_power
        self.operation_maintance_cost = operation_maintance_cost  
        self.cut_in = cut_in
        self.cut_out = cut_out  
        self.nominal = nominal
        self.power_output = 0
        
    def get_wind_speed(self, time):
        return self.wind_data[time]

    def power_generated(self, time):
        
        wind_speed = self.wind_data[time]
        
        if(wind_speed > self.nominal):
            power_generated = self.max_electric_power 
        elif(wind_speed > self.cut_in):
            power_generated = self.max_electric_power * (wind_speed - self.cut_in) / (self.nominal - self.cut_in) 
        else:
            power_generated = 0

        self.power_output = power_generated
        
        return power_generated 

    # def calculate_cost(self):
    #     return self.operation_maintance_cost     

class Grid:
    def __init__(self, down_reg,up_reg, exp_fees, imp_fees):
        self.sell_prices = down_reg
        self.buy_prices = up_reg
        self.exp_fees=exp_fees
        self.imp_fees = imp_fees
        self.time = 0

    def sell(self, E):
        return (self.sell_prices[self.time] + self.exp_fees) * E

    def buy(self, E):
        return -(self.buy_prices[self.time] + self.imp_fees) * E

    #
    # def get_price(self,time):
    #     return self.prices[time]

    def set_time(self, time):
        self.time = time

    def total_cost(self,prices, energy):
        return sum(prices * energy / 100 + self.imp_fees * energy)

class Load:
    def __init__(self, price_sens, base_load, max_v_load,patience):
        self.price_sens = max(0,price_sens)
        self.orig_price_sens = max(0,price_sens)
        self.base_load = base_load
        self.max_v_load = max_v_load
        self.response = 0
        self.shifted_loads={}
        self.patience=max(patience,1)
        self.dr_load=0

    def react(self, price_tier , time_day):
        self.dr_load=self.base_load[time_day]
        response = self.price_sens * (price_tier - 2)
        if response != 0 :
            self.dr_load -= self.base_load[time_day] * response
            self.shifted_loads[time_day] = self.base_load[time_day] * response
        for k in list(self.shifted_loads):
            probability_of_execution = -self.shifted_loads[k]*(price_tier - 2) + (time_day-k)/self.patience
            if random.random()<=probability_of_execution:
                self.dr_load+=self.shifted_loads[k]
                del self.shifted_loads[k]

    def load(self):
        return max(self.dr_load, 0)


if __name__ == '__main__':
    
    MMBTU = 2
    
    # WT constants
    max_electric_power = 500
    operation_maintance_cost = 0.01
    cut_in = 3.0
    cut_out = 25.0
    nominal = 9.0
    
    #chp constants
    efficiency = 0.8
    chp_production = 2.0
    fuel_cost_dg = 55 * MMBTU # dollar per 1000 kg/h
    externalities = 2 # dollar per unit
    emission_factor = 0.2
    max_thermal_power = 2000 # gigajouls
    max_boiler_fuel = 2000 # Range: 500 kg/h to 1500 kg/h
    max_chp_fuel = 2000 # Range: 800 kg/h to 1800 kg/h
    
    #dg constants
    efficiency = 0.8
    fuel_cost = 10 # dollar per 1000 kg/h
    externalities = 2 # dollar per unit
    emission_factor = 0.2
    a_coefficient = 2.4 * 10**(-5)
    b_coefficient = 4.7 * 10**(-5)
    c_coefficient = 2.9 * 10**(-5)
    min_power_output = 50 # kwh
    max_power_output = 500 # kwh
    start_up_cost = 20 # unknown
    max_start_up_count = 100
    max_ramp_up = 200 # unknown
    max_ramp_down = 300 # unknown
    
    
    wind = WT(max_electric_power, operation_maintance_cost, cut_in,
              cut_out, nominal, wind_data = WIND_SPEED_DATA)

    dg = DieselGenerator(efficiency, externalities,
                         emission_factor, max_ramp_up, max_ramp_down,
                         start_up_cost,a_coefficient, b_coefficient,
                         c_coefficient, min_power_output, max_power_output,
                         max_start_up_count)
    
    chp = CHP(efficiency, externalities, chp_production,
                 fuel_cost, emission_factor, max_thermal_power, max_boiler_fuel, max_chp_fuel)
    
    power_dg = dg.power_generated(power = 32000)
    cost_dg = dg.calculate_total_cost()
    power_dg = dg.power_generated(power = 32000)
    cost_dg = dg.calculate_total_cost()
    power = wind.power_generated(time = 1044)
    power_chp = chp.power_generated(2000) / 10
    chp.calculate_total_cost()
    print(power)