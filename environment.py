
import gymnasium as gym
import math
import pandas as pd
import numpy as np
from gym import spaces
from gymnasium.utils import seeding
from gymnasium import spaces, logger
from gymnasium.core import ActType, ObsType
import os
import torch
import torch.nn.functional as F

hmax = 10 # number of shares oer trade-share
initial_balance = 100_000 # initial ammount in tyhe accout
trans_fee = 1/1000 # fixed transaction fee per trade



class tradeEnv(gym.Env):
    ''''
    Environment for asset allocating based on OpenAI gym and FinRL application

    Attributes
    ----------
    df: DataFrame
            input data < >
    nstocks : int
            number of unique stocks in portfolio
    hmax : int
            normalized maximum number of shares to trade
    initial_balance : int
            cash amount at start
    transaction_cost_pct: float
            transaction cost percentage per trade
    reward_scaling: float
        scaling factor for reward, good for training
    state_space: int
        the dimension of input features
    action_space: int
        equals stock dimension
    day: int
            an increment number to control date


    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then 
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step
    '''
    metadata = {"render.modes": ["human"]}
    

    def __init__(self,
                 df: pd.DataFrame,
                 day: int=0,
                 initial:bool=True):
        
        self.df = df
        self.day = day
        self.initial = initial

        # Input dataset: Take a n(days)*n(number of assets) dataframe as input
        self.nstocks = self.df.shape[1] # number of stock


        # Characterize <Action Space>
        # Actions = weights assgin to each asset in the portfolio
        self.action_space = spaces.Box(low = -1, high = 1,
                                 shape = (self.nstocks,))


        # Characterize <
        # states dimentsion = [current_balance]_(1*1)+[prices]_(m*1)+[weights]_(m*1)
        self.observation_space = spaces.Box(low = 0, high = np.inf,
                                    shape = (1+2*self.nstocks,) )



        # Initialize
        self.data = self.df.iloc[self.day,: ] # subset of the whole dataframe
        self.terminal = False


        self.initial_balance = 0
        self.portfolio_value = self.initial_balance

        self.reward = 0
        self.state = [initial_balance] + self.data.values.tolist() + [0]*self.nstocks
        self.cum_reward = 0
        
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory = [ [1/self.nstocks] * self.nstocks ]
        # self.date_memory=[self.data.date.unique()[0]]


    def _sell_stock(self, index, action):
        '''
        Inputs
        ----------
        index : stock index to sell
        action: shares
        '''
        action = np.floor(action)
        if self.state[index+self.nstocks+1] > 0: # feasible to sell
            # update balance
            self.state[0] += self.state[index+1]*min(abs(action), self.state[index+self.nstocks+1]) * (1 - trans_fee)
            # update owned shares
            self.state[index+self.nstocks+1] -= min(abs(action), self.state[index+self.nstocks+1])
        else:
            pass


    def _buy_stock(self, index:int, action:float) -> None:
        '''
        Inputs
        ----------
        index : stock index to sell
        action: shares
        '''
        action = np.floor(action)
        # Buy --> decrease in cash account
        self.state[0] -= self.state[index+1] * action * (1 + trans_fee)
        self.state[1 + self.nstocks + index] += action


    def step(self, actions):
        '''
        at each step the agent will return actions, then 
        we will calculate the reward, and return the next observation.
        '''
        self.terminal = self.day >= len(self.df.index.unique())-1 # stops when reaching the end of dataframe (period)
       
       
        if self.terminal:
            return self.state, self.reward, self.terminal, False, {"episodic_return": self.cum_reward}

        else:
            actions *= hmax 
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.nstocks+1)])*np.array(self.state[(self.nstocks+1):(self.nstocks*2+1)]))
            
            argsort_actions = np.argsort(actions) # indices: 
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self._sell_stock(index, actions[index])
        
            for index in buy_index:
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.iloc[self.day,:]

            #load next state i.e. the new value of the stocks
            self.state =  [self.state[0]] + self.data.values.tolist() + \
                            list(self.state[(self.nstocks+1):(self.nstocks*2+1)])
            
            #total asset = cash + stock_price_vector * stock_share_vector
            end_total_asset = self.state[0] + \
                            sum(np.array(self.state[1:(self.nstocks+1)])*np.array(self.state[(self.nstocks+1):(self.nstocks*2+1)]))
            
            self.reward = end_total_asset - begin_total_asset   

            weights = self.normalization( np.array(self.state[(self.nstocks+1):(self.nstocks*2+1)]) )

            self.actions_memory.append(weights.tolist())

            #document cumulative reward
            self.cum_reward += (end_total_asset - begin_total_asset)

        return self.state, self.reward, self.terminal, False, {}




    def reset(self, seed = 6885, options = None) -> tuple[ObsType, dict]:  
        '''
        Reset the environment
        '''
        self.day = 0
        self.data = self.df.iloc[self.day,:]
        self.terminal = False 

        self.actions_memory=[[1/self.nstocks]*self.nstocks]

        self.state = [initial_balance] + self.data.values.tolist() + [0]*self.nstocks
        self.cum_reward = 0
        return self.state, {}
    
    
    
    def normalization(self, actions):
        '''
        Normalize actions to true weights using softmax
        '''
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        output = numerator/denominator
        #output =  F.softmax(torch.tensor(actions), dim=-1).numpy()
        return output


    def save_action_memory(self):
        return self.actions_memory
    

    def render(self, mode='human',close=False):
        return self.state
    

    def set_seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # def get_sb_env(self):
    #     e = DummyVecEnv([lambda: self])
    #     obs = e.reset()
    #     return e, obs


if __name__ == "__main__":
    print("Trading environment")