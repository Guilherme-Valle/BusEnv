import gym
from gym.spaces import Box, Dict as GymDict
import numpy as np
import collections
import math
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from pettingzoo.mpe import simple_spread_v2
from supersuit import pad_observations_v0, pad_action_space_v0

class NonStationaryMPEWrapper(MultiAgentEnv):
    """
    A custom wrapper around MPE simple_spread to benchmark Continual MARL.
    Introduces regime shifts (friction/noise) and ESCP context injection.
    """
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}
            
        # 1. Instantiate the inner environment
        # We use continuous actions to easily multiply/add noise for friction/chaos regimes.
        self.inner_env = simple_spread_v2.parallel_env(max_cycles=25, continuous_actions=True)
        self.inner_env.reset()
        
        # PettingZoo variables required by RLlib/MARLlib
        self.agents = self.inner_env.possible_agents.copy()
        self.num_agents = len(self.agents)
        self.possible_agents = self.inner_env.possible_agents.copy()
        
        # State & Regimes
        self.episode_count = 0
        self.regime_switch_every = config.get("regime_switch_every", 50)
        self.current_regime = 0
        self.use_context = config.get("use_context", True)
        
        # ESCP Context: Rolling window of team average reward (proxy for physical residual)
        self.context_window_size = 10
        self.reward_window = collections.deque(maxlen=self.context_window_size)
        self.context_val = 0.0
        
        # Spaces modification
        self.observation_spaces = {}
        self.action_spaces = {}
        for agent in self.possible_agents:
            orig_obs = self.inner_env.observation_space(agent)
            
            # Extend Box space by +1 for the ESCP context scalar
            low = np.append(orig_obs.low, 0.0)
            # context log1p(variance) is generally bounded, we can set high to a reasonable limit or inf
            high = np.append(orig_obs.high, 20.0)
            
            self.observation_spaces[agent] = Box(low=low, high=high, dtype=np.float32)
            self.action_spaces[agent] = self.inner_env.action_space(agent)
            
        # RLlib and MARLlib standard attributes
        # MARLlib requires observation_space to be wrapped in a dict with key "obs" for Base_Model
        self.observation_space = GymDict({"obs": self.observation_spaces[self.possible_agents[0]]})
        self.action_space = self.action_spaces[self.possible_agents[0]]
        
    def reset(self):
        self.episode_count += 1
        # Regime logic: 0 -> Normal, 1 -> Friction, 2 -> Chaos
        self.current_regime = (self.episode_count // self.regime_switch_every) % 3
        
        obs = self.inner_env.reset()
        
        self.reward_window.clear()
        self.context_val = 0.0
        self.agents = list(obs.keys())
        
        return self._format_obs(obs)
        
    def step(self, action_dict):
        # 1. Non-Stationarity: Modify actions based on current regime
        modified_actions = {}
        for agent, action in action_dict.items():
            if self.current_regime == 0:
                # Normal
                modified_actions[agent] = action
            elif self.current_regime == 1:
                # High Friction: sluggish movement
                modified_actions[agent] = action * 0.5
            elif self.current_regime == 2:
                # Chaos: noise injection
                noise = np.random.normal(0, 0.4, size=action.shape)
                modified_actions[agent] = np.clip(
                    action + noise, 
                    self.action_spaces[agent].low, 
                    self.action_spaces[agent].high
                )
                                                
        # 2. Step the inner environment
        ret = self.inner_env.step(modified_actions)
        
        # PettingZoo v1.18+ tuple format handling
        if len(ret) == 5:
            obs, rewards, terminations, truncations, infos = ret
            dones = {a: (terminations.get(a, False) or truncations.get(a, False)) for a in obs.keys()}
        else:
            obs, rewards, dones, infos = ret
            
        dones["__all__"] = all(dones.values()) if dones else True
        
        # 3. ESCP Context Injection: Update rolling window with team average reward
        if rewards:
            avg_reward = sum(rewards.values()) / len(rewards)
            self.reward_window.append(avg_reward)
            
        if len(self.reward_window) > 1:
            variance = np.var(self.reward_window)
            self.context_val = math.log1p(variance)
        else:
            self.context_val = 0.0
            
        self.agents = [a for a in self.possible_agents if not dones.get(a, False)]
        
        return self._format_obs(obs), rewards, dones, infos

    def _format_obs(self, obs_dict):
        """Append the context scalar and wrap in the expected dictionary structure."""
        formatted = {}
        for agent, obs in obs_dict.items():
            val = self.context_val if self.use_context else 0.0
            obs_with_context = np.append(obs, val).astype(np.float32)
            formatted[agent] = {"obs": obs_with_context}
        return formatted
        
    def get_env_info(self):
        """Standard MARLlib environment info dictionary."""
        return {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 25,
            "agent_id": self.possible_agents,
            "share_observation_space": self.observation_space,
            "policy_mapping_info": {
                "non_stationary_mpe": {
                    "all_agents_one_policy": False,
                    "one_agent_one_policy": True,
                    "policy_map": {
                        agent_id: f"policy_{i}" for i, agent_id in enumerate(self.possible_agents)
                    }
                }
            }
        }
