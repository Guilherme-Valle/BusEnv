import functools
from pettingzoo import ParallelEnv
import networkx as nx
from gym import spaces
import numpy as np
import random
import pickle
import gym.utils.seeding  # import seeding
from gym.spaces import Discrete
import csv
import os

class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "graph_exploration_v0"}

    # network: A networkx.Graph type object representing the graph
    # actions_amount: The number of possible actions (generally, the maximum number of neighbors a node can have)
    # stopClass: Custom stop class (optional)
    # rewardClass: Custom reward class (optional)
    # initial_nodes, target_nodes: start and target nodes (if not passed, they are chosen randomly)
    # render_mode: "human" or "rgb_array" (optional)
    # avg_travel_time_AB, future_demand_at_B, occupancy_rate, uptime_normalized: dicts with precomputed data
    # real_routes: dict mapping agent_id to a fixed route (list of nodes)
    # route_metadata: dict with metadata for each route (optional)
    def __init__(self, network: nx.Graph, actions_amount: int, max_steps: int, num_agents=2,
                 stopClass=None, rewardClass=None, initial_nodes=None, target_nodes=None,
                 render_mode=None, avg_travel_time_AB=None, future_demand_at_B=None,
                 occupancy_rate=None, uptime_normalized=None,
                 real_routes=None, route_metadata=None,
                 worker_index=0, num_workers=1, regime_multipliers=None,
                 regime_switch_every=50, context_window_size=10, use_context=True):

        # --- Basic configuration ---
        self.network = network
        self.actions_amount = actions_amount
        self.max_steps = max_steps 
        self.render_mode = render_mode

        # --- Agentes ---
        self._num_agents = num_agents
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agents = self.possible_agents.copy()  # <- MARLlib needs this
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}

        # --- Stop/Reward classes ---
        self.stop = DefaultStopClass() if stopClass is None else stopClass
        self.reward = DefaultReward() if rewardClass is None else rewardClass

        # --- Map nodes to indexes ---
        self.node_to_idx = {str(n): i for i, n in enumerate(self.network.nodes)}  
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}

        # --- Internal states per agent ---
        self.states = {}
        self.targets = {}
        self.steps = {}
        self.delays = {}
        self.estimated_times = {}
        self.expected_times = {}

        self.initial_nodes = initial_nodes
        self.target_nodes = target_nodes

        # --- External data / features ---
        self.avg_travel_time_AB = avg_travel_time_AB or {}
        self.future_demand_at_B = future_demand_at_B or {}
        self.occupancy_rate = occupancy_rate or {}
        self.uptime_normalized = uptime_normalized or {}

        # --- Global clock and statistics ---
        self.agent_times = {agent: 6 * 60 * 60 for agent in self.possible_agents} # Every agent starts at 6:00 AM 
        self.headways = {}
        self.sync_stats = {}

        self.service_center_node = random.choice(list(self.network.nodes))

        # --- Internal structure of agents ---
        self.agent_states = {}
        for agent_id in self.possible_agents:
            self.agent_states[agent_id] = {
                "location": None,
                "occupancy": 0.0,
                "uptime": 1.0,
                "fuel": 100.0,
                "maintenance_status": "ok",
                "schedule": [],
                "route": None,
                "route_idx": 0,
                "needs_service": False,
                "going_forward": True,
            }

        # --- Reward parameters ---
        self.reward_weights = {
            "occ_penalty": 1.0,
            "uptime_bonus": 1.0,
            "sync_score": 1.0,
            "energy_efficiency": 1.0
        }
        self.occupancy_range = (0.6, 0.9)
        self.use_context = bool(use_context)

        # --- Observation and action spaces ---
        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self.action_space(agent) for agent in self.possible_agents
        }

        # --- Defaults / limits ---
        if self.avg_travel_time_AB:
            self.default_travel_time = np.mean(list(self.avg_travel_time_AB.values()))
        else:
            self.default_travel_time = 1.0

        self.max_travel_time = 3250.0
        self.max_capacity = 80

        self.real_routes = real_routes or {}
        self.route_metadata = route_metadata or {}
        self.agent_routes = {}

        # Per-second physical rates. Regime fuel/uptime knobs scale THESE, not
        # travel_time, so a 2x Rain travel shock does not silently become a 4x
        # fuel drain.
        self._fuel_capacity = 100.0
        self._base_fuel_per_sec = 1.0 / 300.0
        self._base_uptime_per_sec = 1.0 / (12.0 * 3600.0)
        self._wait_elapsed_sec = 60.0
        self._service_path_factor = 0.3

        # --- Controlled non-stationarity (multivariate regime switcher) ---
        # 0 = Normal, 1 = Peak Hour, 2 = Rain
        # Each axis is an independent physical knob:
        #   travel     -> edge travel time only (_lookup_travel_time)
        #   fuel       -> per-second fuel drain
        #   uptime     -> per-second uptime decay
        #   occupancy  -> EWMA occupancy *target*, then clipped to [0, 1]
        self.current_regime = 0
        self.episode_count = 0
        self.regime_switch_every = int(regime_switch_every)
        self.regime_multipliers = regime_multipliers or {
            0: {"travel": 1.0, "fuel": 1.0, "uptime": 1.0, "occupancy": 1.0},
            1: {"travel": 1.5, "fuel": 1.2, "uptime": 1.1, "occupancy": 1.3},
            2: {"travel": 2.0, "fuel": 1.4, "uptime": 1.3, "occupancy": 1.1},
        }
        self.worker_index = int(worker_index)
        self.num_workers = int(num_workers)
        self.context_window_size = int(context_window_size)
        self._context_windows = {}
        self._context_ptrs = {}
        self._context_counts = {}

        # Per-worker file avoids two Ray actors appending to the same CSV.
        self.metrics_file = f"env_metrics_w{self.worker_index}.csv"

        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode", "env_steps", "mean_reward", "total_reward",
                    "fairness", "regime_id", "worker_index",
                    "travel_mult", "fuel_mult", "uptime_mult", "occupancy_mult",
                ])


    @property
    def num_agents(self):
        return self._num_agents

    def _regime_params(self) -> dict:
        """Return the four independent knobs for the active regime.

        A legacy scalar multiplier is treated as travel-only so older configs
        keep their previous behaviour.
        """
        raw = self.regime_multipliers.get(self.current_regime, {})
        if isinstance(raw, dict):
            return {
                "travel": float(raw.get("travel", 1.0)),
                "fuel": float(raw.get("fuel", 1.0)),
                "uptime": float(raw.get("uptime", 1.0)),
                "occupancy": float(raw.get("occupancy", 1.0)),
            }
        scale = float(raw) if raw != {} else 1.0
        return {"travel": scale, "fuel": 1.0, "uptime": 1.0, "occupancy": 1.0}

    def _regime_scale(self, key: str) -> float:
        return self._regime_params()[key]

    def _travel_time_multiplier(self) -> float:
        """Return the travel-time scale factor for the active regime."""
        return self._regime_scale("travel")

    def _lookup_travel_time(self, start, target) -> float:
        """Base edge travel time scaled by the travel knob only."""
        base = self.avg_travel_time_AB.get((start, target), self.default_travel_time)
        return base * self._travel_time_multiplier()

    def _apply_fuel_drain(self, state, elapsed_sec: float) -> None:
        drain = elapsed_sec * self._base_fuel_per_sec * self._regime_scale("fuel")
        state["fuel"] = max(state["fuel"] - drain, 0.0)

    def _apply_uptime_decay(self, state, elapsed_sec: float) -> None:
        decay = elapsed_sec * self._base_uptime_per_sec * self._regime_scale("uptime")
        state["uptime"] = max(state["uptime"] - decay, 0.0)

    def _occupancy_target(self, raw_expected: float) -> float:
        """Shift the occupancy EWMA target, then keep the state in [0, 1]."""
        return float(np.clip(float(raw_expected) * self._regime_scale("occupancy"), 0.0, 1.0))

    def _clip_reward(self, reward: float) -> float:
        return float(np.clip(reward, -1.0, 1.0))

    def _base_travel_time(self, start, target) -> float:
        """Unscaled table travel time (the published timetable)."""
        return float(self.avg_travel_time_AB.get((start, target), self.default_travel_time))

    def _reset_context_windows(self, agents) -> None:
        h = self.context_window_size
        self._context_windows = {agent: np.zeros(h, dtype=np.float32) for agent in agents}
        self._context_ptrs = {agent: 0 for agent in agents}
        self._context_counts = {agent: 0 for agent in agents}

    def _push_context_residual(self, agent, residual: float) -> None:
        window = self._context_windows[agent]
        ptr = self._context_ptrs[agent]
        window[ptr] = float(residual)
        self._context_ptrs[agent] = (ptr + 1) % self.context_window_size
        self._context_counts[agent] = min(self.context_window_size, self._context_counts[agent] + 1)

    def _context_feature(self, agent) -> float:
        """Regime identity: mean |residual| over the last H steps (stays high all regime)."""
        n = self._context_counts.get(agent, 0)
        if n <= 0:
            return 0.0
        window = self._context_windows[agent]
        filled = window[:n] if n < self.context_window_size else window
        return float(np.clip(np.mean(np.abs(filled)), 0.0, 10.0))

    def _record_physical_surprise(
            self, agent, elapsed_sec, travel_realized=None, travel_base=None,
            occ_prev=None, occ_now=None) -> float:
        """ESCP-style surprise. Travel term is (realized/base - 1) so Peak≈0.5, Rain≈1.0."""
        travel_resid = 0.0
        if travel_realized is not None and travel_base is not None:
            base = float(travel_base)
            travel_resid = (float(travel_realized) - base) / (base + 1e-8)

        occ_resid = 0.0
        if occ_prev is not None and occ_now is not None:
            occ_resid = float(occ_now) - float(occ_prev)

        fuel_extra = float(elapsed_sec) * self._base_fuel_per_sec * (self._regime_scale("fuel") - 1.0)
        fuel_resid = fuel_extra / self._fuel_capacity

        surprise = abs(travel_resid) + abs(occ_resid) + abs(fuel_resid)
        self._push_context_residual(agent, surprise)
        return self._context_feature(agent)

    def _make_obs(self, agent, curr_node, next_node, occupancy, uptime, maintenance_ok):
        travel_time = self._lookup_travel_time(curr_node, next_node)
        normalized_travel_time = min(travel_time / self.max_travel_time, 1.0)
        channels = [
            self.agent_times[agent] / (24 * 60 * 60),
            occupancy,
            normalized_travel_time,
            self.future_demand_at_B.get(next_node, 0.0),
            uptime,
            1.0 if maintenance_ok else 0.0,
            self.node_to_idx[str(curr_node)],
            self.node_to_idx[str(next_node)],
        ]
        if self.use_context:
            channels.append(self._context_feature(agent))
        obs_array = np.array(channels, dtype=np.float32)
        return np.clip(
            obs_array,
            self.observation_space(agent).low,
            self.observation_space(agent).high,
        )

    def set_regime(self, regime_id: int) -> None:
        """Pin the active regime (driver callbacks can keep Ray workers aligned)."""
        regime_id = int(regime_id)
        if regime_id not in self.regime_multipliers:
            raise ValueError(
                f"Unknown regime_id={regime_id}. "
                f"Known regimes: {list(self.regime_multipliers)}"
            )
        self.current_regime = regime_id

    def _update_regime_on_episode_start(self, options=None):
        """Advance the episode counter and rotate, unless a regime is pinned.

        `expected_times` stay on the unscaled timetable on purpose: the
        published schedule does not know about Peak/Rain, so efficiency
        degradation is part of the distribution shift.
        """
        self.episode_count += 1
        options = options or {}
        if "regime_id" in options:
            self.set_regime(options["regime_id"])
            return
        num_regimes = len(self.regime_multipliers)
        self.current_regime = ((self.episode_count - 1) // self.regime_switch_every) % num_regimes
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self._update_regime_on_episode_start(options)

        self.agents = self.possible_agents[:]  
        self.states = {}
        self.targets = {}
        self.steps = {}
        self.delays = {}
        self.estimated_times = {}
        self.expected_times = {}
        self.agent_times = {agent: 6 * 60 * 60 for agent in self.possible_agents}  # 6:00 AM
        self.headways = {}
        self.sync_stats = {}
        self.agent_states = {}

        observations = {}
        self.infos = {}
        self._reset_context_windows(self.agents)

        for agent in self.agents:
            if agent not in self.agent_routes:  
                trip_id, path = random.choice(list(self.real_routes.items()))
                print(f"[DEBUG] Route chosen for {agent} (Trip ID: {trip_id}): {path}")
                self.agent_routes[agent] = path

            path = self.agent_routes[agent]
            if len(path) < 2:
                raise ValueError(f"Invalid route for {agent}: {path}")

            initial = path[0]
            target = path[-1]

            self.agent_states[agent] = {
                "location": initial,
                "occupancy": self._occupancy_target(
                    float(self.occupancy_rate.get(int(initial), 0.0))
                ),
                "uptime": float(self.uptime_normalized.get(initial, 1.0)),
                "fuel": self._fuel_capacity,
                "maintenance_status": "ok",
                "schedule": [],
                "route": path,
                "route_idx": 0,
            }

            self.states[agent] = initial
            self.targets[agent] = target
            self.steps[agent] = 0
            self.estimated_times[agent] = 0
            self.delays[agent] = {}

            # Unscaled timetable on purpose: Peak/Rain are shocks relative to
            # the published schedule, so energy_efficiency degrades under load.
            self.expected_times[agent] = sum(
                self.avg_travel_time_AB.get((path[i], path[i + 1]), self.default_travel_time)
                for i in range(len(path) - 1)
            )

            next_node = path[1]
            observations[agent] = self._make_obs(
                agent,
                initial,
                next_node,
                self.agent_states[agent]["occupancy"],
                self.agent_states[agent]["uptime"],
                self.agent_states[agent]["maintenance_status"] == "ok",
            )


            self.infos[agent] = {
                "chosen_route": path,
                "expected_time": self.expected_times[agent],
            }

        #print(f"[RESET] Environment reset. Agents: {self.agents}")

        self.current_episode_metrics = {  # Metrics for the current episode
            "rewards": {agent: 0.0 for agent in self.agents},
            "steps": {agent: 0 for agent in self.agents},
            "done": False
        }

        # return only observations
        return observations

    
    def step(self, actions):
        if not actions:  # If there are no actions, return empty observations
            self.agents = []
            return {}, {}, {}, {}, {}  # Observations, rewards, terminations, truncations, infos

        observations = {}  # Observations for each agent
        rewards = {}       # Rewards for each agent
        terminations = {}  # Terminations for each agent
        truncations = {}   # Truncations for each agent
        infos = {}

        for agent in self.agents:  # Checks if the agent is active
            self.steps[agent] += 1  # Increments the agent's step counter
            state = self.agent_states[agent]  # Internal state of the agent
            route = state["route"]  # Route of the agent
            idx = state["route_idx"]  # Current index in the route

            if idx >= len(route):  # Verifies if the agent has exceeded the route
                #print(f"[ERROR] Agent {agent} exceeded the route. IDX={idx}, LEN={len(route)}")
                terminations[agent] = True
                truncations[agent] = False
                rewards[agent] = self._clip_reward(-1.0)
                continue

            curr_node = route[idx]  # Current node of the agent
            self.states[agent] = curr_node  # Ensures synchronization

            action = actions[agent]  # Action chosen by the agent
            #print(f"[DEBUG] action: {action} for agent: {agent}")
            elapsed_sec = 0.0
            travel_realized = None
            travel_base = None
            occ_before = float(state.get("occupancy", 0.0))

            # ================= WAIT =================
            if action == 0:  
                reward = self._clip_reward(-0.1)
                elapsed_sec = self._wait_elapsed_sec
                self.agent_times[agent] += elapsed_sec
                self._apply_uptime_decay(state, elapsed_sec)
                self._apply_fuel_drain(state, elapsed_sec)
                terminated = self.agent_times[agent] >= 24 * 3600
                truncated = False

            # ================= MOVE =================
            elif action == 1:  
                going_forward = state.get("going_forward", True)
                route_length = len(route)

                if going_forward:
                    if idx + 1 < route_length:
                        next_node = route[idx + 1]
                        self.agent_states[agent]["route_idx"] += 1
                    else:
                        state["going_forward"] = False
                        self.agent_states[agent]["route_idx"] -= 1
                        next_node = route[self.agent_states[agent]["route_idx"]]
                else:  
                    if idx > 0:
                        self.agent_states[agent]["route_idx"] -= 1
                        next_node = route[self.agent_states[agent]["route_idx"]]
                    else:
                        state["going_forward"] = True
                        self.agent_states[agent]["route_idx"] += 1
                        next_node = route[self.agent_states[agent]["route_idx"]]

                direction = "➡️ going forward" if state.get("going_forward", True) else "⬅️ going backward"
                #print(f"[MOVE] Agent {agent} | {direction} | {curr_node} -> {next_node} "
                #      f"(t={self.current_time/3600:.2f}h, occ={state.get('occupancy',0):.1f}, "
                #      f"fuel={state.get('fuel',0):.1f}, uptime={state.get('uptime',0):.2f})")

                # Travel knob only; fuel/uptime use independent per-second rates.
                travel_time = self._lookup_travel_time(curr_node, next_node)

                prev_occ = state.get("occupancy", 0.0)
                if int(curr_node) in self.occupancy_rate:
                    expected_occ = self._occupancy_target(self.occupancy_rate[int(curr_node)])
                    alpha = 0.5
                    new_occ = (1 - alpha) * prev_occ + alpha * expected_occ
                    occupancy = max(0.0, min(new_occ, 1.0))
                else:
                    occupancy = prev_occ

                state["occupancy"] = occupancy
                self.agent_times[agent] += travel_time
                self.estimated_times[agent] += travel_time
                self._apply_uptime_decay(state, travel_time)
                self._apply_fuel_drain(state, travel_time)
                self.states[agent] = next_node
                elapsed_sec = travel_time
                travel_realized = travel_time
                travel_base = self._base_travel_time(curr_node, next_node)

                if next_node not in self.headways:
                    self.headways[next_node] = []
                self.headways[next_node].append(self.agent_times[agent])

                reward = self.reward.getReward(
                    new_state=next_node,
                    previous_state=curr_node,
                    action=action,
                    target=route[-1],
                    network=self.network,
                    estimated_time=self.estimated_times[agent],
                    expected_time=self.expected_times[agent],
                    delay=0,
                    agent_state=state,
                    headways=self.headways[next_node]
                )

                terminated = self.agent_times[agent] >= 24 * 3600
                truncated = self.steps[agent] >= self.max_steps

            # ================= SERVICE CENTER =================
            elif action == 2:  
                sc_node = self.get_nearest_service_center(curr_node)

                try:
                    path = nx.shortest_path(
                        self.network, source=curr_node, target=sc_node,
                        weight=lambda u, v, d: self._lookup_travel_time(u, v)
                    )

                    total_travel_time = 0.0
                    total_fuel_cost = 0.0
                    total_base_travel = 0.0
                    fuel_rate = self._base_fuel_per_sec * self._regime_scale("fuel")

                    for u, v in zip(path[:-1], path[1:]):
                        edge_time = self._lookup_travel_time(u, v) * self._service_path_factor
                        total_travel_time += edge_time
                        total_base_travel += self._base_travel_time(u, v) * self._service_path_factor
                        total_fuel_cost += edge_time * fuel_rate
                except nx.NetworkXNoPath:
                    reward = self._clip_reward(-1.0)
                    terminated = False
                    truncated = False
                else:
                    travel_norm = min(total_travel_time / self.max_travel_time, 1.0)
                    unnecessary = (
                        state["fuel"] > 0.8 * self._fuel_capacity
                        and state["uptime"] > 0.8
                    )

                    if state["fuel"] < total_fuel_cost:
                        reward = self._clip_reward(-1.0)
                    else:
                        self.agent_times[agent] += total_travel_time
                        self.estimated_times[agent] += total_travel_time
                        state["fuel"] = max(state["fuel"] - total_fuel_cost, 0.0)
                        self._apply_uptime_decay(state, total_travel_time)

                        state["fuel"] = self._fuel_capacity
                        state["uptime"] = 1.0
                        state["maintenance_status"] = "ok"
                        self.states[agent] = sc_node
                        # Detour cost in [-1, 0]; extra hit if the bus did not need service.
                        reward = -travel_norm
                        if unnecessary:
                            reward -= 0.25
                        reward = self._clip_reward(reward)
                        elapsed_sec = total_travel_time
                        travel_realized = total_travel_time
                        travel_base = total_base_travel

                terminated = self.agent_times[agent] >= 24 * 3600
                truncated = self.steps[agent] >= self.max_steps

            else:
                reward = self._clip_reward(-1.0)
                terminated = False
                truncated = True

            # ================= OBSERVATION UPDATE =================
            if self.use_context:
                self._record_physical_surprise(
                    agent,
                    elapsed_sec=elapsed_sec,
                    travel_realized=travel_realized,
                    travel_base=travel_base,
                    occ_prev=occ_before,
                    occ_now=float(state.get("occupancy", occ_before)),
                )
            route_idx = self.agent_states[agent]["route_idx"]
            curr_node = self.states[agent]
            next_node = route[route_idx + 1] if route_idx + 1 < len(route) else curr_node
            observations[agent] = self._make_obs(
                agent,
                curr_node,
                next_node,
                state["occupancy"],
                state["uptime"],
                state["maintenance_status"] == "ok",
            )

            rewards[agent] = self._clip_reward(reward)
            terminations[agent] = terminated
            truncations[agent] = truncated
            regime = self._regime_params()
            infos[agent] = {
                "count": self.steps[agent],
                "occupancy": state["occupancy"],
                "location": curr_node,
                "next_stop": next_node,
                "headways": self.headways.get(curr_node, []),
                "regime_id": self.current_regime,
                "episode_count": self.episode_count,
                "worker_index": self.worker_index,
                "regime_travel": regime["travel"],
                "regime_fuel": regime["fuel"],
                "regime_uptime": regime["uptime"],
                "regime_occupancy": regime["occupancy"],
                "context_var": self._context_feature(agent),
            }
            
            if self.agent_times[agent] >= 24 * 3600:
                print(f"[END OF DAY] Simulation ended at {self.agent_times[agent]/3600:.2f}h (>= 24h).")

        self.agents = [agent for agent in self.agents if not (terminations[agent] or truncations[agent])]

        # Update current episode metrics
        total_reward = sum(rewards.values())
        mean_reward = np.mean(list(rewards.values()))

        # Fairness (Gini coefficient sobre recompensas)
        def gini(x):
            if np.amin(x) < 0:
                x = np.array(x) - np.amin(x)  # shift values to be non-negative
            x = np.sort(np.array(x))
            n = len(x)
            if n == 0:
                return 0.0
            index = np.arange(1, n + 1)
            return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x) + 1e-8)

        fairness = 1 - gini(list(rewards.values())) if rewards else 0.0

        if not hasattr(self, "metrics_history"):  # Initialize metrics history if not present
            self.metrics_history = []
        
        self.metrics_history.append({
            "step": sum(self.steps.values()),  # Total steps taken by all agents
            "total_reward": total_reward,
            "mean_reward": mean_reward,
            "fairness": fairness
        })

        # === save metrics on CSV ===
        if not hasattr(self, "episode_counter"):
            self.episode_counter = 0
        self.episode_counter += 1

        env_steps = sum(self.steps.values())
        regime = self._regime_params()
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_counter,
                env_steps,
                mean_reward,
                total_reward,
                fairness,
                self.current_regime,
                self.worker_index,
                regime["travel"],
                regime["fuel"],
                regime["uptime"],
                regime["occupancy"],
            ])


        # fusing terminations + truncations → dones
        dones = {a: (terminations[a] or truncations[a]) for a in rewards}
        dones["__all__"] = all(dones.values())

        for agent in self.agents: # Add episode reward and fairness to infos
            infos[agent] = {
                **infos.get(agent, {}),
                "episode_reward": rewards[agent],
                "mean_reward_episode": mean_reward,
                "fairness": fairness
            }

        return observations, rewards, dones, infos



    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        low = [
            0.0,    # time_of_day_norm
            0.0,    # occupancy_rate
            0.0,    # avg_travel_time_AB (normalizado)
            0.0,    # future_demand_at_B
            0.0,    # uptime
            0.0,    # maintenance_status
            0.0,    # curr_node_id
            0.0,    # next_node_id
        ]
        high = [
            1.0,    # time_of_day_norm
            1.0,    # occupancy_rate
            1.0,    # avg_travel_time_AB (normalizado!)
            1e6,    # future_demand_at_B (mantém valor realista)
            1.0,    # uptime
            1.0,    # manutenção ok
            2e9,    # curr_node_id
            2e9,    # next_node_id
        ]
        if self.use_context:
            low.append(0.0)    # context_var
            high.append(10.0)
        return spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent): # Define the action space for each agent
        return spaces.Discrete(self.actions_amount)
    
    def generate_random_delay(self, start, target):
        try:
            # Finds the shortest path between the start and target
            shortest_path = nx.shortest_path(self.network, source=start, target=target, weight='weight')
            path_edges = list(zip(shortest_path, shortest_path[1:])) # Creates a list of edges from the shortest path
            total_time = 0

            if not path_edges:
                return  # No edges to delay

            # Calculates total time of the edges in the path
            for a, b in path_edges:
                a, b = str(a), str(b)
                edge_key = (min(a, b), max(a, b)) # Sorts the nodes of the edge to avoid duplication
                if edge_key in self.reward.waitTimeDict:
                    edge_time = self.reward.waitTimeDict[edge_key][0]
                    total_time += edge_time # Sums the waiting time of the edge
                else:
                    x = 0 
                    #print(f"[AVISO] Aresta {edge_key} não está no waitTimeDict!")

            average_time = total_time / len(path_edges)
            #print(f"Média de tempo das arestas do caminho ótimo: {average_time}")

            # Chooses a random edge in the path to apply the delay
            delay_u, delay_v = random.choice(path_edges)
            delay_u, delay_v = str(delay_u), str(delay_v) # Sorts the nodes of the edge to avoid duplication
            delay_edge_key = (min(delay_u, delay_v), max(delay_u, delay_v)) # Chosen edge for delay

            if delay_edge_key in self.reward.waitTimeDict: # Checks if the chosen edge is in waitTimeDict
                delay = average_time * 5  # Simulates heavy congestion
                self.dynamicDelays = {
                    delay_edge_key: delay # Chosen edge for delay with applied delay time
                }
                #print(f"Aresta atrasada: {delay_edge_key}, atraso aplicado: {delay}")
            else:
                #print(f"[ERRO] Aresta escolhida para atraso {delay_edge_key} não está no waitTimeDict.")
                self.dynamicDelays = {}

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.dynamicDelays = {}

    def get_nearest_service_center(self, current_node):
        # Finds the nearest service center node, not dynamic yet
        return self.service_center_node

# This is the base class for reward classes
class RewardBaseClass():
    def getReward(self, state, previousState, action, target, graph):
        raise NotImplementedError

# This is the base class for stop classes
class StopConditionBaseClass():
    def isTerminated(self, state, previousState, action, target, graph):
        raise NotImplementedError
        
class DefaultReward(RewardBaseClass):
    """
    Compound reward and NORMALIZED to [-1, +1] per step.
    Components:
      - occ: penalizes out of ideal range (quadratic, 0..1, sign -)
      - uptime: direct bonus (0..1, sign +)
      - sync: measures regularity of headways vs target (0..1, sign +)
      - efficiency: 1 - (estimated/expected) truncated (0..1, sign +)
    """
    def __init__(self, waitTimeDict=None, reward_weights=None, occupancy_range=(0.6, 0.9),
                 target_headway_seconds: float = 600.0,  # 10 minutos
                 max_sync_rel_std: float = 1.0          # >1 é truncado
                 ):
        super().__init__()
        # self.waitTimeDict can be used if needed for other metrics
        self.waitTimeDict = waitTimeDict or {}

        # Adjustable weights (sum doesn't need to be 1; we do weighted average)
        self.reward_weights = reward_weights or {
            "occ_penalty": 0.5,        # less than 1 to not dominate
            "uptime_bonus": 0.7,
            "sync_score": 0.5,         
            "energy_efficiency": 0.6
        }

        self.occupancy_range = occupancy_range
        self.target_headway = float(target_headway_seconds)
        self.max_sync_rel_std = float(max_sync_rel_std)

    def _occ_component(self, occupancy: float) -> float:
        """
        Returns a value in [0, 1], where 0 = perfect in ideal range; 1 = far off.
        Then we apply negative sign when composing the reward
        """
        min_occ, max_occ = self.occupancy_range
        if occupancy < min_occ:
            return min(1.0, (min_occ - occupancy) ** 2 / (min_occ ** 2 + 1e-8))
        if occupancy > max_occ:
            return min(1.0, (occupancy - max_occ) ** 2 / ((1.0 - max_occ) ** 2 + 1e-8))
        return 0.0

    def _sync_component(self, headways: list) -> float:
        """
        Measures regularity in [0, 1]: 1 = perfect (intervals very close to target),
        0 = very irregular (relative deviation >= max_sync_rel_std)
        """
        if not headways or len(headways) < 3:
            return 0.0  # Not enough information to assess regularity

        # intervals in seconds
        intervals = [headways[i + 1] - headways[i] for i in range(len(headways) - 1)]
        # remove noise/invalid intervals
        intervals = [x for x in intervals if x > 0]
        if len(intervals) < 2:
            return 0.0

        # RMS deviation versus target
        diffs = [(x - self.target_headway) for x in intervals]
        mean_sq = sum(d * d for d in diffs) / len(diffs)
        rms = mean_sq ** 0.5  # in seconds

        # Normalized relative deviation (0=perfect, 1=bad limit)
        rel = min(1.0, rms / (self.max_sync_rel_std * self.target_headway + 1e-8))

        # Convert to "score" in [0,1], where 1 is good
        return 1.0 - rel

    def _efficiency_component(self, estimated_time: float, expected_time: float) -> float:
        """
        Travel efficiency in [0,1]. 1 = equal/to less than expected; 0 = worse than expected.
        """
        if expected_time <= 0:
            return 0.0
        ratio = estimated_time / (expected_time + 1e-8)
        return float(np.clip(1.0 - ratio, 0.0, 1.0))

    def getReward(
        self,
        new_state, previous_state, action, target, network,
        estimated_time, expected_time, delay,
        agent_state=None, headways=None
    ):
        # Normalized components
        occ_pen = 0.0
        uptime = 0.0
        sync = 0.0
        eff = 0.0

        if agent_state is not None:
            occ_pen = self._occ_component(float(agent_state.get("occupancy", 0.0)))
            uptime = float(np.clip(agent_state.get("uptime", 1.0), 0.0, 1.0))

        sync = self._sync_component(headways or [])
        eff = self._efficiency_component(float(estimated_time), float(expected_time))

        # Weighted combination (keeping each term in [-1, +1])
        # occ_pen enters with NEGATIVE sign
        w = self.reward_weights
        reward = (
            -w["occ_penalty"] * occ_pen +
             w["uptime_bonus"] * uptime +
             w["sync_score"] * sync +
             w["energy_efficiency"] * eff
        )

        # Normalize by the sum of weights to keep magnitude around ~[-1, +1]
        weight_sum = (abs(w["occ_penalty"]) + w["uptime_bonus"] + w["sync_score"] + w["energy_efficiency"])
        if weight_sum > 0:
            reward = reward / weight_sum

        # Clip final for numerical stability
        # reward += 0.2
        reward = float(np.clip(reward, -1.0, 1.0))
        return reward


# This is the default stop class, which terminates the episode when the agent reaches the target node or takes the SERVICE_CENTER action
class DefaultStopClass(StopConditionBaseClass):
    def isTerminated(self, state, previousState, action, target, graph):
        return state == target or action == 2
