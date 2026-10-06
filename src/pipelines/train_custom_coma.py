import argparse
import os
import pickle
import shutil
import time
import numpy as np
from gym.spaces import Dict as GymDict

from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.marl.algos.core.CC import coma as marl_coma_core
from marllib.marl.algos.scripts import coma as marl_coma_script
_ALGO_ATTR = "coma"

from ray.rllib.models import ModelCatalog
from envs.sunt_env import parallel_env
import envs.sunt_env as sunt_env_module
from supersuit import pad_observations_v0, pad_action_space_v0
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from models.base_mlp import BaseMLPCustom
from models.custom_coma_torch_policy import CustomCOMATorchPolicy, CustomCOMATrainer

# Absolute src/ root via the imported package (stable under Ray Tune CWD changes)
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(sunt_env_module.__file__)))


def _as_bool(val, default=True):
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes", "y")


def _resolve_worker_index(env_config):
    if "worker_index" in env_config:
        return int(env_config["worker_index"])
    try:
        from ray.rllib.evaluation.rollout_worker import get_global_worker
        return int(get_global_worker().worker_index)
    except Exception:
        return 0


# ------------------------------
# Custom Environment Wrapper
# ------------------------------
class RLlibSuntBus(MultiAgentEnv):
    def __init__(self, env_config):
        BASE_DIR = _SRC_DIR

        # Load graph
        graph_path = os.path.join(BASE_DIR, "viz", "graph_gtfs_fev_2024.gpickle")
        if not os.path.isfile(graph_path):
            raise FileNotFoundError(
                f"Graph not found at {graph_path}. "
                f"Set PYTHONPATH to include the project src/ directory."
            )
        with open(graph_path, "rb") as f:
            G = pickle.load(f)

        # Load observations
        obs_dir = os.path.join(BASE_DIR, "training_observation")

        def load_pickle(filename):
            path = os.path.join(obs_dir, filename)
            with open(path, "rb") as f:
                return pickle.load(f)

        avg_travel_time_AB = load_pickle("avg_travel_time_AB.pkl")
        future_demand_at_B = load_pickle("future_demand_at_B.pkl")
        occupancy_rate = load_pickle("occupancy_rate.pkl")
        uptime_normalized = load_pickle("uptime_normalized.pkl")
        real_routes = load_pickle("real_routes.pkl")
        route_metadata = load_pickle("route_metadata.pkl")

        # Parallel environment
        self.env = parallel_env(
            network=G,
            actions_amount=3,
            max_steps=1000000,
            num_agents=env_config.get("num_buses", 5),
            avg_travel_time_AB=avg_travel_time_AB,
            future_demand_at_B=future_demand_at_B,
            occupancy_rate=occupancy_rate,
            uptime_normalized=uptime_normalized,
            real_routes=real_routes,
            route_metadata=route_metadata,
            worker_index=_resolve_worker_index(env_config),
            num_workers=env_config.get("num_workers", 1),
            regime_switch_every=int(env_config.get("regime_switch_every", 50)),
            use_context=_as_bool(env_config.get("use_context", True)),
        )

        # Supersuit wrappers
        self.env = pad_observations_v0(self.env)
        self.env = pad_action_space_v0(self.env)

        # Agents
        self.agents = self.env.possible_agents.copy()
        self.num_agents = len(self.agents)

        # Spaces
        self.observation_space = GymDict({"obs": self.env.observation_space(self.agents[0])})
        self.action_space = self.env.action_space(self.agents[0])
        self.action_spaces = {agent: self.env.action_space(agent) for agent in self.agents}

    def reset(self):
        obs = self.env.reset()
        self.agents = list(obs.keys())
        return {a: {"obs": np.array(o)} for a, o in obs.items()}

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        obs = {a: {"obs": np.array(o[a])} for a in o.keys()}
        rewards = {a: r.get(a, 0.0) for a in r.keys()}
        dones = {"__all__": all(d.values())}
        infos = {a: info.get(a, {}) for a in info.keys()}
        self.agents = [a for a in self.agents if not d.get(a, False)]
        return obs, rewards, dones, infos

    def render(self, mode=None):
        self.env.render()
        time.sleep(0.05)
        return True

    def close(self):
        self.env.close()

    def _inner_parallel_env(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return getattr(env, "unwrapped", env)

    def set_regime(self, regime_id):
        """Forward a driver-side regime pin through SuperSuit wrappers."""
        self._inner_parallel_env().set_regime(regime_id)

    def get_env_info(self):
        return {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 1000000,
            "agent_id": self.agents,
            "share_observation_space": self.observation_space,
            "policy_mapping_info": {
                "sunt_bus": {
                    "all_agents_one_policy": False,
                    "one_agent_one_policy": True,
                    "policy_map": {
                        agent_id: f"policy_{i}" for i, agent_id in enumerate(self.agents)
                    }
                }
            }
        }

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Custom IPPO with ESCP context + Fast TRAC")
    parser.add_argument("--stop-timesteps", type=int, default=1_000_000)
    parser.add_argument("--stop-iters", type=int, default=None,
                        help="Optional cap on training iterations (useful for smoke tests)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-buses", type=int, default=5,
                        help="Número de agentes (ônibus) operando simultaneamente no ambiente")
    parser.add_argument("--local-mode", action="store_true")
    parser.add_argument("--regime-switch-every", type=int, default=50)
    parser.add_argument("--fixed-batch-timesteps", type=int, default=1000)
    parser.add_argument("--checkpoint-freq", type=int, default=200)
    parser.add_argument("--experiment-name", type=str, default="", 
                        help="Nome customizado da pasta dentro de exp_results para organizar no TensorBoard")
    parser.add_argument(
        "--no-context", action="store_true",
        help="Ablation: keep regime switches but drop the 9th ESCP context channel")
    parser.add_argument(
        "--no-trac", action="store_true",
        help="Ablation: disable Fast-TRAC L2 (lambda_min=lambda_max=0)")
    parser.add_argument(
        "--ablation-baseline", action="store_true",
        help="Shorthand for --no-context --no-trac (regimes still on)")
    return parser.parse_known_args()[0]


def _ensure_marllib_env_yaml():
    try:
        import marllib
    except ImportError:
        return
    dest_dir = os.path.join(
        os.path.dirname(marllib.__file__), "envs", "base_env", "config")
    dest = os.path.join(dest_dir, "sunt_bus.yaml")
    src = os.path.join(_SRC_DIR, "sunt_bus.yaml")
    if os.path.isfile(src):
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"[train_custom_a2c] wrote env yaml -> {dest}")


_args = _parse_args()
if _args.ablation_baseline:
    _args.no_context = True
    _args.no_trac = True
_use_context = not _args.no_context
_ensure_marllib_env_yaml()

# MARLlib COMA hardcodes stock COMATorchPolicy; swap in the TRAC loss.
if hasattr(marl_coma_script, "COMATrainer"):
    marl_coma_script.COMATrainer = CustomCOMATrainer
if hasattr(marl_coma_core, "COMATrainer"):
    marl_coma_core.COMATrainer = CustomCOMATrainer

# Fix MARLlib COMA hardcoding sgd_minibatch_size > episode_limit which breaks Ray
_orig_tune_run = marl_coma_script.tune.run
def _patched_tune_run(*args, **kwargs):
    config = kwargs.get("config")
    if config:
        # Override the weird MARLlib logic
        if "sgd_minibatch_size" in config: del config["sgd_minibatch_size"]
        if "num_sgd_iter" in config: del config["num_sgd_iter"]
        
        config["train_batch_size"] = _args.fixed_batch_timesteps
        config["model"]["max_seq_len"] = 1000  # Avoid OOM in models if ever used
    
    # MARLlib forcibly ignores the user's local_dir. We restore it here.
    if _args.experiment_name:
        kwargs["local_dir"] = local_dir_path
    else:
        # Default to coma_gru_sunt_bus instead of ppo
        kwargs["local_dir"] = os.path.join(_SRC_DIR, "exp_results", "coma_gru_sunt_bus")
        
    if "name" in kwargs:
        kwargs["name"] = kwargs["name"].replace("COMA", "CustomCOMA")
    
    return _orig_tune_run(*args, **kwargs)
marl_coma_script.tune.run = _patched_tune_run

_orig_register = ModelCatalog.register_custom_model

def _register_custom_model(name, cls):
    if name == "Base_Model":
        cls = BaseMLPCustom
        print("✅ Forcing Base_Model -> BaseMLPCustom (ESCP + TRAC)")
    return _orig_register(name, cls)

ModelCatalog.register_custom_model = _register_custom_model

# ------------------------------
# Register environment
# ------------------------------
ENV_REGISTRY["sunt_bus"] = RLlibSuntBus
_made_env = marl.make_env(
    environment_name="sunt_bus",
    map_name="sunt_bus",
    force_coop=False,
    regime_switch_every=_args.regime_switch_every,
    use_context=_use_context,
    num_buses=_args.num_buses,
)
# Local MARLlib returns (env_instance, config); older conda package returns a dict.
_new_marllib_api = isinstance(_made_env, tuple)
env_handle = _made_env

# ------------------------------
# Select algorithm
# ------------------------------
algo = getattr(marl.algos, _ALGO_ATTR)(hyperparam_source="common")

# ------------------------------
# Execution configurations
# ------------------------------
local_dir_path = "exp_results"
if _args.experiment_name:
    local_dir_path = os.path.join("exp_results", _args.experiment_name)

run_config = {
    "local_mode": _args.local_mode,
    "local_dir": local_dir_path,
    "stop": {"timesteps_total": _args.stop_timesteps},
    "checkpoint_freq": _args.checkpoint_freq,
    "num_gpus": 0,
    "num_workers": _args.num_workers,
    "share_policy": "individual",
    "fixed_batch_timesteps": _args.fixed_batch_timesteps,
    "model_arch_args": {
        "core_arch": "gru",
        "encode_layer": "128-128",
        "custom_model": "CCMLPCustom",
        "custom_model_config": {
            "use_context": _use_context,
            "hidden_state_size": 128,
            "global_state_flag": False,
            "opp_action_in_cc": True,
        }
    }
}

custom_config = {
    "lr": 0.0003,
    "batch_episode": 20,
    "gamma": 0.99,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.05, 
    "entropy_coeff_schedule": [[0, 0.05], [800000, 0.001], [1000000, 0.0001]],
    "use_gae": True,
    "lambda": 0.95,
    "use_context": _use_context,
    "trac_lambda_min": 0.0,
    "trac_lambda_max": 0.0 if _args.no_trac else 0.5,
    "trac_lambda_k": 2.0, 
    "trac_anchor_every": 100,
    "trac_refresh_alert_max": 0.15,
    "trac_context_weight": 1.0,
}

final_config = run_config.copy()
final_config.update(custom_config)
stop_conditions = final_config.pop("stop")
if _args.stop_iters is not None:
    stop_conditions["training_iteration"] = _args.stop_iters

print(
    "[train_custom_ppo] "
    f"regimes=ON context={'ON' if _use_context else 'OFF'} "
    f"trac={'OFF' if _args.no_trac else 'ON'} | "
    f"stop={stop_conditions} workers={_args.num_workers} "
    f"regime_every={_args.regime_switch_every} batch_ts={_args.fixed_batch_timesteps}"
)

# ------------------------------
# Train
# ------------------------------
if _new_marllib_api:
    model_pref = {
        "core_arch": "gru",
        "encode_layer": "128-128",
        "custom_model": "CCMLPCustom",
        "custom_model_config": {
            "use_context": _use_context,
            "hidden_state_size": 128,
            "global_state_flag": False,
            "opp_action_in_cc": True,
        }
    }
    _model_cls, model_config = marl.build_model(env_handle, algo, model_pref)
    
    # Pass the actual model_config instead of a tuple to MARLlib's fit
    model_dict = {
        "custom_model": "CCMLPCustom",
        "custom_model_config": model_config["custom_model_config"],
    }
    # Need to update final config with model dictionary for COMA instead of just passing the tuple
    final_config["model"] = model_dict
    
    ModelCatalog.register_custom_model("CCMLPCustom", BaseMLPCustom)
    algo.fit(env_handle, model_config, stop=stop_conditions, **final_config)
else:
    algo.fit(env_handle, stop=stop_conditions, **final_config)
