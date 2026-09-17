import os
import argparse
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.marl.algos.core.IL import a2c as marl_a2c_core
try:
    from marllib.marl.algos.scripts import ia2c as marl_a2c_script
    _ALGO_ATTR = "ia2c"
except ImportError:
    from marllib.marl.algos.scripts import a2c as marl_a2c_script
    _ALGO_ATTR = "a2c"

from src.envs.non_stationary_mpe import NonStationaryMPEWrapper
from src.models.base_mlp import BaseMLPCustom
from src.models.custom_a3c_torch_policy import CustomIA2CTrainer, CustomA3CTorchPolicy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-timesteps", type=int, default=200_000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--regime-switch-every", type=int, default=50)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--no-trac", action="store_true")
    args = parser.parse_args()

    use_context = not args.no_context

    # ------------------------------
    # 1. Monkey-Patch Fast TRAC Policy into MARLlib
    # ------------------------------
    if hasattr(marl_a2c_script, "IA2CTrainer"):
        marl_a2c_script.IA2CTrainer = CustomIA2CTrainer
    if hasattr(marl_a2c_core, "IA2CTrainer"):
        marl_a2c_core.IA2CTrainer = CustomIA2CTrainer
    if hasattr(marl_a2c_core, "IA2CTorchPolicy"):
        marl_a2c_core.IA2CTorchPolicy = CustomA3CTorchPolicy

    # ------------------------------
    # 2. Register Custom Model
    # ------------------------------
    _orig_register = ModelCatalog.register_custom_model
    def _register_custom_model(name, cls):
        if name == "Base_Model":
            cls = BaseMLPCustom
            print("✅ Overriding Base_Model -> BaseMLPCustom (ESCP + TRAC)")
        return _orig_register(name, cls)
    ModelCatalog.register_custom_model = _register_custom_model
    _orig_register("Base_Model", BaseMLPCustom)
    
    # ------------------------------
    # 3. Register Custom Environment
    # ------------------------------
    env_name = "non_stationary_mpe"
    ENV_REGISTRY[env_name] = NonStationaryMPEWrapper
    
    # Fake the YAML required by MARLlib internal builder
    try:
        import marllib
        dest_dir = os.path.join(os.path.dirname(marllib.__file__), "envs", "base_env", "config")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, f"{env_name}.yaml")
        with open(dest, "w") as f:
            f.write(f"env: {env_name}\n")
            f.write("env_args:\n")
            f.write("  map_name: non_stationary_mpe\n")
            f.write("  regime_switch_every: 50\n")
            f.write("  use_context: True\n")
            f.write("mask_flag: False\n")
            f.write("global_state_flag: False\n")
            f.write("opp_action_in_cc: False\n")
            f.write("agent_level_batch_update: False\n")
    except Exception as e:
        print(f"Warning: could not write {env_name}.yaml: {e}")

    # Build environment via MARLlib wrapper factory
    env_handle = marl.make_env(
        environment_name=env_name,
        map_name=env_name,
        force_coop=False,
        regime_switch_every=args.regime_switch_every,
        use_context=use_context
    )
    
    # Check if new or old MARLlib API return format
    _new_marllib_api = isinstance(env_handle, tuple)
    if _new_marllib_api:
        env_instance, env_config_dict = env_handle
    else:
        env_instance = env_handle

    # ------------------------------
    # 4. Initialize IA2C Algorithm
    # ------------------------------
    # Load MPE hyperparameters
    algo = getattr(marl.algos, _ALGO_ATTR)(hyperparam_source="mpe")

    # ------------------------------
    # 5. Configuration & Training
    # ------------------------------
    run_config = {
        "stop": {"timesteps_total": args.stop_timesteps},
        "checkpoint_freq": 50,
        "num_gpus": 0,
        "num_workers": args.num_workers,
        "share_policy": "individual",
    }
    
    custom_config = {
        "lr": 0.0005,
        "gamma": 0.99,
        "vf_loss_coeff": 1.0,
        "entropy_coeff": 0.01,
        "entropy_coeff_schedule": [[0, 0.01], [args.stop_timesteps, 0.0001]],
        "use_gae": True,
        "lambda": 1.0,
        
        # Fast TRAC Configurations
        "use_context": use_context,
        "trac_lambda_min": 0.0,
        "trac_lambda_max": 0.0 if args.no_trac else 8.0,
        "trac_lambda_k": 4.0,
        "trac_anchor_every": 20,
        "trac_refresh_alert_max": 0.2,
        "trac_context_weight": 1.0,
    }
    
    final_config = run_config.copy()
    final_config.update(custom_config)
    stop_conditions = final_config.pop("stop")

    print(f"\n🚀 Starting Fast Benchmark Training on {env_name}")
    print(f"Regimes: ON | Context: {'ON' if use_context else 'OFF'} | TRAC: {'OFF' if args.no_trac else 'ON'}\n")

    if _new_marllib_api:
        model_pref = {
            "core_arch": "mlp",
            "fc_layer": 2,
            "out_dim_fc_0": 128,
            "out_dim_fc_1": 128,
            "encode_layer": "128-128",
        }
        _model_cls, model_config = marl.build_model(env_handle, algo, model_pref)
        model = (BaseMLPCustom, model_config)
        algo.fit(env_handle, model, stop=stop_conditions, **final_config)
    else:
        algo.fit(env_handle, stop=stop_conditions, **final_config)

if __name__ == "__main__":
    main()
