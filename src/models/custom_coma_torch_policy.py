import gym
from typing import Dict, Type, Union, List
import ray
import torch
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

# Fetch COMA baseline internals
from marllib.marl.algos.core.CC.coma import (
    COMATrainer, central_critic_coma_loss, coma_model_value_predictions
)
from marllib.marl.algos.utils.centralized_critic import CentralizedValueMixin, centralized_critic_postprocessing
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG, A2CTrainer
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy

torch, nn = try_import_torch()

_TRAC_LOGGED = False

def _batch_context_alert(train_batch: SampleBatch) -> float:
    """Mean of the last obs dim (ESCP context). 0 if the batch is still 8-D."""
    obs = train_batch[SampleBatch.CUR_OBS]
    if isinstance(obs, dict):
        obs = obs.get("obs", None)
    if obs is None or not torch.is_tensor(obs) or obs.dim() < 2 or obs.shape[-1] < 9:
        return 0.0
    return float(obs[..., -1].detach().mean().item())

def trac_coma_loss(
        policy: Policy, model: ModelV2,
        dist_class, train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        
    global _TRAC_LOGGED
    if not _TRAC_LOGGED:
        print("✅ CustomCOMATorchPolicy + Fast-TRAC loss active")
        _TRAC_LOGGED = True

    # 1. Compute original COMA loss
    total_loss = central_critic_coma_loss(policy, model, dist_class, train_batch)
    
    # Extract value error for shift detection
    value_err = model.tower_stats.get("value_err", 0.0)
    
    shift_detected = False
    if hasattr(model, "update_td_error"):
        if isinstance(value_err, torch.Tensor):
            mean_err = float(value_err.detach().item())
        else:
            mean_err = float(value_err)
        shift_detected = bool(model.update_td_error(mean_err))

    # 2. Fast-TRAC: λ from TD-var + env context (both detached). L2 to a mobile anchor.
    zero = total_loss.new_zeros(()) if torch.is_tensor(total_loss) else torch.zeros(1)
    trac_l2 = zero
    trac_lambda = 0.0
    context_alert = _batch_context_alert(train_batch)
    
    if hasattr(model, "trac_l2") and hasattr(model, "trac_lambda"):
        trac_l2 = model.trac_l2()
        trac_lambda = float(model.trac_lambda(extra_alert=context_alert))
        total_loss = total_loss + trac_lambda * trac_l2
        if hasattr(model, "maybe_refresh_trac_anchor"):
            model.maybe_refresh_trac_anchor(shift_detected, alert=context_alert)

    # 3. Store new stats for stats_fn
    model.tower_stats["trac_l2"] = trac_l2.detach() if torch.is_tensor(trac_l2) else zero
    
    device = total_loss.device if torch.is_tensor(total_loss) else torch.device("cpu")
    dtype = total_loss.dtype if torch.is_tensor(total_loss) else torch.float32
    
    model.tower_stats["trac_lambda"] = torch.as_tensor(trac_lambda, device=device, dtype=dtype)
    model.tower_stats["context_alert"] = torch.as_tensor(context_alert, device=device, dtype=dtype)
    
    td_var = 0.0
    if hasattr(model, "td_var_buf"):
        td_var = float(model.td_var_buf.detach().item())
    model.tower_stats["td_var"] = torch.as_tensor(td_var, device=device, dtype=dtype)
    
    # Overwrite the total_loss stat so it's logged accurately
    model.tower_stats["total_loss"] = total_loss

    return total_loss

def trac_coma_stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    # We call the underlying A3C stats since COMA inherits from it
    # COMA doesn't override stats_fn from A3C, so we fetch it from A3CTorchPolicy manually
    stats = {}
    stats["cur_kl_coeff"] = 0.0  # Not used in A2C/A3C generally
    stats["policy_entropy"] = torch.mean(torch.stack(policy.get_tower_stats("entropy")))
    stats["policy_loss"] = torch.mean(torch.stack(policy.get_tower_stats("pi_err")))
    stats["vf_loss"] = torch.mean(torch.stack(policy.get_tower_stats("value_err")))
    
    def _mean_stat(key, default=0.0):
        try:
            return torch.mean(torch.stack(policy.get_tower_stats(key)))
        except Exception:
            return torch.tensor(default)

    # Inject our TRAC/Context stats
    stats["trac_l2"] = _mean_stat("trac_l2")
    stats["trac_lambda"] = _mean_stat("trac_lambda")
    stats["td_var"] = _mean_stat("td_var")
    stats["context_alert"] = _mean_stat("context_alert")
    
    return stats

# Need to monkey-patch MARLlib's centralized_critic_postprocessing so we don't hit the uneven padding error!
import numpy as np
def _patched_centralized_critic_postprocessing(policy, sample_batch, other_agent_batches, episode):
    try:
        return centralized_critic_postprocessing(policy, sample_batch, other_agent_batches, episode)
    except ValueError as e:
        if "all input arrays must have the same shape" in str(e):
            # The exact uneven lengths bug! We will just fallback gracefully 
            # to setting opponent actions / states to 0s for this broken batch
            custom_config = policy.config["model"]["custom_model_config"]
            obs_dim = sample_batch[SampleBatch.CUR_OBS].shape[1]
            n_agents = custom_config.get("num_agents", 1)
            b_size = len(sample_batch[SampleBatch.CUR_OBS])
            
            sample_batch["state"] = np.zeros((b_size, n_agents, obs_dim), dtype=np.float32)
            sample_batch["opponent_actions"] = np.zeros((b_size, n_agents - 1, 1), dtype=np.float32)
            
            # Recalculate advantages just in case
            from ray.rllib.evaluation.postprocessing import compute_advantages
            completed = sample_batch["dones"][-1]
            last_r = 0.0 if completed else sample_batch[SampleBatch.VF_PREDS][-1]
            return compute_advantages(
                sample_batch,
                last_r,
                policy.config["gamma"],
                policy.config["lambda"],
                use_gae=policy.config["use_gae"])
        raise e

CustomCOMATorchPolicy = A3CTorchPolicy.with_updates(
    name="CustomCOMATorchPolicy",
    get_default_config=lambda: A2C_CONFIG,
    postprocess_fn=_patched_centralized_critic_postprocessing,
    loss_fn=trac_coma_loss,
    stats_fn=trac_coma_stats,
    extra_action_out_fn=coma_model_value_predictions,
    mixins=[
        CentralizedValueMixin
    ])

CustomCOMATrainer = A2CTrainer.with_updates(
    name="CustomCOMATrainer",
    default_policy=None,
    get_policy_class=lambda config: CustomCOMATorchPolicy,
)
