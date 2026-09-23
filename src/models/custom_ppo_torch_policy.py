import gym
from typing import Dict, Type, Union, List
import ray
from ray.rllib.agents.ppo.ppo_torch_policy import (
    ppo_surrogate_loss,
    kl_and_loss_stats,
    vf_preds_fetches,
    setup_config,
    setup_mixins,
    KLCoeffMixin
)
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import apply_grad_clipping
from ray.rllib.utils.typing import TensorType

from ray.rllib.agents.ppo.ppo_torch_policy import ValueNetworkMixin

torch, nn = try_import_torch()

_TRAC_LOGGED = False

def _batch_context_alert(train_batch: SampleBatch) -> float:
    """Mean of the last obs dim (ESCP context). 0 if the batch is still 8-D."""
    obs = train_batch[SampleBatch.OBS]
    if isinstance(obs, dict):
        obs = obs.get("obs", None)
    if obs is None or not torch.is_tensor(obs) or obs.dim() < 2 or obs.shape[-1] < 9:
        return 0.0
    return float(obs[..., -1].detach().mean().item())


def trac_ppo_surrogate_loss(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        
    global _TRAC_LOGGED
    if not _TRAC_LOGGED:
        print("✅ CustomPPOTorchPolicy + Fast-TRAC loss active")
        _TRAC_LOGGED = True

    # 1. Compute original PPO loss
    total_loss = ppo_surrogate_loss(policy, model, dist_class, train_batch)
    
    # Extract value error for shift detection
    # `ppo_surrogate_loss` has already populated model.tower_stats["mean_vf_loss"]
    value_err = model.tower_stats.get("mean_vf_loss", 0.0)
    
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

    # 3. Store new stats for kl_and_loss_stats
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

def trac_kl_and_loss_stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    # Get standard PPO stats
    stats = kl_and_loss_stats(policy, train_batch)
    
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

CustomPPOTorchPolicy = build_policy_class(
    name="CustomPPOTorchPolicy",
    framework="torch",
    get_default_config=lambda: PPO_CONFIG,
    loss_fn=trac_ppo_surrogate_loss,
    stats_fn=trac_kl_and_loss_stats,
    extra_action_out_fn=vf_preds_fetches,
    postprocess_fn=compute_gae_for_sample_batch,
    extra_grad_process_fn=apply_grad_clipping,
    before_init=setup_config,
    before_loss_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ],
)

# Custom Trainer mapping to this policy
CustomIPPOTrainer = PPOTrainer.with_updates(
    name="CustomIPPOTrainer",
    default_policy=None,
    get_policy_class=lambda config: CustomPPOTorchPolicy,
)
