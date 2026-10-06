import gym
from typing import Dict, Type, Union, List
import ray
import torch
from ray.rllib.agents.dqn.dqn_torch_policy import (
    DQNTorchPolicy,
    build_q_losses,
    build_q_model_and_distribution,
    get_distribution_inputs_and_class,
    build_q_stats,
    setup_early_mixins,
    before_loss_init,
    adam_optimizer,
    extra_action_out_fn,
    grad_process_and_td_error_fn
)
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio
from ray.rllib.agents.dqn.simple_q_torch_policy import TargetNetworkMixin
from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.agents.dqn.dqn import DQNTrainer, DEFAULT_CONFIG as DQN_CONFIG
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_ops import concat_multi_gpu_td_errors
from ray.rllib.utils.typing import TensorType

# Since DQNTorchPolicy has ComputeTDErrorMixin
from ray.rllib.agents.dqn.dqn_torch_policy import ComputeTDErrorMixin

_TRAC_LOGGED = False

def _batch_context_alert(train_batch: SampleBatch) -> float:
    """Mean of the last obs dim (ESCP context). 0 if the batch is still 8-D."""
    obs = train_batch[SampleBatch.CUR_OBS]
    if isinstance(obs, dict):
        obs = obs.get("obs", None)
    if obs is None or not torch.is_tensor(obs) or obs.dim() < 2 or obs.shape[-1] < 9:
        return 0.0
    return float(obs[..., -1].detach().mean().item())


def trac_dqn_loss(
        policy: Policy, model: ModelV2,
        dist_class, train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
        
    global _TRAC_LOGGED
    if not _TRAC_LOGGED:
        print("✅ CustomDQNTorchPolicy + Fast-TRAC loss active")
        _TRAC_LOGGED = True

    # 1. Compute original DQN loss
    total_loss = build_q_losses(policy, model, dist_class, train_batch)
    
    # In DQN, loss is usually the td_error squared/huber.
    # The actual loss stat might be "mean_q" or "td_error".
    # Let's use the td_error to detect shifts
    td_error = train_batch.get("td_error") 
    value_err = torch.mean(torch.abs(policy.get_tower_stats("td_error")[0])) if policy.get_tower_stats("td_error") else 0.0
    
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

    # 3. Store new stats
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

def trac_dqn_stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    # Get standard DQN stats
    stats = build_q_stats(policy, train_batch)
    
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

CustomDQNTorchPolicy = build_policy_class(
    name="CustomDQNTorchPolicy",
    framework="torch",
    loss_fn=trac_dqn_loss,
    get_default_config=lambda: DQN_CONFIG,
    make_model_and_action_dist=build_q_model_and_distribution,
    action_distribution_fn=get_distribution_inputs_and_class,
    stats_fn=trac_dqn_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    optimizer_fn=adam_optimizer,
    extra_grad_process_fn=grad_process_and_td_error_fn,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    extra_action_out_fn=extra_action_out_fn,
    before_init=setup_early_mixins,
    before_loss_init=before_loss_init,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ])

# Custom Trainer mapping to this policy
CustomIDQNTrainer = DQNTrainer.with_updates(
    name="CustomIDQNTrainer",
    default_policy=None,
    get_policy_class=lambda config: CustomDQNTorchPolicy,
)
