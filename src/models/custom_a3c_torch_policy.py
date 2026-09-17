import gym
from typing import Optional, Dict

import ray
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG, A2CTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import ValueNetworkMixin
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, \
    Postprocessing
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import Deprecated
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import apply_grad_clipping, sequence_mask
from ray.rllib.utils.typing import TrainerConfigDict, TensorType, \
    PolicyID, LocalOptimizer

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


@Deprecated(
    old="rllib.agents.a3c.a3c_torch_policy.add_advantages",
    new="ray.rllib.evaluation.postprocessing.compute_gae_for_sample_batch",
    error=False)
def add_advantages(
        policy: Policy,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[PolicyID, SampleBatch]] = None,
        episode: Optional[MultiAgentEpisode] = None) -> SampleBatch:

    return compute_gae_for_sample_batch(policy, sample_batch,
                                        other_agent_batches, episode)


def actor_critic_loss(policy: Policy, model: ModelV2,
                      dist_class: ActionDistribution,
                      train_batch: SampleBatch) -> TensorType:
    global _TRAC_LOGGED
    if not _TRAC_LOGGED:
        print("✅ CustomA3CTorchPolicy + Fast-TRAC loss active")
        _TRAC_LOGGED = True

    logits, _ = model.from_batch(train_batch)
    values = model.value_function()

    if policy.is_recurrent():
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask_orig = sequence_mask(train_batch[SampleBatch.SEQ_LENS],
                                  max_seq_len)
        valid_mask = torch.reshape(mask_orig, [-1])
    else:
        valid_mask = torch.ones_like(values, dtype=torch.bool)

    dist = dist_class(logits, model)
    log_probs = dist.logp(train_batch[SampleBatch.ACTIONS]).reshape(-1)
    pi_err = -torch.sum(
        torch.masked_select(log_probs * train_batch[Postprocessing.ADVANTAGES],
                            valid_mask))

    if policy.config["use_critic"]:
        value_err = 0.5 * torch.sum(
            torch.pow(
                torch.masked_select(
                    values.reshape(-1) -
                    train_batch[Postprocessing.VALUE_TARGETS], valid_mask),
                2.0))
    else:
        value_err = 0.0

    entropy = torch.sum(torch.masked_select(dist.entropy(), valid_mask))

    total_loss = (pi_err + value_err * policy.config["vf_loss_coeff"] -
                  entropy * policy.config["entropy_coeff"])

    model.tower_stats["entropy"] = entropy
    model.tower_stats["pi_err"] = pi_err
    model.tower_stats["value_err"] = value_err

    shift_detected = False
    if hasattr(model, "update_td_error"):
        if isinstance(value_err, torch.Tensor):
            n_valid = max(1, int(valid_mask.sum().detach().item()))
            mean_err = float((value_err.detach() / n_valid).item())
        else:
            mean_err = float(value_err)
        shift_detected = bool(model.update_td_error(mean_err))

    # Fast-TRAC: λ from TD-var + env context (both detached). L2 to a mobile anchor.
    zero = pi_err.new_zeros(())
    trac_l2 = zero
    trac_lambda = 0.0
    context_alert = _batch_context_alert(train_batch)
    if hasattr(model, "trac_l2") and hasattr(model, "trac_lambda"):
        trac_l2 = model.trac_l2()
        trac_lambda = float(model.trac_lambda(extra_alert=context_alert))
        total_loss = total_loss + trac_lambda * trac_l2
        if hasattr(model, "maybe_refresh_trac_anchor"):
            model.maybe_refresh_trac_anchor(shift_detected, alert=context_alert)

    model.tower_stats["trac_l2"] = trac_l2.detach() if torch.is_tensor(trac_l2) else zero
    model.tower_stats["trac_lambda"] = torch.as_tensor(
        trac_lambda, device=pi_err.device, dtype=pi_err.dtype)
    model.tower_stats["context_alert"] = torch.as_tensor(
        context_alert, device=pi_err.device, dtype=pi_err.dtype)
    td_var = 0.0
    if hasattr(model, "td_var_buf"):
        td_var = float(model.td_var_buf.detach().item())
    model.tower_stats["td_var"] = torch.as_tensor(
        td_var, device=pi_err.device, dtype=pi_err.dtype)

    return total_loss


def loss_and_entropy_stats(policy: Policy,
                           train_batch: SampleBatch) -> Dict[str, TensorType]:

    def _mean_stat(key, default=0.0):
        try:
            return torch.mean(torch.stack(policy.get_tower_stats(key)))
        except Exception:
            return torch.tensor(default)

    return {
        "policy_entropy": _mean_stat("entropy"),
        "policy_loss": _mean_stat("pi_err"),
        "vf_loss": _mean_stat("value_err"),
        "trac_l2": _mean_stat("trac_l2"),
        "trac_lambda": _mean_stat("trac_lambda"),
        "td_var": _mean_stat("td_var"),
        "context_alert": _mean_stat("context_alert"),
    }


def model_value_predictions(
        policy: Policy, input_dict: Dict[str, TensorType], state_batches,
        model: ModelV2,
        action_dist: ActionDistribution) -> Dict[str, TensorType]:
    return {SampleBatch.VF_PREDS: model.value_function()}


def torch_optimizer(policy: Policy,
                    config: TrainerConfigDict) -> LocalOptimizer:
    return torch.optim.Adam(policy.model.parameters(), lr=config["lr"])


def setup_mixins(policy: Policy, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 config: TrainerConfigDict) -> None:
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)


CustomA3CTorchPolicy = build_policy_class(
    name="CustomA3CTorchPolicy",
    framework="torch",
    get_default_config=lambda: A2C_CONFIG,
    loss_fn=actor_critic_loss,
    stats_fn=loss_and_entropy_stats,
    postprocess_fn=compute_gae_for_sample_batch,
    extra_action_out_fn=model_value_predictions,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=torch_optimizer,
    before_loss_init=setup_mixins,
    mixins=[ValueNetworkMixin],
)


def get_policy_class_custom_ia2c(config_):
    if config_["framework"] == "torch":
        return CustomA3CTorchPolicy


CustomIA2CTrainer = A2CTrainer.with_updates(
    name="CustomIA2CTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_custom_ia2c,
)
