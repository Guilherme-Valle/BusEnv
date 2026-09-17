# Custom Base MLP model for IA2C

from ray.rllib.utils.torch_ops import FLOAT_MIN
from functools import reduce
import math
import numpy as np
from gym.spaces import Dict as GymDict
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List

torch, nn = try_import_torch()

try:
    from marllib.marl.models.zoo.encoder.base_encoder import BaseEncoder
except ImportError:  # stock MARLlib 1.x ships RNN zoo only
    BaseEncoder = None


class BaseMLPCustom(TorchModelV2, nn.Module):
    """Generic fully connected network with a learner-side TD window and TRAC anchors."""

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        print(f"✅ BaseMLPCustom initialized: {name}")

        self.custom_config = model_config["custom_model_config"]
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.n_agents = self.custom_config.get("num_agents", 1)
        self.activation = model_config.get("fcnet_activation")

        in_size = self._infer_obs_size()
        self._obs_in_size = in_size
        self.p_encoder = self._build_encoder(in_size)
        self.vf_encoder = self._build_encoder(in_size)

        self.p_branch = SlimFC(
            in_size=self.p_encoder.output_dim,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        self.vf_branch = SlimFC(
            in_size=self.vf_encoder.output_dim,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        self._features = None
        self._last_obs = None
        self.q_flag = False

        self.actors = [self.p_encoder, self.p_branch]
        self.critics = [self.vf_encoder, self.vf_branch]
        self.actor_initialized_parameters = self.actor_parameters()

        cfg = self.custom_config if isinstance(self.custom_config, dict) else {}
        self.td_window_size = int(cfg.get("td_window_size", 10))
        self.shift_threshold_multiplier = float(cfg.get("shift_threshold_multiplier", 3.0))
        self.trac_lambda_min = float(cfg.get("trac_lambda_min", 0.0))
        self.trac_lambda_max = float(cfg.get("trac_lambda_max", 8.0))
        self.trac_lambda_k = float(cfg.get("trac_lambda_k", 4.0))
        self.trac_anchor_every = int(cfg.get("trac_anchor_every", 200))
        self.trac_refresh_alert_max = float(cfg.get("trac_refresh_alert_max", 0.2))
        self.trac_context_weight = float(cfg.get("trac_context_weight", 1.0))

        # Learner-side truncated TD window (not fed back into obs).
        self.register_buffer("td_window", torch.zeros(self.td_window_size))
        self.register_buffer("td_write_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("td_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("td_var_buf", torch.zeros(1))
        self.register_buffer("td_mean_buf", torch.zeros(1))
        self.register_buffer("shift_flag", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("trac_update_count", torch.zeros(1, dtype=torch.long))
        self.shift_detected = False

        self._register_trac_anchors()

    def _infer_obs_size(self) -> int:
        space = self.full_obs_space
        if isinstance(space, GymDict) and "obs" in space.spaces:
            size = int(np.prod(space["obs"].shape))
        else:
            size = int(np.prod(getattr(space, "shape", (9,))))
        return size if size > 0 else 8

    def _build_encoder(self, in_size: int):
        if BaseEncoder is not None:
            return BaseEncoder(self.model_config, self.full_obs_space)

        arch = {}
        if isinstance(self.custom_config, dict):
            arch = dict(self.custom_config.get("model_arch_args") or {})
        if "fc_layer" not in arch:
            arch = {"fc_layer": 2, "out_dim_fc_0": 128, "out_dim_fc_1": 128}

        layers = []
        dim = in_size
        for i in range(int(arch["fc_layer"])):
            out_dim = int(arch.get("out_dim_fc_{}".format(i), 128))
            layers.append(SlimFC(
                dim, out_dim,
                initializer=normc_initializer(1.0),
                activation_fn=nn.ReLU,
            ))
            dim = out_dim
        encoder = nn.Sequential(*layers)
        encoder.output_dim = dim
        return encoder

    def _register_trac_anchors(self) -> None:
        """Frozen copies of current weights (Actor only). Buffers ride along in state_dict()."""
        self._trac_pairs = []
        for name, param in list(self.named_parameters()):
            # Decouple: Anchor only the Actor. The Critic must be completely free to learn the shifted value function.
            if not (name.startswith("p_encoder") or name.startswith("p_branch")):
                continue
            buf_name = "trac_ref__" + name.replace(".", "_")
            self.register_buffer(buf_name, param.detach().clone())
            self._trac_pairs.append((name, buf_name))

    def refresh_trac_anchor(self) -> None:
        with torch.no_grad():
            params = dict(self.named_parameters())
            for name, buf_name in self._trac_pairs:
                getattr(self, buf_name).copy_(params[name].detach())

    def maybe_refresh_trac_anchor(self, shift_detected: bool, alert: float = 0.0) -> None:
        """Move the TRAC reference only while the world looks calm."""
        with torch.no_grad():
            self.trac_update_count.add_(1)
            if shift_detected or float(alert) > self.trac_refresh_alert_max:
                return
            if int(self.trac_update_count.item()) % max(1, self.trac_anchor_every) == 0:
                self.refresh_trac_anchor()

    def trac_l2(self):
        """Sum of squared distance to the mobile anchor (not averaged — must be able to matter)."""
        penalty = None
        params = dict(self.named_parameters())
        for name, buf_name in self._trac_pairs:
            param = params[name]
            if not param.requires_grad:
                continue
            ref = getattr(self, buf_name)
            term = (param - ref.detach()).pow(2).sum()
            penalty = term if penalty is None else penalty + term
        if penalty is None:
            return self.td_var_buf.new_zeros(())
        return penalty

    def trac_lambda(self, extra_alert: float = 0.0) -> float:
        """High alert (TD var and/or env context) → plastic. Calm → consolidate."""
        raw_var_h = float(self.td_var_buf.detach().item())
        # Normalize Var_TD to a [0, 1) scale (same as context `c`) so neither dominates arbitrarily
        norm_var_h = math.tanh(raw_var_h)
        alert = norm_var_h + self.trac_context_weight * float(extra_alert)
        return self.trac_lambda_min + (self.trac_lambda_max - self.trac_lambda_min) * math.exp(
            -self.trac_lambda_k * alert
        )

    def update_td_error(self, batch_mean_err: float) -> bool:
        """Slide a length-H TD-error window and flag a critic spike.

        Returns True when the new residual is a shift relative to the window.
        """
        batch_mean_err = float(batch_mean_err)
        h = self.td_window_size

        with torch.no_grad():
            n = int(self.td_count.item())
            filled = self.td_window[:n] if 0 < n < h else self.td_window
            shift = False
            if n >= 3:
                mean = float(filled.mean().item())
                std = float(filled.std(unbiased=False).item())
                self.td_mean_buf.fill_(mean)
                if std > 1e-8 and batch_mean_err > mean + self.shift_threshold_multiplier * std:
                    shift = True
                    print(
                        "🚨 [Change Detector] Distribution Shift Detected! "
                        f"TD-Error spiked (mean_err={batch_mean_err:.6f}, "
                        f"window_mean={mean:.6f}, window_std={std:.6f})."
                    )

            ptr = int(self.td_write_ptr.item())
            self.td_window[ptr] = batch_mean_err
            self.td_write_ptr.fill_((ptr + 1) % h)
            self.td_count.fill_(min(h, n + 1))

            n_after = int(self.td_count.item())
            filled_after = self.td_window[:n_after] if n_after < h else self.td_window
            var = float(filled_after.var(unbiased=False).item()) if n_after > 1 else 0.0
            self.td_var_buf.fill_(var)
            self.td_mean_buf.fill_(float(filled_after.mean().item()))
            self.shift_flag.fill_(shift)
            self.shift_detected = shift

        return shift

    def _ensure_obs_dim(self, flat_inputs):
        """Pad a short obs vector to the encoder's expected width."""
        need = int(getattr(self, "_obs_in_size", flat_inputs.shape[-1]))
        got = int(flat_inputs.shape[-1])
        if got < need:
            return torch.cat(
                [flat_inputs, flat_inputs.new_zeros(flat_inputs.shape[0], need - got)],
                dim=1,
            )
        return flat_inputs

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        if self.custom_config.get("global_state_flag") or self.custom_config.get("mask_flag"):
            flat_inputs = input_dict["obs"]["obs"].float()
            if self.custom_config.get("mask_flag"):
                action_mask = input_dict["obs"]["action_mask"]
                inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        else:
            flat_inputs = input_dict["obs"]["obs"].float()

        self.inputs = self._ensure_obs_dim(flat_inputs)
        self._features = self.p_encoder(self.inputs)

        output = self.p_branch(self._features)

        if self.custom_config["mask_flag"]:
            output = output + inf_mask

        return output, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        B = self._features.shape[0]
        x = self.vf_encoder(self.inputs)

        if self.q_flag:
            return torch.reshape(self.vf_branch(x), [B, -1])
        else:
            return torch.reshape(self.vf_branch(x), [-1])

    def actor_parameters(self):
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.actors))

    def critic_parameters(self):
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.critics))
