import typing
from typing import Iterable

import torch
from torch import nn

from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from collections.abc import Callable, Iterable
from vllm.model_executor.layers.shared_fused_moe import SharedFusedMoE
from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM, get_spec_layer_idx_from_weight_name, DeepseekV2Model, ParallelLMHead, PPMissingLayer, LogitsProcessor, DeepseekV2DecoderLayer, DeepseekV2MoE, get_pp_group, maybe_prefix
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.config import VllmConfig
from vllm.config import get_current_vllm_config
from vllm_ascend.ascend_config import get_ascend_config

class CustomDeepseekV2ForCausalLM(DeepseekV2ForCausalLM):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self.vllm_config=get_current_vllm_config()
        ascend_config = get_ascend_config()
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]

        mix_placement = getattr(ascend_config, "mix_placement", False)
        # print(f"{mix_placement=}")
        
        expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if mix_placement #getattr(self.config, "mix_placement", False)
                else 0
            ),
            num_redundant_experts=self.num_redundant_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue

            is_fuse_shared_experts_layer = (
                mix_placement#getattr(self.config, "mix_placement", False)
                and ("mlp.shared_experts" in name)
            )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fuse_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                if (param_name == "fused_qkv_a_proj") and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict.keys():
                    continue
                
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                num_chunks = 1
                if is_fuse_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    split_dim = 1 if "down_proj.weight" in name else 0
                    total = loaded_weight.shape[split_dim]
                    assert total % num_chunks == 0, (
                        f"Shared expert weight dim {total} not divisible by num_chunks {num_chunks}"
                    )
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fuse_shared_experts_layer:
                        if split_dim == 0:
                            weight_to_load = loaded_weight[j * chunk_size : (j + 1) * chunk_size, :]
                        else:
                            weight_to_load = loaded_weight[:, j * chunk_size : (j + 1) * chunk_size]
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )
                        # if 'shared_experts' in name:
                        #     print(f"{name=}")
                        #     print(f"{chunk_name=}")

                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in chunk_name:
                            continue

                        is_expert_weight = True
                        name_mapped = chunk_name.replace(weight_name, param_name)

                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        if name_mapped not in params_dict.keys():
                            continue
                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fuse_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
                            break
                    else:
                        if is_expert_weight:
                            continue
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        name = maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue
                        if is_pp_missing_parameter(name, self):
                            continue
                        if name not in params_dict.keys():
                            continue
                        
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
            if not is_fuse_shared_experts_layer:
                loaded_params.add(name)
        # exit()
        return loaded_params


# DeepseekV2ForCausalLM.__init__ = CustomDeepseekV2ForCausalLM.__init__
DeepseekV2ForCausalLM.load_weights = CustomDeepseekV2ForCausalLM.load_weights