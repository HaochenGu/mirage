#!/usr/bin/env python3
"""
Starting to build final ea_model with mpk:

Step1 Operation to be implemented:
    - [] kernel of target model, 
    - [] kernel of draft model, 
    - [] kernel of tree generation, 
    - [] kernel of verification, 
 
Step2 Bridging tensors across different kernels
"""
import os
import sys
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List, Tuple, Optional
import math
import json
import argparse
from dataclasses import dataclass

# Add mirage to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import mirage as mi


@dataclass
class EAGLEConfig:
    """Configuration for EAGLE model."""
    # Base model config
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    vocab_size: int = 128256
    intermediate_size: int = 11008
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # EAGLE-specific config
    draft_hidden_layers: int = 1  # Single decoder layer for draft model
    tree_size: int = 64  # Maximum tree size
    top_k: int = 10  # Top-k for each tree expansion
    tree_depth: int = 6  # Maximum tree depth
    temperature: float = 0.6  # Temperature for draft model
    
    # EAGLE-3 specific
    use_multi_layer_fusion: bool = True  # Use features from multiple layers
    low_layer_idx: int = 0  # Index for low-level features
    mid_layer_idx: int = 16  # Index for mid-level features
    high_layer_idx: int = 31  # Index for high-level features
    
    # MPK config
    max_batch_size: int = 1
    max_seq_len: int = 4096
    max_new_tokens: int = 512


class EAGLE_MPK:

    def __init__(
        self,
        config: EAGLEConfig,
        world_size: int = 1,
        rank: int = 0,
        meta_tensor: tuple[torch.Tensor, torch.Tensor] = None,
    ):
        self.config = config
        self.world_size = world_size
        self.rank = rank
        self.meta_tensor = meta_tensor
        
        # Initialize MPK
        mi.init()
        self.mpk = mi.PersistentKernel(
            world_size=world_size,
            mpi_rank=rank,
            meta_tensor=self.meta_tensor,
        )

        # self._initialize_target_model_weights()
        # self._initialize_draft_model_weights()
        # self._target_model()
        # self._draft_model()
        # self._tree_generation()
        # self._verification()
        # self.mpk.compile()
    
    def _initialize_draft_model_weights(self):
        """
        Initialize weights for draft model.
        """
        return

    def _initialize_draft_tensors(self):
        """
        Initialize tensors for draft model following Qwen3 pattern.
        """
        
        return
    
    def _initialize_tree_tensors(self):
        """
        Initialize tensors for tree generation.
        """
    
    
    def _draft_model(self):
        """
        mpk implementation of forward pass of draft model.
        """

        self._initialize_draft_tensors()

        config = self.config
        
        self.mpk.concat_layer( # NOT IMPLEMENTED YET
            inputs=[self.low_features, self.mid_features, self.high_features],
            outputs=self.concat_features,
            grid_dim=(1, 1, 1),
            block_dim=(512, 1, 1),
            axis=2,
        )
        
        self.mpk.linear_with_residual_layer(
            input=self.concat_features,
            weight=self.draft_fc_weight,
            residue = self.zero_residue
            output=self.fused_features,
            grid_dim=(96, 1, 1),
            block_dim=(512, 1, 1),
        )
        

        self.mpk.multi_token_embed_layer(
            token_ids=self.current_token_ids,
            weight=self.draft_embed_weight,
            output=self.token_embeds,
            grid_dim=(1, 1, 1),
            block_dim=(512, 1, 1),
            max_tokens=config.tree_size,
        )

        self.mpk.multi_token_rmsnorm_layer( # NOT IMPLEMENTED YETs
            input=self.token_embeds,
            weight=self.draft_norm1_weight,
            output=self.token_embeds_normed,
            grid_dim=(96, 1, 1),
            block_dim=(512, 1, 1),
        )

        self.mpk.multi_token_rmsnorm_layer( # NOT IMPLEMENTED YET
            input=self.fused_features,
            weight=self.draft_norm1_weight,
            output=self.fused_features_normed,
            grid_dim=(96, 1, 1),
            block_dim=(512, 1, 1),
        )

        self.mpk.concat_layer( # NOT IMPLEMENTED YET
            inputs=[self.token_embeds_normed, self.fused_features_normed],
            outputs=self.attn_input,
            grid_dim=(96, 1, 1),
            block_dim=(512, 1, 1),
            embed_dim=config.hidden_size,
            num_tokens=config.tree_size,
        )

        self.mpk.multi_token_rmsnorm_linear_layer( # NOT IMPLEMENTED YET
            input=self.attn_input,
            weight=self.draft_norm1_weight,
            output=self.attn_in,
            grid_dim=(96, 1, 1),
            block_dim=(512, 1, 1),
        )
        
        self.mpk.single_batch_multitoken_decoding(
            qkv=self.attn_in,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            kernel_outputs=self.attn_out,
            seq_len=config.max_seq_len,
            qk_norm=False,
            rotary_emd=False,
            qnorm_weight=None,
            knorm_weight=None,
            cos=None,
            sin=None,
            q_eps=0.0,
            k_eps=0.0,
            prompt_len=0,
            mask_words_per_token=config.max_seq_len // 64,
            attn_mask=attention_mask,
        )
        
        # Output projection and residual
        self.mpk.linear_with_residual_layer(
            input=self.attn_out,
            weight=self.draft_o_proj,
            residual=self.fused_features_normed,
            output=self.hidden_after_attn,
            grid_dim=(96, 1, 1),
            block_dim=(512, 1, 1),
        )
        
        # Post-attention layer norm
        self.mpk.rmsnorm_layer(
            input=self.hidden_after_attn,
            weight=self.draft_norm2_weight,
            output=self.normed_hidden2,
            grid_dim=(96, 1, 1),
            block_dim=(512, 1, 1),
            eps=config.rms_norm_eps,
        )
        
        """
        Some activation to produce the final hidden state, leave it for now
        """

    def _tree_generation(self):
        """
        mpk implementation of topK_generate in eagle model, requires carefully design of tree storage
        """
        config = self.config

        self._initialize_tree_tensors()

        self._draft_model() # initial hidden states

        self.mpk.linear_with_residual_layer(
            input=self.normed_hidden2,
            weight=self.draft_lm_head,
            output=self.draft_logits,
            residual=self.zero_residue,
            grid_dim=(96, 1, 1),
            block_dim=(512, 1, 1),
        )

        self.mpk.log_softmax_layer( # NOT IMPLEMENTED YET
            input=self.draft_logits,
            output=self.draft_log_probs,
            grid_dim=(1, 1, 1),
            block_dim=(1024, 1, 1),
            dim=-1, 
        )
        
        # Select top-k initial candidates
        self.mpk.customized(
            op_type="topk",  # NOT IMPLEMENTED YET
            inputs=[self.last_log_probs],
            outputs=[self.initial_tokens, self.initial_scores],
            grid_dim=(1, 1, 1),
            block_dim=(512, 1, 1),
            k=config.top_k,
        )
        
        # Initialize tree storage
        self.mpk.customized(
            op_type="init_tree_storage",  # NOT IMPLEMENTED YET
            inputs=[self.initial_tokens, self.initial_scores],
            outputs=[self.scores_list, self.parents_list, self.ss_token_list],
            grid_dim=(1, 1, 1),
            block_dim=(512, 1, 1),
            top_k=config.top_k,
        )

    
    def __call__(self, tokens):
        """Execute the persistent kernel."""
        step = len(tokens)
        self.meta_tensor = (tokens, step)
        self.mpk()


def main():
    # Create config
    config = EAGLEConfig(
        tree_size=args.tree_size,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    
    eagle = EAGLE_MPK(config)
    eagle()



if __name__ == "__main__":
    main()