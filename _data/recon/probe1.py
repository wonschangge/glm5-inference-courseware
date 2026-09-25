import json, sys, warnings
warnings.filterwarnings("ignore")
import transformers, torch
print("transformers", transformers.__version__)
from transformers import Glm5NextConfig, Glm5NextTextConfig
c = Glm5NextTextConfig()
print("head_dim          =", c.head_dim)
print("qk_head_dim       =", c.qk_head_dim)
print("layer_types[:8]   =", c.layer_types[:8])
print("mlp_layer_types[:6]=", c.mlp_layer_types[:6])
print("indexer_types[:8] =", c.indexer_types[:8])
n_kda = sum(1 for x in c.layer_types if x == "linear_attention")
n_idx = sum(1 for x in c.layer_types if x == "indexed_attention")
print("KDA layers        =", n_kda, " indexed(MLA+DSA) =", n_idx)
n_dense = sum(1 for x in c.mlp_layer_types if x == "dense")
print("dense MLP layers  =", n_dense, " sparse =", len(c.mlp_layer_types)-n_dense)
