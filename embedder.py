import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5EncoderModel
from transformers.modeling_outputs import TokenClassifierOutput


class MTDP(nn.Module):
    def __init__(
            self,
            vocab_size=128,
            pad_token_id=0,
            d_ff=2048,
            num_layers=6,
            d_model=512,
            num_heads=8,
            dropout=0.1,
            hidden_size=1280,
    ):
        super(MTDP, self).__init__()
        self.config = T5Config(classifier_dropout=0.0,
                               d_ff=d_ff,
                               d_kv=64,
                               d_model=d_model,
                               dense_act_fn="relu",
                               dropout_rate=dropout,
                               eos_token_id=1,
                               feed_forward_proj="relu",
                               initializer_factor=1.0,
                               is_encoder_decoder=True,
                               is_gated_act=False,
                               layer_norm_epsilon=1e-06,
                               model_type="t5",
                               num_decoder_layers=num_layers,
                               num_heads=num_heads,
                               num_layers=num_layers,
                               pad_token_id=pad_token_id,
                               relative_attention_max_distance=128,
                               relative_attention_num_buckets=32,
                               transformers_version="4.38.1",
                               use_cache=True,
                               vocab_size=vocab_size)

        self.T5Encoder = T5EncoderModel(self.config)
        self.embed = nn.Linear(512, hidden_size)

    def forward(self, input_ids=None, attention_mask=None, esm=None):
        embedding_repr = self.T5Encoder(input_ids=input_ids, attention_mask=attention_mask)
        T5Out = embedding_repr.last_hidden_state
        embed = self.embed(torch.mean(T5Out, axis=1))
        
        
        loss = None            

        return TokenClassifierOutput(
            loss=loss,
            logits=embed,
            hidden_states=T5Out
        )
        
