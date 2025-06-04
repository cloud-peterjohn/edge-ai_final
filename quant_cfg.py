from hqq.core.quantize import BaseQuantizeConfig

def get_quant_config_slm(model):
    quant_config = {}
    
    n_layers = model.config.num_hidden_layers
    q8_128_config = BaseQuantizeConfig(nbits=8, group_size=128) 
    q8_64_config = BaseQuantizeConfig(nbits=8, group_size=64) 
    q8_32_config = BaseQuantizeConfig(nbits=8, group_size=32) 
    q4_128_config = BaseQuantizeConfig(nbits=4, group_size=128) 
    q4_256_config = BaseQuantizeConfig(nbits=4, group_size=256) 
    q4_64_config = BaseQuantizeConfig(nbits=4, group_size=64) 
    q4_32_config = BaseQuantizeConfig(nbits=4, group_size=32) 
    q2_128_config = BaseQuantizeConfig(nbits=2, group_size=128) 
    q2_64_config = BaseQuantizeConfig(nbits=2, group_size=64) 
    q2_32_config = BaseQuantizeConfig(nbits=2, group_size=32) 
    
    for i in range(n_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q4_256_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q4_256_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q4_256_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q4_128_config
        
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q4_256_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q4_256_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q4_128_config
        
    return quant_config
