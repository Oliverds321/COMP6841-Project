import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
#done
modelid = "./gemma"
targetlayer = 28
strength = 50.0 
device = "cuda"

print(f"loading model{modelid}")
tokenizer = AutoTokenizer.from_pretrained(modelid)
model = AutoModelForCausalLM.from_pretrained(
    modelid,
    torch_dtype=torch.bfloat16,
).to(device)

#vector must match the main hidden dimension
hidden_dim = model.config.hidden_size
randomvec = torch.randn(hidden_dim, dtype=torch.bfloat16).to(device) * strength

def steer(module, input, output):
      
    #output of DecoderLayer tuple(hidden_states, optional_kv_cache), use first element
    hidden_states = output[0]
    
    print(f"Hook on decoderlayer, tensor shape is: {hidden_states.shape}")

    #check alteration is applied
    original_norm = torch.linalg.norm(hidden_states).item()
    print(f"original hidden_states {original_norm:.4f}")
    
    #use addition instead of making a new one and inserting/replacing
    hidden_states.add_(randomvec)
    
    modified_norm = torch.linalg.norm(hidden_states).item()
    print(f"new hidden_states  {modified_norm:.4f}")

    return output

prompt = "The capital of Australia is"
inputs = tokenizer(tokenizer.bos_token + prompt, return_tensors="pt").to(device)

print("\n1. generating baseline response")
baseoutputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(baseoutputs[0], skip_special_tokens=True))

print("\n2. generating steered response")
layer = model.model.layers[targetlayer]
hook = layer.register_forward_hook(steer)
steeredoutputs = model.generate(**inputs, max_new_tokens=20)
hook.remove()
print(tokenizer.decode(steeredoutputs[0], skip_special_tokens=True))