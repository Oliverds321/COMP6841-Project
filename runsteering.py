import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
#done
MODEL_ID = "./gemma"
TARGET_LAYER = 20
VECTOR_PATH = "steervec.pt"
STRENGTH = 1 #set to negative to reverse
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def steer_inference(model, tokenizer, prompt, vector, strength):


    steer_vec = vector.to(DEVICE) * strength

    def steer(module, a_input, a_output):
        #hook function that applies the steering vector
        #targets the output of an entire decoder layer
        hidden_states = a_output[0]
        
        #apply the steering vector to hidden states
        hidden_states.add_(steer_vec)
        
        #return the modified output tuple
        return a_output

    #entire decoder layer object
    layer_to_steer = model.model.layers[TARGET_LAYER]
    hook = layer_to_steer.register_forward_hook(steer)
    
    #generate steered response
    inputs = tokenizer(tokenizer.bos_token + prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, top_k=50, top_p=0.95)
    
    #remove the hook otherwise bad things happen
    hook.remove()
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python runsteering.py \"prompt\"")
        sys.exit(1)
    
    user_prompt = sys.argv[1]
    
    print(f"loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    #model needs to be in eval mode
    model.eval()

    print(f"Loading vector from {VECTOR_PATH}...")
    #load vector and ensure it has the same data type as the model
    steer_vec = torch.load(VECTOR_PATH).to(dtype=model.dtype)
    
    print(f"\nPrompt: '{user_prompt}'")
    print("\ngenerating steered response")
    steered_text = steer_inference(model, tokenizer, user_prompt, steer_vec, STRENGTH)
    print(steered_text)