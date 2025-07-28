import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
#done
MODEL_ID = "./gemma"
CSV_PATH = "finaldataset.csv"
TARGET_LAYER = 20
VECTOR_PATH = "steervec.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

acts = {}

def get_activation(name):
    #hook function to capture hidden states tensor from the decoder layer output tuple.
    def hook(module, a_input, a_output):
        #first element for hidden_states
        acts[name] = a_output[0].detach()
    return hook

def get_vector(model, tokenizer, datareader):
    #calculate the steering vector by averaging the difference between activation pairs.
    model.eval()
    vectors = []
    layer_to_hook = model.model.layers[TARGET_LAYER]
    total_processed = 0

    print("processing activation pairs")
    for chunk in datareader:
        data = chunk.to_dict(orient='records')
        for item in data:
            #DESIRED activation
            hook = layer_to_hook.register_forward_hook(get_activation("desired"))
            prompt = tokenizer.bos_token + item['Question'] + "\n" + item['desired response']
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            model(**inputs)
            hook.remove()
            #slice the activation of the LAST token only
            desired_vec = acts["desired"][0, -1, :].cpu()
            
            #UNDESIRED activation
            hook = layer_to_hook.register_forward_hook(get_activation("undesired"))
            prompt = tokenizer.bos_token + item['Question'] + "\n" + item['undesired response']
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            model(**inputs)
            hook.remove()
            #slice the activation of the LAST token only
            undesired_vec = acts["undesired"][0, -1, :].cpu()
            
            #calculate the difference and store
            vectors.append(desired_vec - undesired_vec)
            total_processed += 1
            if total_processed % 10 == 0:
                print(f"  ...processed {total_processed} pairs")

    print(f"Finished processing. Averaging {len(vectors)} vectors.")
    #stack all difference vectors and compute the mean
    return torch.mean(torch.stack(vectors), dim=0)

if __name__ == "__main__":
    print(f"loading model {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print(f"loading data from {CSV_PATH}")
    datareader = pd.read_csv(CSV_PATH, chunksize=10)

    print("\ncalculating new steering vector")
    steer_vec = get_vector(model, tokenizer, datareader)
    
    # Save the new vector
    torch.save(steer_vec, VECTOR_PATH)
    print(f"\nvector calculation done, saved to {VECTOR_PATH}")