import torch
#done
#checks the steering vector file for issues
vectorpath = "steervec.pt"
print(f"loading vector from {vectorpath}")

try:
    vec = torch.load(vectorpath)
    magnitude = torch.linalg.norm(vec)

    print(f"\nvector loaded successfully")
    print(f"vectors magnitude: {magnitude.item()}")
    print(f"First 5 values: {vec[:5]}")

except FileNotFoundError:
    print("\nsteervec.pt not found.")