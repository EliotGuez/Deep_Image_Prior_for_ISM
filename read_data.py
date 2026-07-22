import torch

data = torch.load("data/02_TUB_data.pth", map_location="cpu", weights_only=False)

print(data.keys())

for k, v in data.items():
    print("\nKEY:", k)
    print("TYPE:", type(v))
    print("SHAPE:", getattr(v, "shape", None))

    if isinstance(v, dict):
        print("SUBKEYS:", v.keys())