from typing import Dict, Any

def extract_base_weights(model, bank_id: str = "unknown", round_num: int = 0, 
                         num_samples: int = 0, metadata: Dict[str, Any] = None) -> dict:
    """
    Extracts the shared base layer weights from a trained model.
    """
    if metadata is None:
        metadata = {}
        
    weights = {}
    for name, param in model.state_dict().items():
        if name.startswith("base_layers"):
            weights[name] = param.cpu().numpy().tolist()
        elif name.startswith("top_layers"):
            continue
        else:
            raise ValueError(f"Unexpected layer name found during extraction: {name}")
            
    return {
        "bank_id": bank_id,
        "round": round_num,
        "num_samples": num_samples,
        "weights": weights,
        "metadata": metadata
    }

def load_base_weights(model, weight_dict: dict):
    """
    Applies a received global weight dictionary back into the model's base_layers.
    """
    import torch
    state_dict = model.state_dict()
    for name, list_weights in weight_dict.get("weights", weight_dict).items():
        if name in state_dict:
            state_dict[name] = torch.tensor(list_weights)
    model.load_state_dict(state_dict)
    return model
