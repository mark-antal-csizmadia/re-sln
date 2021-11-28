import torch
from pathlib import Path
from models.wideresnet import Wide_ResNet


def get_model(model_name, n_classes, device):
    if model_name == "wrn-28-2":
        model = Wide_ResNet(num_classes=n_classes).to(device=device)
    else:
        raise Exception
    return model


def load_model(exp_id, dataset_name, n_classes, device):
    # get models for naive and ema (depends on dataset)
    model_name = "wrn-28-2" if dataset_name in ["cifar10", "cifar100"] else "MODEL_NAME_FOR_CLOTHING1M"
    model = get_model(model_name=model_name, n_classes=n_classes, device=device)
    # if multi gpu
    if device == "cuda":
        if 1 < torch.cuda.device_count():
            model = torch.nn.DataParallel(model)
    model.to(device)
    
    model_load_path = Path(f"saved_models/{exp_id}/model_ema.pth")
    # if momentum model exists, load that
    if model_load_path.exists():
        model.load_state_dict(torch.load(model_load_path))
    # else load the no momentum model
    else:
        model_load_path = Path(f"saved_models/{exp_id}/model.pth")
        model.load_state_dict(torch.load(model_load_path))
    print(f"loaded model at {model_load_path}")  
    
    return model
