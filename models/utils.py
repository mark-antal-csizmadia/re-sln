from models.wideresnet import Wide_ResNet


def get_model(model_name, n_classes, device):
    if model_name == "wrn-28-2":
        model = Wide_ResNet(num_classes=n_classes).to(device=device)
    else:
        raise Exception
    return model