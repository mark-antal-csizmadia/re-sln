import yaml
from pathlib import Path


def save_config(exp_id, dataset_name, batch_size, n_epochs, lr, noise_mode, p, custom_noise, make_new_custom_noise, 
                sigma, mo, lc_n_epoch, seed):
    config_save_path = Path(f"configs/{exp_id}")
    config_save_path.mkdir(parents=True, exist_ok=True)
    config_save_path = config_save_path / "config.yml"
    config_data = dict(
        exp_id=exp_id,
        dataset_name = dataset_name,
        batch_size = batch_size,
        n_epochs=n_epochs,
        lr=lr,
        noise_mode = noise_mode,
        p = p,
        custom_noise = custom_noise,
        make_new_custom_noise=make_new_custom_noise,
        sigma=sigma,
        mo=mo,
        lc_n_epoch=lc_n_epoch,
        seed=seed
    )
    with open(config_save_path, 'w') as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False)
    
    print(f"saved config for exp_id:{exp_id}")
    

def load_config(exp_id):
    config_load_path = Path(f"configs/{exp_id}/config.yml")
    # dataset_name, batch_size, noise_mode, p, custom_noise, seed
    with open(config_load_path, "r") as stream:
        try:
            config_data = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            raise e
            
    return config_data
