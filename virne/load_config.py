import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../settings", config_name="main")
def load_config(cfg: DictConfig) -> DictConfig:
    """
    Load the configuration file using Hydra.

    Args:
        cfg (DictConfig): The configuration object.

    Returns:
        DictConfig: The loaded configuration object.
    """
    import pdb; pdb.set_trace()
    # Convert the config to a dictionary
    # config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Print the configuration for debugging
    # print(OmegaConf.to_yaml(cfg))
    
    return cfg

def main():
    load_config()  # Hydra will handle CLI args

if __name__ == "__main__":
    main()