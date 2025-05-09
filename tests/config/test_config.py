def test_load_settings():
    """
    Test the loading of settings using Hydra.
    """
    from virne.load_config import load_config
    from omegaconf import DictConfig

    # Load the configuration
    cfg: DictConfig = load_config()

    # Check if the configuration is loaded correctly
    assert cfg is not None, "Configuration should not be None"
    assert 'settings' in cfg, "Configuration should contain 'settings'"
    assert 'main' in cfg, "Configuration should contain 'main'"
    assert 'p_net_dataset_dir' in cfg.settings, "Configuration should contain 'p_net_dataset_dir'"
    assert 'p_net_dataset_dir' in cfg.settings.main, "Configuration should contain 'p_net_dataset_dir' in main settings"
    assert cfg.settings.main.p_net_dataset_dir == 'path/to/dataset', "p_net_dataset_dir should match the expected value"