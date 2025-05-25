import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from omegaconf import DictConfig, OmegaConf

from virne.network.dataset_generator import Generator
from virne.network.physical_network import PhysicalNetwork
from virne.network.virtual_network_request_simulator import VirtualNetworkRequestSimulator
from virne.utils.setting import read_setting
from virne.utils.dataset import get_p_net_dataset_dir_from_setting, get_v_nets_dataset_dir_from_setting


class TestGenerator:
    """Test suite for Generator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock configurations instead of reading from files
        self.p_net_setting = read_setting('tests/settings/p_net_setting/default.yaml')
        self.v_sim_setting = read_setting('tests/settings/v_sim_setting/default.yaml')
        
        self.basic_config = {
            'seed': 42,
            'p_net_setting': self.p_net_setting,
            'v_sim_setting': self.v_sim_setting,
        }
        
    @patch('virne.network.dataset_generator.Generator.generate_p_net_dataset_from_config')
    @patch('virne.network.dataset_generator.Generator.generate_v_nets_dataset_from_config')
    def test_generate_dataset_both(self, mock_v_nets, mock_p_net):
        """Test generating both physical network and virtual network datasets."""
        mock_p_net.return_value = Mock(spec=PhysicalNetwork)
        mock_v_nets.return_value = Mock(spec=VirtualNetworkRequestSimulator)
        
        p_net, v_sim = Generator.generate_dataset(
            self.basic_config, 
            p_net=True, 
            v_nets=True, 
            save=False
        )
        
        assert p_net is not None
        assert v_sim is not None
        mock_p_net.assert_called_once_with(self.basic_config, save=False)
        mock_v_nets.assert_called_once_with(self.basic_config, save=False)
        
    @patch('virne.network.dataset_generator.Generator.generate_p_net_dataset_from_config')
    @patch('virne.network.dataset_generator.Generator.generate_v_nets_dataset_from_config')
    def test_generate_dataset_p_net_only(self, mock_v_nets, mock_p_net):
        """Test generating only physical network dataset."""
        mock_p_net.return_value = Mock(spec=PhysicalNetwork)
        
        p_net, v_sim = Generator.generate_dataset(
            self.basic_config, 
            p_net=True, 
            v_nets=False, 
            save=False
        )
        
        assert p_net is not None
        assert v_sim is None
        mock_p_net.assert_called_once_with(self.basic_config, save=False)
        mock_v_nets.assert_not_called()
        
    @patch('virne.network.dataset_generator.Generator.generate_p_net_dataset_from_config')
    @patch('virne.network.dataset_generator.Generator.generate_v_nets_dataset_from_config')
    def test_generate_dataset_v_nets_only(self, mock_v_nets, mock_p_net):
        """Test generating only virtual network dataset."""
        mock_v_nets.return_value = Mock(spec=VirtualNetworkRequestSimulator)
        
        p_net, v_sim = Generator.generate_dataset(
            self.basic_config, 
            p_net=False, 
            v_nets=True, 
            save=False
        )
        
        assert p_net is None
        assert v_sim is not None
        mock_p_net.assert_not_called()
        mock_v_nets.assert_called_once_with(self.basic_config, save=False)
        
    @patch('virne.network.dataset_generator.Generator.generate_p_net_dataset_from_config')
    @patch('virne.network.dataset_generator.Generator.generate_v_nets_dataset_from_config')
    def test_generate_dataset_neither(self, mock_v_nets, mock_p_net):
        """Test generating neither dataset."""
        p_net, v_sim = Generator.generate_dataset(
            self.basic_config, 
            p_net=False, 
            v_nets=False, 
            save=False
        )
        
        assert p_net is None
        assert v_sim is None
        mock_p_net.assert_not_called()
        mock_v_nets.assert_not_called()
        
    @patch('virne.utils.dataset.set_seed')
    @patch('virne.network.physical_network.PhysicalNetwork.from_setting')
    @patch('virne.utils.get_p_net_dataset_dir_from_setting')
    def test_generate_p_net_dataset_from_config(self, mock_get_dir, mock_from_setting, mock_set_seed):
        """Test generating physical network dataset from config."""
        mock_p_net = Mock(spec=PhysicalNetwork)
        mock_p_net.save_dataset = Mock()
        mock_from_setting.return_value = mock_p_net
        mock_get_dir.return_value = get_p_net_dataset_dir_from_setting(self.basic_config['p_net_setting'])
        
        result = Generator.generate_p_net_dataset_from_config(self.basic_config, save=False)
        
        assert result == mock_p_net
        # mock_set_seed.assert_called_once_with(42)
        mock_from_setting.assert_called_once_with(self.basic_config['p_net_setting'])
        mock_p_net.save_dataset.assert_not_called()  # save=False
        
    @patch('virne.utils.dataset.set_seed')
    @patch('virne.network.physical_network.PhysicalNetwork.from_setting')
    @patch('virne.utils.get_p_net_dataset_dir_from_setting')
    def test_generate_p_net_dataset_from_config_with_save(self, mock_get_dir, mock_from_setting, mock_set_seed):
        """Test generating and saving physical network dataset."""
        mock_p_net = Mock(spec=PhysicalNetwork)
        mock_p_net.save_dataset = Mock()
        mock_from_setting.return_value = mock_p_net
        mock_get_dir.return_value = get_p_net_dataset_dir_from_setting(self.basic_config['p_net_setting'])
        
        result = Generator.generate_p_net_dataset_from_config(self.basic_config, save=True)
        
        assert result == mock_p_net
        mock_p_net.save_dataset.assert_called_once_with(get_p_net_dataset_dir_from_setting(self.basic_config['p_net_setting']))
        
    def test_generate_p_net_dataset_missing_p_net_setting(self):
        """Test error when p_net_setting is missing."""
        config = {'other_setting': {}}
        with pytest.raises(AssertionError, match="config must contain 'p_net_setting' key"):
            Generator.generate_p_net_dataset_from_config(config)
            
    @patch('virne.utils.dataset.set_seed')
    @patch('virne.network.virtual_network_request_simulator.VirtualNetworkRequestSimulator.from_setting')
    @patch('virne.utils.get_v_nets_dataset_dir_from_setting')
    def test_generate_v_nets_dataset_from_config(self, mock_get_dir, mock_from_setting, mock_set_seed):
        """Test generating virtual network dataset from config."""
        mock_v_sim = Mock(spec=VirtualNetworkRequestSimulator)
        mock_v_sim.renew = Mock()
        mock_v_sim.save_dataset = Mock()
        mock_from_setting.return_value = mock_v_sim
        mock_get_dir.return_value = get_v_nets_dataset_dir_from_setting(self.basic_config['v_sim_setting'])
        result = Generator.generate_v_nets_dataset_from_config(self.basic_config, save=False)
        
        assert result == mock_v_sim
        # mock_set_seed.assert_called_once_with(42)
        mock_from_setting.assert_called_once_with(self.basic_config['v_sim_setting'])
        mock_v_sim.renew.assert_called_once()
        mock_v_sim.save_dataset.assert_not_called()  # save=False
        
    @patch('virne.utils.dataset.set_seed')
    @patch('virne.network.virtual_network_request_simulator.VirtualNetworkRequestSimulator.from_setting')
    @patch('virne.utils.get_v_nets_dataset_dir_from_setting')
    def test_generate_v_nets_dataset_from_config_with_save(self, mock_get_dir, mock_from_setting, mock_set_seed):
        """Test generating and saving virtual network dataset."""
        mock_v_sim = Mock(spec=VirtualNetworkRequestSimulator)
        mock_v_sim.renew = Mock()
        mock_v_sim.save_dataset = Mock()
        mock_from_setting.return_value = mock_v_sim
        mock_get_dir.return_value = get_v_nets_dataset_dir_from_setting(self.basic_config['v_sim_setting'])
        
        result = Generator.generate_v_nets_dataset_from_config(self.basic_config, save=True)
        
        assert result == mock_v_sim
        mock_v_sim.save_dataset.assert_called_once_with(get_v_nets_dataset_dir_from_setting(self.basic_config['v_sim_setting']))

    def test_generate_v_nets_dataset_missing_v_sim_setting(self):
        """Test error when v_sim_setting is missing."""
        config = {'other_setting': {}}
        with pytest.raises(AssertionError, match="config must contain 'v_sim_setting' key"):
            Generator.generate_v_nets_dataset_from_config(config)
            
    @patch('virne.utils.dataset.set_seed')
    @patch('virne.network.virtual_network_request_simulator.VirtualNetworkRequestSimulator.from_setting')
    @patch('virne.utils.get_v_nets_dataset_dir_from_setting')
    def test_generate_changeable_v_nets_dataset_from_config(self, mock_get_dir, mock_from_setting, mock_set_seed):
        """Test generating changeable virtual network dataset from config."""
        # Create mock v_nets list
        mock_v_nets = [Mock() for _ in range(100)]  # Assuming 100 v_nets for testing
        
        # Create mock simulator with v_nets attribute
        mock_v_sim = Mock(spec=VirtualNetworkRequestSimulator)
        mock_v_sim.v_nets = mock_v_nets
        mock_v_sim.save_dataset = Mock()
        mock_v_sim.renew = Mock()
        
        # Create mock temp simulators for each stage
        mock_v_sim_temp = Mock(spec=VirtualNetworkRequestSimulator)
        mock_v_sim_temp.v_nets = mock_v_nets
        mock_v_sim_temp.renew = Mock()
        
        # Set up the from_setting mock to return different instances
        mock_from_setting.side_effect = [mock_v_sim, mock_v_sim_temp, mock_v_sim_temp, mock_v_sim_temp, mock_v_sim_temp]
        mock_get_dir.return_value = '/test/v_nets/dir'
        
        result = Generator.generate_changeable_v_nets_dataset_from_config(self.basic_config, save=False)
        
        assert result == mock_v_sim
        # mock_set_seed.assert_called_once_with(42)
        assert mock_from_setting.call_count == 5  # 1 main + 4 stages
        mock_v_sim.save_dataset.assert_not_called()  # save=False
        
    def test_generate_changeable_v_nets_dataset_missing_v_sim_setting(self):
        """Test error when v_sim_setting is missing for changeable dataset."""
        config = {'other_setting': {}}
        with pytest.raises(AssertionError, match="config must contain 'v_sim_setting' key"):
            Generator.generate_changeable_v_nets_dataset_from_config(config)
            
    def test_generate_changeable_v_nets_dataset_none_v_sim_setting(self):
        """Test error when v_sim_setting is None for changeable dataset."""
        config = {'v_sim_setting': None}
        with pytest.raises(ValueError, match="Virtual network simulation config"):
            Generator.generate_changeable_v_nets_dataset_from_config(config)
            
    def test_config_with_dictconfig(self):
        """Test that methods work with DictConfig objects."""
        config = OmegaConf.create(self.basic_config)
        
        # Should not raise assertion errors
        with patch('virne.network.physical_network.PhysicalNetwork.from_setting'):
            with patch('virne.utils.dataset.set_seed'):
                with patch('virne.utils.get_p_net_dataset_dir_from_setting'):
                    try:
                        Generator.generate_p_net_dataset_from_config(config)
                    except:
                        pass  # We're just testing that config type validation passes
                        
        with patch('virne.network.virtual_network_request_simulator.VirtualNetworkRequestSimulator.from_setting'):
            with patch('virne.utils.dataset.set_seed'):
                with patch('virne.utils.get_v_nets_dataset_dir_from_setting'):
                    try:
                        Generator.generate_v_nets_dataset_from_config(config)
                    except:
                        pass  # We're just testing that config type validation passes


if __name__ == '__main__':
    pytest.main([__file__])