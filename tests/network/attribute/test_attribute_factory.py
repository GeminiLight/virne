import pytest
import numpy as np
import networkx as nx
from unittest.mock import Mock, patch
from omegaconf import DictConfig, OmegaConf

from virne.network.attribute import (
    create_attr_from_dict, 
    create_node_attrs_from_dict, 
    create_link_attrs_from_dict,
    create_attrs_from_setting,
    create_node_attrs_from_setting,
    create_link_attrs_from_setting,
    ATTRIBUTES_DICT
)
from virne.network.attribute.node_attribute import NodeAttribute
from virne.network.attribute.link_attribute import LinkAttribute


class TestAttributeFactory:
    """Test suite for attribute factory functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.node_resource_config = {
            'name': 'cpu',
            'owner': 'node',
            'type': 'resource',
            'distribution': 'uniform',
            'low': 1,
            'high': 10
        }
        
        self.link_resource_config = {
            'name': 'bandwidth',
            'owner': 'link', 
            'type': 'resource',
            'distribution': 'uniform',
            'low': 1,
            'high': 100
        }
        
        self.node_status_config = {
            'name': 'status',
            'owner': 'node',
            'type': 'status',
            'default': True
        }
        
    def test_attributes_dict_structure(self):
        """Test that ATTRIBUTES_DICT has expected structure."""
        assert isinstance(ATTRIBUTES_DICT, dict)
        
        # Check some expected keys exist
        expected_keys = [
            ('node', 'status'),
            ('link', 'status'),
            ('node', 'resource'),
            ('link', 'resource')
        ]
        
        for key in expected_keys:
            assert key in ATTRIBUTES_DICT, f"Key {key} should exist in ATTRIBUTES_DICT"
            
    def test_create_attr_from_dict_node_resource(self):
        """Test creating node resource attribute from dict."""
        attr = create_attr_from_dict(self.node_resource_config)
        
        assert attr.name == 'cpu'
        assert hasattr(attr, 'owner')
        assert hasattr(attr, 'type')
        
    def test_create_attr_from_dict_link_resource(self):
        """Test creating link resource attribute from dict."""
        attr = create_attr_from_dict(self.link_resource_config)
        
        assert attr.name == 'bandwidth'
        assert hasattr(attr, 'owner')
        assert hasattr(attr, 'type')
        
    def test_create_attr_from_dict_missing_owner(self):
        """Test error when owner is missing."""
        config = self.node_resource_config.copy()
        del config['owner']
        
        with pytest.raises(AssertionError):
            create_attr_from_dict(config)
            
    def test_create_attr_from_dict_missing_type(self):
        """Test error when type is missing."""
        config = self.node_resource_config.copy()
        del config['type']
        
        with pytest.raises(AssertionError):
            create_attr_from_dict(config)
            
    def test_create_attr_from_dict_invalid_combination(self):
        """Test error with invalid owner/type combination."""
        config = {
            'name': 'invalid',
            'owner': 'invalid_owner',
            'type': 'invalid_type'
        }
        
        with pytest.raises(ValueError, match="Attribute class for \\(invalid_owner, invalid_type\\) is not defined"):
            create_attr_from_dict(config)
            
    def test_create_attr_from_dictconfig(self):
        """Test creating attribute from DictConfig."""
        config = OmegaConf.create(self.node_resource_config)
        attr = create_attr_from_dict(config)
        
        assert attr.name == 'cpu'
        
    def test_create_node_attrs_from_dict(self):
        """Test creating node attribute specifically."""
        attr = create_node_attrs_from_dict(self.node_resource_config)
        
        assert isinstance(attr, NodeAttribute)
        assert attr.name == 'cpu'
        
    def test_create_link_attrs_from_dict(self):
        """Test creating link attribute specifically."""
        attr = create_link_attrs_from_dict(self.link_resource_config)
        
        assert isinstance(attr, LinkAttribute)
        assert attr.name == 'bandwidth'
        
    def test_create_node_attrs_from_dict_wrong_owner(self):
        """Test error when creating node attr with link config."""
        with pytest.raises(TypeError):
            create_node_attrs_from_dict(self.link_resource_config)
            
    def test_create_link_attrs_from_dict_wrong_owner(self):
        """Test error when creating link attr with node config."""
        with pytest.raises(TypeError):
            create_link_attrs_from_dict(self.node_resource_config)
            
    def test_create_attrs_from_setting(self):
        """Test creating multiple attributes from setting."""
        setting = [
            self.node_resource_config,
            self.node_status_config
        ]
        
        attrs = create_attrs_from_setting(setting)
        
        assert len(attrs) == 2
        assert 'cpu' in attrs
        assert 'status' in attrs
        assert attrs['cpu'].name == 'cpu'
        assert attrs['status'].name == 'status'
        
    def test_create_node_attrs_from_setting(self):
        """Test creating multiple node attributes from setting."""
        setting = [
            self.node_resource_config,
            self.node_status_config
        ]
        
        attrs = create_node_attrs_from_setting(setting)
        
        assert len(attrs) == 2
        assert all(isinstance(attr, NodeAttribute) for attr in attrs.values())
        
    def test_create_link_attrs_from_setting(self):
        """Test creating multiple link attributes from setting."""
        link_status_config = {
            'name': 'status',
            'owner': 'link',
            'type': 'status',
            'default': True
        }
        
        setting = [
            self.link_resource_config,
            link_status_config
        ]
        
        attrs = create_link_attrs_from_setting(setting)
        
        assert len(attrs) == 2
        assert all(isinstance(attr, LinkAttribute) for attr in attrs.values())
        
    def test_create_attrs_from_setting_invalid_dict(self):
        """Test error with invalid dict in setting."""
        setting = [
            self.node_resource_config,
            "invalid_config"  # Should be dict or DictConfig
        ]
        
        with pytest.raises(AssertionError):
            create_attrs_from_setting(setting)
            
    def test_create_attrs_from_setting_empty(self):
        """Test creating attributes from empty setting."""
        attrs = create_attrs_from_setting([])
        assert len(attrs) == 0
        assert isinstance(attrs, dict)
        
    def test_create_attrs_from_dictconfig_setting(self):
        """Test creating attributes from DictConfig setting."""
        setting = OmegaConf.create([self.node_resource_config, self.node_status_config])
        
        attrs = create_attrs_from_setting(setting)
        
        assert len(attrs) == 2
        assert 'cpu' in attrs
        assert 'status' in attrs
        
    def test_kwargs_passing(self):
        """Test that extra kwargs are passed to attribute constructor."""
        config = self.node_resource_config.copy()
        config['extra_param'] = 'test_value'
        
        # Should not raise error (extra kwargs should be passed to constructor)
        attr = create_attr_from_dict(config)
        assert attr.name == 'cpu'


if __name__ == '__main__':
    pytest.main([__file__])
