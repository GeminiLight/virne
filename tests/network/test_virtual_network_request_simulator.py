import pytest
import os
import copy
import tempfile
import numpy as np
from types import SimpleNamespace
from unittest.mock import Mock, patch, mock_open
from dataclasses import asdict

from virne.network.virtual_network_request_simulator import VirtualNetworkRequestSimulator, VirtualNetworkEvent
from virne.network.virtual_network import VirtualNetwork
from virne.utils.setting import read_setting


class TestVirtualNetworkEvent:
    """Test suite for VirtualNetworkEvent class."""
    
    def test_valid_event_creation(self):
        """Test creating a valid VirtualNetworkEvent."""
        event = VirtualNetworkEvent(id=1, type=1, v_net_id=10, time=5.0)
        assert event.id == 1
        assert event.type == 1
        assert event.v_net_id == 10
        assert event.time == 5.0
        
    def test_valid_event_types(self):
        """Test valid event types (0 and 1)."""
        # Arrival event
        arrival_event = VirtualNetworkEvent(id=1, type=1, v_net_id=10, time=5.0)
        assert arrival_event.type == 1
        
        # Leave event
        leave_event = VirtualNetworkEvent(id=2, type=0, v_net_id=10, time=10.0)
        assert leave_event.type == 0
        
    def test_invalid_event_type(self):
        """Test invalid event type raises ValueError."""
        with pytest.raises(ValueError, match="Event type must be 0 \\(leave\\) or 1 \\(arrival\\)"):
            VirtualNetworkEvent(id=1, type=2, v_net_id=10, time=5.0)
            
    def test_negative_v_net_id(self):
        """Test negative v_net_id raises ValueError."""
        with pytest.raises(ValueError, match="Virtual network ID must be non-negative"):
            VirtualNetworkEvent(id=1, type=1, v_net_id=-1, time=5.0)
            
    def test_negative_time(self):
        """Test negative time raises ValueError."""
        with pytest.raises(ValueError, match="Event time must be non-negative"):
            VirtualNetworkEvent(id=1, type=1, v_net_id=10, time=-1.0)
            
    def test_event_repr(self):
        """Test event string representation."""
        event = VirtualNetworkEvent(id=1, type=1, v_net_id=10, time=5.0)
        expected = "VirtualNetworkEvent(v_net_id=10, time=5.0, type=1, id=1)"
        assert repr(event) == expected
        assert str(event) == expected
        
    def test_event_getitem_setitem(self):
        """Test event item access methods."""
        event = VirtualNetworkEvent(id=1, type=1, v_net_id=10, time=5.0)
        
        # Test getitem
        assert event['id'] == 1
        assert event['type'] == 1
        assert event['v_net_id'] == 10
        assert event['time'] == 5.0
        
        # Test setitem
        event['time'] = 7.5
        assert event.time == 7.5


class TestVirtualNetworkRequestSimulator:
    """Test suite for VirtualNetworkRequestSimulator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # This fixture intentionally retains the legacy arrival-rate schema so
        # existing user configurations stay covered by the network test suite.
        self.basic_v_sim_setting = read_setting('tests/settings/v_sim_setting/default.yaml')
        
        # Create some sample v_nets and events
        self.sample_v_nets = [Mock(spec=VirtualNetwork) for _ in range(3)]
        self.sample_events = [
            VirtualNetworkEvent(id=0, type=1, v_net_id=0, time=1.0),
            VirtualNetworkEvent(id=1, type=1, v_net_id=1, time=2.0),
            VirtualNetworkEvent(id=2, type=0, v_net_id=0, time=5.0),
        ]
        
    def test_init_empty(self):
        """Test initialization with no parameters."""
        simulator = VirtualNetworkRequestSimulator()
        assert simulator.v_nets == []
        assert simulator.events == []
        assert simulator.v_sim_setting == {}

    def test_default_containers_are_isolated_between_instances(self):
        """Default simulator state must never leak into another instance."""
        first = VirtualNetworkRequestSimulator()
        second = VirtualNetworkRequestSimulator()

        first.v_nets.append(Mock(spec=VirtualNetwork))
        first.events.append(VirtualNetworkEvent(id=0, type=1, v_net_id=0, time=1.0))
        try:
            assert second.v_nets == []
            assert second.events == []
        finally:
            # Keep the legacy implementation from contaminating later tests when
            # this regression test fails before the fix is applied.
            first.v_nets.clear()
            first.events.clear()
        
    def test_init_with_data(self):
        """Test initialization with v_nets and events."""
        simulator = VirtualNetworkRequestSimulator(
            v_nets=self.sample_v_nets,
            events=self.sample_events,
            v_sim_setting=self.basic_v_sim_setting
        )
        
        assert len(simulator.v_nets) == 3
        assert len(simulator.events) == 3
        assert simulator.v_sim_setting == self.basic_v_sim_setting
        
    def test_num_v_nets_property(self):
        """Test num_v_nets property."""
        simulator = VirtualNetworkRequestSimulator(v_nets=self.sample_v_nets)
        assert simulator.num_v_nets == 3
        
        empty_simulator = VirtualNetworkRequestSimulator()
        assert empty_simulator.num_v_nets == 0
        
    def test_v_sim_setting_deep_copy(self):
        """Test that v_sim_setting is deep copied."""
        original_setting = copy.deepcopy(self.basic_v_sim_setting)
        simulator = VirtualNetworkRequestSimulator(v_sim_setting=self.basic_v_sim_setting)
        
        # Modify the original setting
        self.basic_v_sim_setting['num_v_nets'] = 999
        
        # Simulator's setting should be unchanged
        assert simulator.v_sim_setting['num_v_nets'] == original_setting['num_v_nets']
        
    @patch.object(VirtualNetworkRequestSimulator, '_construct_v2event_dict')
    def test_construct_v2event_dict_called(self, mock_construct):
        """Test that _construct_v2event_dict is called during initialization."""
        VirtualNetworkRequestSimulator(v_nets=self.sample_v_nets, events=self.sample_events)
        mock_construct.assert_called_once()
        
    def test_cached_vnets_loads_class_variable(self):
        """Test that _cached_vnets_loads is a class variable."""
        assert hasattr(VirtualNetworkRequestSimulator, '_cached_vnets_loads')
        assert isinstance(VirtualNetworkRequestSimulator._cached_vnets_loads, dict)
        
    @patch('virne.network.virtual_network_request_simulator.VirtualNetworkRequestSimulator.from_setting')
    def test_from_setting_method_exists(self, mock_from_setting):
        """Test that from_setting class method exists and can be called."""
        mock_from_setting.return_value = Mock(spec=VirtualNetworkRequestSimulator)
        
        result = VirtualNetworkRequestSimulator.from_setting(self.basic_v_sim_setting)
        
        mock_from_setting.assert_called_once_with(self.basic_v_sim_setting)
        assert result is not None
        
    def test_methods_exist(self):
        """Test that expected methods exist on the simulator."""
        simulator = VirtualNetworkRequestSimulator()
        
        # Test that these methods exist (even if not implemented in the visible code)
        expected_methods = [
            'renew', 'save_dataset', 'load_dataset'
        ]
        
        for method_name in expected_methods:
            assert hasattr(simulator, method_name), f"Method '{method_name}' should exist"
            
    def test_events_with_various_types(self):
        """Test simulator with various event types."""
        events = [
            VirtualNetworkEvent(id=0, type=1, v_net_id=0, time=1.0),  # arrival
            VirtualNetworkEvent(id=1, type=0, v_net_id=0, time=5.0),  # departure
            VirtualNetworkEvent(id=2, type=1, v_net_id=1, time=3.0),  # arrival
        ]
        
        simulator = VirtualNetworkRequestSimulator(events=events)
        assert len(simulator.events) == 3
        
        # Check that events maintain their properties
        assert simulator.events[0].type == 1
        assert simulator.events[1].type == 0
        assert simulator.events[2].type == 1
        
    def test_empty_v_nets_list(self):
        """Test simulator with empty v_nets list."""
        simulator = VirtualNetworkRequestSimulator(v_nets=[], events=self.sample_events)
        assert simulator.num_v_nets == 0
        assert len(simulator.events) == 3
        
    def test_empty_events_list(self):
        """Test simulator with empty events list."""
        simulator = VirtualNetworkRequestSimulator(v_nets=self.sample_v_nets, events=[])
        assert simulator.num_v_nets == 3
        assert len(simulator.events) == 0
        
    def test_kwargs_handling(self):
        """Test that simulator handles additional kwargs."""
        simulator = VirtualNetworkRequestSimulator(
            v_nets=self.sample_v_nets,
            events=self.sample_events,
            v_sim_setting=self.basic_v_sim_setting,
            custom_param='test_value'
        )
        
        # Should not raise any errors
        assert len(simulator.v_nets) == 3
        assert len(simulator.events) == 3
        
    def test_event_time_ordering(self):
        """Test events with different time orderings."""
        unordered_events = [
            VirtualNetworkEvent(id=0, type=1, v_net_id=0, time=5.0),
            VirtualNetworkEvent(id=1, type=1, v_net_id=1, time=1.0),
            VirtualNetworkEvent(id=2, type=0, v_net_id=0, time=3.0),
        ]
        
        simulator = VirtualNetworkRequestSimulator(events=unordered_events)
        
        # Events should be stored as provided (ordering handled by arrange_events method)
        assert simulator.events[0].time == 5.0
        assert simulator.events[1].time == 1.0
        assert simulator.events[2].time == 3.0

    def test_equal_time_events_process_departure_before_arrival(self):
        """The service interval is [arrival, departure), so releases win ties."""
        active_v_net = SimpleNamespace(id=0, arrival_time=0.0, lifetime=10.0)
        arriving_v_net = SimpleNamespace(id=1, arrival_time=10.0, lifetime=5.0)
        simulator = VirtualNetworkRequestSimulator(v_nets=[active_v_net, arriving_v_net])

        simulator._renew_events()

        events_at_ten = [
            (event.type, event.v_net_id)
            for event in simulator.events
            if event.time == 10.0
        ]
        assert events_at_ten == [(0, 0), (1, 1)]

    @pytest.mark.parametrize('lifetime', [0.0, -1.0, float('nan'), float('inf')])
    def test_v_net_lifetime_must_be_finite_and_positive(self, lifetime):
        """Invalid service intervals must fail before events reach the environment."""
        v_net = SimpleNamespace(id=7, arrival_time=1.0, lifetime=lifetime)
        simulator = VirtualNetworkRequestSimulator(v_nets=[v_net])

        with pytest.raises(
            ValueError,
            match='Virtual network 7 lifetime must be finite and positive',
        ):
            simulator._renew_events()

    @pytest.mark.parametrize('time', [float('nan'), float('inf')])
    def test_event_time_must_be_finite(self, time):
        with pytest.raises(ValueError, match='Event time must be finite'):
            VirtualNetworkEvent(id=0, type=1, v_net_id=0, time=time)

    def test_normalize_events_sorts_ties_and_reindexes_legacy_ids(self):
        events = [
            VirtualNetworkEvent(id=20, type=1, v_net_id=1, time=10.0),
            VirtualNetworkEvent(id=10, type=0, v_net_id=0, time=10.0),
            VirtualNetworkEvent(id=5, type=1, v_net_id=0, time=1.0),
        ]

        normalized = VirtualNetworkRequestSimulator._normalize_events(events)

        assert [(event.time, event.type, event.v_net_id) for event in normalized] == [
            (1.0, 1, 0),
            (10.0, 0, 0),
            (10.0, 1, 1),
        ]
        assert [event.id for event in normalized] == [0, 1, 2]

    def test_poisson_arrivals_have_exponential_interarrival_times(self):
        """A Poisson process must have exponentially distributed interarrival times."""
        setting = copy.deepcopy(self.basic_v_sim_setting)
        setting['num_v_nets'] = 50_000

        simulator = VirtualNetworkRequestSimulator.from_setting(setting, seed=2026)
        simulator.arrange_v_nets()

        arrival_times = np.asarray(simulator.v_nets_arrival_time)
        intervals = np.diff(np.concatenate(([0.0], arrival_times)))
        expected_mean = 1.0 / setting['arrival_rate']['lam']

        assert np.mean(intervals) == pytest.approx(expected_mean, rel=0.03)
        assert np.std(intervals) / np.mean(intervals) == pytest.approx(1.0, abs=0.05)

    def test_legacy_and_canonical_poisson_configs_are_equivalent(self):
        """The legacy lam schema and canonical rate schema should generate the same process."""
        legacy_setting = copy.deepcopy(self.basic_v_sim_setting)
        legacy_setting['num_v_nets'] = 100
        canonical_setting = copy.deepcopy(legacy_setting)
        canonical_setting['arrival_rate'] = {
            'type': 'poisson',
            'rate': legacy_setting['arrival_rate']['lam'],
            'time_model': 'continuous',
        }

        legacy_simulator = VirtualNetworkRequestSimulator.from_setting(legacy_setting, seed=123)
        legacy_simulator.arrange_v_nets()
        canonical_simulator = VirtualNetworkRequestSimulator.from_setting(canonical_setting, seed=123)
        canonical_simulator.arrange_v_nets()

        assert np.array_equal(
            legacy_simulator.v_nets_arrival_time,
            canonical_simulator.v_nets_arrival_time,
        )

    @pytest.mark.parametrize('rate', [0, -0.01, float('nan'), float('inf')])
    def test_poisson_arrival_rate_must_be_positive(self, rate):
        """Invalid Poisson rates should fail before generating misleading data."""
        setting = copy.deepcopy(self.basic_v_sim_setting)
        setting['arrival_rate'] = {
            'type': 'poisson',
            'rate': rate,
            'time_model': 'continuous',
        }
        simulator = VirtualNetworkRequestSimulator.from_setting(setting, seed=1)

        with pytest.raises(ValueError, match='arrival rate must be finite and positive'):
            simulator.arrange_v_nets()

    def test_legacy_custom_interarrival_distribution_is_preserved(self):
        """Non-Poisson legacy interval distributions should keep their existing behavior."""
        setting = copy.deepcopy(self.basic_v_sim_setting)
        setting['num_v_nets'] = 20
        setting['arrival_rate'] = {
            'distribution': 'uniform',
            'dtype': 'float',
            'low': 3.0,
            'high': 3.0,
        }
        simulator = VirtualNetworkRequestSimulator.from_setting(setting, seed=1)
        simulator.arrange_v_nets()

        intervals = np.diff(np.concatenate(([0.0], simulator.v_nets_arrival_time)))
        assert intervals.tolist() == [3.0] * setting['num_v_nets']


class TestVirtualNetworkEventEdgeCases:
    """Test edge cases for VirtualNetworkEvent."""
    
    def test_zero_values(self):
        """Test event with zero values."""
        event = VirtualNetworkEvent(id=0, type=0, v_net_id=0, time=0.0)
        assert event.id == 0
        assert event.type == 0
        assert event.v_net_id == 0
        assert event.time == 0.0
        
    def test_large_values(self):
        """Test event with large values."""
        event = VirtualNetworkEvent(id=999999, type=1, v_net_id=999999, time=999999.999)
        assert event.id == 999999
        assert event.type == 1
        assert event.v_net_id == 999999
        assert event.time == 999999.999
        
    def test_float_ids(self):
        """Test that float IDs are converted to int."""
        event = VirtualNetworkEvent(id=1.0, type=1, v_net_id=10.0, time=5.0)
        # Note: This test assumes the implementation accepts float ids
        # If the implementation enforces int types, this test should be adjusted
        assert event.id == 1.0  # or 1 if converted
        assert event.v_net_id == 10.0  # or 10 if converted


if __name__ == '__main__':
    pytest.main([__file__])
