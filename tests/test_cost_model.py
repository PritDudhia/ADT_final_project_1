"""
Cost Model Unit Tests

Tests for vector and SQL cost models.
"""

import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cost_model.vector_cost import VectorCostModel
from cost_model.sql_cost import SQLCostModel


class TestVectorCostModel:
    """Tests for VectorCostModel"""
    
    @pytest.fixture
    def cost_model(self):
        config = {
            'vector_distance_cost': 1.0,
            'vector_comparison_cost': 0.1,
            'cpu_operator_cost': 0.0025
        }
        return VectorCostModel(config)
    
    def test_hnsw_cost_scales_logarithmically(self, cost_model):
        """HNSW cost should scale logarithmically with dataset size"""
        cost_10k = cost_model.estimate_hnsw_search_cost(10000, k=10)
        cost_100k = cost_model.estimate_hnsw_search_cost(100000, k=10)
        
        # Cost should increase, but not linearly
        assert cost_100k > cost_10k
        assert cost_100k < cost_10k * 5  # Much less than 10x
    
    def test_ivfflat_cost_increases_with_probes(self, cost_model):
        """IVFFlat cost should increase with more probes"""
        cost_5_probes = cost_model.estimate_ivfflat_search_cost(
            100000, k=10, n_lists=100, n_probes=5
        )
        cost_20_probes = cost_model.estimate_ivfflat_search_cost(
            100000, k=10, n_lists=100, n_probes=20
        )
        
        assert cost_20_probes > cost_5_probes
    
    def test_sequential_scan_linear(self, cost_model):
        """Sequential scan cost should be linear in dataset size"""
        cost_10k = cost_model.estimate_sequential_vector_scan(10000, k=10)
        cost_20k = cost_model.estimate_sequential_vector_scan(20000, k=10)
        
        # Should be approximately 2x
        assert 1.8 < (cost_20k / cost_10k) < 2.2
    
    def test_filter_first_recommendation(self, cost_model):
        """High selectivity should recommend filter-first"""
        strategy = cost_model.recommend_strategy(
            n_vectors=100000,
            k=10,
            filter_selectivity=0.01,  # Very selective (1%)
            index_type='hnsw'
        )
        
        assert strategy == 'filter_first'
    
    def test_vector_first_recommendation(self, cost_model):
        """Low selectivity should recommend vector-first"""
        strategy = cost_model.recommend_strategy(
            n_vectors=100000,
            k=10,
            filter_selectivity=0.9,  # Not selective (90%)
            index_type='hnsw'
        )
        
        assert strategy == 'vector_first'


class TestSQLCostModel:
    """Tests for SQLCostModel"""
    
    @pytest.fixture
    def sql_cost(self):
        config = {
            'seq_scan_cost': 1.0,
            'random_page_cost': 4.0,
            'cpu_tuple_cost': 0.01,
            'cpu_index_tuple_cost': 0.005,
            'cpu_operator_cost': 0.0025
        }
        return SQLCostModel(config)
    
    def test_seqscan_cost_linear(self, sql_cost):
        """Sequential scan cost should scale linearly"""
        cost_10k = sql_cost.estimate_sequential_scan_cost(10000)
        cost_20k = sql_cost.estimate_sequential_scan_cost(20000)
        
        assert 1.8 < (cost_20k / cost_10k) < 2.2
    
    def test_index_scan_cheaper_for_selective_queries(self, sql_cost):
        """Index scan should be cheaper for selective queries"""
        n_tuples = 100000
        n_matching = 100  # 0.1% selectivity
        
        seqscan_cost = sql_cost.estimate_sequential_scan_cost(n_tuples)
        indexscan_cost = sql_cost.estimate_index_scan_cost(n_tuples, n_matching)
        
        assert indexscan_cost < seqscan_cost
    
    def test_seqscan_cheaper_for_large_results(self, sql_cost):
        """Sequential scan should be cheaper for large result sets"""
        n_tuples = 100000
        n_matching = 90000  # 90% selectivity
        
        seqscan_cost = sql_cost.estimate_sequential_scan_cost(n_tuples)
        indexscan_cost = sql_cost.estimate_index_scan_cost(n_tuples, n_matching)
        
        assert seqscan_cost < indexscan_cost
    
    def test_selectivity_estimation_equality(self, sql_cost):
        """Equality selectivity should be 1/n_distinct"""
        selectivity = sql_cost.estimate_selectivity('equality', n_distinct=50)
        assert abs(selectivity - 0.02) < 0.001  # 1/50 = 0.02
    
    def test_combined_selectivity(self, sql_cost):
        """Combined selectivity should multiply individual selectivities"""
        sel1 = 0.1  # 10%
        sel2 = 0.2  # 20%
        
        combined = sql_cost.estimate_combined_selectivity([sel1, sel2])
        
        assert abs(combined - 0.02) < 0.001  # 0.1 * 0.2 = 0.02


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
