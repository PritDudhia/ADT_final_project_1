"""
SQL Cost Model for Traditional Operations

Estimates costs for SQL filters, joins, and scans using PostgreSQL cost model.
Integrates with vector cost model for hybrid query optimization.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class SQLCostModel:
    """
    Cost model for traditional SQL operations.
    
    Based on PostgreSQL's cost model:
    - Sequential scans
    - Index scans
    - Filter operations
    - Selectivity estimation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize SQL cost model with PostgreSQL-compatible parameters.
        
        Args:
            config: Configuration dictionary with cost parameters
        """
        # I/O costs
        self.seq_page_cost = config.get('seq_scan_cost', 1.0)
        self.random_page_cost = config.get('random_page_cost', 4.0)
        
        # CPU costs
        self.cpu_tuple_cost = config.get('cpu_tuple_cost', 0.01)
        self.cpu_index_tuple_cost = config.get('cpu_index_tuple_cost', 0.005)
        self.cpu_operator_cost = config.get('cpu_operator_cost', 0.0025)
        
        # Page size (PostgreSQL default)
        self.page_size = 8192  # bytes
        
    def estimate_sequential_scan_cost(
        self, 
        n_tuples: int, 
        avg_tuple_size: int = 200
    ) -> float:
        """
        Estimate cost of sequential table scan.
        
        Cost = (pages * seq_page_cost) + (tuples * cpu_tuple_cost)
        
        Args:
            n_tuples: Number of rows in table
            avg_tuple_size: Average row size in bytes
            
        Returns:
            Estimated cost
        """
        # Calculate number of pages
        tuples_per_page = max(1, self.page_size // avg_tuple_size)
        n_pages = max(1, (n_tuples + tuples_per_page - 1) // tuples_per_page)
        
        # I/O cost
        io_cost = n_pages * self.seq_page_cost
        
        # CPU cost
        cpu_cost = n_tuples * self.cpu_tuple_cost
        
        total_cost = io_cost + cpu_cost
        
        logger.debug(f"SeqScan: {n_tuples} tuples, {n_pages} pages -> cost={total_cost:.2f}")
        
        return total_cost
    
    def estimate_index_scan_cost(
        self, 
        n_tuples: int, 
        n_matching: int,
        avg_tuple_size: int = 200
    ) -> float:
        """
        Estimate cost of B-tree index scan.
        
        Cost = (index_pages * random_page_cost)       # Random I/O for index
               + (matching_tuples * random_page_cost) # Random I/O for heap
               + (matching_tuples * cpu_index_tuple_cost)
        
        Args:
            n_tuples: Total rows in table
            n_matching: Estimated rows matching index condition
            avg_tuple_size: Average row size
            
        Returns:
            Estimated cost
        """
        # Estimate index height (B-tree)
        # Assume ~200 keys per page for typical B-tree
        keys_per_page = 200
        index_height = max(1, int(np.log(max(n_tuples, 1)) / np.log(keys_per_page)))
        
        # Cost of traversing index (height * random I/O)
        index_traversal_cost = index_height * self.random_page_cost
        
        # Cost of fetching matching tuples from heap (random I/O)
        heap_fetch_cost = n_matching * self.random_page_cost
        
        # CPU cost
        cpu_cost = n_matching * self.cpu_index_tuple_cost
        
        total_cost = index_traversal_cost + heap_fetch_cost + cpu_cost
        
        logger.debug(f"IndexScan: {n_matching}/{n_tuples} tuples -> cost={total_cost:.2f}")
        
        return total_cost
    
    def estimate_filter_cost(self, n_tuples: int, n_predicates: int = 1) -> float:
        """
        Estimate cost of applying filter predicates.
        
        Cost = tuples * predicates * cpu_operator_cost
        
        Args:
            n_tuples: Number of tuples to filter
            n_predicates: Number of filter conditions
            
        Returns:
            Estimated cost
        """
        cost = n_tuples * n_predicates * self.cpu_operator_cost
        
        logger.debug(f"Filter: {n_tuples} tuples, {n_predicates} predicates -> cost={cost:.2f}")
        
        return cost
    
    def estimate_selectivity(
        self, 
        predicate_type: str, 
        n_distinct: Optional[int] = None,
        range_fraction: Optional[float] = None
    ) -> float:
        """
        Estimate filter selectivity (fraction of rows passing filter).
        
        Uses PostgreSQL's selectivity estimation heuristics.
        
        Args:
            predicate_type: Type of predicate ('equality', 'range', 'like', etc.)
            n_distinct: Number of distinct values for equality predicates
            range_fraction: Fraction of range for range predicates (0.0 to 1.0)
            
        Returns:
            Selectivity estimate (0.0 to 1.0)
        """
        if predicate_type == 'equality':
            # For equality: selectivity = 1 / n_distinct
            if n_distinct and n_distinct > 0:
                return 1.0 / n_distinct
            else:
                return 0.005  # Default: 0.5%
        
        elif predicate_type == 'range':
            # For range queries: use provided fraction or default
            if range_fraction is not None:
                return max(0.0, min(1.0, range_fraction))
            else:
                return 0.1  # Default: 10%
        
        elif predicate_type == 'like':
            # For LIKE queries
            return 0.05  # Default: 5%
        
        elif predicate_type == 'in':
            # For IN queries: depends on list size
            if n_distinct:
                # Assume IN list has sqrt(n_distinct) values
                return min(1.0, (n_distinct ** 0.5) / n_distinct)
            else:
                return 0.1  # Default: 10%
        
        else:
            # Unknown predicate type
            return 0.1  # Conservative default
    
    def estimate_combined_selectivity(self, selectivities: List[float]) -> float:
        """
        Estimate combined selectivity of multiple predicates (AND conditions).
        
        Assumes independence: sel(A AND B) = sel(A) * sel(B)
        
        Args:
            selectivities: List of individual selectivity estimates
            
        Returns:
            Combined selectivity
        """
        if not selectivities:
            return 1.0
        
        combined = 1.0
        for sel in selectivities:
            combined *= sel
        
        return combined
    
    def choose_scan_method(
        self, 
        n_tuples: int, 
        selectivity: float,
        has_index: bool = True,
        avg_tuple_size: int = 200
    ) -> Tuple[str, float]:
        """
        Choose between sequential scan and index scan based on cost.
        
        Args:
            n_tuples: Total rows in table
            selectivity: Filter selectivity
            has_index: Whether an index exists on filter column
            avg_tuple_size: Average row size
            
        Returns:
            Tuple of (scan_method, estimated_cost)
            scan_method: 'seqscan' or 'indexscan'
        """
        n_matching = int(n_tuples * selectivity)
        
        # Sequential scan cost
        seqscan_cost = self.estimate_sequential_scan_cost(n_tuples, avg_tuple_size)
        seqscan_cost += self.estimate_filter_cost(n_tuples)
        
        if has_index:
            # Index scan cost
            indexscan_cost = self.estimate_index_scan_cost(n_tuples, n_matching, avg_tuple_size)
            
            # Choose cheaper method
            if indexscan_cost < seqscan_cost:
                logger.debug(f"Choose IndexScan: {indexscan_cost:.2f} < {seqscan_cost:.2f}")
                return 'indexscan', indexscan_cost
        
        logger.debug(f"Choose SeqScan: {seqscan_cost:.2f}")
        return 'seqscan', seqscan_cost



# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Initialize SQL cost model
    config = {
        'seq_scan_cost': 1.0,
        'random_page_cost': 4.0,
        'cpu_tuple_cost': 0.01,
        'cpu_index_tuple_cost': 0.005,
        'cpu_operator_cost': 0.0025
    }
    
    sql_cost = SQLCostModel(config)
    
    # Test scenario: 100K products table
    n_tuples = 100000
    
    print("=" * 60)
    print("SQL Cost Model Analysis")
    print("=" * 60)
    
    # Sequential scan
    seqscan_cost = sql_cost.estimate_sequential_scan_cost(n_tuples)
    print(f"\nSequential Scan ({n_tuples:,} tuples): {seqscan_cost:.2f}")
    
    # Selectivity estimation
    print("\n" + "=" * 60)
    print("Selectivity Estimation")
    print("=" * 60)
    
    # Equality: brand = 'Apple'
    brand_selectivity = sql_cost.estimate_selectivity('equality', n_distinct=50)
    print(f"brand = 'Apple' (50 brands): {brand_selectivity:.4f} ({brand_selectivity*100:.2f}%)")
    
    # Range: price < 1000
    price_selectivity = sql_cost.estimate_selectivity('range', range_fraction=0.3)
    print(f"price < 1000: {price_selectivity:.4f} ({price_selectivity*100:.2f}%)")
    
    # Combined: brand = 'Apple' AND price < 1000
    combined = sql_cost.estimate_combined_selectivity([brand_selectivity, price_selectivity])
    print(f"brand = 'Apple' AND price < 1000: {combined:.4f} ({combined*100:.2f}%)")
    
    # Scan method selection
    print("\n" + "=" * 60)
    print("Scan Method Selection")
    print("=" * 60)
    
    for selectivity in [0.001, 0.01, 0.1, 0.5]:
        method, cost = sql_cost.choose_scan_method(n_tuples, selectivity, has_index=True)
        n_matching = int(n_tuples * selectivity)
        print(f"\nSelectivity: {selectivity:.1%} ({n_matching:,} rows)")
        print(f"  → Method: {method.upper()}, Cost: {cost:.2f}")
