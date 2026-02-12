"""
Plan Selector - Cost-Based Optimization

Evaluates generated plans using cost models and selects the optimal one.
This is where SQL cost model + Vector cost model come together.
"""

import logging
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_model.vector_cost import VectorCostModel
from cost_model.sql_cost import SQLCostModel
from optimizer.plan_generator import ExecutionPlan, PlanType

logger = logging.getLogger(__name__)


class PlanSelector:
    """
    Selects the best execution plan using cost-based optimization.
    
    Combines SQL and vector cost models to estimate total query cost.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize plan selector with cost models.
        
        Args:
            config: Configuration dictionary with cost model parameters
        """
        cost_config = config.get('cost_model', {})
        
        self.vector_cost_model = VectorCostModel(cost_config)
        self.sql_cost_model = SQLCostModel(cost_config)
        
    def select_best_plan(
        self,
        plans: List[ExecutionPlan],
        table_stats: Dict[str, Any]
    ) -> ExecutionPlan:
        """
        Select the plan with minimum estimated cost.
        
        Args:
            plans: List of candidate execution plans
            table_stats: Table statistics for cost estimation
            
        Returns:
            ExecutionPlan with lowest estimated cost
        """
        if not plans:
            raise ValueError("No plans provided for selection")
        
        # Estimate cost for each plan
        for plan in plans:
            plan.estimated_cost = self.estimate_plan_cost(plan, table_stats)
            logger.info(f"{plan.plan_id} ({plan.plan_type.value}): cost={plan.estimated_cost:.2f}")
        
        # Select minimum cost plan
        best_plan = min(plans, key=lambda p: p.estimated_cost)
        
        logger.info(f"Selected {best_plan.plan_id}: {best_plan.plan_type.value} (cost={best_plan.estimated_cost:.2f})")
        
        return best_plan
    
    def estimate_plan_cost(
        self,
        plan: ExecutionPlan,
        table_stats: Dict[str, Any]
    ) -> float:
        """
        Estimate total cost of an execution plan.
        
        Walks through plan operations and sums costs.
        
        Args:
            plan: Execution plan to estimate
            table_stats: Table statistics
            
        Returns:
            Total estimated cost
        """
        total_cost = 0.0
        current_rows = table_stats.get('n_tuples', 100000)
        
        for op in plan.operations:
            op_type = op['type']
            
            if op_type == 'filter':
                # SQL filter operation
                filter_cost, output_rows = self._estimate_filter_cost(
                    op, current_rows, table_stats
                )
                total_cost += filter_cost
                current_rows = output_rows
                logger.debug(f"  Filter: cost={filter_cost:.2f}, output={output_rows} rows")
                
            elif op_type == 'vector_search':
                # Vector similarity search
                vector_cost = self._estimate_vector_search_cost(
                    op, current_rows, table_stats
                )
                total_cost += vector_cost
                current_rows = min(op['k'], current_rows)  # Top-k results
                logger.debug(f"  VectorSearch: cost={vector_cost:.2f}, output={current_rows} rows")
                
            elif op_type == 'limit':
                # Just a limit operation (minimal cost)
                current_rows = min(op['k'], current_rows)
                logger.debug(f"  Limit: output={current_rows} rows")
        
        # Add small baseline cost to avoid zero costs
        if total_cost == 0.0:
            total_cost = 1.0
            
        return total_cost
    
    def _estimate_filter_cost(
        self,
        filter_op: Dict[str, Any],
        n_rows: int,
        table_stats: Dict[str, Any]
    ) -> tuple:
        """
        Estimate cost of SQL filter operation.
        
        Returns:
            Tuple of (cost, output_rows)
        """
        predicates = filter_op['predicates']
        method = filter_op.get('method', 'seqscan')
        
        # Estimate combined selectivity
        combined_selectivity = 1.0
        for pred in predicates:
            if pred.selectivity:
                combined_selectivity *= pred.selectivity
            else:
                # Estimate selectivity based on predicate type
                selectivity = self.sql_cost_model.estimate_selectivity(
                    pred.predicate_type,
                    n_distinct=table_stats.get(f'{pred.column}_distinct', None)
                )
                combined_selectivity *= selectivity
        
        output_rows = int(n_rows * combined_selectivity)
        
        # Estimate cost based on scan method
        if method == 'seqscan':
            # Sequential scan cost
            scan_cost = self.sql_cost_model.estimate_sequential_scan_cost(
                n_rows,
                table_stats.get('avg_tuple_size', 200)
            )
            filter_cost = self.sql_cost_model.estimate_filter_cost(n_rows, len(predicates))
            cost = scan_cost + filter_cost
            
        elif method == 'indexscan':
            # Index scan cost (already includes filter application)
            cost = self.sql_cost_model.estimate_index_scan_cost(
                n_rows,
                output_rows,
                table_stats.get('avg_tuple_size', 200)
            )
            # Add explicit filter cost
            cost += self.sql_cost_model.estimate_filter_cost(output_rows, len(predicates))
            
        else:  # sequential on small dataset
            cost = self.sql_cost_model.estimate_filter_cost(n_rows, len(predicates))
        
        return cost, output_rows
    
    def _estimate_vector_search_cost(
        self,
        vector_op: Dict[str, Any],
        n_vectors: int,
        table_stats: Dict[str, Any]
    ) -> float:
        """
        Estimate cost of vector similarity search.
        
        Returns:
            Estimated cost
        """
        k = vector_op['k']
        index_type = vector_op.get('index_type', 'hnsw')
        
        if index_type == 'hnsw':
            # Get HNSW parameters from config or defaults
            m = table_stats.get('hnsw_m', 16)
            ef_search = table_stats.get('hnsw_ef_search', 40)
            
            cost = self.vector_cost_model.estimate_hnsw_search_cost(
                n_vectors, k, m, ef_search
            )
            
        elif index_type == 'ivfflat':
            # Get IVFFlat parameters
            n_lists = table_stats.get('ivfflat_lists', 100)
            n_probes = table_stats.get('ivfflat_probes', 10)
            
            cost = self.vector_cost_model.estimate_ivfflat_search_cost(
                n_vectors, k, n_lists, n_probes
            )
            
        else:  # sequential scan
            cost = self.vector_cost_model.estimate_sequential_vector_scan(
                n_vectors, k
            )
        
        return cost
    
    def explain_plan(self, plan: ExecutionPlan, table_stats: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of execution plan.
        
        Similar to PostgreSQL's EXPLAIN output.
        
        Args:
            plan: Execution plan to explain
            table_stats: Table statistics
            
        Returns:
            Formatted explanation string
        """
        lines = []
        lines.append(f"Execution Plan: {plan.plan_id}")
        lines.append(f"Type: {plan.plan_type.value}")
        lines.append(f"Estimated Cost: {plan.estimated_cost:.2f}")
        lines.append("")
        lines.append("Operations:")
        
        current_rows = table_stats.get('n_tuples', 100000)
        
        for i, op in enumerate(plan.operations, 1):
            op_type = op['type']
            
            if op_type == 'filter':
                predicates = op['predicates']
                method = op.get('method', 'seqscan')
                
                pred_strs = []
                for pred in predicates:
                    pred_strs.append(f"{pred.column} {pred.operator} {pred.value}")
                
                lines.append(f"  {i}. {method.upper()}")
                lines.append(f"     Filters: {' AND '.join(pred_strs)}")
                
                # Estimate output
                _, output_rows = self._estimate_filter_cost(op, current_rows, table_stats)
                lines.append(f"     Input: {current_rows:,} rows → Output: {output_rows:,} rows")
                current_rows = output_rows
                
            elif op_type == 'vector_search':
                k = op['k']
                index_type = op.get('index_type', 'hnsw')
                
                lines.append(f"  {i}. VECTOR SEARCH ({index_type.upper()})")
                lines.append(f"     Top-{k} by {op.get('distance_metric', 'cosine')} similarity")
                lines.append(f"     Input: {current_rows:,} vectors → Output: {min(k, current_rows):,} rows")
                current_rows = min(k, current_rows)
                
            elif op_type == 'limit':
                k = op['k']
                lines.append(f"  {i}. LIMIT {k}")
                lines.append(f"     Output: {min(k, current_rows):,} rows")
                current_rows = min(k, current_rows)
        
        lines.append("")
        lines.append(f"Final Output: {current_rows:,} rows")
        
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from optimizer.plan_generator import PlanGenerator, QueryPredicate, VectorOperation
    
    # Example query components
    predicates = [
        QueryPredicate(
            column='brand',
            operator='=',
            value='Apple',
            predicate_type='equality',
            selectivity=0.02
        ),
        QueryPredicate(
            column='price',
            operator='<',
            value=1000,
            predicate_type='range',
            selectivity=0.3
        )
    ]
    
    vector_op = VectorOperation(
        embedding_column='embedding',
        query_vector=[0.1] * 768,
        k=10,
        distance_metric='cosine',
        index_type='hnsw'
    )
    
    table_stats = {
        'n_tuples': 100000,
        'indexed_columns': {'brand', 'price'},
        'avg_tuple_size': 200,
        'brand_distinct': 50,
        'hnsw_m': 16,
        'hnsw_ef_search': 40
    }
    
    # Generate plans
    config = {
        'enable_filter_first': True,
        'enable_vector_first': True,
        'enable_hybrid': False,
        'cost_model': {
            'vector_distance_cost': 1.0,
            'vector_comparison_cost': 0.1,
            'seq_scan_cost': 1.0,
            'random_page_cost': 4.0,
            'cpu_tuple_cost': 0.01,
            'cpu_index_tuple_cost': 0.005,
            'cpu_operator_cost': 0.0025
        }
    }
    
    generator = PlanGenerator(config)
    plans = generator.generate_plans(predicates, vector_op, table_stats)
    
    # Select best plan
    selector = PlanSelector(config)
    best_plan = selector.select_best_plan(plans, table_stats)
    
    print("\n" + "=" * 60)
    print("PLAN SELECTION RESULTS")
    print("=" * 60)
    
    for plan in plans:
        print(f"\n{plan}")
        if plan == best_plan:
            print("  ✓ SELECTED")
    
    print("\n" + "=" * 60)
    print("EXPLAIN BEST PLAN")
    print("=" * 60)
    print(selector.explain_plan(best_plan, table_stats))
