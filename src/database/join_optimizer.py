"""
Join Optimizer - Database Architect Component
Handles multi-table join ordering and optimization for hybrid queries
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import itertools
import numpy as np


class JoinType(Enum):
    """Types of join operations"""
    INNER = "INNER JOIN"
    LEFT = "LEFT JOIN"
    RIGHT = "RIGHT JOIN"
    CROSS = "CROSS JOIN"
    SEMI = "SEMI JOIN"


class JoinMethod(Enum):
    """Physical join implementation methods"""
    NESTED_LOOP = "nested_loop"
    HASH_JOIN = "hash_join"
    MERGE_JOIN = "merge_join"
    INDEX_NESTED_LOOP = "index_nested_loop"


@dataclass
class JoinCondition:
    """Represents a join condition between two tables"""
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    join_type: JoinType = JoinType.INNER
    
    def __hash__(self):
        return hash((self.left_table, self.left_column, self.right_table, self.right_column))


@dataclass
class TableStats:
    """Statistics for a table used in join ordering"""
    table_name: str
    row_count: int
    avg_row_size: int  # bytes
    has_index: Dict[str, bool]  # column -> has_index
    selectivity: float = 1.0  # After filters applied
    has_vector_search: bool = False
    vector_search_k: int = 0


@dataclass
class JoinNode:
    """Represents a node in the join tree"""
    tables: Set[str]
    row_estimate: int
    cost: float
    method: JoinMethod
    left_child: Optional['JoinNode'] = None
    right_child: Optional['JoinNode'] = None
    join_condition: Optional[JoinCondition] = None
    
    def is_leaf(self) -> bool:
        return self.left_child is None and self.right_child is None


class JoinOrderOptimizer:
    """
    Implements join ordering algorithms for multi-table queries
    Uses dynamic programming (similar to System R) and greedy heuristics
    """
    
    def __init__(self, cost_model=None):
        self.cost_model = cost_model
        self.memo = {}  # DP memoization table
        
    def optimize_join_order(
        self,
        tables: List[TableStats],
        join_conditions: List[JoinCondition],
        use_dp: bool = True
    ) -> JoinNode:
        """
        Find optimal join order using dynamic programming or greedy algorithm
        
        Args:
            tables: List of tables with statistics
            join_conditions: Join predicates between tables
            use_dp: Use DP (exact) vs greedy (approximate)
            
        Returns:
            Optimal join tree
        """
        if len(tables) <= 1:
            return self._create_leaf_node(tables[0])
        
        if use_dp and len(tables) <= 8:
            # DP is exponential, only use for small queries
            return self._dp_join_order(tables, join_conditions)
        else:
            # Greedy for larger queries
            return self._greedy_join_order(tables, join_conditions)
    
    def _dp_join_order(
        self,
        tables: List[TableStats],
        join_conditions: List[JoinCondition]
    ) -> JoinNode:
        """
        Dynamic programming join ordering (System R algorithm)
        Time: O(3^n), Space: O(2^n)
        """
        self.memo.clear()
        table_names = {t.table_name for t in tables}
        table_map = {t.table_name: t for t in tables}
        
        # Build join graph
        join_graph = self._build_join_graph(join_conditions)
        
        # Initialize leaves (single tables)
        for table in tables:
            subset = frozenset([table.table_name])
            self.memo[subset] = self._create_leaf_node(table)
        
        # DP: build up from smaller to larger subsets
        for size in range(2, len(tables) + 1):
            for subset in itertools.combinations(table_names, size):
                subset_frozen = frozenset(subset)
                best_plan = None
                best_cost = float('inf')
                
                # Try all ways to split this subset
                for left_size in range(1, size):
                    for left_subset in itertools.combinations(subset, left_size):
                        left_frozen = frozenset(left_subset)
                        right_frozen = subset_frozen - left_frozen
                        
                        # Check if this is a valid join (connected by join condition)
                        join_cond = self._find_join_condition(
                            left_frozen, right_frozen, join_conditions
                        )
                        if join_cond is None:
                            continue
                        
                        # Get best plans for left and right
                        left_plan = self.memo.get(left_frozen)
                        right_plan = self.memo.get(right_frozen)
                        
                        if left_plan is None or right_plan is None:
                            continue
                        
                        # Estimate cost of joining these two
                        for method in JoinMethod:
                            cost, rows = self._estimate_join_cost(
                                left_plan, right_plan, join_cond, method
                            )
                            
                            total_cost = left_plan.cost + right_plan.cost + cost
                            
                            if total_cost < best_cost:
                                best_cost = total_cost
                                best_plan = JoinNode(
                                    tables=subset_frozen,
                                    row_estimate=rows,
                                    cost=total_cost,
                                    method=method,
                                    left_child=left_plan,
                                    right_child=right_plan,
                                    join_condition=join_cond
                                )
                
                if best_plan is not None:
                    self.memo[subset_frozen] = best_plan
        
        return self.memo.get(frozenset(table_names))
    
    def _greedy_join_order(
        self,
        tables: List[TableStats],
        join_conditions: List[JoinCondition]
    ) -> JoinNode:
        """
        Greedy join ordering heuristic
        - Start with smallest table (after filters/vector search)
        - Iteratively join next table that minimizes cost
        """
        remaining_tables = {t.table_name: t for t in tables}
        
        # Find smallest table to start with
        start_table = min(tables, key=lambda t: t.row_count * t.selectivity)
        current_plan = self._create_leaf_node(start_table)
        del remaining_tables[start_table.table_name]
        
        # Greedily add tables
        while remaining_tables:
            best_cost = float('inf')
            best_plan = None
            best_table_name = None
            
            for table_name, table_stats in remaining_tables.items():
                # Find join condition
                join_cond = self._find_join_condition(
                    current_plan.tables,
                    frozenset([table_name]),
                    join_conditions
                )
                
                if join_cond is None:
                    continue  # No direct join available
                
                # Try each join method
                right_plan = self._create_leaf_node(table_stats)
                for method in JoinMethod:
                    cost, rows = self._estimate_join_cost(
                        current_plan, right_plan, join_cond, method
                    )
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_table_name = table_name
                        best_plan = JoinNode(
                            tables=current_plan.tables | {table_name},
                            row_estimate=rows,
                            cost=current_plan.cost + cost,
                            method=method,
                            left_child=current_plan,
                            right_child=right_plan,
                            join_condition=join_cond
                        )
            
            if best_plan is None:
                # No valid join found - do cross join with cheapest remaining
                table_name = min(remaining_tables.keys(), 
                               key=lambda t: remaining_tables[t].row_count)
                table_stats = remaining_tables[table_name]
                right_plan = self._create_leaf_node(table_stats)
                
                cost, rows = self._estimate_cross_join_cost(current_plan, right_plan)
                best_plan = JoinNode(
                    tables=current_plan.tables | {table_name},
                    row_estimate=rows,
                    cost=current_plan.cost + cost,
                    method=JoinMethod.NESTED_LOOP,
                    left_child=current_plan,
                    right_child=right_plan
                )
                best_table_name = table_name
            
            current_plan = best_plan
            del remaining_tables[best_table_name]
        
        return current_plan
    
    def _build_join_graph(self, join_conditions: List[JoinCondition]) -> Dict:
        """Build adjacency list for join graph"""
        graph = {}
        for jc in join_conditions:
            if jc.left_table not in graph:
                graph[jc.left_table] = []
            if jc.right_table not in graph:
                graph[jc.right_table] = []
            graph[jc.left_table].append((jc.right_table, jc))
            graph[jc.right_table].append((jc.left_table, jc))
        return graph
    
    def _find_join_condition(
        self,
        left_tables: Set[str],
        right_tables: Set[str],
        join_conditions: List[JoinCondition]
    ) -> Optional[JoinCondition]:
        """Find a join condition connecting two table sets"""
        for jc in join_conditions:
            if ((jc.left_table in left_tables and jc.right_table in right_tables) or
                (jc.left_table in right_tables and jc.right_table in left_tables)):
                return jc
        return None
    
    def _create_leaf_node(self, table: TableStats) -> JoinNode:
        """Create a leaf node for a single table"""
        effective_rows = int(table.row_count * table.selectivity)
        if table.has_vector_search and table.vector_search_k > 0:
            effective_rows = min(effective_rows, table.vector_search_k)
        
        return JoinNode(
            tables={table.table_name},
            row_estimate=effective_rows,
            cost=0.0,  # Leaf nodes have zero join cost
            method=JoinMethod.NESTED_LOOP  # Placeholder
        )
    
    def _estimate_join_cost(
        self,
        left: JoinNode,
        right: JoinNode,
        join_cond: JoinCondition,
        method: JoinMethod
    ) -> Tuple[float, int]:
        """
        Estimate cost and output cardinality of a join
        
        Returns:
            (cost, output_rows)
        """
        left_rows = left.row_estimate
        right_rows = right.row_estimate
        
        # Estimate output cardinality (simplified)
        # In reality, use histograms and correlation statistics
        output_rows = int(left_rows * right_rows * 0.01)  # 1% selectivity assumption
        
        if method == JoinMethod.NESTED_LOOP:
            # Cost: outer_rows * (inner_rows + startup_cost)
            cost = left_rows * (right_rows * 0.01 + 1.0)
            
        elif method == JoinMethod.INDEX_NESTED_LOOP:
            # Cost: outer_rows * (log(inner_rows) + startup)
            if right_rows > 0:
                cost = left_rows * (np.log2(max(right_rows, 1)) * 2.0 + 1.0)
            else:
                cost = left_rows
                
        elif method == JoinMethod.HASH_JOIN:
            # Cost: build_hash(left) + probe(right)
            build_cost = left_rows * 0.02  # Hash table build
            probe_cost = right_rows * 0.01  # Hash lookup
            cost = build_cost + probe_cost + 100.0  # Startup cost
            
        elif method == JoinMethod.MERGE_JOIN:
            # Cost: sort(left) + sort(right) + merge
            if left_rows > 0 and right_rows > 0:
                sort_cost = (left_rows * np.log2(max(left_rows, 1)) +
                           right_rows * np.log2(max(right_rows, 1)))
                merge_cost = left_rows + right_rows
                cost = sort_cost * 0.05 + merge_cost * 0.01
            else:
                cost = 0.0
        else:
            cost = left_rows * right_rows * 0.01
        
        return cost, output_rows
    
    def _estimate_cross_join_cost(
        self,
        left: JoinNode,
        right: JoinNode
    ) -> Tuple[float, int]:
        """Estimate cost of cross join (no condition)"""
        output_rows = left.row_estimate * right.row_estimate
        cost = output_rows * 0.01  # Sequential scan cost
        return cost, output_rows
    
    def explain_plan(self, node: JoinNode, indent: int = 0) -> str:
        """Generate EXPLAIN-style output for join tree"""
        if node is None:
            return ""
        
        prefix = "  " * indent
        lines = []
        
        if node.is_leaf():
            lines.append(f"{prefix}Scan {list(node.tables)[0]} (rows={node.row_estimate})")
        else:
            lines.append(
                f"{prefix}{node.method.value} "
                f"(cost={node.cost:.2f}, rows={node.row_estimate})"
            )
            if node.join_condition:
                jc = node.join_condition
                lines.append(
                    f"{prefix}  Condition: {jc.left_table}.{jc.left_column} = "
                    f"{jc.right_table}.{jc.right_column}"
                )
            lines.append(self.explain_plan(node.left_child, indent + 1))
            lines.append(self.explain_plan(node.right_child, indent + 1))
        
        return "\n".join(lines)


class HybridJoinOptimizer(JoinOrderOptimizer):
    """
    Extended join optimizer that handles hybrid SQL + vector queries
    - Pushes vector search early when selectivity is high
    - Coordinates between SQL filters and vector similarity
    """
    
    def optimize_hybrid_query(
        self,
        tables: List[TableStats],
        join_conditions: List[JoinCondition],
        vector_search_table: Optional[str] = None,
        vector_k: int = 100
    ) -> JoinNode:
        """
        Optimize join order for hybrid queries with vector search
        
        Strategy:
        1. If vector search is very selective, do it first
        2. Apply SQL filters on vector results
        3. Join with other tables
        """
        # Mark which table has vector search
        if vector_search_table:
            for table in tables:
                if table.table_name == vector_search_table:
                    table.has_vector_search = True
                    table.vector_search_k = vector_k
                    # Vector search is typically very selective
                    table.selectivity = min(table.selectivity, vector_k / table.row_count)
        
        # Use standard join ordering with updated selectivity
        return self.optimize_join_order(tables, join_conditions, use_dp=True)
    
    def should_push_vector_first(
        self,
        vector_table: TableStats,
        sql_selectivity: float,
        vector_k: int
    ) -> bool:
        """
        Decide whether to do vector search before or after SQL filters
        
        Returns:
            True if vector search should be done first
        """
        vector_output = vector_k
        sql_output = int(vector_table.row_count * sql_selectivity)
        
        # If vector search is more selective, do it first
        if vector_output < sql_output:
            return True
        
        # If SQL is very selective (< 1%), do SQL first
        if sql_selectivity < 0.01:
            return False
        
        # For moderate selectivity, vector-first is often better
        # because it reduces the candidate set significantly
        return vector_k < vector_table.row_count * 0.1
