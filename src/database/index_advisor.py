"""
Index Advisor - Database Architect Component
Recommends optimal indexes for hybrid SQL+Vector workloads
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class IndexType(Enum):
    """Types of indexes"""
    BTREE = "btree"
    HASH = "hash"
    GIN = "gin"  # Generalized Inverted Index
    GIST = "gist"  # Generalized Search Tree
    HNSW = "hnsw"  # Hierarchical Navigable Small World (vector)
    IVFFLAT = "ivfflat"  # Inverted File Flat (vector)


@dataclass
class IndexCandidate:
    """Represents a potential index"""
    table_name: str
    columns: Tuple[str, ...]  # Can be composite
    index_type: IndexType
    estimated_size_mb: float
    estimated_benefit: float  # Query speedup factor
    creation_cost: float  # Time to build index
    maintenance_cost: float  # Cost per insert/update
    
    def benefit_cost_ratio(self) -> float:
        """Calculate benefit/cost ratio for prioritization"""
        if self.creation_cost + self.maintenance_cost == 0:
            return float('inf')
        return self.estimated_benefit / (self.creation_cost + self.maintenance_cost)


@dataclass
class WorkloadQuery:
    """Represents a query in the workload"""
    query_id: str
    frequency: float  # Executions per second
    table_scans: Dict[str, int]  # table -> row count scanned
    filter_columns: Set[Tuple[str, str]]  # (table, column) pairs
    join_columns: Set[Tuple[str, str]]  # (table, column) pairs
    vector_search: Optional[Tuple[str, str]] = None  # (table, vector_column)
    order_by_columns: Set[Tuple[str, str]] = None
    current_cost: float = 0.0


class IndexAdvisor:
    """
    Recommends indexes based on query workload analysis
    Considers both traditional SQL indexes and vector indexes
    """
    
    def __init__(
        self,
        max_indexes: int = 10,
        max_total_size_mb: float = 10000.0,
        random_page_cost: float = 1.5,
        seq_page_cost: float = 1.0
    ):
        self.max_indexes = max_indexes
        self.max_total_size_mb = max_total_size_mb
        self.random_page_cost = random_page_cost
        self.seq_page_cost = seq_page_cost
        
    def recommend_indexes(
        self,
        workload: List[WorkloadQuery],
        table_stats: Dict[str, Dict]
    ) -> List[IndexCandidate]:
        """
        Analyze workload and recommend optimal index configuration
        
        Args:
            workload: List of representative queries with frequencies
            table_stats: Statistics for each table (row_count, avg_row_size, etc.)
            
        Returns:
            Prioritized list of index recommendations
        """
        candidates = []
        
        # 1. Generate index candidates from workload patterns
        candidates.extend(self._generate_filter_indexes(workload, table_stats))
        candidates.extend(self._generate_join_indexes(workload, table_stats))
        candidates.extend(self._generate_vector_indexes(workload, table_stats))
        candidates.extend(self._generate_composite_indexes(workload, table_stats))
        
        # 2. Estimate benefits for each candidate
        for candidate in candidates:
            candidate.estimated_benefit = self._estimate_benefit(
                candidate, workload, table_stats
            )
        
        # 3. Select best indexes under constraints
        selected = self._select_index_configuration(candidates)
        
        return selected
    
    def _generate_filter_indexes(
        self,
        workload: List[WorkloadQuery],
        table_stats: Dict[str, Dict]
    ) -> List[IndexCandidate]:
        """Generate B-tree indexes for filter predicates"""
        candidates = []
        column_usage = {}  # (table, column) -> frequency
        
        # Count how often each column appears in filters
        for query in workload:
            for table, column in query.filter_columns:
                key = (table, column)
                column_usage[key] = column_usage.get(key, 0) + query.frequency
        
        # Create index candidates for frequently filtered columns
        for (table, column), freq in column_usage.items():
            if freq < 0.1:  # Skip rarely used columns
                continue
            
            stats = table_stats.get(table, {})
            row_count = stats.get('row_count', 100000)
            avg_row_size = stats.get('avg_row_size', 200)
            
            # Estimate index size (B-tree: ~70% of column data)
            index_size_mb = (row_count * 20) / (1024 * 1024) * 0.7
            
            # Creation cost proportional to table size
            creation_cost = row_count * 0.0001  # Simplified
            
            # Maintenance cost for writes
            maintenance_cost = freq * 0.01
            
            candidates.append(IndexCandidate(
                table_name=table,
                columns=(column,),
                index_type=IndexType.BTREE,
                estimated_size_mb=index_size_mb,
                estimated_benefit=0.0,  # Computed later
                creation_cost=creation_cost,
                maintenance_cost=maintenance_cost
            ))
        
        return candidates
    
    def _generate_join_indexes(
        self,
        workload: List[WorkloadQuery],
        table_stats: Dict[str, Dict]
    ) -> List[IndexCandidate]:
        """Generate indexes for join columns"""
        candidates = []
        join_usage = {}
        
        for query in workload:
            for table, column in query.join_columns:
                key = (table, column)
                join_usage[key] = join_usage.get(key, 0) + query.frequency
        
        for (table, column), freq in join_usage.items():
            if freq < 0.1:
                continue
            
            stats = table_stats.get(table, {})
            row_count = stats.get('row_count', 100000)
            
            index_size_mb = (row_count * 20) / (1024 * 1024) * 0.7
            creation_cost = row_count * 0.0001
            maintenance_cost = freq * 0.01
            
            # Joins benefit greatly from indexes (hash join candidate)
            candidates.append(IndexCandidate(
                table_name=table,
                columns=(column,),
                index_type=IndexType.BTREE,
                estimated_size_mb=index_size_mb,
                estimated_benefit=0.0,
                creation_cost=creation_cost,
                maintenance_cost=maintenance_cost
            ))
        
        return candidates
    
    def _generate_vector_indexes(
        self,
        workload: List[WorkloadQuery],
        table_stats: Dict[str, Dict]
    ) -> List[IndexCandidate]:
        """Generate HNSW and IVFFlat indexes for vector searches"""
        candidates = []
        vector_usage = {}
        
        for query in workload:
            if query.vector_search:
                table, column = query.vector_search
                key = (table, column)
                vector_usage[key] = vector_usage.get(key, 0) + query.frequency
        
        for (table, column), freq in vector_usage.items():
            stats = table_stats.get(table, {})
            row_count = stats.get('row_count', 100000)
            vector_dim = stats.get('vector_dimensions', 384)
            
            # HNSW index
            hnsw_size_mb = self._estimate_hnsw_size(row_count, vector_dim)
            hnsw_build_cost = row_count * 0.001  # More expensive to build
            
            candidates.append(IndexCandidate(
                table_name=table,
                columns=(column,),
                index_type=IndexType.HNSW,
                estimated_size_mb=hnsw_size_mb,
                estimated_benefit=0.0,
                creation_cost=hnsw_build_cost,
                maintenance_cost=freq * 0.05
            ))
            
            # IVFFlat index (alternative, cheaper to build)
            ivfflat_size_mb = self._estimate_ivfflat_size(row_count, vector_dim)
            ivfflat_build_cost = row_count * 0.0005
            
            candidates.append(IndexCandidate(
                table_name=table,
                columns=(column,),
                index_type=IndexType.IVFFLAT,
                estimated_size_mb=ivfflat_size_mb,
                estimated_benefit=0.0,
                creation_cost=ivfflat_build_cost,
                maintenance_cost=freq * 0.03
            ))
        
        return candidates
    
    def _generate_composite_indexes(
        self,
        workload: List[WorkloadQuery],
        table_stats: Dict[str, Dict]
    ) -> List[IndexCandidate]:
        """Generate composite indexes for common column combinations"""
        candidates = []
        
        # Find columns that appear together in filters
        column_pairs = {}  # (table, (col1, col2)) -> frequency
        
        for query in workload:
            # Group filter columns by table
            by_table = {}
            for table, column in query.filter_columns:
                if table not in by_table:
                    by_table[table] = []
                by_table[table].append(column)
            
            # Create pairs for each table
            for table, columns in by_table.items():
                if len(columns) >= 2:
                    # Sort to get canonical order
                    for i, col1 in enumerate(columns):
                        for col2 in columns[i+1:]:
                            pair = tuple(sorted([col1, col2]))
                            key = (table, pair)
                            column_pairs[key] = column_pairs.get(key, 0) + query.frequency
        
        # Create composite index candidates for frequent pairs
        for (table, columns), freq in column_pairs.items():
            if freq < 0.5:  # Higher threshold for composite
                continue
            
            stats = table_stats.get(table, {})
            row_count = stats.get('row_count', 100000)
            
            # Composite index is larger
            index_size_mb = (row_count * 40) / (1024 * 1024) * 0.7
            creation_cost = row_count * 0.0002
            maintenance_cost = freq * 0.02
            
            candidates.append(IndexCandidate(
                table_name=table,
                columns=columns,
                index_type=IndexType.BTREE,
                estimated_size_mb=index_size_mb,
                estimated_benefit=0.0,
                creation_cost=creation_cost,
                maintenance_cost=maintenance_cost
            ))
        
        return candidates
    
    def _estimate_benefit(
        self,
        candidate: IndexCandidate,
        workload: List[WorkloadQuery],
        table_stats: Dict[str, Dict]
    ) -> float:
        """
        Estimate query speedup from adding this index
        
        Returns:
            Speedup factor (e.g., 5.0 means 5x faster)
        """
        total_benefit = 0.0
        
        for query in workload:
            benefit = 0.0
            
            # Check if this index helps this query
            if candidate.index_type == IndexType.HNSW or candidate.index_type == IndexType.IVFFLAT:
                # Vector index
                if query.vector_search:
                    table, column = query.vector_search
                    if table == candidate.table_name and column in candidate.columns:
                        # Vector index provides huge benefit
                        stats = table_stats.get(table, {})
                        row_count = stats.get('row_count', 100000)
                        
                        # Sequential scan cost
                        seq_cost = row_count * self.seq_page_cost * 0.01
                        
                        # HNSW/IVFFlat cost
                        if candidate.index_type == IndexType.HNSW:
                            index_cost = np.log2(max(row_count, 1)) * 10.0
                        else:  # IVFFlat
                            index_cost = np.sqrt(row_count) * 5.0
                        
                        benefit = max(0, seq_cost - index_cost)
            else:
                # SQL index (B-tree, hash, etc.)
                for table, column in query.filter_columns:
                    if table == candidate.table_name and column in candidate.columns:
                        # Index scan instead of sequential scan
                        stats = table_stats.get(table, {})
                        row_count = stats.get('row_count', 100000)
                        selectivity = stats.get('selectivity', 0.1)
                        
                        # Sequential scan cost
                        seq_cost = row_count * self.seq_page_cost * 0.01
                        
                        # Index scan cost
                        output_rows = row_count * selectivity
                        index_cost = (np.log2(max(row_count, 1)) * 2.0 + 
                                    output_rows * self.random_page_cost * 0.01)
                        
                        benefit += max(0, seq_cost - index_cost)
                
                # Join benefit
                for table, column in query.join_columns:
                    if table == candidate.table_name and column in candidate.columns:
                        # Enables index nested loop instead of hash join
                        benefit += query.current_cost * 0.3  # 30% improvement estimate
            
            # Weight by query frequency
            total_benefit += benefit * query.frequency
        
        return total_benefit
    
    def _select_index_configuration(
        self,
        candidates: List[IndexCandidate]
    ) -> List[IndexCandidate]:
        """
        Select best subset of indexes under space constraint
        This is a knapsack problem - use greedy heuristic
        """
        # Sort by benefit/cost ratio
        candidates.sort(key=lambda x: x.benefit_cost_ratio(), reverse=True)
        
        selected = []
        total_size = 0.0
        
        for candidate in candidates:
            if len(selected) >= self.max_indexes:
                break
            
            if total_size + candidate.estimated_size_mb > self.max_total_size_mb:
                continue
            
            # Avoid duplicate indexes on same column
            is_duplicate = False
            for existing in selected:
                if (existing.table_name == candidate.table_name and
                    set(existing.columns) == set(candidate.columns)):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                selected.append(candidate)
                total_size += candidate.estimated_size_mb
        
        return selected
    
    def _estimate_hnsw_size(self, row_count: int, vector_dim: int) -> float:
        """Estimate HNSW index size in MB"""
        # HNSW stores: vectors + graph edges
        # Graph typically uses m=16, avg edges ~32 per node
        vector_size = row_count * vector_dim * 4  # float32
        graph_size = row_count * 32 * 8  # 32 edges * 8 bytes per edge
        total_bytes = vector_size + graph_size
        return total_bytes / (1024 * 1024)
    
    def _estimate_ivfflat_size(self, row_count: int, vector_dim: int) -> float:
        """Estimate IVFFlat index size in MB"""
        # IVFFlat stores: centroids + vectors + cluster assignments
        n_lists = int(np.sqrt(row_count))
        centroid_size = n_lists * vector_dim * 4
        vector_size = row_count * vector_dim * 4
        assignment_size = row_count * 4
        total_bytes = centroid_size + vector_size + assignment_size
        return total_bytes / (1024 * 1024)
    
    def generate_sql(self, index: IndexCandidate) -> str:
        """Generate SQL CREATE INDEX statement"""
        col_list = ", ".join(index.columns)
        index_name = f"idx_{index.table_name}_{'_'.join(index.columns)}"
        
        if index.index_type == IndexType.HNSW:
            return f"""
CREATE INDEX {index_name}
    ON {index.table_name} USING hnsw ({index.columns[0]} vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
""".strip()
        
        elif index.index_type == IndexType.IVFFLAT:
            return f"""
CREATE INDEX {index_name}
    ON {index.table_name} USING ivfflat ({index.columns[0]} vector_cosine_ops)
    WITH (lists = 100);
""".strip()
        
        else:
            return f"""
CREATE INDEX {index_name} ON {index.table_name} ({col_list});
""".strip()
    
    def generate_report(
        self,
        recommendations: List[IndexCandidate]
    ) -> str:
        """Generate human-readable report of recommendations"""
        report = ["=" * 80]
        report.append("INDEX RECOMMENDATIONS")
        report.append("=" * 80)
        report.append("")
        
        total_size = sum(idx.estimated_size_mb for idx in recommendations)
        total_benefit = sum(idx.estimated_benefit for idx in recommendations)
        
        report.append(f"Total indexes recommended: {len(recommendations)}")
        report.append(f"Total estimated size: {total_size:.2f} MB")
        report.append(f"Total estimated benefit: {total_benefit:.2f}")
        report.append("")
        
        for i, idx in enumerate(recommendations, 1):
            report.append(f"{i}. {idx.table_name}.{','.join(idx.columns)}")
            report.append(f"   Type: {idx.index_type.value}")
            report.append(f"   Size: {idx.estimated_size_mb:.2f} MB")
            report.append(f"   Benefit: {idx.estimated_benefit:.2f}")
            report.append(f"   Benefit/Cost Ratio: {idx.benefit_cost_ratio():.2f}")
            report.append("")
            report.append(f"   SQL:")
            for line in self.generate_sql(idx).split('\n'):
                report.append(f"   {line}")
            report.append("")
        
        return "\n".join(report)
