"""
Distributed Query Coordinator - Distributed Systems Component
Coordinates query execution across multiple database nodes
"""

import threading
import socket
import pickle
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib


@dataclass
class DatabaseNode:
    """Represents a database node in the cluster"""
    node_id: str
    host: str
    port: int
    db_name: str
    user: str
    password: str
    is_active: bool = True
    load_factor: float = 0.0  # Current load (0.0 to 1.0)
    
    # Partition assignment
    partitions: List[str] = None  # List of partition IDs this node owns
    
    def __post_init__(self):
        if self.partitions is None:
            self.partitions = []
    
    def get_connection_params(self) -> Dict[str, str]:
        """Get psycopg2 connection parameters"""
        return {
            'host': self.host,
            'port': self.port,
            'dbname': self.db_name,
            'user': self.user,
            'password': self.password
        }


@dataclass
class PartitionInfo:
    """Information about a data partition"""
    partition_id: str
    table_name: str
    partition_key: str
    key_range: Tuple[any, any]  # (min, max) for range partitioning
    row_count: int
    size_mb: float
    node_id: str  # Which node owns this partition


class PartitionStrategy:
    """
    Partitioning strategy for distributed tables
    Supports hash, range, and list partitioning
    """
    
    def __init__(self, strategy_type: str = 'hash', num_partitions: int = 4):
        """
        Args:
            strategy_type: 'hash', 'range', or 'list'
            num_partitions: Number of partitions
        """
        self.strategy_type = strategy_type
        self.num_partitions = num_partitions
    
    def get_partition_id(self, partition_key: any) -> int:
        """
        Determine which partition a key belongs to
        
        Returns:
            Partition ID (0 to num_partitions-1)
        """
        if self.strategy_type == 'hash':
            # Hash partitioning
            key_str = str(partition_key)
            hash_val = int(hashlib.md5(key_str.encode()).hexdigest(), 16)
            return hash_val % self.num_partitions
        
        elif self.strategy_type == 'range':
            # Range partitioning (requires metadata about ranges)
            # Simplified: assume partition_key is already a partition ID
            return int(partition_key) % self.num_partitions
        
        else:
            # List partitioning
            return 0  # Simplified
    
    def get_partitions_for_predicate(
        self,
        predicate: Dict[str, any]
    ) -> List[int]:
        """
        Determine which partitions need to be scanned for a predicate
        
        Args:
            predicate: e.g., {'partition_key': {'>=': 100, '<': 200}}
            
        Returns:
            List of partition IDs to scan
        """
        if not predicate or 'partition_key' not in predicate:
            # No partition filter - scan all partitions
            return list(range(self.num_partitions))
        
        if self.strategy_type == 'hash':
            # Hash partitioning: can only prune if exact equality
            if '=' in predicate['partition_key']:
                key = predicate['partition_key']['=']
                return [self.get_partition_id(key)]
            else:
                # Range or inequality - must scan all
                return list(range(self.num_partitions))
        
        elif self.strategy_type == 'range':
            # Range partitioning: can prune based on ranges
            # Simplified implementation
            return list(range(self.num_partitions))
        
        return list(range(self.num_partitions))


class DistributedQueryCoordinator:
    """
    Coordinates distributed query execution across multiple nodes
    Handles:
    - Query decomposition into sub-queries
    - Partition pruning
    - Parallel execution on multiple nodes
    - Result aggregation
    """
    
    def __init__(self, nodes: List[DatabaseNode]):
        self.nodes = {node.node_id: node for node in nodes}
        self.executor = ThreadPoolExecutor(max_workers=len(nodes) * 2)
        self.partition_map: Dict[str, PartitionInfo] = {}
        self.partition_strategy = PartitionStrategy(strategy_type='hash', num_partitions=len(nodes))
    
    def register_partition(self, partition: PartitionInfo):
        """Register a partition with the coordinator"""
        self.partition_map[partition.partition_id] = partition
    
    def execute_distributed_query(
        self,
        query: str,
        table_name: str,
        predicate: Optional[Dict[str, any]] = None
    ) -> List[Dict]:
        """
        Execute a query across distributed partitions
        
        Args:
            query: SQL query to execute
            table_name: Which table to query
            predicate: Predicate for partition pruning
            
        Returns:
            Combined results from all nodes
        """
        # 1. Determine which partitions to query (partition pruning)
        target_partitions = self._get_target_partitions(table_name, predicate)
        
        # 2. Group partitions by node
        partitions_by_node = self._group_partitions_by_node(target_partitions)
        
        # 3. Execute sub-queries in parallel on each node
        futures = []
        for node_id, partition_ids in partitions_by_node.items():
            node = self.nodes[node_id]
            
            future = self.executor.submit(
                self._execute_on_node,
                node,
                query,
                partition_ids
            )
            futures.append((node_id, future))
        
        # 4. Collect and combine results
        all_results = []
        for node_id, future in futures:
            try:
                node_results = future.result(timeout=60)
                all_results.extend(node_results)
            except Exception as e:
                print(f"Error executing on node {node_id}: {e}")
        
        return all_results
    
    def execute_distributed_join(
        self,
        left_table: str,
        right_table: str,
        join_key: str,
        join_type: str = 'inner'
    ) -> List[Dict]:
        """
        Execute a distributed join
        
        Strategies:
        1. Broadcast join: if one table is small, broadcast to all nodes
        2. Shuffle join: redistribute data on join key, then local join
        3. Co-located join: if both tables partitioned on join key
        """
        # Check if tables are co-located (same partitioning on join key)
        if self._are_colocated(left_table, right_table, join_key):
            return self._execute_colocated_join(left_table, right_table, join_key)
        
        # Check table sizes
        left_size = self._estimate_table_size(left_table)
        right_size = self._estimate_table_size(right_table)
        
        if left_size < 1000 or right_size < 1000:
            # Use broadcast join
            smaller_table = left_table if left_size < right_size else right_table
            larger_table = right_table if left_size < right_size else left_table
            return self._execute_broadcast_join(smaller_table, larger_table, join_key)
        else:
            # Use shuffle join
            return self._execute_shuffle_join(left_table, right_table, join_key)
    
    def _get_target_partitions(
        self,
        table_name: str,
        predicate: Optional[Dict[str, any]]
    ) -> List[str]:
        """Determine which partitions to query (partition pruning)"""
        # Get all partitions for this table
        table_partitions = [
            p for p in self.partition_map.values()
            if p.table_name == table_name
        ]
        
        if not predicate:
            return [p.partition_id for p in table_partitions]
        
        # Use partition strategy to prune
        partition_indices = self.partition_strategy.get_partitions_for_predicate(predicate)
        
        # Map indices to actual partition IDs
        target_ids = []
        for idx in partition_indices:
            for p in table_partitions:
                if p.partition_id.endswith(f"_{idx}"):
                    target_ids.append(p.partition_id)
        
        return target_ids if target_ids else [p.partition_id for p in table_partitions]
    
    def _group_partitions_by_node(
        self,
        partition_ids: List[str]
    ) -> Dict[str, List[str]]:
        """Group partition IDs by the node that owns them"""
        result = {}
        for partition_id in partition_ids:
            partition = self.partition_map.get(partition_id)
            if partition:
                node_id = partition.node_id
                if node_id not in result:
                    result[node_id] = []
                result[node_id].append(partition_id)
        return result
    
    def _execute_on_node(
        self,
        node: DatabaseNode,
        query: str,
        partition_ids: List[str]
    ) -> List[Dict]:
        """Execute query on a specific node"""
        import psycopg2
        import psycopg2.extras
        
        if not node.is_active:
            return []
        
        conn = None
        try:
            conn = psycopg2.connect(**node.get_connection_params())
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Modify query to target specific partitions
            # This is simplified - in practice would use UNION ALL or partition-specific tables
            cursor.execute(query)
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
        
        except Exception as e:
            print(f"Error on node {node.node_id}: {e}")
            return []
        
        finally:
            if conn:
                conn.close()
    
    def _are_colocated(
        self,
        left_table: str,
        right_table: str,
        join_key: str
    ) -> bool:
        """Check if two tables are co-located (same partitioning)"""
        # Simplified: check if both tables have same partition strategy
        left_partitions = [p for p in self.partition_map.values() if p.table_name == left_table]
        right_partitions = [p for p in self.partition_map.values() if p.table_name == right_table]
        
        # Check if partition keys match
        if left_partitions and right_partitions:
            left_key = left_partitions[0].partition_key
            right_key = right_partitions[0].partition_key
            return left_key == right_key == join_key
        
        return False
    
    def _execute_colocated_join(
        self,
        left_table: str,
        right_table: str,
        join_key: str
    ) -> List[Dict]:
        """Execute join when tables are co-located"""
        # Join can be done locally on each node
        query = f"""
            SELECT * FROM {left_table} l
            INNER JOIN {right_table} r ON l.{join_key} = r.{join_key}
        """
        
        # Execute on all nodes in parallel
        futures = []
        for node in self.nodes.values():
            future = self.executor.submit(
                self._execute_on_node,
                node,
                query,
                []
            )
            futures.append(future)
        
        # Combine results
        results = []
        for future in as_completed(futures):
            results.extend(future.result())
        
        return results
    
    def _execute_broadcast_join(
        self,
        small_table: str,
        large_table: str,
        join_key: str
    ) -> List[Dict]:
        """Broadcast smaller table to all nodes, join locally"""
        # 1. Collect small table from all nodes
        small_data = self.execute_distributed_query(
            f"SELECT * FROM {small_table}",
            small_table
        )
        
        # 2. For each node, send small_data and execute join
        # Simplified: assume we can create temp table
        query = f"""
            SELECT * FROM {large_table} l
            INNER JOIN temp_small_table s ON l.{join_key} = s.{join_key}
        """
        
        # In practice, would serialize small_data and send to nodes
        # For now, execute standard distributed query
        return self.execute_distributed_query(query, large_table)
    
    def _execute_shuffle_join(
        self,
        left_table: str,
        right_table: str,
        join_key: str
    ) -> List[Dict]:
        """Shuffle both tables on join key, then join"""
        # This is complex - requires data redistribution
        # Simplified implementation
        
        # 1. Redistribute left table by hash(join_key)
        # 2. Redistribute right table by hash(join_key)
        # 3. Perform local joins on each node
        # 4. Collect results
        
        # For now, fall back to collecting all data and joining locally
        left_data = self.execute_distributed_query(f"SELECT * FROM {left_table}", left_table)
        right_data = self.execute_distributed_query(f"SELECT * FROM {right_table}", right_table)
        
        # Local hash join
        hash_table = {}
        for row in left_data:
            key = row[join_key]
            if key not in hash_table:
                hash_table[key] = []
            hash_table[key].append(row)
        
        results = []
        for row in right_data:
            key = row[join_key]
            if key in hash_table:
                for left_row in hash_table[key]:
                    joined = {**left_row, **row}
                    results.append(joined)
        
        return results
    
    def _estimate_table_size(self, table_name: str) -> int:
        """Estimate number of rows in table"""
        partitions = [p for p in self.partition_map.values() if p.table_name == table_name]
        return sum(p.row_count for p in partitions)
    
    def get_cluster_status(self) -> Dict:
        """Get status of all nodes in the cluster"""
        status = {
            'total_nodes': len(self.nodes),
            'active_nodes': sum(1 for n in self.nodes.values() if n.is_active),
            'total_partitions': len(self.partition_map),
            'nodes': []
        }
        
        for node in self.nodes.values():
            node_partitions = [
                p for p in self.partition_map.values()
                if p.node_id == node.node_id
            ]
            
            status['nodes'].append({
                'node_id': node.node_id,
                'host': node.host,
                'is_active': node.is_active,
                'load_factor': node.load_factor,
                'num_partitions': len(node_partitions),
                'total_rows': sum(p.row_count for p in node_partitions),
                'total_size_mb': sum(p.size_mb for p in node_partitions)
            })
        
        return status
    
    def shutdown(self):
        """Shutdown coordinator"""
        self.executor.shutdown(wait=True)
