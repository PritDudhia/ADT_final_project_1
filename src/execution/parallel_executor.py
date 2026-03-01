"""
Parallel Query Executor - Execution Engine Component
Executes query plans with parallel processing and pipelining
"""

import threading
import queue
import time
import psycopg2
from typing import List, Dict, Iterator, Optional, Callable, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np


@dataclass
class ExecutionOperator:
    """Represents an operator in the execution plan"""
    operator_type: str  # 'scan', 'filter', 'join', 'vector_search', 'project'
    params: Dict[str, Any]
    input_operators: List['ExecutionOperator'] = None
    
    def __post_init__(self):
        if self.input_operators is None:
            self.input_operators = []


class TupleBuffer:
    """Thread-safe buffer for passing tuples between operators"""
    
    def __init__(self, max_size: int = 1000):
        self.queue = queue.Queue(maxsize=max_size)
        self.finished = False
        self.lock = threading.Lock()
    
    def put(self, tuple_data: Dict):
        """Add a tuple to the buffer"""
        self.queue.put(tuple_data)
    
    def get(self, timeout: float = 1.0) -> Optional[Dict]:
        """Get a tuple from the buffer"""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def mark_finished(self):
        """Signal that no more tuples will be added"""
        with self.lock:
            self.finished = True
    
    def is_finished(self) -> bool:
        """Check if buffer is finished and empty"""
        with self.lock:
            return self.finished and self.queue.empty()
    
    def size(self) -> int:
        """Current buffer size"""
        return self.queue.qsize()


class ParallelExecutor:
    """
    Executes query plans with parallelism and pipelining
    Implements volcano-style iterator model with parallel operators
    """
    
    def __init__(
        self,
        db_connection_params: Dict[str, str],
        num_workers: int = 4,
        buffer_size: int = 1000
    ):
        self.db_params = db_connection_params
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.active_futures: List[Future] = []
    
    def execute_plan(
        self,
        root_operator: ExecutionOperator,
        limit: Optional[int] = None
    ) -> Iterator[Dict]:
        """
        Execute a query plan and stream results
        
        Args:
            root_operator: Root of the execution plan tree
            limit: Maximum number of results to return
            
        Yields:
            Result tuples as dictionaries
        """
        # Create output buffer for root
        output_buffer = TupleBuffer(self.buffer_size)
        
        # Start execution in background
        future = self.executor.submit(
            self._execute_operator,
            root_operator,
            output_buffer
        )
        self.active_futures.append(future)
        
        # Stream results
        count = 0
        while not output_buffer.is_finished():
            tuple_data = output_buffer.get()
            if tuple_data is not None:
                yield tuple_data
                count += 1
                if limit and count >= limit:
                    break
        
        # Wait for execution to complete
        future.result()
    
    def _execute_operator(
        self,
        operator: ExecutionOperator,
        output_buffer: TupleBuffer
    ):
        """Execute a single operator"""
        try:
            if operator.operator_type == 'scan':
                self._execute_scan(operator, output_buffer)
            elif operator.operator_type == 'index_scan':
                self._execute_index_scan(operator, output_buffer)
            elif operator.operator_type == 'vector_search':
                self._execute_vector_search(operator, output_buffer)
            elif operator.operator_type == 'filter':
                self._execute_filter(operator, output_buffer)
            elif operator.operator_type == 'hash_join':
                self._execute_hash_join(operator, output_buffer)
            elif operator.operator_type == 'nested_loop_join':
                self._execute_nested_loop_join(operator, output_buffer)
            elif operator.operator_type == 'project':
                self._execute_project(operator, output_buffer)
            elif operator.operator_type == 'sort':
                self._execute_sort(operator, output_buffer)
            elif operator.operator_type == 'aggregate':
                self._execute_aggregate(operator, output_buffer)
            else:
                raise ValueError(f"Unknown operator: {operator.operator_type}")
        finally:
            output_buffer.mark_finished()
    
    def _execute_scan(
        self,
        operator: ExecutionOperator,
        output_buffer: TupleBuffer
    ):
        """Sequential table scan"""
        table_name = operator.params['table']
        columns = operator.params.get('columns', ['*'])
        
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()
        
        try:
            col_list = ', '.join(columns)
            query = f"SELECT {col_list} FROM {table_name}"
            
            cursor.execute(query)
            
            # Fetch in batches
            while True:
                rows = cursor.fetchmany(100)
                if not rows:
                    break
                
                for row in rows:
                    tuple_data = dict(zip(columns, row))
                    output_buffer.put(tuple_data)
        
        finally:
            cursor.close()
            conn.close()
    
    def _execute_index_scan(
        self,
        operator: ExecutionOperator,
        output_buffer: TupleBuffer
    ):
        """Index scan with predicate"""
        table_name = operator.params['table']
        columns = operator.params.get('columns', ['*'])
        predicate = operator.params['predicate']  # e.g., "price < 100"
        
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()
        
        try:
            col_list = ', '.join(columns)
            query = f"SELECT {col_list} FROM {table_name} WHERE {predicate}"
            
            cursor.execute(query)
            
            while True:
                rows = cursor.fetchmany(100)
                if not rows:
                    break
                
                for row in rows:
                    tuple_data = dict(zip(columns, row))
                    output_buffer.put(tuple_data)
        
        finally:
            cursor.close()
            conn.close()
    
    def _execute_vector_search(
        self,
        operator: ExecutionOperator,
        output_buffer: TupleBuffer
    ):
        """Vector similarity search"""
        table_name = operator.params['table']
        vector_column = operator.params['vector_column']
        query_vector = operator.params['query_vector']
        k = operator.params.get('k', 10)
        columns = operator.params.get('columns', ['*'])
        
        conn = psycopg2.connect(**self.db_params)
        cursor = conn.cursor()
        
        try:
            # Convert query vector to pgvector format
            vector_str = '[' + ','.join(map(str, query_vector)) + ']'
            
            col_list = ', '.join(columns)
            query = f"""
                SELECT {col_list},
                       {vector_column} <=> %s::vector AS distance
                FROM {table_name}
                ORDER BY {vector_column} <=> %s::vector
                LIMIT %s
            """
            
            cursor.execute(query, (vector_str, vector_str, k))
            
            rows = cursor.fetchall()
            for row in rows:
                tuple_data = dict(zip(columns + ['distance'], row))
                output_buffer.put(tuple_data)
        
        finally:
            cursor.close()
            conn.close()
    
    def _execute_filter(
        self,
        operator: ExecutionOperator,
        output_buffer: TupleBuffer
    ):
        """Apply filter predicate to input stream"""
        input_op = operator.input_operators[0]
        predicate_fn = operator.params['predicate_fn']
        
        # Create buffer for input
        input_buffer = TupleBuffer(self.buffer_size)
        
        # Execute input operator in parallel
        input_future = self.executor.submit(
            self._execute_operator,
            input_op,
            input_buffer
        )
        
        # Process input stream
        while not input_buffer.is_finished():
            tuple_data = input_buffer.get()
            if tuple_data is not None:
                if predicate_fn(tuple_data):
                    output_buffer.put(tuple_data)
        
        input_future.result()
    
    def _execute_hash_join(
        self,
        operator: ExecutionOperator,
        output_buffer: TupleBuffer
    ):
        """Hash join between two inputs"""
        left_op = operator.input_operators[0]
        right_op = operator.input_operators[1]
        left_key = operator.params['left_key']
        right_key = operator.params['right_key']
        
        # Build phase: hash the smaller input (assumed to be left)
        left_buffer = TupleBuffer(self.buffer_size)
        left_future = self.executor.submit(
            self._execute_operator,
            left_op,
            left_buffer
        )
        
        # Build hash table
        hash_table = {}
        while not left_buffer.is_finished():
            tuple_data = left_buffer.get()
            if tuple_data is not None:
                key = tuple_data[left_key]
                if key not in hash_table:
                    hash_table[key] = []
                hash_table[key].append(tuple_data)
        
        left_future.result()
        
        # Probe phase: stream right input
        right_buffer = TupleBuffer(self.buffer_size)
        right_future = self.executor.submit(
            self._execute_operator,
            right_op,
            right_buffer
        )
        
        while not right_buffer.is_finished():
            right_tuple = right_buffer.get()
            if right_tuple is not None:
                key = right_tuple[right_key]
                if key in hash_table:
                    for left_tuple in hash_table[key]:
                        # Merge tuples
                        joined = {**left_tuple, **right_tuple}
                        output_buffer.put(joined)
        
        right_future.result()
    
    def _execute_nested_loop_join(
        self,
        operator: ExecutionOperator,
        output_buffer: TupleBuffer
    ):
        """Nested loop join (for small inputs or when no join key)"""
        outer_op = operator.input_operators[0]
        inner_op = operator.input_operators[1]
        join_predicate = operator.params.get('predicate_fn', lambda l, r: True)
        
        # Materialize outer
        outer_buffer = TupleBuffer(self.buffer_size)
        outer_future = self.executor.submit(
            self._execute_operator,
            outer_op,
            outer_buffer
        )
        
        outer_tuples = []
        while not outer_buffer.is_finished():
            tuple_data = outer_buffer.get()
            if tuple_data is not None:
                outer_tuples.append(tuple_data)
        
        outer_future.result()
        
        # For each outer tuple, scan inner
        for outer_tuple in outer_tuples:
            inner_buffer = TupleBuffer(self.buffer_size)
            inner_future = self.executor.submit(
                self._execute_operator,
                inner_op,
                inner_buffer
            )
            
            while not inner_buffer.is_finished():
                inner_tuple = inner_buffer.get()
                if inner_tuple is not None:
                    if join_predicate(outer_tuple, inner_tuple):
                        joined = {**outer_tuple, **inner_tuple}
                        output_buffer.put(joined)
            
            inner_future.result()
    
    def _execute_project(
        self,
        operator: ExecutionOperator,
        output_buffer: TupleBuffer
    ):
        """Project (select columns)"""
        input_op = operator.input_operators[0]
        columns = operator.params['columns']
        
        input_buffer = TupleBuffer(self.buffer_size)
        input_future = self.executor.submit(
            self._execute_operator,
            input_op,
            input_buffer
        )
        
        while not input_buffer.is_finished():
            tuple_data = input_buffer.get()
            if tuple_data is not None:
                projected = {col: tuple_data[col] for col in columns if col in tuple_data}
                output_buffer.put(projected)
        
        input_future.result()
    
    def _execute_sort(
        self,
        operator: ExecutionOperator,
        output_buffer: TupleBuffer
    ):
        """Sort input"""
        input_op = operator.input_operators[0]
        sort_keys = operator.params['sort_keys']
        
        # Materialize input
        input_buffer = TupleBuffer(self.buffer_size)
        input_future = self.executor.submit(
            self._execute_operator,
            input_op,
            input_buffer
        )
        
        tuples = []
        while not input_buffer.is_finished():
            tuple_data = input_buffer.get()
            if tuple_data is not None:
                tuples.append(tuple_data)
        
        input_future.result()
        
        # Sort
        def sort_key(t):
            return tuple(t.get(k, 0) for k in sort_keys)
        
        tuples.sort(key=sort_key)
        
        # Output
        for t in tuples:
            output_buffer.put(t)
    
    def _execute_aggregate(
        self,
        operator: ExecutionOperator,
        output_buffer: TupleBuffer
    ):
        """Aggregate (group by + aggregations)"""
        input_op = operator.input_operators[0]
        group_by = operator.params.get('group_by', [])
        aggregates = operator.params.get('aggregates', {})  # {output_col: ('func', input_col)}
        
        # Materialize input
        input_buffer = TupleBuffer(self.buffer_size)
        input_future = self.executor.submit(
            self._execute_operator,
            input_op,
            input_buffer
        )
        
        groups = {}
        while not input_buffer.is_finished():
            tuple_data = input_buffer.get()
            if tuple_data is not None:
                # Extract group key
                group_key = tuple(tuple_data.get(k, None) for k in group_by) if group_by else ()
                
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(tuple_data)
        
        input_future.result()
        
        # Compute aggregates
        for group_key, group_tuples in groups.items():
            result = {}
            
            # Add group by columns
            for i, col in enumerate(group_by):
                result[col] = group_key[i]
            
            # Compute aggregates
            for output_col, (func, input_col) in aggregates.items():
                values = [t.get(input_col, 0) for t in group_tuples]
                
                if func == 'count':
                    result[output_col] = len(values)
                elif func == 'sum':
                    result[output_col] = sum(values)
                elif func == 'avg':
                    result[output_col] = np.mean(values)
                elif func == 'min':
                    result[output_col] = min(values) if values else None
                elif func == 'max':
                    result[output_col] = max(values) if values else None
            
            output_buffer.put(result)
    
    def shutdown(self):
        """Shutdown executor"""
        self.executor.shutdown(wait=True)
