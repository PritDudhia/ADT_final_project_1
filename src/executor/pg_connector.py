"""
PostgreSQL + pgvector Connector

Handles database connection and execution of hybrid queries.
Integrates with our cost-based optimizer.
"""

import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import yaml

logger = logging.getLogger(__name__)


class PostgreSQLConnector:
    """
    Connection manager for PostgreSQL + pgvector database.
    
    Provides methods for:
    - Executing SQL queries
    - Vector similarity search
    - Collecting table statistics
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize database connection.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.db_config = config['database']
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['name'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_config['name']}")
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Disconnected from database")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[tuple]:
        """
        Execute SQL query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters (for parameterized queries)
            
        Returns:
            List of result tuples
        """
        try:
            self.cursor.execute(query, params)
            
            # Check if query returns results
            if self.cursor.description:
                results = self.cursor.fetchall()
                return results
            else:
                self.conn.commit()
                return []
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            self.conn.rollback()
            raise
    
    def vector_search(
        self,
        table: str,
        embedding_column: str,
        query_vector: np.ndarray,
        k: int = 10,
        distance_metric: str = 'cosine',
        where_clause: Optional[str] = None,
        where_params: Optional[tuple] = None
    ) -> List[tuple]:
        """
        Perform vector similarity search using pgvector.
        
        Args:
            table: Table name
            embedding_column: Column containing embeddings
            query_vector: Query embedding vector
            k: Number of top results
            distance_metric: 'cosine', 'l2', or 'inner_product'
            where_clause: Optional SQL filter (e.g., "price < %s AND brand = %s")
            where_params: Parameters for where clause
            
        Returns:
            List of result tuples
        """
        # Convert distance metric to pgvector operator
        if distance_metric == 'cosine':
            operator = '<=>'  # Cosine distance
        elif distance_metric == 'l2':
            operator = '<->'  # L2 distance
        elif distance_metric == 'inner_product':
            operator = '<#>'  # Negative inner product
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        
        # Build query
        query = f"""
            SELECT *
            FROM {table}
        """
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        query += f"""
            ORDER BY {embedding_column} {operator} %s
            LIMIT %s
        """
        
        # Prepare parameters
        vector_param = query_vector.tolist()
        if where_params:
            params = where_params + (vector_param, k)
        else:
            params = (vector_param, k)
        
        return self.execute_query(query, params)
    
    def get_table_statistics(self, table: str) -> Dict[str, Any]:
        """
        Collect table statistics for cost estimation.
        
        Args:
            table: Table name
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        # Get row count
        query = f"SELECT COUNT(*) FROM {table}"
        result = self.execute_query(query)
        stats['n_tuples'] = result[0][0]
        
        # Get table size
        query = f"SELECT pg_total_relation_size('{table}')"
        result = self.execute_query(query)
        stats['table_size_bytes'] = result[0][0]
        
        # Estimate average tuple size
        if stats['n_tuples'] > 0:
            stats['avg_tuple_size'] = stats['table_size_bytes'] // stats['n_tuples']
        else:
            stats['avg_tuple_size'] = 200  # Default
        
        # Get column statistics
        query = f"""
            SELECT 
                attname,
                n_distinct,
                null_frac
            FROM pg_stats
            WHERE tablename = %s
        """
        results = self.execute_query(query, (table,))
        
        for col_name, n_distinct, null_frac in results:
            if n_distinct > 0:
                stats[f'{col_name}_distinct'] = int(n_distinct)
            stats[f'{col_name}_null_frac'] = null_frac
        
        # Get indexes
        query = f"""
            SELECT 
                indexname,
                indexdef
            FROM pg_indexes
            WHERE tablename = %s
        """
        results = self.execute_query(query, (table,))
        
        indexed_columns = set()
        for idx_name, idx_def in results:
            # Parse indexed columns from index definition
            # Simple parsing - could be improved
            if 'btree' in idx_def.lower():
                # Extract column name between parentheses
                start = idx_def.find('(')
                end = idx_def.find(')')
                if start > 0 and end > 0:
                    col = idx_def[start+1:end].strip()
                    indexed_columns.add(col)
        
        stats['indexed_columns'] = indexed_columns
        
        logger.info(f"Collected statistics for {table}: {stats['n_tuples']:,} rows")
        
        return stats
    
    def explain_analyze(self, query: str, params: Optional[tuple] = None) -> str:
        """
        Get PostgreSQL EXPLAIN ANALYZE output for query.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            EXPLAIN output as string
        """
        explain_query = f"EXPLAIN ANALYZE {query}"
        results = self.execute_query(explain_query, params)
        
        # Format results
        output_lines = [row[0] for row in results]
        return '\n'.join(output_lines)
    
    def insert_vectors(
        self,
        table: str,
        data: List[Dict[str, Any]],
        embedding_column: str = 'embedding'
    ):
        """
        Bulk insert data with embeddings.
        
        Args:
            table: Table name
            data: List of dictionaries with column values
            embedding_column: Name of embedding column
        """
        if not data:
            return
        
        # Get column names
        columns = list(data[0].keys())
        
        # Build INSERT query
        cols_str = ', '.join(columns)
        placeholders = ', '.join(['%s'] * len(columns))
        query = f"INSERT INTO {table} ({cols_str}) VALUES ({placeholders})"
        
        # Prepare values
        values = []
        for row in data:
            row_values = []
            for col in columns:
                val = row[col]
                # Convert numpy arrays to lists for pgvector
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                row_values.append(val)
            values.append(tuple(row_values))
        
        # Execute batch insert
        try:
            execute_values(self.cursor, query, values)
            self.conn.commit()
            logger.info(f"Inserted {len(data)} rows into {table}")
            
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            self.conn.rollback()
            raise

    def reconnect(self, max_retries: int = 3, retry_delay: float = 2.0):
        """
        Reconnect to database with retry logic.
        
        Useful when the connection is lost due to network issues or
        PostgreSQL server restarts during long-running benchmarks.
        
        Args:
            max_retries: Maximum number of reconnection attempts
            retry_delay: Seconds to wait between retries
        """
        import time
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Reconnect attempt {attempt}/{max_retries}...")
                self.disconnect()
                self.connect()
                logger.info("Reconnected successfully.")
                return
            except Exception as e:
                logger.warning(f"Reconnect attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
        
        raise ConnectionError(f"Failed to reconnect after {max_retries} attempts.")

    def get_vector_index_stats(
        self,
        table: str,
        embedding_column: str
    ) -> Dict[str, Any]:
        """
        Retrieve vector index metadata (HNSW / IVFFlat) for cost estimation.
        
        The cost model needs the index type and its parameters (e.g., HNSW M,
        ef_construction; IVFFlat lists, probes) to produce accurate cost estimates.
        This method queries pg_indexes and pg_opclass to extract that information.
        
        Args:
            table: Table name (e.g., 'products')
            embedding_column: Vector column name (e.g., 'embedding')
            
        Returns:
            Dictionary with index type and parameters, e.g.:
            {
                'index_type': 'hnsw',
                'index_name': 'products_embedding_idx',
                'hnsw_m': 16,
                'hnsw_ef_construction': 64,
            }
            Returns empty dict if no vector index found.
        """
        query = """
            SELECT
                i.relname        AS index_name,
                am.amname        AS index_type,
                ix.indoption     AS index_options,
                pg_get_indexdef(ix.indexrelid) AS index_def
            FROM
                pg_index ix
                JOIN pg_class  t  ON t.oid  = ix.indrelid
                JOIN pg_class  i  ON i.oid  = ix.indexrelid
                JOIN pg_am     am ON am.oid = i.relam
                JOIN pg_attribute a ON a.attrelid = t.oid
                    AND a.attnum = ANY(ix.indkey)
            WHERE
                t.relname = %s
                AND a.attname = %s
        """
        results = self.execute_query(query, (table, embedding_column))
        
        if not results:
            logger.warning(
                f"No vector index found on {table}.{embedding_column}. "
                f"Cost estimates will use brute-force defaults."
            )
            return {}
        
        index_name, index_type, _, index_def = results[0]
        index_def_lower = index_def.lower()
        
        stats: Dict[str, Any] = {
            'index_name': index_name,
            'index_type': index_type.lower(),  # 'hnsw' or 'ivfflat'
        }
        
        # Parse HNSW parameters from index definition
        # e.g.: USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)
        if 'hnsw' in index_def_lower:
            import re
            m_match = re.search(r'm\s*=\s*(\d+)', index_def_lower)
            ef_match = re.search(r'ef_construction\s*=\s*(\d+)', index_def_lower)
            stats['hnsw_m'] = int(m_match.group(1)) if m_match else 16
            stats['hnsw_ef_construction'] = int(ef_match.group(1)) if ef_match else 64
        
        # Parse IVFFlat parameters
        # e.g.: USING ivfflat (embedding vector_cosine_ops) WITH (lists=100)
        elif 'ivfflat' in index_def_lower:
            import re
            lists_match = re.search(r'lists\s*=\s*(\d+)', index_def_lower)
            stats['ivfflat_lists'] = int(lists_match.group(1)) if lists_match else 100
        
        logger.info(
            f"Vector index on {table}.{embedding_column}: "
            f"{stats['index_type']} — {stats}"
        )
        return stats

    def execute_hybrid_query(
        self,
        table: str,
        embedding_column: str,
        query_vector: np.ndarray,
        k: int,
        where_clause: Optional[str],
        where_params: Optional[tuple],
        plan_type: str = 'auto',
        distance_metric: str = 'cosine'
    ) -> Tuple[List[tuple], Dict[str, Any]]:
        """
        Execute a hybrid SQL + vector query using the optimizer's chosen plan.

        This is the main integration point between the plan selector and the
        database. It receives the plan decision (filter-first, vector-first, or
        auto) from the optimizer and dispatches to the correct execution strategy.

        Execution strategies
        --------------------
        filter-first : Apply SQL predicates first to reduce the candidate set,
                       then run vector similarity search on the smaller result.
                       Best when the SQL filter is highly selective (< ~5%).

        vector-first : Run vector similarity search on the full table first
                       (fetching k * oversample candidates), then apply SQL
                       filters to the result set.
                       Best when the SQL filter has low selectivity (> ~50%).

        auto         : Falls back to filter-first if a where_clause is provided,
                       otherwise vector-first.

        Args:
            table:            Table name
            embedding_column: Vector column name
            query_vector:     Query embedding (numpy array)
            k:                Number of final results to return
            where_clause:     SQL filter string, e.g. "price < %s AND brand = %s"
            where_params:     Parameter tuple for where_clause
            plan_type:        'filter_first' | 'vector_first' | 'auto'
            distance_metric:  'cosine' | 'l2' | 'inner_product'

        Returns:
            Tuple of:
              - results: list of row tuples
              - meta:    execution metadata dict with plan_type and row counts
        """
        import time

        meta: Dict[str, Any] = {
            'plan_type': plan_type,
            'table': table,
            'k': k,
        }

        # Resolve 'auto' strategy
        if plan_type == 'auto':
            plan_type = 'filter_first' if where_clause else 'vector_first'
            meta['plan_type'] = plan_type
            logger.info(f"Auto-selected plan: {plan_type}")

        t_start = time.perf_counter()

        if plan_type == 'filter_first':
            # ── Strategy 1: Filter → then Vector ──────────────────────────────
            # SQL predicates narrow the candidate set first.
            # The ORDER BY vector distance then runs only over matching rows.
            logger.info(
                f"[filter_first] Applying WHERE clause first, "
                f"then ordering by vector distance."
            )
            results = self.vector_search(
                table=table,
                embedding_column=embedding_column,
                query_vector=query_vector,
                k=k,
                distance_metric=distance_metric,
                where_clause=where_clause,
                where_params=where_params,
            )
            meta['candidate_rows_scanned'] = len(results)

        elif plan_type == 'vector_first':
            # ── Strategy 2: Vector → then Filter ──────────────────────────────
            # Fetch an oversampled set of nearest neighbours first (k * 10),
            # then apply the SQL filter in Python over the candidate set.
            # This avoids scanning the whole table when the filter is weak.
            oversample_k = k * 10
            logger.info(
                f"[vector_first] Fetching top-{oversample_k} candidates via "
                f"vector search, then applying SQL filter in post-processing."
            )
            candidates = self.vector_search(
                table=table,
                embedding_column=embedding_column,
                query_vector=query_vector,
                k=oversample_k,
                distance_metric=distance_metric,
                where_clause=None,   # No pre-filter — vector search on full table
                where_params=None,
            )
            meta['candidate_rows_scanned'] = len(candidates)

            # Post-filter: if a where_clause was supplied, evaluate it in Python.
            # TODO (T16 - parallel_executor): push this into the parallel pipeline
            # instead of doing it here in a single thread.
            if where_clause and where_params:
                # For now, return top-k of candidates (sorted by vector distance).
                # Full in-process filtering will move to parallel_executor in T16.
                top_k: List[tuple] = [candidates[i] for i in range(min(k, len(candidates)))]
                results = top_k
                logger.warning(
                    "vector_first post-filter: returning top-k of candidates. "
                    "In-process predicate evaluation is deferred to T16."
                )
            else:
                results = [candidates[i] for i in range(min(k, len(candidates)))]

        else:
            raise ValueError(
                f"Unknown plan_type '{plan_type}'. "
                f"Expected 'filter_first', 'vector_first', or 'auto'."
            )

        elapsed_ms: float = float((time.perf_counter() - t_start) * 1000)
        meta['execution_time_ms'] = int(elapsed_ms * 1000) / 1000  # 3 decimal places
        meta['results_returned'] = len(results)

        logger.info(
            f"execute_hybrid_query [{meta['plan_type']}] → "
            f"{len(results)} rows in {elapsed_ms:.1f} ms"
        )
        return results, meta


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # NOTE: Requires a running PostgreSQL + pgvector instance.
    # Run  sql/setup.sql  and  scripts/load_data.py  first.

    try:
        db = PostgreSQLConnector('config.yaml')
        db.connect()

        # ── 1. Table statistics ───────────────────────────────────────────────
        stats = db.get_table_statistics('products')
        print("\n[1] Table Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # ── 2. Vector index metadata (for cost model) ─────────────────────────
        idx_stats = db.get_vector_index_stats('products', 'embedding')
        print("\n[2] Vector Index Stats:")
        for key, value in idx_stats.items():
            print(f"  {key}: {value}")

        # ── 3. Hybrid query — filter-first ────────────────────────────────────
        query_vector = np.random.randn(384).astype(np.float32)
        results, meta = db.execute_hybrid_query(
            table='products',
            embedding_column='embedding',
            query_vector=query_vector,
            k=10,
            where_clause='price < %s AND brand = %s',
            where_params=(1000, 'Apple'),
            plan_type='filter_first',
        )
        print(f"\n[3] Hybrid query (filter_first): {len(results)} results")
        print(f"    Metadata: {meta}")

        # ── 4. Hybrid query — vector-first ────────────────────────────────────
        results2, meta2 = db.execute_hybrid_query(
            table='products',
            embedding_column='embedding',
            query_vector=query_vector,
            k=10,
            where_clause='price < %s AND brand = %s',
            where_params=(1000, 'Apple'),
            plan_type='vector_first',
        )
        print(f"\n[4] Hybrid query (vector_first): {len(results2)} results")
        print(f"    Metadata: {meta2}")

        db.disconnect()

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print("Note: This demo requires PostgreSQL + pgvector.")
        print("Run sql/setup.sql first!")
