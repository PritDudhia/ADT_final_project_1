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


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Note: This example requires a running PostgreSQL instance
    # with pgvector extension and database setup completed
    
    try:
        # Connect to database
        db = PostgreSQLConnector('config.yaml')
        db.connect()
        
        # Get statistics
        stats = db.get_table_statistics('products')
        print("\nTable Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Example vector search
        # query_vector = np.random.randn(768)  # Example embedding
        # results = db.vector_search(
        #     table='products',
        #     embedding_column='embedding',
        #     query_vector=query_vector,
        #     k=10,
        #     distance_metric='cosine',
        #     where_clause='price < %s AND brand = %s',
        #     where_params=(1000, 'Apple')
        # )
        # print(f"\nFound {len(results)} results")
        
        db.disconnect()
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print("Note: This example requires PostgreSQL + pgvector setup")
        print("Run sql/setup.sql first!")
