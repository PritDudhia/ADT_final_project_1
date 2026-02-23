"""
Mid-Project Demo: Cost-Based Optimization for Hybrid SQL + Vector Queries
=========================================================================

This script demonstrates the components completed so far (~30% progress):

  1. Database Schema Design         (Arbaz)   - COMPLETED
  2. SQL Cost Model                 (Bharat)  - COMPLETED
  3. Vector Cost Model              (Bharat)  - IN PROGRESS
  4. Plan Generator                 (Prit)    - IN PROGRESS
  5. Plan Selector / Cost Compare   (Prit)    - IN PROGRESS
  6. PostgreSQL Connector           (Farhan)  - IN PROGRESS

Run:  python mid_project_demo.py
"""

import sys
import os
import time

# ── path setup ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cost_model.sql_cost import SQLCostModel
from cost_model.vector_cost import VectorCostModel
from optimizer.plan_generator import (
    PlanGenerator, QueryPredicate, VectorOperation, PlanType
)
from optimizer.plan_selector import PlanSelector

# ── helpers ─────────────────────────────────────────────────────────────

DIVIDER  = "=" * 70
SECTION  = "-" * 50

def banner(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

def sub(title):
    print(f"\n  {SECTION}")
    print(f"  {title}")
    print(f"  {SECTION}")

def pause():
    input("\n  [Press Enter to continue...]\n")


# ── configuration (from config.yaml) ───────────────────────────────────

COST_CONFIG = {
    'seq_scan_cost': 1.0,
    'random_page_cost': 4.0,
    'cpu_tuple_cost': 0.01,
    'cpu_index_tuple_cost': 0.005,
    'cpu_operator_cost': 0.0025,
    'vector_distance_cost': 1.0,
    'vector_comparison_cost': 0.1,
}

OPTIMIZER_CONFIG = {
    'enable_filter_first': True,
    'enable_vector_first': True,
    'enable_hybrid': False,       # not yet implemented
    'plan_enumeration_limit': 10,
}

FULL_CONFIG = {
    'cost_model': COST_CONFIG,
    'optimizer': OPTIMIZER_CONFIG,
}


# ────────────────────────────────────────────────────────────────────────
# DEMO  1 — Database Schema
# ────────────────────────────────────────────────────────────────────────

def demo_schema():
    banner("DEMO 1: Database Schema Design  (Arbaz)")

    schema_file = os.path.join(os.path.dirname(__file__),
                               'sql', 'schema_multi_table.sql')

    if os.path.exists(schema_file):
        with open(schema_file, 'r') as f:
            content = f.read()

        # Count tables
        tables = [line for line in content.split('\n')
                  if line.strip().upper().startswith('CREATE TABLE')]
        vector_cols = content.lower().count('vector(')

        print(f"""
  Schema file : sql/schema_multi_table.sql
  Tables      : {len(tables)}
  Vector cols : {vector_cols}  (pgvector embedding columns)
""")
        for t in tables:
            name = t.strip().split('(')[0].replace('CREATE TABLE', '').strip()
            print(f"    - {name}")

        print(f"""
  Key design decisions:
    * Normalized e-commerce schema (3NF)
    * Foreign key constraints between all tables
    * vector(384) columns embedded directly in relational tables
    * Supports hybrid SQL + vector queries on the SAME tables
""")
    else:
        print("  [schema file not found]")

    pause()


# ────────────────────────────────────────────────────────────────────────
# DEMO  2 — SQL Cost Model
# ────────────────────────────────────────────────────────────────────────

def demo_sql_cost():
    banner("DEMO 2: SQL Cost Model  (Bharat)")

    sql_model = SQLCostModel(COST_CONFIG)

    # 2-a  Sequential scan vs index scan
    sub("2a. Sequential Scan Cost — products table (100 000 rows)")

    n = 100_000
    seq_cost = sql_model.estimate_sequential_scan_cost(n, avg_tuple_size=200)
    print(f"    Rows          : {n:,}")
    print(f"    Seq-scan cost : {seq_cost:,.2f}")

    sub("2b. Index Scan Cost — equality on 'brand' (1 000 matching rows)")

    idx_cost = sql_model.estimate_index_scan_cost(n, n_matching=1000)
    print(f"    Total rows    : {n:,}")
    print(f"    Matching rows : 1,000")
    print(f"    Index cost    : {idx_cost:,.2f}")
    print(f"\n    >>> Index scan is {seq_cost/idx_cost:.1f}x cheaper than full scan")

    # 2-b  Selectivity estimation
    sub("2c. Selectivity Estimation")

    scenarios = [
        ('equality', {'n_distinct': 50}),
        ('equality', {'n_distinct': 1000}),
        ('range',    {'range_fraction': 0.25}),
        ('range',    {'range_fraction': 0.01}),
        ('like',     {}),
    ]

    print(f"    {'Predicate Type':<18} {'Parameters':<28} {'Selectivity':>12}")
    print(f"    {'-'*18} {'-'*28} {'-'*12}")
    for ptype, params in scenarios:
        sel = sql_model.estimate_selectivity(ptype, **params)
        pstr = ', '.join(f'{k}={v}' for k, v in params.items()) or '-'
        print(f"    {ptype:<18} {pstr:<28} {sel:>12.4f}")

    # Combined selectivity
    sub("2d. Combined Selectivity (AND conditions)")

    s1 = sql_model.estimate_selectivity('equality', n_distinct=50)
    s2 = sql_model.estimate_selectivity('range', range_fraction=0.25)
    combined = sql_model.estimate_combined_selectivity([s1, s2])
    print(f"    brand = 'Apple'   -> selectivity = {s1:.4f}")
    print(f"    price < 1000      -> selectivity = {s2:.4f}")
    print(f"    Combined (AND)    -> selectivity = {combined:.4f}")
    print(f"    On 100K rows      -> estimated {int(n * combined):,} matching rows")

    pause()


# ────────────────────────────────────────────────────────────────────────
# DEMO  3 — Vector Cost Model
# ────────────────────────────────────────────────────────────────────────

def demo_vector_cost():
    banner("DEMO 3: Vector Cost Model  (Bharat — In Progress)")

    vec_model = VectorCostModel(COST_CONFIG)

    n = 100_000
    k = 10

    sub("3a. HNSW Search Cost")

    hnsw_cost = vec_model.estimate_hnsw_search_cost(n, k, m=16, ef_search=40)
    print(f"    Vectors   : {n:,}")
    print(f"    k         : {k}")
    print(f"    HNSW(m=16, ef=40) cost : {hnsw_cost:,.2f}")

    sub("3b. IVFFlat Search Cost")

    ivf_cost = vec_model.estimate_ivfflat_search_cost(n, k, n_lists=100, n_probes=10)
    print(f"    IVFFlat(lists=100, probes=10) cost : {ivf_cost:,.2f}")

    sub("3c. Sequential (Brute-Force) Scan")

    seq_cost = vec_model.estimate_sequential_vector_scan(n, k)
    print(f"    Brute-force cost : {seq_cost:,.2f}")

    print(f"""
    >>> Comparison (100K vectors, k=10):
        HNSW      : {hnsw_cost:>10,.2f}  ({seq_cost/hnsw_cost:>5.1f}x faster than brute-force)
        IVFFlat   : {ivf_cost:>10,.2f}  ({seq_cost/ivf_cost:>5.1f}x faster)
        BruteForce: {seq_cost:>10,.2f}  (baseline)
""")

    # Filtered search
    sub("3d. Filter-First vs Vector-First Cost Comparison")

    filter_sel = 0.02  # 2% of rows pass SQL filter
    filtered_n = int(n * filter_sel)

    hnsw_filtered = vec_model.estimate_hnsw_search_cost(filtered_n, k)
    hnsw_full = vec_model.estimate_hnsw_search_cost(n, k)

    print(f"    Filter selectivity  : {filter_sel*100:.0f}% ({filtered_n:,} rows)")
    print(f"    Vector-first (full) : {hnsw_full:,.2f}")
    print(f"    Filter-first (2K)   : {hnsw_filtered:,.2f}")
    print(f"\n    >>> Filter-first is {hnsw_full/hnsw_filtered:.1f}x cheaper when selectivity is low")

    print(f"""
    NOTE: Index traversal cost estimation is still being refined.
          Current model handles distance calculations accurately but
          I/O cost for disk-resident index pages is a TODO item.
""")

    pause()


# ────────────────────────────────────────────────────────────────────────
# DEMO  4 — Plan Generator
# ────────────────────────────────────────────────────────────────────────

def demo_plan_generator():
    banner("DEMO 4: Plan Generator  (Prit — In Progress)")

    gen = PlanGenerator(OPTIMIZER_CONFIG)

    # Define a sample hybrid query
    print("""
  Sample Query:
    SELECT * FROM products
    WHERE price < 500 AND brand = 'Apple'
    ORDER BY cosine_similarity(embedding, :query_vector)
    LIMIT 10;
""")

    predicates = [
        QueryPredicate(column='price',  operator='<',  value=500,
                       predicate_type='range', selectivity=0.25),
        QueryPredicate(column='brand',  operator='=',  value='Apple',
                       predicate_type='equality', selectivity=0.02),
    ]

    vector_op = VectorOperation(
        embedding_column='embedding',
        query_vector=[0.1] * 384,  # placeholder 384-dim vector
        k=10,
        distance_metric='cosine',
        index_type='hnsw',
    )

    table_stats = {
        'n_tuples': 100_000,
        'avg_tuple_size': 200,
        'hnsw_m': 16,
        'hnsw_ef_search': 40,
        'price_distinct': 5000,
        'brand_distinct': 50,
    }

    # Generate plans
    plans = gen.generate_plans(predicates, vector_op, table_stats)

    print(f"  Plans generated: {len(plans)}\n")

    for plan in plans:
        print(f"    {plan.plan_id}  |  Type: {plan.plan_type.value}")
        print(f"    {'':>10}  Operations:")
        for i, op in enumerate(plan.operations, 1):
            if op['type'] == 'filter':
                cols = [p.column for p in op['predicates']]
                print(f"    {'':>10}    Step {i}: SQL filter on [{', '.join(cols)}] via {op['method']}")
            elif op['type'] == 'vector_search':
                print(f"    {'':>10}    Step {i}: Vector search ({op['index_type']}, k={op['k']}, {op['distance_metric']})")
            elif op['type'] == 'limit':
                print(f"    {'':>10}    Step {i}: Limit to top-{op['k']}")
        print()

    print("""    NOTE: Hybrid interleaved plan generation is planned but
          not yet implemented (Phase 3 task).
""")

    pause()
    return plans, table_stats


# ────────────────────────────────────────────────────────────────────────
# DEMO  5 — Plan Selector (Cost-Based Selection)
# ────────────────────────────────────────────────────────────────────────

def demo_plan_selector(plans, table_stats):
    banner("DEMO 5: Plan Selector — Cost Comparison  (Prit — In Progress)")

    selector = PlanSelector(FULL_CONFIG)

    print("  Estimating cost for each candidate plan...\n")

    best = selector.select_best_plan(plans, table_stats)

    print(f"\n  {'Plan ID':<12} {'Type':<18} {'Est. Cost':>12}")
    print(f"  {'-'*12} {'-'*18} {'-'*12}")
    for p in sorted(plans, key=lambda x: x.estimated_cost):
        marker = " <-- BEST" if p.plan_id == best.plan_id else ""
        print(f"  {p.plan_id:<12} {p.plan_type.value:<18} {p.estimated_cost:>12,.2f}{marker}")

    print(f"""
  Decision:  {best.plan_type.value.upper()}
  Reason  :  The cost model evaluates both strategies and selects the one
             with the lowest total estimated cost. HNSW graph traversal is
             efficient (O(log n)), so vector-first + post-filter wins here.
""")

    # Show how decision changes with different selectivity
    sub("5b. Sensitivity Analysis — What if table is very small?")

    print("""
  Now simulating a SMALL table scenario (1,000 rows instead of 100K).
  With fewer rows, sequential scan becomes more viable.
""")

    from optimizer.plan_generator import PlanGenerator, QueryPredicate, VectorOperation

    gen = PlanGenerator(OPTIMIZER_CONFIG)

    broad_predicates = [
        QueryPredicate(column='price', operator='<', value=500,
                       predicate_type='range', selectivity=0.25),
        QueryPredicate(column='brand', operator='=', value='Apple',
                       predicate_type='equality', selectivity=0.02),
    ]

    broad_vector = VectorOperation(
        embedding_column='embedding',
        query_vector=[0.1] * 384,
        k=10,
        distance_metric='cosine',
        index_type='hnsw',
    )

    small_table_stats = dict(table_stats)
    small_table_stats['n_tuples'] = 1000

    broad_plans = gen.generate_plans(broad_predicates, broad_vector, small_table_stats)
    best_broad = selector.select_best_plan(broad_plans, small_table_stats)

    print(f"  {'Plan ID':<12} {'Type':<18} {'Est. Cost':>12}")
    print(f"  {'-'*12} {'-'*18} {'-'*12}")
    for p in sorted(broad_plans, key=lambda x: x.estimated_cost):
        marker = " <-- BEST" if p.plan_id == best_broad.plan_id else ""
        print(f"  {p.plan_id:<12} {p.plan_type.value:<18} {p.estimated_cost:>12,.2f}{marker}")

    print(f"""
  Decision:  {best_broad.plan_type.value.upper()}
  Reason  :  Cost estimates change with table size and selectivity.
             The optimizer adapts its choice based on query characteristics.

  This demonstrates the CORE VALUE of our cost-based optimizer:
  it evaluates multiple strategies and picks the cheapest one dynamically,
  rather than always using a fixed execution order.
""")

    pause()


# ────────────────────────────────────────────────────────────────────────
# DEMO  6 — Schema Walkthrough (SQL file on screen)
# ────────────────────────────────────────────────────────────────────────

def demo_schema_detail():
    banner("DEMO 6: Schema Detail — Key Tables & Relationships  (Arbaz)")

    print("""
  E-Commerce Schema (7 tables):

    categories ---+
                  +--> products --+--> order_items --> orders --> customers
    sellers ------+               +--> reviews --------------------^

  Vector Columns Integrated in Relational Schema:
    +---------------+---------------------------+------------------+
    | Table         | Column                    | Dimension        |
    +---------------+---------------------------+------------------+
    | categories    | embedding                 | vector(384)      |
    | sellers       | profile_embedding         | vector(384)      |
    | products      | embedding                 | vector(384)      |
    | products      | image_embedding           | vector(512)      |
    | customers     | preference_embedding      | vector(384)      |
    | reviews       | review_embedding          | vector(384)      |
    +---------------+---------------------------+------------------+

  This is the KEY INNOVATION in our schema:
    Traditional databases keep vector data in separate stores.
    We embed vectors DIRECTLY in relational tables, enabling
    hybrid SQL + vector queries in a SINGLE query plan.
""")

    pause()


# ────────────────────────────────────────────────────────────────────────
# SUMMARY
# ────────────────────────────────────────────────────────────────────────

def demo_summary():
    banner("SUMMARY: Mid-Project Progress (~30%)")

    print("""
  +---+--------------------------------------+------------+----------+
  | # | Component                            | Member     | Status   |
  +---+--------------------------------------+------------+----------+
  | 1 | Database schema (7 tables + vectors) | Arbaz      | DONE     |
  | 2 | SQL cost model (scan, filter, sel.)  | Bharat     | DONE     |
  | 3 | Vector cost model (HNSW, IVFFlat)    | Bharat     | PARTIAL  |
  | 4 | Plan generator (filter/vector-first) | Prit       | PARTIAL  |
  | 5 | Plan selector (cost comparison)      | Prit       | PARTIAL  |
  | 6 | PostgreSQL connector                 | Farhan     | PARTIAL  |
  | 7 | Data scripts (generate, load)        | Farhan     | DONE     |
  +---+--------------------------------------+------------+----------+

  REMAINING WORK:
    - ML-based selectivity estimator (Random Forest)
    - Neural cost model (PyTorch LSTM)
    - Auto-calibration framework
    - Join optimizer (dynamic programming)
    - Index advisor
    - Parallel query executor
    - Distributed query coordinator
    - TPC-H benchmarks & performance dashboard
    - Full system integration
    - Final report & presentation
""")


# ────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────

def main():
    os.system('cls' if os.name == 'nt' else 'clear')

    print(DIVIDER)
    print("  MID-PROJECT DEMONSTRATION")
    print("  Cost-Based Optimization for Hybrid SQL + Vector Queries")
    print("  University of Windsor — Advanced Database Systems")
    print(f"  Team: Prit, Arbaz, Bharat, Farhan")
    print(DIVIDER)
    print("""
  This demo walks through all components completed so far.
  Each section maps to a task in our Task Breakdown Structure.
""")

    pause()

    demo_schema()           # Demo 1 — Schema (Arbaz)
    demo_sql_cost()         # Demo 2 — SQL cost model (Bharat)
    demo_vector_cost()      # Demo 3 — Vector cost model (Bharat)
    plans, stats = demo_plan_generator()  # Demo 4 — Plan gen (Prit)
    demo_plan_selector(plans, stats)       # Demo 5 — Plan select (Prit)
    demo_schema_detail()    # Demo 6 — Schema detail (Arbaz)
    demo_summary()          # Summary

    print(DIVIDER)
    print("  Demo complete. Thank you!")
    print(DIVIDER)


if __name__ == '__main__':
    main()
