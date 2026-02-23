# Mid-Project Evaluation Report

## Cost-Based Optimization for Hybrid SQL + Vector Queries

**Course:** Advanced Database Systems – University of Windsor  
**Team Size:** 7 Members  
**Evaluation Date:** February 2026

---

## 1. Task Breakdown Structure (TBS)

### Overview

The project is divided into **4 phases** with **18 tasks** across 7 team members. Current overall progress is approximately **30%**, with foundational phases completed and core implementation underway.

---

### Phase 1: Foundation & Database Design ✅ (Complete)

| Task ID | Task Description | Assigned To | Status | Dependencies | Remarks / Notes |
|---------|-----------------|-------------|--------|--------------|-----------------|
| T1 | Database schema design (ER diagram, multi-table schema with vector columns) | Member 1 (DB Architect) | ✅ Completed | None | Final schema with 7 tables: categories, sellers, products, customers, orders, order_items, reviews. Includes vector(384) columns for embeddings. See `sql/schema_multi_table.sql` |
| T2 | PostgreSQL + pgvector environment setup | Member 1 (DB Architect) | ✅ Completed | None | Setup script created (`sql/setup.sql`). pgvector extension enabled, three base tables (products, research_papers, images) with vector(768) columns |
| T3 | Configuration and project scaffolding | Member 3 (Lead) | ✅ Completed | None | `config.yaml` with database, vector index (HNSW/IVFFlat), cost model parameters, optimizer, and benchmark settings. Project folder structure organized |
| T4 | Sample data generation scripts | Member 7 (Evaluation) | ✅ Completed | T1, T2 | `scripts/generate_data.py` and `scripts/load_data.py` created. Data pipeline for populating tables with synthetic e-commerce data and vector embeddings |
| T5 | Pre-proposal and literature review | All Members | ✅ Completed | None | IEEE-format pre-proposal covering 8 related works (SIGMOD 2025 references). Novelty claim established: first unified cost-based optimizer for hybrid SQL+vector queries |

**Phase 1 Completion: 5/5 tasks done**

---

### Phase 2: Core Cost Model & Optimizer (In Progress)

| Task ID | Task Description | Assigned To | Status | Dependencies | Remarks / Notes |
|---------|-----------------|-------------|--------|--------------|-----------------|
| T6 | SQL cost model implementation | Member 2 (Cost Model Researcher) | ✅ Completed | T1, T3 | `src/cost_model/sql_cost.py` — Implements traditional PostgreSQL cost formulas: sequential scan, index scan, CPU tuple costs. Selectivity estimation using histogram-based approach |
| T7 | Vector cost model implementation | Member 2 (Cost Model Researcher) | 🔄 In Progress | T3 | `src/cost_model/vector_cost.py` — HNSW and IVFFlat cost formulas partially implemented. Distance calculation costs done; index traversal cost estimation still being refined |
| T8 | Query plan generator | Member 3 (Lead) | 🔄 In Progress | T6, T7 | `src/optimizer/plan_generator.py` — Generating filter-first and vector-first plan skeletons. Hybrid interleaved plan generation not yet started |
| T9 | Query plan selector (cost-based selection) | Member 3 (Lead) | ⬚ Not Started | T7, T8 | `src/optimizer/plan_selector.py` — Will compare generated plans using cost estimates and select optimal plan |
| T10 | PostgreSQL connector / executor interface | Member 4 (Execution Engine) | 🔄 In Progress | T2 | `src/executor/pg_connector.py` — Basic connection pooling and query execution working. Plan-aware execution routing not yet implemented |

**Phase 2 Completion: 1/5 tasks done, 3 in progress**

---

### Phase 3: Advanced Components (Not Started)

| Task ID | Task Description | Assigned To | Status | Dependencies | Remarks / Notes |
|---------|-----------------|-------------|--------|--------------|-----------------|
| T11 | ML-based selectivity estimator | Member 2 (Cost Model) | ⬚ Not Started | T6, T7 | Will use Random Forest for learned selectivity estimation. Planned in `src/ml/selectivity_estimator.py` |
| T12 | Neural cost model (deep learning) | Member 5 (ML Integration) | ⬚ Not Started | T6, T7, T11 | PyTorch LSTM+MLP model for cost prediction. Planned in `src/ml/neural_cost_model.py` |
| T13 | Auto-calibration framework | Member 2 (Cost Model) | ⬚ Not Started | T6, T7, T11 | Automatic tuning of cost model parameters. Planned in `src/calibration/auto_calibrator.py` |
| T14 | Dynamic programming join optimizer | Member 1 (DB Architect) | ⬚ Not Started | T1, T9 | Multi-table join ordering using DP algorithm. Planned in `src/database/join_optimizer.py` |
| T15 | Index advisor | Member 1 (DB Architect) | ⬚ Not Started | T1, T14 | Workload-driven index recommendations for B-tree and vector indexes. Planned in `src/database/index_advisor.py` |
| T16 | Parallel query executor | Member 4 (Execution Engine) | ⬚ Not Started | T9, T10 | Multi-threaded execution operators (parallel scan, hash join, merge join). Planned in `src/execution/parallel_executor.py` |
| T17 | Distributed query coordinator | Member 6 (Distributed Systems) | ⬚ Not Started | T9, T16 | Multi-node query routing and result aggregation. Planned in `src/distributed/query_coordinator.py` |

**Phase 3 Completion: 0/7 tasks done**

---

### Phase 4: Evaluation, Integration & Presentation (Not Started)

| Task ID | Task Description | Assigned To | Status | Dependencies | Remarks / Notes |
|---------|-----------------|-------------|--------|--------------|-----------------|
| T18 | TPC-H benchmark suite | Member 7 (Evaluation) | ⬚ Not Started | T9, T16 | TPC-H queries augmented with vector similarity predicates. Planned in `benchmarks/tpch_benchmark.py` |
| T19 | Performance dashboard | Member 7 (Evaluation) | ⬚ Not Started | T18 | Streamlit dashboard for visualizing benchmark results. Planned in `dashboards/performance_dashboard.py` |
| T20 | System integration (master API) | Member 3 (Lead) | ⬚ Not Started | T11–T17 | Unified system combining all 7 members' components. Planned in `main_system.py` |
| T21 | Final report and presentation | All Members | ⬚ Not Started | T18–T20 | Slides, demo notebook, and final project report |

**Phase 4 Completion: 0/4 tasks done**

---

### Summary Progress Table

| Phase | Tasks | Completed | In Progress | Not Started | Phase Status |
|-------|-------|-----------|-------------|-------------|--------------|
| Phase 1: Foundation & DB Design | 5 | 5 | 0 | 0 | ✅ Complete |
| Phase 2: Core Cost Model & Optimizer | 5 | 1 | 3 | 1 | 🔄 In Progress |
| Phase 3: Advanced Components | 7 | 0 | 0 | 7 | ⬚ Not Started |
| Phase 4: Evaluation & Integration | 4 | 0 | 0 | 4 | ⬚ Not Started |
| **Total** | **21** | **6** | **3** | **12** | **~30%** |

---

### Task Dependency Diagram

```
Phase 1 (Foundation)             Phase 2 (Core)                Phase 3 (Advanced)           Phase 4 (Final)
========================         ====================          ====================         =================

T1 (Schema Design) ✅ ──────┬──► T6 (SQL Cost) ✅ ──────┬──► T11 (ML Selectivity) ────┐
                             │                            │                              │
T2 (Env Setup) ✅ ──────────┤   T7 (Vector Cost) 🔄 ────┤──► T12 (Neural Cost) ────────┤
                             │                            │                              │
T3 (Config) ✅ ─────────────┤   T8 (Plan Generator) 🔄 ──┤──► T13 (Auto-Calibrate) ────┤
                             │                            │                              │
T4 (Data Scripts) ✅ ────────┘   T9 (Plan Selector) ⬚ ◄──┘                              ├──► T20 (Integration) ──► T21 (Report)
                                                    │                                    │
T5 (Pre-proposal) ✅             T10 (Connector) 🔄 ─┤──► T14 (Join Optimizer) ─────────┤
                                                     │──► T15 (Index Advisor) ──────────┤
                                                     │──► T16 (Parallel Exec) ──────────┤──► T18 (Benchmarks) ──► T19 (Dashboard)
                                                     └──► T17 (Distributed) ────────────┘
```

---

### What Has Been Accomplished So Far

1. **Complete database schema** with 7 normalized tables, foreign key constraints, and vector embedding columns integrated into the relational schema
2. **Working PostgreSQL + pgvector environment** with setup scripts for reproducible deployment
3. **SQL cost model** implementing standard PostgreSQL cost formulas for scan, join, and sort operations
4. **Partial vector cost model** with HNSW/IVFFlat distance calculation cost formulas (index traversal estimation ongoing)
5. **Initial plan generator** producing filter-first and vector-first execution plan skeletons
6. **Project infrastructure**: config files, data generation scripts, organized repository structure

### What Remains (Next Phase Priorities)

1. Complete vector cost model (T7) — estimated 1 week
2. Finish plan generator with hybrid plans (T8) — estimated 1 week
3. Implement plan selector for cost-based plan comparison (T9) — estimated 2 weeks
4. Begin ML selectivity estimator (T11) — the key innovation of the project
5. All Phase 3 advanced components and Phase 4 integration/evaluation

---

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ML model accuracy below target | Medium | High | Fall back to enhanced histogram-based estimation |
| Integration complexity across 7 members' code | Medium | Medium | Well-defined API interfaces established early |
| Benchmark dataset size limitations | Low | Medium | Use TPC-H scale factors adjustable from SF-1 to SF-10 |
| pgvector performance variability | Low | Low | Multiple index types (HNSW, IVFFlat) tested |
