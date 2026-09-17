# Cost-Based Optimization for Hybrid SQL + Vector Queries

## 🎓 7-Person Team Research Project
**University of Windsor - Advanced Database Systems Final Project**

A query optimizer that will combine traditional SQL cost models with vector similarity search optimization. This project involves 7 team members covering database architecture, machine learning, distributed systems, and performance evaluation.

**Real-world Application:** RAG (Retrieval-Augmented Generation) Query Optimization  
**Current Status:** Foundation complete, core implementation in progress (~30%)

## 🎯 Project Overview

Modern AI applications combine traditional SQL filters with vector similarity search (semantic search). Current databases execute these queries inefficiently because they lack cost models for vector operations. This project aims to implement a **cost-based query optimizer** that automatically chooses the best execution strategy.

### Problem Statement
When executing hybrid queries like:
```sql
SELECT * FROM products
WHERE price < 1000 AND brand = 'Apple'
ORDER BY cosine_similarity(embedding, :query_vector)
LIMIT 10;
```

**Current approach:** Database uses fixed execution order (inefficient)  
**Our solution:** Cost-based optimizer chooses the best plan dynamically

### Expected Impact
- 3-10x faster queries on real workloads
- Reduced CPU and memory usage
- Applicable to RAG, semantic search, recommendation systems

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SQL Query (User Input)                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Query Parser & Analyzer                   │
│            (Extract filters, vector ops, predicates)         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Plan Generator (Our Core Work)               │
│  • Generate alternative plans (filter-first, vector-first)   │
│  • Enumerate hybrid strategies                               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Cost Model (Our Innovation)                     │
│  • Estimate vector search cost (ANN index stats)             │
│  • Estimate SQL filter selectivity                           │
│  • Choose minimum-cost plan                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Execution Engine                          │
│         Execute optimized plan on PostgreSQL + pgvector      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure (Current)

```
final_project/
├── src/
│   ├── cost_model/            # Cost estimation (in progress)
│   │   ├── vector_cost.py     # HNSW/IVFFlat cost formulas (partial)
│   │   └── sql_cost.py        # Traditional SQL cost (done)
│   ├── optimizer/             # Query plan optimization (in progress)
│   │   ├── plan_generator.py  # Generate alternative plans (partial)
│   │   └── plan_selector.py   # Cost-based plan selection (planned)
│   ├── executor/              # Query execution
│   │   └── pg_connector.py    # PostgreSQL + pgvector interface (basic)
│   └── utils/
├── data/
│   ├── datasets/              # Sample data
│   └── workloads/             # Test queries
├── benchmarks/
│   ├── run_experiments.py     # Automated benchmarking (planned)
│   └── results/               # Performance logs
├── notebooks/
│   └── 01_demo.ipynb          # Interactive demo (planned)
├── tests/
│   └── test_cost_model.py     # Unit tests (partial)
├── sql/
│   ├── setup.sql              # PostgreSQL + pgvector setup
│   └── schema_multi_table.sql # Full relational schema with vector columns
├── scripts/
│   ├── generate_data.py       # Data generation
│   └── load_data.py           # Data loading
├── requirements.txt
├── config.yaml                # Database & optimizer config
└── README.md
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Setup PostgreSQL + pgvector
psql -U postgres -f sql/setup.sql
```

### 2. Load Sample Data
```bash
python scripts/load_data.py --dataset products
```

### 3. Run Demo
```bash
# Compare baseline vs optimized execution
python benchmarks/run_experiments.py --query workloads/rag_queries.sql
```

### 4. View Results
```bash
jupyter notebook notebooks/01_demo.ipynb
```

---

## 🧪 Experiments & Evaluation

### Datasets
1. **E-commerce Products** (100K rows, 768-dim embeddings)
   - Filters: price, brand, category, rating
   - Vector: product description embeddings (BERT)

2. **Research Papers** (50K rows, 1024-dim embeddings)
   - Filters: year, citations, venue
   - Vector: abstract embeddings (SciBERT)

3. **Image Search** (200K rows, 512-dim embeddings)
   - Filters: size, format, tags
   - Vector: CLIP image embeddings

### Workloads
- **RAG queries**: High selectivity filters + top-k similarity
- **Semantic search**: Low selectivity + large k
- **Hybrid search**: Mixed selectivity scenarios

### Metrics
- Query latency (ms)
- CPU time
- Memory usage
- Plan quality (cost estimation accuracy)

---

## 🎓 Learning Outcomes (Planned)

### Database Internals
- Query optimization algorithms  
- Cost model design  
- Index selection strategies  
- Execution plan evaluation

### Modern AI/DB Integration
- Vector databases (pgvector, FAISS)  
- Embedding-based search  
- RAG system architecture  
- Hybrid query processing

### Production Skills
- PostgreSQL administration  
- Performance benchmarking  
- Python for data systems  
- Statistical analysis & visualization

---

## 📊 Expected Results

| Query Type | Baseline (ms) | Optimized (ms) | Speedup |
|------------|---------------|----------------|---------|
| High-selectivity filter | 1200 | 180 | **6.7x** |
| Low-selectivity filter | 450 | 380 | **1.2x** |
| No filter (pure vector) | 320 | 320 | 1.0x |
| Hybrid (balanced) | 800 | 210 | **3.8x** |

**Average improvement: 3-5x faster**

---

## 🔬 Technical Innovation

### 1. Vector Operation Cost Model
```python
def estimate_vector_search_cost(n_vectors, k, index_type='ivfflat'):
    """
    Cost = (probe_lists * avg_list_size * distance_calc_cost) 
           + (k * log(k) * comparison_cost)
    """
    # Based on FAISS/pgvector benchmarking
```

### 2. Plan Enumeration
- **Filter-first**: Apply SQL filters → vector search on subset
- **Vector-first**: Vector search → apply SQL filters
- **Hybrid**: Partial filter → vector → remaining filters

### 3. Adaptive Optimization
- Collects runtime statistics
- Refines cost estimates
- Learns from query patterns

---

## 📝 Deliverables

### Code
- ✅ Database schema design and setup
- ✅ SQL cost model
- 🔄 Vector cost model (in progress)
- 🔄 Plan generator (in progress)
- ⬜ Plan selector
- ⬜ ML selectivity estimator
- ⬜ Neural cost model
- ⬜ Join optimizer & index advisor
- ⬜ Parallel executor
- ⬜ Distributed coordinator
- ⬜ System integration

### Documentation
- ✅ Pre-proposal (IEEE format)
- ✅ README
- ⬜ Architecture design document
- ⬜ Final report

### Evaluation
- ⬜ Benchmark results on 3 datasets
- ⬜ Performance analysis report
- ⬜ Visualization dashboard

### Presentation
- ⬜ Demo notebook
- ⬜ Slide deck

---

## 🛠️ Technology Stack

- **Database**: PostgreSQL 16 + pgvector
- **Vector Search**: pgvector (HNSW, IVFFlat), FAISS (optional)
- **Language**: Python 3.11+
- **Libraries**: 
  - psycopg2 (PostgreSQL connector)
  - numpy, pandas (data processing)
  - sentence-transformers (embeddings)
  - matplotlib, plotly (visualization)
- **Tools**: Jupyter, pytest, EXPLAIN ANALYZE

---

## 📚 References

- [PostgreSQL Query Optimization](https://www.postgresql.org/docs/current/planner-optimizer.html)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [FAISS: A Library for Efficient Similarity Search](https://ai.meta.com/tools/faiss/)
- [RAG Systems Best Practices](https://arxiv.org/abs/2005.11401)

---
 
Advanced Database Systems - Winter 2026  
University of Windsor

---

## 📄 License

MIT License - Educational Use
