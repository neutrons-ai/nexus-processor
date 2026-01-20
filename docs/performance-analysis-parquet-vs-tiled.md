# Performance Analysis: Parquet/Iceberg Lakehouse vs. Tiled + HDF5

This document compares the expected performance of time-slicing neutron event data using two different architectures:

1. **Parquet/Iceberg Lakehouse** (this project + neutron-lakehouse)
2. **Tiled Server** serving original HDF5/NeXus files

## Use Case: Time-Slicing Neutron Events

The time-slicing operation requires:

1. **Loading events** (potentially billions of rows with `pulse_index`, `pulse_time`, `event_id`, `time_offset`)
2. **Filtering/grouping** by `pulse_time` (directly stored in events, no join required)

**Note:** As of the latest version, `pulse_time` is stored directly in each event row during the ETL process. This eliminates the need to load DAS logs or perform a join on `pulse_index`, significantly improving query performance.

---

## Approach 1: Parquet/Iceberg Lakehouse (Spark)

| Aspect | Details |
|--------|---------|
| **Data Format** | Columnar Parquet, partitioned by `(instrument_id, run_number)` |
| **Storage** | S3-compatible (MinIO) via Iceberg tables |
| **Query Engine** | Distributed Spark (can scale to cluster) |

### Performance Characteristics

| Operation | Rating | Notes |
|-----------|--------|-------|
| **Partition Pruning** | ⭐⭐⭐⭐⭐ Excellent | Iceberg metadata allows skipping entire partitions; querying `run_id = 'REF_L:218386'` only reads that run's data |
| **Column Pruning** | ⭐⭐⭐⭐⭐ Excellent | Parquet columnar format means `pulse_time`, `time_offset` read without touching `event_id` |
| **Predicate Pushdown** | ⭐⭐⭐⭐⭐ Excellent | Filter on `pulse_time` pushes down to Parquet row groups |
| **Join Performance** | N/A | Join eliminated—`pulse_time` stored directly in events |
| **Aggregation** | ⭐⭐⭐⭐⭐ Excellent | Native Spark GROUP BY is highly optimized |
| **Scalability** | ⭐⭐⭐⭐⭐ Excellent | Horizontal scaling to cluster, parallelized across workers |
| **Multi-Run Analysis** | ⭐⭐⭐⭐⭐ Excellent | Single query can span thousands of runs |

### Expected Time-Slicing Performance

```
1 billion events, single run:
├── Read events (only needed columns): ~30-60 seconds (depends on cluster size)
├── Filter + aggregate on pulse_time: ~10-20 seconds
└── Total: ~40-80 seconds on single node, <20 seconds on 4-node cluster
```

**Note:** Since `pulse_time` is stored directly in events, no separate DAS log read or join is needed.

---

## Approach 2: Tiled Server + HDF5 (NeXus Files)

| Aspect | Details |
|--------|---------|
| **Data Format** | Original HDF5/NeXus files (row-oriented for events) |
| **Storage** | Filesystem or object storage |
| **Query Engine** | Tiled server (single process, Python-based) |

### Performance Characteristics

| Operation | Rating | Notes |
|-----------|--------|-------|
| **Partition Pruning** | ⭐⭐⭐⭐ Good | Tiled can serve specific files/runs via catalog |
| **Column Pruning** | ⭐⭐ Limited | HDF5 datasets are row-based; selecting `pulse_index` still reads entire event records |
| **Predicate Pushdown** | ⭐ Poor | No native time-based filtering; must read all events, then filter client-side |
| **Join Performance** | ⭐⭐ Limited | No server-side joins; must pull both datasets to client |
| **Aggregation** | ⭐ Poor | No server-side aggregation; all counting done client-side |
| **Scalability** | ⭐⭐ Limited | Single server process, though can cache hot data |
| **Multi-Run Analysis** | ⭐⭐ Limited | Must query each file separately |

### Expected Time-Slicing Performance

```
1 billion events, single run:
├── Fetch events via HTTP (chunked): ~5-15 minutes (network + serialization overhead)
├── Client-side join: ~1-2 minutes (pandas merge in memory)
├── Client-side aggregate: ~30 seconds
└── Total: ~7-18 minutes
```

### Critical HDF5 Limitation

The NeXus event data is stored as **parallel 1D arrays** (`event_id`, `event_time_offset`, `event_index`). There's no way to filter by time at the storage level—you must read all events and filter in Python.

---

## Head-to-Head Comparison

| Factor | Parquet/Iceberg | Tiled + HDF5 | Winner |
|--------|-----------------|--------------|--------|
| **Time-slice query (1B events)** | ~40-80 sec | ~10-18 min | **Parquet** (10-15x faster) |
| **Multi-run aggregation** | Single SQL query | Loop over files | **Parquet** (orders of magnitude) |
| **Storage efficiency** | ~40-60% of HDF5 size | Original size | **Parquet** (columnar compression) |
| **Schema evolution** | Iceberg handles it | Fixed by NeXus schema | **Parquet** |
| **Concurrent users** | Scales horizontally | Limited by server memory | **Parquet** |
| **Setup complexity** | Higher (Docker, Spark) | Lower (just Tiled) | **Tiled** |
| **Preserve original data** | Converted copy | Original HDF5 | **Tiled** |
| **Random array slicing** | Not native | ⭐⭐⭐⭐⭐ Excellent | **Tiled** |
| **Image/detector data** | Would need conversion | Native HDF5 chunks | **Tiled** |

---

## Key Insights

### Why Parquet/Iceberg Wins for Time-Slicing

1. **Columnar Storage**: Reading only `pulse_time` and `time_offset` from 1B events means reading ~16 GB instead of 40+ GB (skipping `event_id`, etc.)

2. **Embedded Pulse Time**: `pulse_time` is stored directly in events during ETL, eliminating the need to load DAS logs and perform joins at query time

3. **Predicate Pushdown**: Parquet row groups (~128 MB each) have min/max statistics. Filtering on `pulse_time < 60.0` allows entire row groups to be skipped.

4. **Distributed Processing**: Spark parallelizes across cores/nodes. A 4-worker cluster processes 4x faster.

5. **Native SQL Aggregation**: `GROUP BY interval` with `COUNT(*)` is a single pass—no client-side loop needed.

### When Tiled + HDF5 is Better

1. **Slicing detector images**: Tiled excels at serving array slices (`client['detector'][100:200, 300:400]`)

2. **Interactive exploration**: Browse NeXus hierarchy without ETL

3. **Preserving original format**: Regulatory/archive requirements

4. **Light usage**: Few users, occasional queries—overhead of lakehouse not justified

---

## Recommendations

### By Scenario

| Scenario | Recommendation |
|----------|----------------|
| **Production analytics** | ✅ Parquet/Iceberg Lakehouse |
| **Ad-hoc exploration of a single file** | ✅ Tiled (or local h5py) |
| **Multi-run time-series analysis** | ✅ Parquet/Iceberg (Spark SQL) |
| **Serving data to web dashboards** | 🤔 Either (depends on query patterns) |

### Summary

The lakehouse approach (nexus-processor → Parquet → Iceberg) is the right architecture for repeated, large-scale time-slicing queries. The one-time ETL cost (which now includes embedding `pulse_time` directly in events) pays off with **10-15x faster queries** by eliminating joins at query time.

---

## Hybrid Architecture (Best of Both)

You can run both systems in parallel:

```
                                    ┌─────────────────┐
                                    │  Tiled Server   │
                                    │                 │
Original HDF5 ──────────────────────►  Interactive    │
       │                            │  exploration    │
       │                            │  Image slicing  │
       │                            └─────────────────┘
       │
       │                            ┌─────────────────┐
       │     ┌────────────────┐     │    Lakehouse    │
       └────►│ nexus-processor │────►│                 │
             │ (ETL to Parquet)│     │  Fast SQL       │
             └────────────────┘     │  Time-slicing   │
                                    │  Multi-run      │
                                    └─────────────────┘
```

This gives you:

- **Tiled**: Quick browsing, image slicing, preserves original files
- **Lakehouse**: Fast SQL analytics, time-slicing, multi-run queries

---

## References

- [Tiled Documentation](https://blueskyproject.io/tiled/)
- [Apache Iceberg](https://iceberg.apache.org/)
- [Apache Parquet](https://parquet.apache.org/)
- [NeXus Format](https://www.nexusformat.org/)
