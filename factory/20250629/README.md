# CogPrime Distributed Cognitive Framework  
*Unified release integrating AtomSpace, node9 namespace and mem0 memory*

---

## 1  Overview  
This repository contains the next-generation implementation of the CogPrime Artificial General Intelligence architecture.  
The codebase now ships with three heavyweight subsystems merged as first-class modules:

| Layer | Purpose |
|-------|---------|
| **AtomSpace (Hyperon/DAS)** | Distributed hypergraph knowledge base & rule engine |
| **node9** | Plan 9 / Inferno inspired micro-kernel that offers a network-transparent namespace for distributed cognition |
| **mem0** | High-level memory stack providing vector semantic search, graph memory, persistence & LLM-powered fact extraction |

CogPrime orchestrates these layers to realise a scalable “distributed OS for distributed cognition”.

---

## 2  Architecture

```
┌──────────────────────────────────────────┐
│      User / Application Interfaces       │
│  •  Python API  •  Lua Shell  •  REST    │
└──────────────┬───────────────────────────┘
               │
┌──────────────▼───────────────────────────┐
│           CogPrime Core Cycle            │
│  perceive → reason → act → learn loop    │
└──────────────┬───────────────┬───────────┘
               │               │
      ┌────────▼────────┐ ┌────▼─────┐
      │  AtomSpace      │ │  mem0     │
      │  (Hyperon DAS)  │ │  Memory   │
      └────────┬────────┘ └────┬─────┘
               │ GraphQL / gRPC │
               │ Pattern Match  │
┌──────────────▼───────────────────────────┐
│         node9 Distributed Namespace      │
│  • Styx protocol exposes AtomSpace files │
│  • Virtual procs host CogPrime modules   │
└──────────────────────────────────────────┘
```

Data and control flow  
1. Cognitive modules run as **node9 vprocs**; each module sees a unified `/cog` namespace.  
2. Knowledge is stored in **AtomSpace**; queries/patterns travel via Styx or gRPC.  
3. **mem0** attaches persistence and vector retrieval to AtomSpace atoms, storing embeddings & history in pluggable back-ends.  
4. CogPrime core binds everything into a single cognitive loop.

---

## 3  Key Features

* **Distributed AtomSpace** ‑ automatic sharding & replication using Hyperon DAS.  
* **Plan 9 style namespace** ‑ every cognitive resource is a file; remote transparency.  
* **Unified memory stack**  
  * Vector DB (Chroma, Pinecone, etc.)  
  * Neo4j graph memory  
  * SQLite history log  
  * LLM-driven entity / relation extraction  
* **PyTorch integration** for neural sub-modules.  
* **LuaJIT** bindings for ultra-low latency rule execution inside node9 kernel.  
* **One-command deployment** with Docker / Nix.

---

## 4  Installation

Prerequisites  
* Linux/macOS, 16 GB RAM  
* Python ≥ 3.10, Rust toolchain, CMake  
* Docker (optional for quick start)

```
# clone repo
git clone https://github.com/your-org/cogprime.git
cd cogprime

# build node9 kernel & AtomSpace
make all                    # or ./scripts/build.sh

# create virtualenv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# launch distributed stack (single-node demo)
./scripts/run_demo.sh
```

---

## 5  Basic Usage

Python (high-level)

```python
from cogprime import CogPrimeCore
from cogprime.memory import Memory         # mem0 wrapper
from cogprime.atomspace import AtomSpace   # Hyperon client

core   = CogPrimeCore()
memory = Memory()          # semantic + graph + history
space  = AtomSpace()       # connect to local DAS server

core.update_goals("explore-environment")

for step in range(100):
    sensory = get_simulated_input()
    reward  = compute_reward()
    action  = core.cognitive_cycle(sensory, reward)
    print("Agent action:", action)
```

Lua (low-level, inside node9 shell)

```lua
local sys = require("sys")          -- node9 syscall layer
local as  = require("atomspace")    -- FFI binding

local a = as.NewLink("Concept", "Hello", "World")
as.Commit(a)
print("Atoms in space:", as.Count())
```

---

## 6  Component Descriptions

| Directory | Description |
|-----------|-------------|
| `src/` | CogPrime Python modules (core + perception/learning/etc.) |
| `integrations/node9_core/` | node9 kernel, Styx server, Lua runtime |
| `integrations/mem0/` | mem0 library sources (vector DB, graph memory) |
| `integrations/atomspace/` | C++ Hyperon DAS client & server wrappers |
| `docker/` | Dockerfiles for all-in-one cluster |
| `scripts/` | Build, run, and migration scripts |

---

## 7  Roadmap

| Milestone | Target |
|-----------|--------|
| **M1** AtomSpace proto-driver | done ✓ |
| **M2** node9 Styx bridge exposes `/cog/space` filesystem | WIP |
| **M3** mem0 vector search integrated into reasoning heuristics | 2025-Q1 |
| **M4** Multi-node DAS replication test-bed | 2025-Q2 |
| **M5** Cognitive service mesh & load balancer | 2025-Q3 |
| **M6** Embodied agent demo in 3-D environment | 2025-Q4 |
| **M7** Public API & SDK (Python, Lua, Rust) | 2026 |

Contributions and issue reports are welcome. Please read `CONTRIBUTING.md` (to be added) for coding standards and workflow guidelines.

---

© 2025 CogPrime Project — Licensed under AGPL-3.0
