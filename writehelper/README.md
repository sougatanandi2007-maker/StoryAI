# StoryWeave AI – Persistent Universe Intelligence Writing Studio

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.10+-e92063.svg)](https://docs.pydantic.dev/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-v1.5+-orange.svg)](https://www.trychroma.com/)
[![Google Gemini](https://img.shields.io/badge/Gemini-Flash--Lite-4285F4.svg)](https://ai.google.dev/)
[![Tests](https://img.shields.io/badge/tests-93%20passed%20%7C%200%20failed-brightgreen.svg)]()

> **StoryWeave AI** is an intelligent authoring platform engineered to eliminate narrative drift and continuity contradictions in long-form creative fiction. Unlike generic chatbots that suffer from context window amnesia, StoryWeave combines deep structured extraction, vector memory embeddings, grounded RAG retrieval, and cross-chapter continuity audits to maintain a persistent, evolving understanding of an author's entire fictional universe.

---

## Table of Contents
1. [Core Capabilities & Completed Phases](#core-capabilities--completed-phases)
   - [Phase 1: FastAPI Foundation & Core Setup](#phase-1-fastapi-foundation--core-setup)
   - [Phase 2: Story & Chapter Manuscript Management](#phase-2-story--chapter-manuscript-management)
   - [Phase 3: Gemini AI Story Analysis & Structured Extraction](#phase-3-gemini-ai-story-analysis--structured-extraction)
   - [Phase 4 – Part A: Sentence Transformers & ChromaDB Vector Memory Engine](#phase-4--part-a-sentence-transformers--chromadb-vector-memory-engine)
   - [Phase 4 – Part B: Intelligent Memory Retrieval & Grounded RAG Chat](#phase-4--part-b-intelligent-memory-retrieval--grounded-rag-chat)
   - [Phase 5: Story Continuity & Consistency Engine](#phase-5-story-continuity--consistency-engine)
   - [Phase 6: Character Intelligence & Relationship Graphs](#phase-6-character-intelligence--relationship-graphs)
2. [System Architecture](#system-architecture)
3. [Technology Stack](#technology-stack)
4. [API Reference](#api-reference)
5. [Automated Verification & Test Suite](#automated-verification--test-suite)
6. [Getting Started & Local Installation](#getting-started--local-installation)
7. [Environment Variables](#environment-variables)
8. [Author Sovereignty & Ethical Boundaries](#author-sovereignty--ethical-boundaries)

---

## Core Capabilities & Completed Phases

### Phase 1: FastAPI Foundation & Core Setup
- **Architecture**: Modular FastAPI backend with lifespan management for database connections and structured JSON logging.
- **Health Probing**: Real-time readiness and liveness endpoints (`/health`, `/health/database`, `/health/ai`).
- **Configuration**: Pydantic Settings with strict validation, environment parsing, and sensible defaults.
- **Error Handling**: Centralized domain exception hierarchy mapping business rules to standard HTTP error codes (`400`, `404`, `409`, `422`, `502`, `503`).

### Phase 2: Story & Chapter Manuscript Management
- **Story CRUD**: Full lifecycle management for fictional universes (title, synopsis, genre, world rules overview, custom metadata).
- **Chapter CRUD**: Sequential chapter ingestion ordered by sequence numbers, with pagination and story-level cascade deletes.
- **Original Manuscript Preservation**: Enforces an immutable `original_content` field that permanently records initial submissions, preventing unintentional manuscript loss even when working drafts are edited.
- **Processing Status Lifecycle**: Tracks chapter states (`pending`, `processing`, `completed`, `failed`) and records retry attempt metrics.
- **Repository Pattern**: Thread-safe abstract data layer decoupling domain services from underlying storage engines, allowing seamless future migration to MongoDB Atlas.

### Phase 3: Gemini AI Story Analysis & Structured Extraction
- **Official SDK**: Built on the official `google-genai` SDK using `gemini-flash-lite-latest` with explicit 30s timeout guards.
- **Resilient Retry Policy**: Bounded exponential backoff with full jitter and provider-directed `Retry-After` header inspection to mitigate transient rate limits and temporary model demand spikes (HTTP 503/504).
- **Structured Pydantic Extraction**: Extracts validated schemas from chapter text:
  - *Chapter Summaries*: Multi-paragraph synopsis, key story beats, POV character, and pacing evaluation.
  - *Characters*: Name, aliases, appearance, traits, abilities, goals, motivations, and verbatim manuscript evidence.
  - *Relationships*: Interpersonal dynamics (ally, enemy, mentor, romantic, rival, family) with source quotes.
  - *Events*: Chronological plot points, participants, locations, and narrative significance.
  - *World Rules & Facts*: Explicit statements vs. inferred lore, confidence scores ($0.0 - 1.0$), and category tags.
  - *Timeline*: Chronological anchors, flashback flags, and temporal ambiguity detection.

### Phase 4 – Part A: Sentence Transformers & ChromaDB Vector Memory Engine
- **Local Neural Embeddings**: Powered by Hugging Face `sentence-transformers/all-MiniLM-L6-v2` producing normalized 384-dimensional vector embeddings with singleton thread-safe caching.
- **Sentence-Boundary Chunking**: Natural language chunker preserving sentence integrity within configurable token/character bounds ($300 - 500$ words with $10\%$ overlap).
- **Persistent Vector Storage**: Local ChromaDB persistent storage client (`data/chroma`) with deterministic chunk IDs (`chunk_{chapter_id}_{index}`) ensuring strict idempotency on re-indexing.
- **Multi-Tenant Isolation**: Hard metadata filtering by `user_id` and `story_id` prevents cross-story or cross-author memory leakage.

### Phase 4 – Part B: Intelligent Memory Retrieval & Grounded RAG Chat
- **Story-Scoped Vector Retrieval**: Calculates cosine similarity against indexed memories with configurable `top_k` and `min_relevance` thresholds.
- **Context Builder**: Formats retrieved chunks with chapter provenance, source attribution, and strict character/token budget enforcement ($4,000$ character ceiling).
- **Grounded Gemini Generation**: Answers author questions regarding their universe exclusively based on verified memory chunks, with anti-hallucination uncertainty detection (`insufficient_context` flags).
- **Prompt Injection Defense**: Treating manuscript chunks strictly as passive data inside delimited blocks neutralizes jailbreak attempts embedded in creative prose.

### Phase 5: Story Continuity & Consistency Engine
- **Cross-Chapter Contradiction Audits**: Compares newly submitted or analyzed chapters against all prior established chapter lore and ChromaDB memories.
- **Multi-Category Issue Detection**:
  - `character_consistency`: Physical trait shifts, eye/hair color changes, abrupt personality breaks.
  - `world_rule_consistency`: Violations of established magic systems, universe laws, or metaphysical rules.
  - `timeline_consistency`: Chronological conflicts, impossible travel durations, or order inconsistencies.
  - `relationship_consistency`: Unexplained relationship inversions.
  - `spatial_consistency`: Conflicting geography or impossible simultaneous locations.
- **False-Positive Prevention**:
  - Explicit facts override inferences; inferred facts cannot flag explicit manuscript quotes as errors.
  - Distinguishes intentional character development from contradictions.
  - Flashback awareness prevents non-linear scenes from being marked as chronological errors.
  - Evaluates plausible off-screen explanations before assigning high severity.
- **Author Sovereignty Triage**: Authors can triage reported issues (`open` $\to$ `accepted`, `dismissed`, `resolved`) with custom justification notes. Triage decisions persist across re-checks.

### Phase 6: Character Intelligence & Relationship Graphs
- **Persistent Profiles**: Stores comprehensive character profiles (`character_id`, `story_id`, `name`, `aliases`, `physical_description`, `personality_traits`, `abilities`, `motivations`, `goals`, `source_chapter_ids`).
- **Automated Chapter Synchronization**: Analyzing a chapter automatically updates character profiles, merges new traits, appends cross-chapter appearances, and indexes relationships.
- **Exact Case-Insensitive Deduplication**: Canonical name matching (`"kael"` == `"Kael"` == `"KAEL"`) and alias mapping prevents duplication while strictly preventing spurious merging of distinct characters with similar names (`"Kael"` vs `"Kaela"`).
- **Character REST API**: Endpoints to list characters by story, fetch by ID, update traits/descriptions, and retrieve relationship matrices.
- **RAG Chat Integration**: Questions about characters dynamically incorporate persistent profile records into prompt context alongside semantic vector memories.

---

## System Architecture

```mermaid
flowchart TD
    subgraph Client Layer
        UI["Author / Swagger UI / Web Client"]
    end

    subgraph API Layer (FastAPI v1)
        StoriesAPI["/api/v1/stories"]
        ChaptersAPI["/api/v1/chapters"]
        CharactersAPI["/api/v1/characters"]
        MemoryAPI["/api/v1/memory"]
        RAGAPI["/api/v1/ai/chat"]
        ContinuityAPI["/api/v1/chapters/{id}/continuity-check"]
    end

    subgraph Intelligence Services
        AnalysisSvc["StoryAnalysisService"]
        ContinuitySvc["ContinuityService"]
        RAGSvc["RAGService"]
        CharSvc["CharacterService"]
        EmbeddingSvc["EmbeddingService (all-MiniLM-L6-v2)"]
        ContextBld["ContextBuilder"]
        GeminiSvc["GeminiService (gemini-flash-lite-latest)"]
    end

    subgraph Storage Layer
        StoryRepo[("Story & Chapter Repo (In-Memory / Atlas Ready)")]
        CharRepo[("Character & Relationship Repo")]
        ContinuityRepo[("Continuity Audit Repo")]
        ChromaDB[("ChromaDB Vector Store (Local Persistent)")]
    end

    UI --> API Layer
    StoriesAPI --> StoryRepo
    ChaptersAPI --> AnalysisSvc
    AnalysisSvc --> GeminiSvc
    AnalysisSvc --> CharSvc
    CharSvc --> CharRepo
    ChaptersAPI --> EmbeddingSvc
    EmbeddingSvc --> ChromaDB
    RAGAPI --> RAGSvc
    RAGSvc --> ChromaDB
    RAGSvc --> CharRepo
    RAGSvc --> ContextBld
    ContextBld --> GeminiSvc
    ContinuityAPI --> ContinuitySvc
    ContinuitySvc --> GeminiSvc
    ContinuitySvc --> ContinuityRepo
```

---

## Technology Stack

| Layer | Component | Version / Specification | Purpose |
|---|---|---|---|
| **Language** | Python | 3.11+ / 3.12 | Modern type-hinted core runtime |
| **Framework** | FastAPI | 0.115+ | High-performance asynchronous REST API |
| **Data Validation** | Pydantic v2 | 2.10+ | Schema enforcement & JSON serialization |
| **Server** | Uvicorn | 0.34+ | ASGI web server |
| **AI LLM** | Google Gemini API | `google-genai` SDK | Structured chapter analysis, RAG, continuity |
| **LLM Model** | `gemini-flash-lite-latest` | 1M Context Window | Low-latency, cost-effective reasoning |
| **Embeddings** | Sentence Transformers | `all-MiniLM-L6-v2` | 384-dimensional dense semantic vectors |
| **Vector DB** | ChromaDB | 1.5+ Persistent | Story-isolated semantic memory storage |
| **Document Store** | Repository Pattern | Thread-Safe In-Memory | MongoDB Atlas-ready decoupled persistence |
| **Testing** | Pytest & pytest-asyncio | 8.3+ | Unit, regression, and integration testing |

---

## API Reference

### Health & System Status
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Core application health and uptime |
| `GET` | `/health/database` | MongoDB and ChromaDB connectivity status |
| `GET` | `/health/ai` | Gemini API and Sentence Transformers model status |

### Stories (`/api/v1/stories`)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/stories` | Create a new story universe |
| `GET` | `/api/v1/stories` | List stories for the author with pagination |
| `GET` | `/api/v1/stories/{id}` | Retrieve story details and chapter counts |
| `PUT` | `/api/v1/stories/{id}` | Update title, synopsis, genre, or world rules |
| `DELETE` | `/api/v1/stories/{id}` | Safely delete story and cascade-delete its chapters |

### Chapters (`/api/v1/chapters`)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/stories/{id}/chapters` | Ingest manuscript; saves original immutable draft |
| `GET` | `/api/v1/stories/{id}/chapters` | List chapters in story ordered by sequence |
| `GET` | `/api/v1/chapters/{id}` | Get working manuscript and original preserved content |
| `PUT` | `/api/v1/chapters/{id}` | Update draft without overwriting `original_content` |
| `DELETE` | `/api/v1/chapters/{id}` | Delete a chapter |
| `POST` | `/api/v1/chapters/{id}/analyze` | Trigger AI deep chapter intelligence extraction |
| `GET` | `/api/v1/chapters/{id}/analysis` | Retrieve stored structured analysis report |
| `POST` | `/api/v1/chapters/{id}/retry-processing` | Retry failed AI analysis |
| `POST` | `/api/v1/chapters/{id}/index` | Vectorize manuscript into ChromaDB memory |

### Character Intelligence (`/api/v1/characters`)
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/stories/{id}/characters` | List all persistent character profiles in a story |
| `GET` | `/api/v1/characters/{id}` | Get detailed character profile |
| `PUT` | `/api/v1/characters/{id}` | Update traits, abilities, description, goals |
| `GET` | `/api/v1/stories/{id}/relationships` | List observed interpersonal relationship records |

### RAG Universe Chat (`/api/v1/ai/chat`)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/ai/chat` | Query story universe with grounded semantic RAG retrieval |

### Continuity & Consistency Engine (`/api/v1/continuity`)
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/chapters/{id}/continuity-check` | Audit chapter against prior canon for contradictions |
| `GET` | `/api/v1/chapters/{id}/continuity` | Retrieve stored chapter continuity report |
| `GET` | `/api/v1/stories/{id}/continuity` | List all continuity audits across the story universe |
| `PATCH` | `/api/v1/continuity/issues/{id}` | Author triage: set issue to `accepted`, `dismissed`, `resolved` |

---

## Automated Verification & Test Suite

The test suite consists of **93 deterministic automated tests** with 100% pass rate:

```powershell
.\venv\Scripts\python.exe -m pytest tests -m "not live_api" -v
```

```
tests/test_health.py               4 passed [ 4%]
tests/test_stories.py              9 passed [14%]
tests/test_chapters.py             9 passed [24%]
tests/test_story_analysis.py       7 passed [31%]
tests/test_memory.py              18 passed [51%]
tests/test_rag.py                 13 passed [65%]
tests/test_continuity.py          18 passed [84%]
tests/test_characters.py          15 passed [100%]
====================== 93 passed, 3 deselected in 79.50s ======================
```

---

## Getting Started & Local Installation

### 1. Prerequisites
- Python 3.11 or newer
- Git
- Google Gemini API Key ([Google AI Studio](https://aistudio.google.com/))

### 2. Setup Environment
```bash
# Clone the repository
git clone https://github.com/saptarshiboni-arch/writehelper.git
cd writehelper/storyweave-ai/backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Credentials
Copy `.env.example` to `.env` and provide your Gemini API key:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL=gemini-flash-lite-latest
CHROMA_PERSIST_DIRECTORY=./data/chroma
```

### 4. Run Development Server
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** to test via Swagger UI.

---

## Author Sovereignty & Ethical Boundaries
1. **Advisory by Design**: StoryWeave AI never automatically modifies manuscript text, alters character profiles, or overwrites established lore without author consent.
2. **Author Decisions are Canon**: Flagged continuity issues can be dismissed, accepted, or resolved by the author, and their decisions persist across future re-audits.
3. **No Ghostwriting Overrides**: The system operates as a tireless research assistant and continuity editor, preserving the author's unique voice and vision.
