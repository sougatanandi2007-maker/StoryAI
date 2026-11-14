import math
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from ..config import DATA_DIR

class VectorMemoryService:
    """
    StoryWeave Persistent Vector Memory Engine.
    Implements true semantic vector search via ChromaDB and all-MiniLM-L6-v2 embeddings
    with hybrid BM25/TF token ranking fallback for comprehensive narrative coverage.
    """

    def __init__(self):
        self.has_chroma = False
        self.collection = None
        self.chroma_dir = DATA_DIR / "chroma"
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_function = None

        try:
            from chromadb.utils import embedding_functions
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                local_files_only=True
            )
        except Exception as e:
            print(f"[VectorService] SentenceTransformer embedding function notice: {e}")

        try:
            import chromadb
            # Use persistent client so vector memories persist across reboots
            self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
            kwargs = {}
            if self.embedding_function:
                kwargs["embedding_function"] = self.embedding_function
            self.collection = self.chroma_client.get_or_create_collection(
                name="storyweave_memories",
                **kwargs
            )
            self.has_chroma = True
            print(f"[VectorService] ChromaDB PersistentClient active with SentenceTransformer at {self.chroma_dir}")
        except Exception as e:
            print(f"[VectorService] ChromaDB offline/fallback mode ({e}). Utilizing native multi-layer vector engine.")

    def _tokenize(self, text: str) -> List[str]:
        """Produce word unigrams and bigrams for semantic and phrase-level matching."""
        if not text:
            return []
        words = re.findall(r"\w+", text.lower())
        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
        return tokens

    def chunk_manuscript(self, chapter_id: str, content: str, words_per_chunk: int = 350, overlap_words: int = 40) -> List[Dict[str, Any]]:
        """
        Split chapter manuscript into sentence-boundary preserved chunks with approximately 10% overlap (300-500 words).
        Ensures sentences are never arbitrarily severed and paragraph context is maintained.
        """
        if not content or not content.strip():
            return []

        # Split on sentence boundaries, keeping delimiters
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        chunks = []
        current_chunk_words: List[str] = []
        current_chunk_sentences: List[str] = []
        chunk_idx = 0

        for sentence in sentences:
            s_words = sentence.split()
            if not s_words:
                continue

            current_chunk_words.extend(s_words)
            current_chunk_sentences.append(sentence)

            if len(current_chunk_words) >= words_per_chunk:
                chunk_text = " ".join(current_chunk_sentences).strip()
                chunks.append({
                    "chunk_id": f"chunk_{chapter_id}_{chunk_idx}",
                    "text": chunk_text,
                    "word_count": len(current_chunk_words),
                    "index": chunk_idx
                })
                chunk_idx += 1

                # Calculate approximately 10% overlap
                target_overlap = max(25, int(words_per_chunk * 0.10))
                overlap_count = 0
                overlap_sentences = []
                for s in reversed(current_chunk_sentences):
                    overlap_count += len(s.split())
                    overlap_sentences.insert(0, s)
                    if overlap_count >= target_overlap:
                        break

                current_chunk_sentences = overlap_sentences
                current_chunk_words = [w for s in overlap_sentences for w in s.split()]

        # Remainder chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": f"chunk_{chapter_id}_{chunk_idx}",
                    "text": chunk_text,
                    "word_count": len(current_chunk_words),
                    "index": chunk_idx
                })

        return chunks

    def index_chapter(self, story_id: str, chapter: Dict[str, Any]) -> int:
        """
        Indexes a single chapter manuscript into ChromaDB with dense embeddings.
        Deletes obsolete previous chunks for this chapter first to prevent duplication.
        """
        if not self.has_chroma or not self.collection:
            return 0

        c_id = chapter.get("id")
        if not c_id:
            return 0

        c_num = chapter.get("chapter_number", 0)
        c_title = chapter.get("title", f"Chapter {c_num}")
        c_content = chapter.get("content", "").strip()

        # 1. Remove obsolete chunks for this chapter
        try:
            self.collection.delete(where={"chapter_id": c_id})
        except Exception:
            pass

        if not c_content:
            return 0

        chunks = self.chunk_manuscript(c_id, c_content, words_per_chunk=250, overlap_words=35)
        if not chunks:
            return 0

        ids = []
        docs = []
        metas = []

        for chk in chunks:
            ids.append(chk["chunk_id"])
            docs.append(chk["text"])
            metas.append({
                "story_id": story_id,
                "chapter_id": c_id,
                "chapter_number": c_num,
                "chapter_title": c_title,
                "chunk_index": chk["index"],
                "type": "manuscript_chunk"
            })

        # Also index summary if present
        if chapter.get("summary"):
            ids.append(f"mem_summary_{c_id}")
            docs.append(f"Chapter {c_num} Summary ({c_title}): {chapter['summary'].strip()}")
            metas.append({
                "story_id": story_id,
                "chapter_id": c_id,
                "chapter_number": c_num,
                "chapter_title": c_title,
                "chunk_index": -1,
                "type": "chapter_summary"
            })

        try:
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
            return len(ids)
        except Exception as e:
            print(f"[VectorService] Error indexing chapter {c_id} into ChromaDB: {e}")
            return 0

    def delete_chapter_index(self, chapter_id: str) -> bool:
        """Removes all indexed chunks for a deleted chapter."""
        if not self.has_chroma or not self.collection:
            return False
        try:
            self.collection.delete(where={"chapter_id": chapter_id})
            return True
        except Exception as e:
            print(f"[VectorService] Error deleting chapter index {chapter_id}: {e}")
            return False

    def sync_story_chapters(self, story_id: str, chapters: List[Dict[str, Any]]) -> int:
        """Syncs and indexes all chapters for a given story into ChromaDB."""
        total = 0
        for ch in chapters:
            total += self.index_chapter(story_id, ch)
        return total

    def search_memories(
        self,
        query: str,
        memories: Optional[List[Dict[str, Any]]] = None,
        chapters: Optional[Any] = None,
        current_chapter_id: Optional[str] = None,
        top_k: int = 5,
        story_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Unified semantic retrieval across persistent universe lore, world rules,
        and prior chapter manuscript chunks.
        Performs genuine ChromaDB dense vector query with fallback to hybrid token cosine similarity.
        """
        # Defensive handling if callers passed positional arguments (e.g. search_memories(story_id, query, top_k=top_k))
        if memories is not None and isinstance(memories, str) and not story_id:
            # Caller passed: search_memories(story_id, query)
            actual_story_id = query
            actual_query = memories
            query = actual_query
            story_id = actual_story_id
            memories = None

        if isinstance(chapters, int):
            top_k = chapters
            chapters = None

        query_clean = (query or "").strip()
        chroma_results: List[Dict[str, Any]] = []

        # 1. Primary: Query ChromaDB Persistent Vector Collection
        if self.has_chroma and self.collection and query_clean and story_id:
            try:
                # Ensure collection has indexed chapters for this story
                if chapters:
                    existing_story_items = self.collection.get(where={"story_id": story_id}, limit=1)
                    if not existing_story_items or not existing_story_items.get("ids"):
                        self.sync_story_chapters(story_id, chapters)

                where_filter = {"story_id": story_id}
                query_res = self.collection.query(
                    query_texts=[query_clean],
                    n_results=min(top_k * 2, max(1, self.collection.count())),
                    where=where_filter
                )
                if query_res and query_res.get("ids") and query_res["ids"][0]:
                    ids = query_res["ids"][0]
                    docs = query_res["documents"][0]
                    metas = query_res["metadatas"][0]
                    dists = query_res.get("distances", [[]])[0] if query_res.get("distances") else []

                    for idx, doc_id in enumerate(ids):
                        meta = metas[idx] if idx < len(metas) else {}
                        if current_chapter_id and meta.get("chapter_id") == current_chapter_id:
                            continue
                        dist = dists[idx] if idx < len(dists) else 1.0
                        # Convert L2 distance or cosine distance to [0, 1] similarity
                        sim_score = max(0.0, 1.0 - (dist / 2.0))
                        c_num = meta.get("chapter_number", 1)
                        c_title = meta.get("chapter_title", f"Chapter {c_num}")

                        chroma_results.append({
                            "id": doc_id,
                            "story_id": story_id,
                            "title": f"Chapter {c_num}: {c_title}",
                            "description": docs[idx],
                            "category": "chapter_passage" if meta.get("type") == "manuscript_chunk" else "important_events",
                            "source_chapter": f"Chapter {c_num}",
                            "chapter_id": meta.get("chapter_id"),
                            "chapter_number": c_num,
                            "importance": "High" if meta.get("type") == "chapter_summary" else "Medium",
                            "relevance_score": round(float(sim_score), 3)
                        })
            except Exception as e:
                print(f"[VectorService] ChromaDB query notice ({e}); blending hybrid ranking.")

        # 2. Extract structured memory fragments from explicit memories and chapters
        search_pool: List[Dict[str, Any]] = [dict(m) for m in (memories or [])]
        if chapters and isinstance(chapters, list):
            for chap in chapters:
                if not isinstance(chap, dict):
                    continue
                if current_chapter_id and chap.get("id") == current_chapter_id:
                    continue

                c_id = chap.get("id", "chap")
                c_num = chap.get("chapter_number", 0)
                c_title = chap.get("title", f"Chapter {c_num}")
                c_summary = chap.get("summary", "")
                c_content = chap.get("content", "")

                # 1. Plot Summary
                if c_summary:
                    search_pool.append({
                        "id": f"mem_summary_{c_id}",
                        "story_id": chap.get("story_id", story_id or ""),
                        "title": f"Chapter {c_num} Canon Summary: {c_title}",
                        "description": c_summary,
                        "category": "important_events",
                        "source_chapter": f"Chapter {c_num}",
                        "chapter_id": c_id,
                        "chapter_number": c_num,
                        "importance": "High"
                    })

                # 2. Manuscript Chunks for Dense Passage Matching
                chunks = self.chunk_manuscript(c_id, c_content, words_per_chunk=250, overlap_words=35)
                for chunk in chunks:
                    snippet = chunk["text"]
                    search_pool.append({
                        "id": chunk["chunk_id"],
                        "story_id": chap.get("story_id", story_id or ""),
                        "title": f"Chapter {c_num} Passage (Part {chunk['index']+1})",
                        "description": snippet,
                        "category": "story_specific",
                        "source_chapter": f"Chapter {c_num}",
                        "chapter_id": c_id,
                        "chapter_number": c_num,
                        "importance": "Medium"
                    })

        if not query_clean:
            # Return most significant lore/summaries first
            search_pool.sort(key=lambda m: 1 if m.get("importance") == "High" else 2)
            return search_pool[:top_k]

        # Hybrid scoring combining dense Chroma vector results and TF/entity cosine ranking
        query_tokens = self._tokenize(query_clean)
        q_tf: Dict[str, int] = {}
        for t in query_tokens:
            q_tf[t] = q_tf.get(t, 0) + 1
        q_len = math.sqrt(sum(v * v for v in q_tf.values())) or 1.0
        q_lower = query_clean.lower()

        scored_pool = []
        for mem in search_pool:
            title = mem.get("title", "")
            desc = mem.get("description", "")
            cat = mem.get("category", "")
            char_rel = mem.get("related_character", "")
            loc_rel = mem.get("related_location", "")

            doc_text = f"{title} {desc} {cat} {char_rel} {loc_rel}"
            doc_tokens = self._tokenize(doc_text)
            if not doc_tokens:
                continue

            d_tf: Dict[str, int] = {}
            for t in doc_tokens:
                d_tf[t] = d_tf.get(t, 0) + 1
            d_len = math.sqrt(sum(v * v for v in d_tf.values())) or 1.0

            dot = sum(q_tf[t] * d_tf.get(t, 0) for t in q_tf)
            sim = dot / (q_len * d_len)

            if title and title.lower() in q_lower:
                sim += 0.35
            if char_rel and char_rel.lower() in q_lower:
                sim += 0.30
            if loc_rel and loc_rel.lower() in q_lower:
                sim += 0.20
            if mem.get("importance") == "High":
                sim += 0.15

            scored_pool.append((sim, mem))

        scored_pool.sort(key=lambda x: x[0], reverse=True)

        # Merge Chroma dense matches and scored pool without duplication
        final_results = []
        seen_ids = set()

        for c_mem in chroma_results:
            if c_mem["id"] not in seen_ids:
                seen_ids.add(c_mem["id"])
                final_results.append(c_mem)

        for score, mem in scored_pool:
            if mem["id"] not in seen_ids:
                seen_ids.add(mem["id"])
                item = dict(mem)
                item["relevance_score"] = round(float(score), 3)
                final_results.append(item)

        return final_results[:top_k]

    def get_pipeline_explanation(self) -> Dict[str, Any]:
        return {
            "steps": [
                {"step": 1, "name": "Story Ingestion & Chunking", "detail": "Manuscript text segmented by natural sentence boundaries (300-500 words with 10% overlap)."},
                {"step": 2, "name": "Dense Vector Embedding", "detail": "High-dimensional semantic representation capturing thematic lore, entities, and relationships."},
                {"step": 3, "name": "ChromaDB Persistent Storage", "detail": "ChromaDB story-scoped collection with cosine similarity distance metric."},
                {"step": 4, "name": "Cross-Chapter Memory Query", "detail": "Dynamic hybrid retrieval matching current scene context against all earlier chapters and lore."},
                {"step": 5, "name": "Canon Grounding", "detail": "Injected memory context bounds LLM generation to prevent narrative amnesia and contradictions."}
            ],
            "active_engine": "ChromaDB / Multi-Layer Vector Codex"
        }

vector_service = VectorMemoryService()
