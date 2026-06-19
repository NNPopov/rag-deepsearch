"""Cross-lingual search (red): question in any language → search in the corpus language, answer in the question language.

Problem (Rag_query_architecture.md §5): the corpus is English, BM25 (`content @@@ q`) is a LEXICAL
full-text search → a Russian/Chinese `bm25_query` won't match any English token (0 hits,
half the hybrid is dead); vector degrades cross-lingually. Solution WITHOUT a new step or extra LLM calls:
  • PLAN  — `vector_queries`/`bm25_queries` are emitted in the CORPUS LANGUAGE (`corpus_language`, default English),
    regardless of the question language (we translate the concepts); `sub_questions` — for reasoning, language doesn't matter;
  • REFLECT — proposed `new_*_queries` are also in the corpus language (they go back into the searchers);
  • SYNTH  — answer in the QUESTION LANGUAGE (sources are English, the answer to the user is in their language);
  • `corpus_language` — a config parameter (`[corpus] language`), read in the composition root and
    threaded into `DeepSearch` → `plan`/`reflect` via DI. Contracts don't change (queries — list[str]).

Pure unit (CLAUDE.md §5): mock gateway, we check WHAT went into the prompt. RED until implementation.
"""
from __future__ import annotations

from rag.contracts import (
    ExpandedSection,
    Reflection,
    RetrievedChunk,
    SearchPlan,
)


class FakeGateway:
    """Mock gateway: completion returns the given JSON per task; records (task, messages)."""

    def __init__(self, *, responses=None, stream_tokens=None):
        self.responses = responses or {}
        self.stream_tokens = stream_tokens or ["x"]
        self.completion_calls: list[tuple[str, list[dict]]] = []
        self.stream_calls: list[tuple[str, list[dict]]] = []

    def completion(self, *, task, messages, **kw):
        self.completion_calls.append((task, messages))
        return self.responses[task]

    def completion_stream(self, *, task, messages, **kw):
        self.stream_calls.append((task, messages))
        yield from self.stream_tokens


def _prompt(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages)


def _plan_obj() -> SearchPlan:
    return SearchPlan(
        sub_questions=["s"], vector_queries=["replication"], bm25_queries=["replication"]
    )


# --- PLAN: queries in the corpus language --------------------------------------------------

def test_plan_instructs_corpus_language_for_queries():
    from rag.steps import plan

    gw = FakeGateway(responses={"plan": _plan_obj().model_dump_json()})
    # question in Russian, corpus in English
    plan("Как работает отказоустойчивость лидера?", gateway=gw, corpus_language="English")

    prompt = _prompt(gw.completion_calls[0][1])
    assert "English" in prompt                       # corpus language named in the prompt
    # the instruction concerns the search queries (vector/bm25), not the sub-questions
    assert "vector_queries" in prompt and "bm25_queries" in prompt


def test_plan_corpus_language_is_parametrized():
    from rag.steps import plan

    gw = FakeGateway(responses={"plan": _plan_obj().model_dump_json()})
    plan("question", gateway=gw, corpus_language="Russian")
    assert "Russian" in _prompt(gw.completion_calls[0][1])


def test_plan_corpus_language_defaults_to_english():
    from rag.steps import plan

    gw = FakeGateway(responses={"plan": _plan_obj().model_dump_json()})
    plan("question", gateway=gw)                      # without explicit corpus_language
    assert "English" in _prompt(gw.completion_calls[0][1])


# --- REFLECT: new queries also in the corpus language ------------------------------------

def test_reflect_instructs_corpus_language_for_new_queries():
    from rag.steps import reflect

    raw = Reflection(is_sufficient=False, gaps=["g"], new_bm25_queries=["x"]).model_dump_json()
    gw = FakeGateway(responses={"reflect": raw})
    chunks = [
        RetrievedChunk(
            chunk_id=1, content="english evidence", document_id=1,
            section_id=1, source="DDIA › 6", score=1.0,
        )
    ]
    reflect("вопрос по-русски", _plan_obj(), chunks, gateway=gw, corpus_language="English")

    prompt = _prompt(gw.completion_calls[0][1])
    assert "English" in prompt
    assert "new_vector_queries" in prompt and "new_bm25_queries" in prompt


# --- SYNTH: answer in the question language ---------------------------------------------------

def test_synth_instructs_answer_in_question_language():
    from rag.steps import synth_stream

    gw = FakeGateway(stream_tokens=["x"])
    blocks = [ExpandedSection(section_id=1, document_id=1, full_text="english source", source="DDIA › 6")]
    list(synth_stream("Как работает репликация?", blocks, gateway=gw))

    prompt = _prompt(gw.stream_calls[0][1]).lower()
    # SYNTH must instruct answering in the question language (sources are English, the answer to the user is in their language)
    assert "language" in prompt and "question" in prompt


# --- LOOP: corpus_language threaded into plan and reflect ----------------------------

def test_deepsearch_threads_corpus_language_into_plan_and_reflect():
    from rag.loop import DeepSearch

    plan_json = SearchPlan(
        sub_questions=["s"], vector_queries=["q1"], bm25_queries=[]
    ).model_dump_json()
    reflect_json = Reflection(is_sufficient=True).model_dump_json()

    class LoopGateway(FakeGateway):
        def completion(self, *, task, messages, **kw):
            self.completion_calls.append((task, messages))
            return plan_json if task == "plan" else reflect_json

    class FakeSearcher:
        def __init__(self, hits): self.hits = hits

        def search(self, query, *, filters=None):
            return [c.model_copy() for c in self.hits]

        def search_many(self, queries, *, filters=None):
            return [self.search(q, filters=filters) for q in queries]

    class FakeExpander:
        def expand(self, chunks):
            return [
                ExpandedSection(section_id=c.section_id, document_id=1,
                                full_text="ft", source=c.source)
                for c in chunks
            ]

    hit = RetrievedChunk(
        chunk_id=1, content="c", document_id=1, section_id=10,
        source="DDIA › 6", vector=[1.0, 0.0],
    )
    gw = LoopGateway(stream_tokens=["ans"])
    ds = DeepSearch(
        vector_searcher=FakeSearcher([hit]),
        bm25_searcher=FakeSearcher([]),
        expander=FakeExpander(),
        gateway=gw,
        corpus_language="English",
    )
    ds.run("Вопрос на русском")

    by_task = {task: msgs for task, msgs in gw.completion_calls}
    assert "English" in _prompt(by_task["plan"])      # corpus_language reached PLAN
    assert "English" in _prompt(by_task["reflect"])   # and REFLECT
