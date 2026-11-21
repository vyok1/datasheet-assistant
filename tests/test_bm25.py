import json

import pytest

from datasheet_assistant.retrieval.bm25 import BM25Retriever, tokenize


@pytest.fixture
def chunks_file(tmp_path):
    data = {
        "chunks": [
            {
                "chunk_id": "gpio-1",
                "doc_id": "stm32",
                "page_number": 269,
                "section_heading": "GPIO introduction",
                "text": "GPIOx_MODER configures GPIO pin modes.",
            },
            {
                "chunk_id": "dma-1",
                "doc_id": "stm32",
                "page_number": 306,
                "section_heading": "DMA description",
                "text": "DMA supports peripheral-to-memory transfers.",
            },
            {
                "chunk_id": "spi-1",
                "doc_id": "stm32",
                "page_number": 871,
                "section_heading": "SPI description",
                "text": "MISO receives data and MOSI transmits data.",
            },
        ]
    }

    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_tokenize_preserves_register_identifiers():
    assert tokenize("GPIOx_MODER configures pin modes") == [
        "gpiox_moder", "configures", "pin", "modes"
    ]


def test_retriever_returns_expected_fields(chunks_file):
    retriever = BM25Retriever(chunks_file)
    results = retriever.search("GPIOx_MODER", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "gpio-1"
    assert results[0]["page_number"] == 269
    assert results[0]["doc_id"] == "stm32"
    assert "score" in results[0]
    assert "text" in results[0]


def test_top_k_limits_results(chunks_file):
    retriever = BM25Retriever(chunks_file)
    assert len(retriever.search("data", top_k=2)) == 2


def test_blank_query_is_rejected(chunks_file):
    retriever = BM25Retriever(chunks_file)

    with pytest.raises(ValueError):
        retriever.search("   ")


def test_invalid_top_k_is_rejected(chunks_file):
    retriever = BM25Retriever(chunks_file)

    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            retriever.search("GPIO", top_k=value)


def test_empty_chunks_are_rejected(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text('{"chunks": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="No chunks"):
        BM25Retriever(path)


# --------------------------------------------------------------------------
# expand_context() tests
# --------------------------------------------------------------------------

def _make_chunk(doc_id, page, index, heading=None):
    return {
        "chunk_id": f"{doc_id}-p{page}-c{index}",
        "doc_id": doc_id,
        "page_number": page,
        "section_heading": heading,
        "text": f"text {doc_id} p{page} c{index}",
    }


def _make_result(chunk, score):
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "page_number": chunk["page_number"],
        "section_heading": chunk["section_heading"],
        "score": score,
        "text": chunk["text"],
    }


def _ids(results):
    return [result["chunk_id"] for result in results]


# Original order: doc A page 1 has 4 chunks, A page 2 has 2, doc B page 1 has 2
# (page 1 of two different documents).
EXPAND_CHUNKS = (
    [_make_chunk("A", 1, i, "1.1 Intro") for i in range(4)]
    + [_make_chunk("A", 2, i) for i in range(2)]
    + [_make_chunk("B", 1, i) for i in range(2)]
)
PAGE_A1 = EXPAND_CHUNKS[0:4]
PAGE_A2 = EXPAND_CHUNKS[4:6]
PAGE_B1 = EXPAND_CHUNKS[6:8]


@pytest.fixture
def make_retriever(tmp_path):
    """Build a real BM25Retriever from a list of chunk dicts."""

    def build(chunks):
        path = tmp_path / "expand_chunks.json"
        path.write_text(json.dumps({"chunks": chunks}), encoding="utf-8")
        return BM25Retriever(path)

    return build


def test_expand_retrieved_chunk_with_one_neighbor(make_retriever):
    retriever = make_retriever(PAGE_A2)  # a page with exactly two chunks
    out = retriever.expand_context([_make_result(PAGE_A2[0], 7.5)], window=1)

    assert _ids(out) == ["A-p2-c0", "A-p2-c1"]
    assert out[0]["expanded"] is False and out[0]["score"] == 7.5
    assert out[1]["expanded"] is True and out[1]["score"] == 0.0


def test_expand_chunk_at_beginning_of_page_gets_only_following_neighbor(make_retriever):
    retriever = make_retriever(EXPAND_CHUNKS)
    out = retriever.expand_context([_make_result(PAGE_A1[0], 3.0)], window=1)

    assert _ids(out) == ["A-p1-c0", "A-p1-c1"]


def test_expand_chunk_at_end_of_page_gets_only_preceding_neighbor(make_retriever):
    retriever = make_retriever(EXPAND_CHUNKS)
    out = retriever.expand_context([_make_result(PAGE_A1[3], 3.0)], window=1)

    assert _ids(out) == ["A-p1-c2", "A-p1-c3"]
    assert [result["expanded"] for result in out] == [True, False]


def test_expand_middle_chunk_gets_both_neighbors_and_window_is_clamped(make_retriever):
    retriever = make_retriever(EXPAND_CHUNKS)
    hit = [_make_result(PAGE_A1[1], 1.0)]

    assert _ids(retriever.expand_context(hit, window=1)) == ["A-p1-c0", "A-p1-c1", "A-p1-c2"]
    assert _ids(retriever.expand_context(hit, window=2)) == [
        "A-p1-c0", "A-p1-c1", "A-p1-c2", "A-p1-c3"
    ]
    assert len(retriever.expand_context(hit, window=50)) == 4  # never beyond the page


def test_expand_overlapping_neighbors_are_not_duplicated(make_retriever):
    retriever = make_retriever(EXPAND_CHUNKS)
    results = [_make_result(PAGE_A1[0], 5.0), _make_result(PAGE_A1[2], 4.0)]  # c1 neighbors both

    out = retriever.expand_context(results, window=1)

    assert _ids(out) == ["A-p1-c0", "A-p1-c1", "A-p1-c2", "A-p1-c3"]
    assert len(set(_ids(out))) == len(out)
    assert {r["chunk_id"]: r["expanded"] for r in out} == {
        "A-p1-c0": False,
        "A-p1-c1": True,
        "A-p1-c2": False,
        "A-p1-c3": True,
    }


def test_expand_retrieved_chunk_that_is_also_a_neighbor_keeps_its_score(make_retriever):
    retriever = make_retriever(EXPAND_CHUNKS)
    # c0 is a neighbor of c1 AND is retrieved itself.
    results = [_make_result(PAGE_A1[1], 2.0), _make_result(PAGE_A1[0], 9.0)]

    out = {r["chunk_id"]: r for r in retriever.expand_context(results, window=1)}

    assert out["A-p1-c0"]["score"] == 9.0 and out["A-p1-c0"]["expanded"] is False
    assert out["A-p1-c1"]["score"] == 2.0 and out["A-p1-c1"]["expanded"] is False


def test_expand_duplicate_retrieved_results_appear_once_first_wins(make_retriever):
    retriever = make_retriever(EXPAND_CHUNKS)
    results = [_make_result(PAGE_A1[0], 5.0), _make_result(PAGE_A1[0], 1.0)]

    out = retriever.expand_context(results, window=1)

    assert _ids(out) == ["A-p1-c0", "A-p1-c1"]
    assert out[0]["score"] == 5.0


def test_expand_does_not_cross_pages_or_documents(make_retriever):
    retriever = make_retriever(EXPAND_CHUNKS)

    # Last chunk of A p1 is followed by the first chunk of A p2 in self.chunks: must not cross.
    out = retriever.expand_context([_make_result(PAGE_A1[3], 1.0)], window=3)
    assert _ids(out) == ["A-p1-c0", "A-p1-c1", "A-p1-c2", "A-p1-c3"]

    # First chunk of A p2 is preceded by A p1's last chunk: must not cross.
    out = retriever.expand_context([_make_result(PAGE_A2[0], 1.0)], window=3)
    assert _ids(out) == ["A-p2-c0", "A-p2-c1"]

    # Same page number in a different document is a different page.
    out = retriever.expand_context([_make_result(PAGE_B1[0], 1.0)], window=3)
    assert _ids(out) == ["B-p1-c0", "B-p1-c1"]


def test_expand_orders_chunks_within_page_and_keeps_page_retrieval_order(make_retriever):
    retriever = make_retriever(EXPAND_CHUNKS)
    results = [
        _make_result(PAGE_B1[1], 9.0),  # best hit first
        _make_result(PAGE_A1[2], 5.0),
        _make_result(PAGE_A1[0], 1.0),
    ]

    out = retriever.expand_context(results, window=1)

    assert _ids(out) == [
        "B-p1-c0", "B-p1-c1",                                 # B page 1 first (best result)
        "A-p1-c0", "A-p1-c1", "A-p1-c2", "A-p1-c3",           # then A page 1, in chunk order
    ]


def test_expand_preserves_fields_and_neighbors_have_the_same_shape(make_retriever):
    retriever = make_retriever(EXPAND_CHUNKS)
    out = retriever.expand_context([_make_result(PAGE_A1[0], 3.25)], window=1)

    expected_keys = {
        "chunk_id", "doc_id", "page_number", "section_heading", "score", "text", "expanded",
    }
    assert all(set(result) == expected_keys for result in out)
    assert out[0] == {**_make_result(PAGE_A1[0], 3.25), "expanded": False}
    assert out[1]["section_heading"] == "1.1 Intro"
    assert out[1]["text"] == PAGE_A1[1]["text"]


def test_expand_does_not_modify_input_results(make_retriever):
    retriever = make_retriever(EXPAND_CHUNKS)
    result = _make_result(PAGE_A1[0], 3.0)
    snapshot = dict(result)

    retriever.expand_context([result], window=1)

    assert result == snapshot
    assert "expanded" not in result


def test_expand_empty_results_give_empty_list(make_retriever):
    assert make_retriever(EXPAND_CHUNKS).expand_context([], window=1) == []


def test_expand_default_window_is_one(make_retriever):
    retriever = make_retriever(EXPAND_CHUNKS)
    out = retriever.expand_context([_make_result(PAGE_A1[1], 1.0)])

    assert _ids(out) == ["A-p1-c0", "A-p1-c1", "A-p1-c2"]


@pytest.mark.parametrize("window", [0, -1, 1.5, 2.0, True, False, None, "1", [1]])
def test_expand_invalid_window_values_are_rejected(make_retriever, window):
    retriever = make_retriever(EXPAND_CHUNKS)

    with pytest.raises(ValueError, match="window must be a positive integer"):
        retriever.expand_context([_make_result(PAGE_A1[0], 1.0)], window=window)


def test_expand_invalid_window_is_rejected_even_for_empty_results(make_retriever):
    with pytest.raises(ValueError):
        make_retriever(EXPAND_CHUNKS).expand_context([], window=0)


def test_expand_unmatched_results_raise_clear_errors(make_retriever):
    retriever = make_retriever(EXPAND_CHUNKS)

    unknown_chunk = {**_make_result(PAGE_A1[0], 1.0), "chunk_id": "A-p1-c99"}
    with pytest.raises(ValueError, match="A-p1-c99"):
        retriever.expand_context([unknown_chunk], window=1)

    unknown_page = {**_make_result(PAGE_A1[0], 1.0), "page_number": 77}
    with pytest.raises(ValueError, match="no chunks found"):
        retriever.expand_context([unknown_page], window=1)


def test_expand_works_on_real_search_output(make_retriever):
    retriever = make_retriever(EXPAND_CHUNKS)
    hits = retriever.search("c2", top_k=1)  # only chunk A-p1-c2 contains the token "c2"
    snapshot = [dict(hit) for hit in hits]

    out = retriever.expand_context(hits, window=1)

    assert [hit["chunk_id"] for hit in hits] == ["A-p1-c2"]
    assert _ids(out) == ["A-p1-c1", "A-p1-c2", "A-p1-c3"]
    assert out[1]["score"] == hits[0]["score"] > 0.0 and out[1]["expanded"] is False
    assert out[0]["score"] == 0.0 and out[2]["score"] == 0.0
    assert hits == snapshot  # search results are left untouched