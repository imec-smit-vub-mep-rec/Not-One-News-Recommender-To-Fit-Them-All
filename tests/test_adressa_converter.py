"""Tests for Adressa converter, including session continuity across chunks."""

from pathlib import Path

import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import DatasetConfig
from src.converters.adressa import AdressaConverter


def _make_chunk(user_id, times, url="https://www.adressa.no/nyheter/article-1", session_starts=None):
    """Create a minimal Adressa-style chunk DataFrame."""
    n = len(times)
    session_starts = session_starts if session_starts is not None else [False] * n
    return pd.DataFrame({
        "userId": [user_id] * n,
        "time": times,
        "url": [url] * n,
        "eventId": list(range(1000, 1000 + n)),
        "id": ["art1"] * n,
        "activeTime": [5.0] * n,
        "sessionStart": session_starts,
    })


def test_cross_chunk_session_continuity():
    """User with events in chunk 1 (end) and chunk 2 (start) within timeout gets same session_id."""
    config = DatasetConfig(
        name="adressa",
        input_path="/tmp",
        format="jsonl",
        session_timeout_seconds=1800,
    )
    converter = AdressaConverter(config=config)

    # Chunk 1: user u1 has event at t=1000
    chunk1 = _make_chunk("u1", [1000])
    imp1 = converter._process_chunk(chunk1)
    session_chunk1 = imp1["session_id"].iloc[0]

    # Chunk 2: user u1 has event at t=1000 + 600 (10 min later, within 30 min timeout)
    chunk2 = _make_chunk("u1", [1600])
    imp2 = converter._process_chunk(chunk2)
    session_chunk2 = imp2["session_id"].iloc[0]

    assert session_chunk1 == session_chunk2, (
        f"Session should continue across chunks: {session_chunk1} != {session_chunk2}"
    )


def test_timeout_starts_new_session():
    """Events separated by more than timeout get different session_ids."""
    config = DatasetConfig(
        name="adressa",
        input_path="/tmp",
        format="jsonl",
        session_timeout_seconds=1800,
    )
    converter = AdressaConverter(config=config)

    # Chunk 1: user u1 at t=1000
    chunk1 = _make_chunk("u1", [1000])
    imp1 = converter._process_chunk(chunk1)
    session1 = imp1["session_id"].iloc[0]

    # Chunk 2: user u1 at t=1000 + 2000 (beyond 30 min timeout)
    chunk2 = _make_chunk("u1", [3000])
    imp2 = converter._process_chunk(chunk2)
    session2 = imp2["session_id"].iloc[0]

    assert session1 != session2, (
        f"Timeout should start new session: {session1} == {session2}"
    )
    assert session1 == "u1_1"
    assert session2 == "u1_2"


def test_session_start_overrides_continuation():
    """sessionStart=True starts new session even within timeout."""
    config = DatasetConfig(
        name="adressa",
        input_path="/tmp",
        format="jsonl",
        session_timeout_seconds=1800,
    )
    converter = AdressaConverter(config=config)

    # Chunk 1: user u1 at t=1000
    chunk1 = _make_chunk("u1", [1000])
    imp1 = converter._process_chunk(chunk1)
    session1 = imp1["session_id"].iloc[0]

    # Chunk 2: user u1 at t=1100 (within timeout) but sessionStart=True
    chunk2 = _make_chunk("u1", [1100], session_starts=[True])
    imp2 = converter._process_chunk(chunk2)
    session2 = imp2["session_id"].iloc[0]

    assert session1 != session2
    assert session1 == "u1_1"
    assert session2 == "u1_2"


def test_new_user_in_chunk2_gets_session_one():
    """User appearing for first time in chunk 2 gets session_id starting at 1."""
    config = DatasetConfig(
        name="adressa",
        input_path="/tmp",
        format="jsonl",
        session_timeout_seconds=1800,
    )
    converter = AdressaConverter(config=config)

    # Chunk 1: only user u1
    chunk1 = _make_chunk("u1", [1000])
    converter._process_chunk(chunk1)

    # Chunk 2: new user u2 (never seen before)
    chunk2 = _make_chunk("u2", [2000])
    imp2 = converter._process_chunk(chunk2)
    session_u2 = imp2["session_id"].iloc[0]

    assert session_u2 == "u2_1"


def test_multiple_sessions_per_user_in_chunk():
    """User with timeout gap within a single chunk gets multiple sessions."""
    config = DatasetConfig(
        name="adressa",
        input_path="/tmp",
        format="jsonl",
        session_timeout_seconds=1800,
    )
    converter = AdressaConverter(config=config)

    # User has events at t=1000, t=3000 (gap > 1800)
    chunk = _make_chunk("u1", [1000, 3000])
    imp = converter._process_chunk(chunk)

    assert imp["session_id"].iloc[0] == "u1_1"
    assert imp["session_id"].iloc[1] == "u1_2"
