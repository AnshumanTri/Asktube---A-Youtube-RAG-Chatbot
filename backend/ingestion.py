"""
Ingestion pipeline:
1. Fetch transcript (with timestamp metadata preserved)
2. Semantic-aware chunking (better than fixed-size)
3. Generate HuggingFace embeddings locally
4. Return chunks ready for vector store
"""

import re
from typing import List, Tuple

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
)
import yt_dlp
import requests
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL


def extract_video_id(url_or_id: str) -> str:
    """
    Accept full YouTube URL or bare video ID.
    Handles: youtu.be/ID, youtube.com/watch?v=ID, youtube.com/shorts/ID
    """
    patterns = [
        r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    # If no pattern matched, assume it's already a bare ID
    if re.match(r"^[A-Za-z0-9_-]{11}$", url_or_id):
        return url_or_id
    raise ValueError(f"Could not extract video ID from: {url_or_id}")


def fetch_transcript(video_id: str) -> Tuple[List[dict], str]:
    """
    Fetch transcript using youtube-transcript-api v1.x instance API.
    Falls back to yt-dlp if the primary method fails (e.g. PO-token block).
    Returns (raw_transcript_list, plain_text) where raw_transcript_list is
    a list of {"text": str, "start": float} dicts (normalized format).
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(video_id, languages=["en"])
        raw = [{"text": s.text, "start": s.start} for s in fetched.snippets]
    except NoTranscriptFound:
        # Try any available language, translated to English
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.list(video_id)
            transcript = next(iter(transcript_list))
            fetched = transcript.translate("en").fetch()
            raw = [{"text": s.text, "start": s.start} for s in fetched.snippets]
        except Exception:
            raw = _fetch_transcript_via_ytdlp(video_id)
    except TranscriptsDisabled:
        raise RuntimeError(f"Transcripts are disabled for video {video_id}.")
    except VideoUnavailable:
        raise RuntimeError(f"Video {video_id} is unavailable (private, deleted, or invalid ID).")
    except Exception:
        # Covers PoTokenRequired and any other transient/blocking errors
        raw = _fetch_transcript_via_ytdlp(video_id)

    if not raw:
        raise RuntimeError(
            f"Could not fetch a transcript for video {video_id} via any method. "
            "The video may have no captions, or YouTube is currently blocking automated access."
        )

    plain_text = " ".join(
        seg["text"].strip().replace("\n", " ") for seg in raw
    )
    return raw, plain_text


def _fetch_transcript_via_ytdlp(video_id: str) -> List[dict]:
    """
    Fallback transcript fetch using yt-dlp.
    Extracts the caption track URL (manual or auto-generated, json3 format)
    and parses it in-memory — no files written to disk.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "json3",
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        raise RuntimeError(f"yt-dlp fallback failed for video {video_id}: {e}")

    subs = info.get("subtitles", {}) or {}
    auto_subs = info.get("automatic_captions", {}) or {}

    track = subs.get("en") or auto_subs.get("en")
    if not track:
        raise RuntimeError(f"No English captions found for video {video_id} (yt-dlp fallback).")

    json3_entry = next((t for t in track if t.get("ext") == "json3"), track[0])
    caption_url = json3_entry["url"]

    response = requests.get(caption_url, timeout=15)
    response.raise_for_status()
    data = response.json()

    raw = []
    for event in data.get("events", []):
        if "segs" not in event:
            continue
        text = "".join(seg.get("utf8", "") for seg in event["segs"]).strip()
        if text:
            raw.append({"text": text, "start": event.get("tStartMs", 0) / 1000})

    return raw


def build_chunks(raw_transcript: List[dict], plain_text: str) -> List[Document]:
    """
    Better chunking strategy vs fixed RecursiveCharacterTextSplitter:
    - Split on sentence boundaries first (paragraph-aware)
    - Attach timestamp metadata to each chunk for citations
    """
    # Build time-indexed segments for metadata mapping
    # Each segment: {"start": float, "text": str}
    time_map = [{"start": seg["start"], "text": seg["text"]} for seg in raw_transcript]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],  # sentence-aware order
        length_function=len,
    )

    chunks = splitter.create_documents([plain_text])

    # Attach timestamp metadata to each chunk
    # We find the approximate start time by matching chunk text back to raw segments
    char_cursor = 0
    full_text = plain_text

    for chunk in chunks:
        chunk_start_char = full_text.find(chunk.page_content[:40])  # first 40 chars as anchor
        # Map character position → approximate timestamp
        approx_time = _char_to_timestamp(chunk_start_char, full_text, time_map)
        chunk.metadata["start_time"] = approx_time
        chunk.metadata["timestamp_url_suffix"] = f"&t={int(approx_time)}s"

    return chunks


def _char_to_timestamp(char_pos: int, full_text: str, time_map: List[dict]) -> float:
    """Approximate: which transcript segment does this character position fall in."""
    if char_pos <= 0 or not time_map:
        return 0.0
    # Build cumulative character counts per segment
    cumulative = 0
    for seg in time_map:
        seg_len = len(seg["text"]) + 1  # +1 for space
        if cumulative + seg_len >= char_pos:
            return seg["start"]
        cumulative += seg_len
    return time_map[-1]["start"]


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Local HuggingFace embeddings - no API key, runs on CPU fine for this size.
    Cached after first load by sentence-transformers library.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest_video(url_or_id: str) -> Tuple[List[Document], str]:
    """
    Full ingestion pipeline. Returns (chunks, video_id).
    Raises RuntimeError with user-friendly message on failure.
    """
    video_id = extract_video_id(url_or_id)
    raw_transcript, plain_text = fetch_transcript(video_id)
    chunks = build_chunks(raw_transcript, plain_text)
    return chunks, video_id