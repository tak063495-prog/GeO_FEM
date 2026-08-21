"""GF1 byte payload discovery and text extraction helpers."""

from __future__ import annotations

import json
import zlib


def gf1_payload_text_candidates(data: bytes) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for blob_label, blob in gf1_payload_blobs(data):
        for text_label, text in decoded_text_candidates(blob):
            for candidate_label, candidate in gf1_text_candidates(text):
                label = f"{blob_label}:{text_label}:{candidate_label}"
                key = candidate.strip()
                if key and key not in seen:
                    seen.add(key)
                    out.append((label, candidate))
    return out


def gf1_payload_blobs(data: bytes) -> list[tuple[str, bytes]]:
    out: list[tuple[str, bytes]] = [("raw", data)]
    seen = {data}
    for offset, payload in zlib_payloads(data):
        if payload and payload not in seen:
            seen.add(payload)
            out.append((f"zlib@{offset}", payload))
    return out


def zlib_payloads(data: bytes) -> list[tuple[int, bytes]]:
    offsets = {0}
    for marker in (b"\x78\x01", b"\x78\x5e", b"\x78\x9c", b"\x78\xda"):
        start = data.find(marker)
        while start >= 0:
            offsets.add(start)
            start = data.find(marker, start + 1)
    out: list[tuple[int, bytes]] = []
    for offset in sorted(offsets):
        try:
            payload = zlib.decompress(data[offset:])
        except zlib.error:
            try:
                decompressor = zlib.decompressobj()
                payload = decompressor.decompress(data[offset:])
                if not decompressor.eof:
                    continue
            except zlib.error:
                continue
        if payload:
            out.append((offset, payload))
    return out


def decoded_text_candidates(data: bytes) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp932"):
        text = decode_with_encoding(data, encoding)
        if text and text not in seen:
            seen.add(text)
            out.append((encoding, text))
    for marker, encoding in ((b"{\x00", "utf-16-le"), (b"[\x00", "utf-16-le"), (b"\x00{", "utf-16-be"), (b"\x00[", "utf-16-be")):
        start = data.find(marker)
        while start >= 0 and len(out) < 16:
            text = decode_with_encoding(data[start:], encoding)
            if text and text not in seen:
                seen.add(text)
                out.append((f"{encoding}@{start}", text))
            start = data.find(marker, start + 1)
    return out


def decode_with_encoding(data: bytes, encoding: str) -> str:
    try:
        return data.decode(encoding)
    except UnicodeError:
        return data.decode(encoding, errors="ignore")


def gf1_text_candidates(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    stripped = text.lstrip("\ufeff\x00\r\n\t ")
    if looks_like_gf1_text(stripped):
        out.append(("text", stripped))
    out.extend((f"json@{start}", payload) for start, payload in json_payloads(text))
    return out


def json_payloads(text: str) -> list[tuple[int, str]]:
    decoder = json.JSONDecoder()
    out: list[tuple[int, str]] = []
    checked = 0
    for start, char in enumerate(text):
        if char not in "{[":
            continue
        checked += 1
        try:
            _, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            if checked >= 64:
                break
            continue
        out.append((start, text[start : start + end]))
        if checked >= 64:
            break
    return out


def looks_like_gf1_text(text: str) -> bool:
    return text.startswith(("{", "[", "geometry:", "layers:", "lines:", "regions:", "tunnels:", "hatches:"))


def best_effort_decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp932"):
        text = decode_with_encoding(data, encoding)
        if text.strip("\ufeff\x00\r\n\t "):
            return text
    return ""


def gf1_binary_payload_marker(data: bytes, label: str) -> bool:
    return label.startswith("zlib@") or ":json@" in label or looks_binary(data)


def looks_binary(data: bytes) -> bool:
    sample = data[:4096]
    return any(byte == 0 or byte < 9 or (13 < byte < 32) for byte in sample)


__all__ = [
    "best_effort_decode",
    "decoded_text_candidates",
    "decode_with_encoding",
    "gf1_binary_payload_marker",
    "gf1_payload_blobs",
    "gf1_payload_text_candidates",
    "gf1_text_candidates",
    "json_payloads",
    "looks_binary",
    "looks_like_gf1_text",
    "zlib_payloads",
]
