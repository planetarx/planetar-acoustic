"""Pure-Python zmesg envelope encoder.

Mirrors the binary layout from ~/github/sness23/zmesg/zmesg.h. Wire format
matches the C reference exactly so envelopes produced here are parseable by
planetar-broker without conversion.

Layout (little-endian, fixed 66-byte header + variable strings + payload):

    0   4  magic = 0x5A4D5347 "ZMSG"
    4   1  version = 1
    5   1  flags
    6   2  header_len (66 + len(topic|source|schema|corr|cause))
    8  16  id (UUIDv7)
   24   8  created_at_ns
   32   8  stored_at_ns
   40   8  published_at_ns
   48   2  topic_len
   50   2  source_len
   52   2  schema_name_len
   54   2  correlation_id_len
   56   2  causation_id_len
   58   4  schema_version
   62   4  payload_len
   66   .. topic, source, schema_name, correlation_id, causation_id, payload
"""
from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass, field

MAGIC = 0x5A4D5347
VERSION = 1
FIXED_HDR = 66


def now_ns() -> int:
    return time.time_ns()


def uuid7() -> bytes:
    """RFC 9562 UUIDv7: 48-bit ms timestamp + 74 random bits + version/variant nibbles."""
    ms = time.time_ns() // 1_000_000
    rand = bytearray(os.urandom(16))
    rand[0] = (ms >> 40) & 0xFF
    rand[1] = (ms >> 32) & 0xFF
    rand[2] = (ms >> 24) & 0xFF
    rand[3] = (ms >> 16) & 0xFF
    rand[4] = (ms >> 8) & 0xFF
    rand[5] = ms & 0xFF
    rand[6] = (rand[6] & 0x0F) | 0x70
    rand[8] = (rand[8] & 0x3F) | 0x80
    return bytes(rand)


def uuid_str(id_: bytes) -> str:
    """Canonical dashed-hex string form of a 16-byte envelope id.

    Matches the bridge's bytesToUuid and the ontology's uuidToString, so
    correlation/causation ids compare equal across the whole stack.
    """
    h = id_.hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


@dataclass
class Envelope:
    topic: str
    payload: bytes
    source: str = "planetar-acoustic"
    schema_name: str = ""
    schema_version: int = 0
    correlation_id: str = ""
    causation_id: str = ""
    flags: int = 0
    id: bytes = field(default_factory=uuid7)
    created_at_ns: int = field(default_factory=now_ns)
    stored_at_ns: int = 0
    published_at_ns: int = 0

    def serialize(self) -> bytes:
        topic_b = self.topic.encode("utf-8")
        source_b = self.source.encode("utf-8")
        schema_b = self.schema_name.encode("utf-8")
        corr_b = self.correlation_id.encode("utf-8")
        cause_b = self.causation_id.encode("utf-8")
        if len(self.id) != 16:
            raise ValueError("id must be 16 bytes")
        header_len = FIXED_HDR + len(topic_b) + len(source_b) + len(schema_b) + len(corr_b) + len(cause_b)
        hdr = struct.pack(
            "<IBBH16sQQQHHHHHII",
            MAGIC,
            VERSION,
            self.flags,
            header_len,
            self.id,
            self.created_at_ns,
            self.stored_at_ns,
            self.published_at_ns,
            len(topic_b),
            len(source_b),
            len(schema_b),
            len(corr_b),
            len(cause_b),
            self.schema_version,
            len(self.payload),
        )
        return hdr + topic_b + source_b + schema_b + corr_b + cause_b + self.payload


def parse(buf: bytes) -> Envelope:
    """Inverse of serialize — useful for tests."""
    if len(buf) < FIXED_HDR:
        raise ValueError("buffer too short for fixed header")
    (
        magic, version, flags, header_len, id_, created, stored, published,
        topic_len, source_len, schema_len, corr_len, cause_len,
        schema_version, payload_len,
    ) = struct.unpack("<IBBH16sQQQHHHHHII", buf[:FIXED_HDR])
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08x}")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    off = FIXED_HDR
    def take(n: int) -> str:
        nonlocal off
        s = buf[off:off + n].decode("utf-8")
        off += n
        return s
    topic = take(topic_len)
    source = take(source_len)
    schema = take(schema_len)
    corr = take(corr_len)
    cause = take(cause_len)
    payload = buf[off:off + payload_len]
    return Envelope(
        topic=topic,
        payload=payload,
        source=source,
        schema_name=schema,
        schema_version=schema_version,
        correlation_id=corr,
        causation_id=cause,
        flags=flags,
        id=id_,
        created_at_ns=created,
        stored_at_ns=stored,
        published_at_ns=published,
    )
