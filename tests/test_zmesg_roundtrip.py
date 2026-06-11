"""Roundtrip + wire-layout tests for the zmesg encoder.

The encoded bytes must match the C reference in ~/github/sness23/zmesg/zmesg.h
exactly, or planetar-broker will reject them.
"""
from __future__ import annotations

import struct

from planetar_acoustic.bus.zmesg import FIXED_HDR, MAGIC, VERSION, Envelope, parse


def test_roundtrip_minimal():
    e = Envelope(topic="acoustic.detect", payload=b"hello")
    e2 = parse(e.serialize())
    assert e2.topic == "acoustic.detect"
    assert e2.payload == b"hello"
    assert e2.source == "planetar-acoustic"
    assert e2.id == e.id


def test_roundtrip_all_fields():
    e = Envelope(
        topic="acoustic.classify",
        payload=b'{"class":"tug","score":0.91}',
        source="planetar-acoustic:test",
        schema_name="planetar.acoustic.classify",
        schema_version=1,
        correlation_id="clip-abc-123",
        causation_id="parent-uuid",
        flags=0,
    )
    e2 = parse(e.serialize())
    for attr in ("topic", "payload", "source", "schema_name", "schema_version",
                 "correlation_id", "causation_id", "flags", "id", "created_at_ns"):
        assert getattr(e, attr) == getattr(e2, attr), attr


def test_wire_layout_matches_c_reference():
    e = Envelope(topic="t", payload=b"")
    buf = e.serialize()
    assert len(buf) >= FIXED_HDR + 1
    (magic,) = struct.unpack("<I", buf[0:4])
    assert magic == MAGIC
    assert buf[4] == VERSION
    (header_len,) = struct.unpack("<H", buf[6:8])
    assert header_len == FIXED_HDR + len(b"t") + len(b"planetar-acoustic")
    (topic_len,) = struct.unpack("<H", buf[48:50])
    assert topic_len == 1
    assert buf[FIXED_HDR:FIXED_HDR + 1] == b"t"


def test_uuid7_version_and_variant():
    e = Envelope(topic="t", payload=b"")
    # RFC 9562: version nibble = 0x7, variant high bits = 0b10.
    assert (e.id[6] & 0xF0) == 0x70
    assert (e.id[8] & 0xC0) == 0x80
