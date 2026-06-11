"""TCP publisher to planetar-broker.

Connects to the broker's publish port (default 127.0.0.1:12001), then writes
length-prefixed zmesg frames matching the broker's TCP wire format:

    [4-byte BIG-endian envelope length, network byte order][envelope bytes]

The length prefix is network-byte-order (big-endian) — the broker calls
ntohl() on every prefix it reads (see planetar-broker.c:995). Note that the
zmesg envelope *itself* is little-endian internally; only the framing prefix
is BE.
"""
from __future__ import annotations

import logging
import socket
import struct
import time
from typing import Optional

from planetar_acoustic.bus.zmesg import Envelope

log = logging.getLogger(__name__)

FRAME_MAX = 1 << 20  # broker rejects > 1 MiB


class Publisher:
    def __init__(self, host: str = "127.0.0.1", port: int = 12001, connect_timeout: float = 2.0):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self._sock: Optional[socket.socket] = None

    @classmethod
    def from_endpoint(cls, endpoint: str) -> "Publisher":
        host, _, port_s = endpoint.partition(":")
        return cls(host=host or "127.0.0.1", port=int(port_s or 12001))

    def connect(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.connect_timeout)
        s.connect((self.host, self.port))
        s.settimeout(None)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = s
        log.info("connected to planetar-broker at %s:%d", self.host, self.port)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def publish(self, env: Envelope) -> int:
        """Serialize and ship one envelope. Returns bytes written (including 4-byte prefix)."""
        if self._sock is None:
            self.connect()
        env.published_at_ns = time.time_ns()
        frame = env.serialize()
        if len(frame) > FRAME_MAX:
            raise ValueError(f"envelope {len(frame)} B exceeds FRAME_MAX={FRAME_MAX}")
        prefix = struct.pack(">I", len(frame))  # network byte order: broker calls ntohl
        assert self._sock is not None
        self._sock.sendall(prefix + frame)
        return len(prefix) + len(frame)

    def __enter__(self) -> "Publisher":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
