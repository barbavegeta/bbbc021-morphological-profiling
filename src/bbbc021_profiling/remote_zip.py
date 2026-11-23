"""Read single members of a remote ZIP archive with HTTP range requests.

The central directory at the end of the archive lists every member with its offset, sizes and
CRC-32. A member's bytes run from its local header to the next member's header (or the central
directory), so one range request fetches one member, which is then decompressed and checked
against the CRC-32 and size recorded in the directory.
"""

from __future__ import annotations

import http.client
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

EOCD = struct.Struct("<4sHHHHIIH")             # end of central directory record
CDH = struct.Struct("<4sHHHHHHIIIHHHHHII")     # central directory file header
LFH = struct.Struct("<4sHHHHHIIIHH")           # local file header
EOCD_SIG, CDH_SIG, LFH_SIG = b"PK\x05\x06", b"PK\x01\x02", b"PK\x03\x04"
MAX_COMMENT = 0xFFFF


class ArchiveChanged(RuntimeError):
    """The archive on the server no longer matches the one whose directory was read."""


class MemberError(ValueError):
    """A member's bytes do not match its central directory entry."""


@dataclass(frozen=True)
class Member:
    name: str
    method: int
    flags: int
    crc32: int
    compressed_size: int
    size: int
    header_offset: int
    end: int  # first byte after this member's data (next member's header or the central directory)


class RangeReader(Protocol):
    size: int

    def read(self, start: int, end: int) -> bytes:
        """Bytes start..end inclusive."""


def read_directory(reader: RangeReader) -> list[Member]:
    """Parse the central directory of the archive behind reader (ZIP64 archives are rejected)."""
    tail_start = max(0, reader.size - (EOCD.size + MAX_COMMENT))
    tail = reader.read(tail_start, reader.size - 1)
    i = tail.rfind(EOCD_SIG)
    if i < 0 or len(tail) - i < EOCD.size:
        raise ValueError("end of central directory record not found")
    _, disk, cd_disk, n_disk, n_total, cd_size, cd_offset, _ = EOCD.unpack_from(tail, i)
    if 0xFFFF in (n_disk, n_total) or 0xFFFFFFFF in (cd_size, cd_offset):
        raise ValueError("ZIP64 archives are not supported")
    if disk or cd_disk or n_disk != n_total:
        raise ValueError("multi-disk archives are not supported")
    if cd_offset + cd_size != tail_start + i:
        raise ValueError("central directory does not end where the end record starts")

    cd = (tail[cd_offset - tail_start:i] if cd_offset >= tail_start
          else reader.read(cd_offset, cd_offset + cd_size - 1))
    raw, p = [], 0
    while p < len(cd):
        (sig, _, _, flags, method, _, _, crc, csize, usize, nlen, elen, clen, _, _, _, offset) = CDH.unpack_from(cd, p)
        if sig != CDH_SIG:
            raise ValueError(f"bad central directory entry at byte {p}")
        name_bytes = cd[p + CDH.size:p + CDH.size + nlen]
        name = name_bytes.decode("utf-8" if flags & 0x800 else "cp437")
        raw.append((name, method, flags, crc, csize, usize, offset))
        p += CDH.size + nlen + elen + clen
    if len(raw) != n_total:
        raise ValueError(f"directory lists {len(raw)} entries, end record says {n_total}")

    offsets = sorted({r[6] for r in raw}) + [cd_offset]
    next_offset = dict(zip(offsets[:-1], offsets[1:]))
    return [Member(*r, end=next_offset[r[6]]) for r in raw]


def extract(raw: bytes, m: Member) -> bytes:
    """Decompress one member from its bytes [header_offset, end) and check size and CRC-32."""
    if len(raw) != m.end - m.header_offset:
        raise MemberError(f"{m.name}: got {len(raw)} bytes, expected {m.end - m.header_offset}")
    sig, _, flags, method, _, _, _, _, _, nlen, elen = LFH.unpack_from(raw, 0)
    if sig != LFH_SIG:
        raise MemberError(f"{m.name}: no local file header")
    local_name = raw[LFH.size:LFH.size + nlen].decode("utf-8" if flags & 0x800 else "cp437")
    if local_name != m.name or method != m.method:
        raise MemberError(f"{m.name}: local header names {local_name!r} with method {method}")
    if m.flags & 0x1:
        raise MemberError(f"{m.name}: encrypted members are not supported")
    start = LFH.size + nlen + elen
    data = raw[start:start + m.compressed_size]
    if len(data) != m.compressed_size:
        raise MemberError(f"{m.name}: compressed data is truncated")
    if m.method == 0:
        out = data
    elif m.method == 8:
        d = zlib.decompressobj(-zlib.MAX_WBITS)
        try:
            out = d.decompress(data) + d.flush()
        except zlib.error as e:
            raise MemberError(f"{m.name}: {e}") from e
        if not d.eof:
            raise MemberError(f"{m.name}: deflate stream is incomplete")
    else:
        raise MemberError(f"{m.name}: compression method {m.method} is not supported")
    if len(out) != m.size or zlib.crc32(out) != m.crc32:
        raise MemberError(f"{m.name}: size or CRC-32 does not match the central directory")
    return out


class HttpSession:
    """Persistent HTTP(S) connections, one per thread and host, with retries and backoff."""

    def __init__(self, user_agent: str, timeout: float = 60, retries: int = 5):
        self.user_agent, self.timeout, self.retries = user_agent, timeout, retries
        self._local = threading.local()

    def _conn(self, scheme: str, host: str, port: int | None) -> http.client.HTTPConnection:
        conns = self._local.__dict__.setdefault("conns", {})
        key = (scheme, host, port)
        if key not in conns:
            cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
            conns[key] = cls(host, port, timeout=self.timeout)
        return conns[key]

    def _drop(self, scheme: str, host: str, port: int | None) -> None:
        conn = self._local.__dict__.get("conns", {}).pop((scheme, host, port), None)
        if conn is not None:
            conn.close()

    def request(self, url: str, method: str, headers: dict, expect_range: tuple[int, int, int] | None = None):
        """HEAD or GET url. With expect_range=(start, end, total) the response must be that exact
        206 range; otherwise the response must be 200 and its headers are returned."""
        u = urlsplit(url)
        if u.scheme not in ("http", "https"):
            raise ValueError(f"unsupported URL scheme: {url}")
        target = (u.scheme, u.hostname, u.port)
        headers = {"User-Agent": self.user_agent, **headers}
        for attempt in range(self.retries + 1):
            try:
                conn = self._conn(*target)
                conn.request(method, u.path, headers=headers)
                resp = conn.getresponse()
                if expect_range is None:
                    resp.read()
                    if resp.status != 200:
                        raise OSError(f"{url}: HTTP {resp.status}")
                    return {k.lower(): v for k, v in resp.getheaders()}
                start, end, total = expect_range
                if resp.status == 200:
                    self._drop(*target)  # the whole archive is on its way instead of the range: don't read it
                    raise ArchiveChanged(f"{url}: server sent the full file instead of a range (archive changed?)")
                if resp.status != 206:
                    resp.read()
                    raise OSError(f"{url}: HTTP {resp.status}")
                expected = f"bytes {start}-{end}/{total}"
                if resp.getheader("Content-Range") != expected:
                    self._drop(*target)
                    raise ArchiveChanged(f"{url}: Content-Range {resp.getheader('Content-Range')!r}, expected {expected!r}")
                body = resp.read()
                if len(body) != end - start + 1:
                    raise OSError(f"{url}: short read ({len(body)} of {end - start + 1} bytes)")
                return body
            except ArchiveChanged:
                raise
            except (OSError, http.client.HTTPException):
                self._drop(*target)
                if attempt == self.retries:
                    raise
                time.sleep(2 ** (attempt + 1))
        raise AssertionError("unreachable")


class HttpRangeReader:
    """Range reads of one remote file.

    The archive size, ETag and Last-Modified come from a HEAD request. Every range request sends
    If-Range with the ETag (or Last-Modified when the ETag is weak or missing), so if the archive
    changes the server answers with the whole file, which is refused, instead of bytes from a
    different version.
    """

    def __init__(self, url: str, session: HttpSession):
        self.url, self.session = url, session
        h = session.request(url, "HEAD", {})
        self.size = int(h["content-length"])
        self.etag, self.last_modified = h.get("etag"), h.get("last-modified")
        if self.etag and not self.etag.startswith("W/"):
            self.validator = self.etag
        else:
            self.validator = self.last_modified

    def read(self, start: int, end: int) -> bytes:
        headers = {"Range": f"bytes={start}-{end}"}
        if self.validator:
            headers["If-Range"] = self.validator
        return self.session.request(self.url, "GET", headers, expect_range=(start, end, self.size))
