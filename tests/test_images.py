"""Range-based TIFF download on synthetic ZIP archives (no BBBC021 data)."""

import hashlib
import io
import random
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import pytest

from bbbc021_profiling import images, remote_zip


def fake_tiff(seed: int) -> bytes:
    """Synthetic file content: a random part (incompressible) and a zero run (compressible)."""
    r = random.Random(seed)
    return r.randbytes(r.randint(200, 800)) + bytes(r.randint(500, 3000))


class _NoSeek:
    """A write-only stream, which makes zipfile write data descriptors after each member."""

    def __init__(self):
        self.buf = io.BytesIO()

    def write(self, b):
        return self.buf.write(b)

    def flush(self):
        pass


def make_zip(plate: str, files: dict[str, bytes], descriptors: bool = False) -> bytes:
    target = _NoSeek() if descriptors else io.BytesIO()
    with zipfile.ZipFile(target, "w") as zf:
        zf.writestr(zipfile.ZipInfo(f"{plate}/"), b"")  # the real archives start with a folder entry
        for i, (name, data) in enumerate(files.items()):
            zf.writestr(f"{plate}/{name}", data,
                        compress_type=zipfile.ZIP_STORED if i % 3 == 0 else zipfile.ZIP_DEFLATED)
    return (target.buf if descriptors else target).getvalue()


class BytesReader:
    def __init__(self, data: bytes, etag: str = '"v1"'):
        self.data, self.size, self.etag, self.last_modified, self.calls = data, len(data), etag, None, 0

    def read(self, start, end):
        assert 0 <= start <= end < self.size
        self.calls += 1
        return self.data[start:end + 1]


@pytest.mark.parametrize("descriptors", [False, True])
def test_directory_and_extract(descriptors):
    files = {f"f{i}.tif": fake_tiff(i) for i in range(7)}
    data = make_zip("P1", files, descriptors)
    members = remote_zip.read_directory(BytesReader(data))
    assert [m.name for m in members] == ["P1/"] + [f"P1/{n}" for n in files]
    assert {m.method for m in members[1:]} == {0, 8}
    for m in members[1:]:
        assert remote_zip.extract(data[m.header_offset:m.end], m) == files[m.name.split("/")[1]]


@pytest.mark.parametrize("which", [1, 2])  # a stored and a deflated member
def test_corrupt_member_rejected(which):
    data = bytearray(make_zip("P1", {f"f{i}.tif": fake_tiff(i) for i in range(3)}))
    m = remote_zip.read_directory(BytesReader(bytes(data)))[which]
    data[m.end - 10] ^= 0xFF  # a byte inside the member's data
    with pytest.raises(remote_zip.MemberError):
        remote_zip.extract(bytes(data[m.header_offset:m.end]), m)


def _selected(plates_files):
    rows = []
    for plate, wells in plates_files.items():
        for well in wells:
            rows.append({"plate": plate, **{f"path_{ch}": f"Week1/{plate}/{well}_{tag}.tif"
                                            for ch, tag in [("DAPI", "w1"), ("Tubulin", "w2"), ("Actin", "w4")]}})
    return pd.DataFrame(rows)


@pytest.fixture
def archive(tmp_path):
    """Two plates; plate P1 has an extra well (C03) that is not selected."""
    wells = {"P1": ["B02", "C03", "D05"], "P2": ["B02"]}
    contents, zips = {}, {}
    for plate, ws in wells.items():
        files = {}
        for w in ws:
            for tag in ["w1", "w2", "w4"]:
                files[f"{w}_{tag}.tif"] = contents[f"Week1/{plate}/{w}_{tag}.tif"] = fake_tiff(len(contents))
        zips[plate] = make_zip(plate, files)
    selected = _selected({"P1": ["B02", "D05"], "P2": ["B02"]})
    return {"zips": zips, "contents": contents, "needed": images.needed_files(selected),
            "root": tmp_path / "images", "manifest": tmp_path / "manifest"}


def _run(a, readers, **kw):
    return images.download_images(a["needed"], a["root"], a["manifest"], lambda p: readers[p],
                                  workers=2, log=lambda s: None, **kw)


def test_download_writes_files_and_manifest(archive):
    readers = {p: BytesReader(z) for p, z in archive["zips"].items()}
    s = _run(archive, readers)
    assert s["tiffs_needed"] == s["tiffs_done"] == s["tiffs_fetched_now"] == 9
    for rel in archive["needed"]["path"]:
        assert (archive["root"] / rel).read_bytes() == archive["contents"][rel]
    assert not (archive["root"] / "Week1/P1/C03_w1.tif").exists()
    t = pd.read_csv(archive["manifest"] / "tiffs.csv")
    assert len(t) == 9
    for r in t.to_dict("records"):
        assert r["sha256"] == hashlib.sha256(archive["contents"][r["path"]]).hexdigest()
    p = pd.read_csv(archive["manifest"] / "plates.csv")
    assert list(p["state"]) == ["done", "done"] and list(p["members"]) == [9, 3]


def test_rerun_resumes(archive):
    _run(archive, {p: BytesReader(z) for p, z in archive["zips"].items()})
    readers = {p: BytesReader(z) for p, z in archive["zips"].items()}
    assert _run(archive, readers)["tiffs_fetched_now"] == 0
    assert sum(r.calls for r in readers.values()) == 0  # directories come from the cache
    (archive["root"] / "Week1/P2/B02_w2.tif").unlink()
    readers = {p: BytesReader(z) for p, z in archive["zips"].items()}
    assert _run(archive, readers)["tiffs_fetched_now"] == 1
    assert readers["P2"].calls == 1 and readers["P1"].calls == 0


def test_truncated_file_is_dropped_from_manifest_and_refetched(archive):
    _run(archive, {p: BytesReader(z) for p, z in archive["zips"].items()})
    rel = "Week1/P1/D05_w1.tif"
    (archive["root"] / rel).write_bytes(b"")
    s = _run(archive, {p: BytesReader(z) for p, z in archive["zips"].items()})
    assert s["tiffs_fetched_now"] == 1
    assert (archive["root"] / rel).read_bytes() == archive["contents"][rel]
    t = pd.read_csv(archive["manifest"] / "tiffs.csv")
    assert len(t) == 9 and t["path"].is_unique


def test_verify_finds_changed_content(archive):
    _run(archive, {p: BytesReader(z) for p, z in archive["zips"].items()})
    v = images.verify_images(archive["needed"], archive["root"], archive["manifest"])
    assert v["verified"] == 9 and v["failed"] == 0 and v["not_in_manifest"] == 0
    f = archive["root"] / "Week1/P2/B02_w4.tif"
    f.write_bytes(bytes(len(f.read_bytes())))  # same size, other content
    v = images.verify_images(archive["needed"], archive["root"], archive["manifest"])
    assert v["verified"] == 8 and v["failed_paths"] == ["Week1/P2/B02_w4.tif"]
    s = _run(archive, {p: BytesReader(z) for p, z in archive["zips"].items()})
    assert s["tiffs_fetched_now"] == 1
    assert f.read_bytes() == archive["contents"]["Week1/P2/B02_w4.tif"]


def test_directories_only(archive):
    s = _run(archive, {p: BytesReader(z) for p, z in archive["zips"].items()}, directories_only=True)
    assert s["tiffs_done"] == 0 and s["needed_bytes"] == sum(len(v) for k, v in archive["contents"].items()
                                                             if k in set(archive["needed"]["path"]))
    assert not archive["root"].exists() or not any(archive["root"].rglob("*.tif"))


def test_missing_member_and_changed_archive(archive):
    readers = {p: BytesReader(z) for p, z in archive["zips"].items()}
    bad = dict(archive, needed=images.needed_files(_selected({"P1": ["E07"]})))
    with pytest.raises(remote_zip.MemberError):
        _run(bad, readers)
    _run(archive, readers)
    with pytest.raises(remote_zip.ArchiveChanged):
        _run(archive, {p: BytesReader(z, etag='"v2"') for p, z in archive["zips"].items()})


def test_paths_stay_inside_root(tmp_path):
    with pytest.raises(ValueError):
        images.safe_path(tmp_path, "../outside.tif")
    with pytest.raises(ValueError):
        images.needed_files(_selected({"P1": ["B02"]}).assign(path_DAPI="Week1/P9/B02_w1.tif"))


class _RangeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_HEAD(self):
        self._reply(head=True)

    def do_GET(self):
        self._reply(head=False)

    def _reply(self, head):
        data, etag = self.server.data, self.server.etag
        rng, if_range = self.headers.get("Range"), self.headers.get("If-Range")
        if rng and not head and (if_range is None or if_range == etag):
            a, b = map(int, rng.removeprefix("bytes=").split("-"))
            body = data[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", etag)
        self.end_headers()
        if not head:
            try:
                self.wfile.write(body)
            except OSError:
                pass


def test_http_range_reader():
    files = {f"f{i}.tif": fake_tiff(i) for i in range(4)}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    server.data, server.etag = make_zip("P1", files), '"v1"'
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/P1.zip"
        reader = remote_zip.HttpRangeReader(url, remote_zip.HttpSession("test", retries=0))
        assert reader.size == len(server.data) and reader.validator == '"v1"'
        members = remote_zip.read_directory(reader)
        for m in members[1:]:
            assert remote_zip.extract(reader.read(m.header_offset, m.end - 1), m) == files[m.name.split("/")[1]]
        server.etag = '"v2"'  # the archive is replaced on the server
        with pytest.raises(remote_zip.ArchiveChanged):
            reader.read(members[1].header_offset, members[1].end - 1)
    finally:
        server.shutdown()
        server.server_close()
