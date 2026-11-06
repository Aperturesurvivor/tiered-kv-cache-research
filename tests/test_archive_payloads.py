import json
import hashlib
import tarfile
import io
import os
from pathlib import Path

import pytest

from research.archive_payloads import pack, reclaim, verify


def payload(tmp_path):
    root = tmp_path/'source'
    root.mkdir()
    (root/'data').write_bytes(b'original KV bytes' * 100)
    complete = tmp_path/'results/source/complete.json'
    complete.parent.mkdir(parents=True)
    complete.write_text(json.dumps({'rows': 2, 'weights_unchanged': True}))
    archive = tmp_path/'backup/source.tar'
    receipt = pack(root, complete, archive)
    metadata = {'id': 'test-only', 'name': archive.name, 'size': str(receipt['size']),
                'md5Checksum': receipt['md5'], 'sha256Checksum': receipt['sha256']}
    cloud = tmp_path/'cloud.json'
    cloud.write_text(json.dumps(metadata))
    return root, archive, cloud


@pytest.mark.parametrize('change', ['cloud_digest', 'new_file', 'changed_file', 'symlink'])
def test_reclaim_refuses_unverified_or_changed_data(tmp_path, change):
    root, archive, cloud = payload(tmp_path)
    if change == 'cloud_digest':
        metadata = json.loads(cloud.read_text())
        metadata['md5Checksum'] = '0' * 32
        cloud.write_text(json.dumps(metadata))
    elif change == 'new_file':
        (root/'new').write_text('not archived')
    elif change == 'changed_file':
        (root/'data').write_text('newer evidence')
    else:
        (root/'link').symlink_to(root/'data')
    with pytest.raises(ValueError):
        reclaim(archive, cloud)
    assert root.is_dir() and (root/'data').exists()
    assert not archive.with_suffix('.tar.reclaimed.json').exists()


def test_reclaim_preserves_restorable_verified_archive(tmp_path):
    root, archive, cloud = payload(tmp_path)
    original = (root/'data').read_bytes()
    record = reclaim(archive, cloud)
    assert record['status'] == 'local_duplicates_reclaimed'
    assert record['source_verification']['sha256_files'] == 1
    assert record['source_verification']['unchanged_signature_files'] == 0
    assert not root.exists()
    verify(archive)
    with tarfile.open(archive) as source:
        source.extractall(tmp_path/'restored', filter='data')
    assert (tmp_path/'restored/data').read_bytes() == original


def test_missing_provider_checksum_requires_explicit_recorded_fallback(tmp_path):
    root, archive, cloud = payload(tmp_path)
    metadata = json.loads(cloud.read_text())
    del metadata['md5Checksum'], metadata['sha256Checksum']
    cloud.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='provider checksum required'):
        reclaim(archive, cloud)
    assert (root/'data').exists()
    record = reclaim(archive, cloud, allow_provider_size_only=True)
    assert record['cloud_verification'] == 'provider_identity_and_size_only; checksum unavailable'
    verify(archive)


def test_size_only_flag_cannot_override_provider_mismatch(tmp_path):
    root, archive, cloud = payload(tmp_path)
    metadata = json.loads(cloud.read_text())
    metadata['md5Checksum'] = '0' * 32
    cloud.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='cloud size/digest'):
        reclaim(archive, cloud, allow_provider_size_only=True)
    assert (root/'data').exists()


@pytest.mark.parametrize('location', ['payload', 'trailing_padding'])
@pytest.mark.parametrize('signature_path', [False, True])
def test_stream_verification_covers_payload_and_tar_padding(tmp_path, location, signature_path):
    root, archive, cloud = payload(tmp_path)
    with archive.open('r+b') as stream:
        if location=='payload':
            with tarfile.open(archive) as index:
                offset=index.getmember('data').offset_data
        else:
            offset=archive.stat().st_size-1
        stream.seek(offset)
        old=stream.read(1)
        stream.seek(offset)
        stream.write(bytes([old[0]^1]))
    with pytest.raises(ValueError):
        reclaim(archive, cloud, trust_unchanged_source_signatures=signature_path)
    assert (root/'data').exists()


def test_stream_verification_crosses_large_buffer_boundary(tmp_path):
    root, _, _ = payload(tmp_path)
    data=b'0123456789abcdef'*(9*1024*1024//16)+b'odd-size'
    (root/'data').write_bytes(data)
    archive=tmp_path/'backup/large.tar'
    receipt=pack(root,tmp_path/'results/source/complete.json',archive)
    assert verify(archive)==receipt
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
    with tarfile.open(archive) as source:
        assert source.extractfile('data').read()==data


def test_explicit_signature_path_still_verifies_archive_without_rereading_source(tmp_path, monkeypatch):
    root, archive, cloud = payload(tmp_path)
    original_open = Path.open
    def guarded_open(self, *args, **kwargs):
        if self == root/'data':
            raise AssertionError('unchanged source was unnecessarily reread')
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    record = reclaim(archive, cloud, trust_unchanged_source_signatures=True)
    assert record['source_verification']['unchanged_signature_files'] == 1
    assert record['source_verification']['sha256_files'] == 0
    verify(archive)


@pytest.mark.parametrize('replace_inode', [False, True])
def test_changed_bytes_with_restored_mtime_are_not_trusted(tmp_path, replace_inode):
    root, archive, cloud = payload(tmp_path)
    entry = root/'data'
    before = entry.stat()
    changed = b'X' * before.st_size
    if replace_inode:
        replacement = root/'replacement'
        replacement.write_bytes(changed)
        replacement.replace(entry)
    else:
        entry.write_bytes(changed)
    os.utime(entry, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match='source content changed'):
        reclaim(archive, cloud, trust_unchanged_source_signatures=True)
    assert entry.read_bytes() == changed
    assert not archive.with_suffix('.tar.reclaimed.json').exists()


def test_changed_signature_with_same_bytes_falls_back_to_source_hash(tmp_path):
    root, archive, cloud = payload(tmp_path)
    entry = root/'data'
    before = entry.stat()
    os.utime(entry, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    record = reclaim(archive, cloud, trust_unchanged_source_signatures=True)
    assert record['source_verification']['sha256_files'] == 1
    assert record['source_verification']['unchanged_signature_files'] == 0


def test_legacy_inventory_without_signatures_uses_full_source_hash(tmp_path):
    root, archive, cloud = payload(tmp_path)
    receipt = json.loads(archive.with_suffix('.tar.receipt.json').read_text())
    with tarfile.open(archive) as source:
        inventory = json.load(source.extractfile('ARCHIVE_INVENTORY.json'))
    for row in inventory['files']:
        del row['source_signature']
    # Rebuild a genuine legacy-format archive and its independently computed
    # receipt/provider hashes; the legacy file records contain only data hashes.
    with tarfile.open(archive, 'w') as output:
        output.add(root/'data', arcname='data')
        data = json.dumps(inventory).encode()
        info = tarfile.TarInfo('ARCHIVE_INVENTORY.json')
        info.size = len(data)
        output.addfile(info, io.BytesIO(data))
    raw = archive.read_bytes()
    receipt.update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest(), md5=hashlib.md5(raw).hexdigest())
    archive.with_suffix('.tar.receipt.json').write_text(json.dumps(receipt))
    metadata = json.loads(cloud.read_text())
    metadata.update(size=str(len(raw)), sha256Checksum=receipt['sha256'], md5Checksum=receipt['md5'])
    cloud.write_text(json.dumps(metadata))
    record = reclaim(archive, cloud, trust_unchanged_source_signatures=True)
    assert record['source_verification']['sha256_files'] == 1
    assert record['source_verification']['unchanged_signature_files'] == 0


def test_source_change_after_verification_is_rejected_before_unlink(tmp_path, monkeypatch):
    root, archive, cloud = payload(tmp_path)
    entry = root/'data'
    before = entry.stat()
    original_lstat = Path.lstat
    record_path = archive.with_suffix('.tar.reclaimed.json')
    changed = False
    def change_before_unlink(self, *args, **kwargs):
        nonlocal changed
        if self == entry and record_path.exists() and not changed:
            changed = True
            entry.write_bytes(b'X' * before.st_size)
            os.utime(entry, ns=(before.st_atime_ns, before.st_mtime_ns))
        return original_lstat(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'lstat', change_before_unlink)
    with pytest.raises(ValueError, match='source changed before release'):
        reclaim(archive, cloud, trust_unchanged_source_signatures=True)
    assert changed and entry.read_bytes() == b'X' * before.st_size
    assert json.loads(record_path.read_text())['status'] == 'reclaim_started'


def test_separate_entry_archives_restore_one_original_tree(tmp_path):
    root, _, _ = payload(tmp_path)
    for name in ('prefix', 'overlay'):
        (root/name).mkdir()
        (root/name/'layer.kv').write_bytes(name.encode() * 100)
    originals = {p.relative_to(root).as_posix(): p.read_bytes()
                 for p in root.rglob('*') if p.is_file()}
    archives = []
    for i, entries in enumerate((['prefix'], ['overlay', 'data'])):
        archive = tmp_path/f'part-{i}.tar'
        receipt = pack(root, tmp_path/'results/source/complete.json', archive, entries=entries)
        assert receipt['original_root'] == str(root.resolve())
        assert receipt['entries'] == sorted(entries)
        cloud = tmp_path/f'part-{i}.cloud.json'
        cloud.write_text(json.dumps({'id': str(i), 'name': archive.name,
                                    'size': receipt['size'], 'md5Checksum': receipt['md5']}))
        record = reclaim(archive, cloud, trust_unchanged_source_signatures=True)
        assert record['status'] == 'local_duplicates_reclaimed'
        if i == 0:
            assert (root/'overlay/layer.kv').exists() and (root/'data').exists()
        archives.append(archive)
    assert not root.exists()
    restored = tmp_path/'restored'
    for archive in archives:
        verify(archive)
        with tarfile.open(archive) as source:
            source.extractall(restored, members=[m for m in source
                if m.name != 'ARCHIVE_INVENTORY.json'], filter='data')
    assert {p.relative_to(restored).as_posix(): p.read_bytes()
            for p in restored.rglob('*') if p.is_file()} == originals


def test_entry_reclaim_preserves_unselected_and_new_outside_files(tmp_path):
    root, _, _ = payload(tmp_path)
    archive = tmp_path/'selected.tar'
    receipt = pack(root, tmp_path/'results/source/complete.json', archive, entries=['data'])
    (root/'new-evidence').write_bytes(b'never archived')
    (root/'unselected-empty-directory').mkdir()
    cloud = tmp_path/'selected.cloud.json'
    cloud.write_text(json.dumps({'id': 'selected', 'name': archive.name,
                                'size': receipt['size'], 'md5Checksum': receipt['md5']}))
    reclaim(archive, cloud)
    assert not (root/'data').exists()
    assert (root/'new-evidence').read_bytes() == b'never archived'
    assert (root/'unselected-empty-directory').is_dir()


def test_entry_reclaim_refuses_new_files_inside_selected_directory(tmp_path):
    root, _, _ = payload(tmp_path)
    (root/'selected').mkdir()
    (root/'selected/data').write_bytes(b'original')
    archive = tmp_path/'selected.tar'
    receipt = pack(root, tmp_path/'results/source/complete.json', archive, entries=['selected'])
    (root/'selected/new').write_bytes(b'new evidence')
    cloud = tmp_path/'selected.cloud.json'
    cloud.write_text(json.dumps({'id': 'selected', 'name': archive.name,
                                'size': receipt['size'], 'md5Checksum': receipt['md5']}))
    with pytest.raises(ValueError, match='source inventory changed'):
        reclaim(archive, cloud, trust_unchanged_source_signatures=True)
    assert (root/'selected/data').read_bytes() == b'original'
    assert (root/'selected/new').read_bytes() == b'new evidence'


@pytest.mark.parametrize('entries', [[], ['..'], ['/tmp'], ['data/../data'], ['data', 'data']])
def test_entry_scope_rejects_empty_ambiguous_or_escaping_names(tmp_path, entries):
    root, _, _ = payload(tmp_path)
    with pytest.raises(ValueError, match='unique immediate child names'):
        pack(root, tmp_path/'results/source/complete.json', tmp_path/'invalid.tar', entries=entries)
    assert (root/'data').exists()


def test_selected_symlink_is_rejected_before_traversal(tmp_path):
    root, _, _ = payload(tmp_path)
    (root/'link').symlink_to(tmp_path/'results', target_is_directory=True)
    with pytest.raises(ValueError, match='non-regular source entry'):
        pack(root, tmp_path/'results/source/complete.json', tmp_path/'invalid.tar', entries=['link'])


def test_receipt_cannot_change_archived_entry_scope(tmp_path):
    root, _, _ = payload(tmp_path)
    archive = tmp_path/'selected.tar'
    receipt = pack(root, tmp_path/'results/source/complete.json', archive, entries=['data'])
    receipt['entries'] = ['other']
    archive.with_suffix('.tar.receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='entry scope differs'):
        verify(archive)


def failed_payload(tmp_path):
    root = tmp_path/'failed-run'
    root.mkdir()
    (root/'partial-prefix').mkdir()
    (root/'partial-prefix/layer.kv').write_bytes(b'partial')
    refs = {}
    for name in ('rows', 'state', 'log'):
        path = tmp_path/name
        path.write_text(name)
        refs[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    document = {'classification': 'FAILED', 'run': root.name, 'original_root': str(root),
                'source_revision': 'a' * 40, 'worker': {'pid': 1, 'create_time': 1.2, 'returncode': 1},
                'supervisor': {'pid': 2, 'create_time': 1.3, 'returncode': 1}, 'evidence_files': refs}
    failure = tmp_path/'failure.json'
    failure.write_text(json.dumps(document))
    archive = tmp_path/'failed.tar'
    return root, failure, archive


def test_failed_archive_requires_evidence_and_provider_before_reclaim(tmp_path):
    root, failure, archive = failed_payload(tmp_path)
    receipt = pack(root, failure, archive, failed=True)
    assert receipt['classification'] == 'FAILED'
    assert verify(archive)['classification'] == 'FAILED'
    cloud = tmp_path/'cloud.json'
    cloud.write_text(json.dumps({'id': 'verified', 'name': archive.name,
                                 'size': receipt['size'], 'md5Checksum': receipt['md5']}))
    (tmp_path/'log').write_text('changed')
    with pytest.raises(ValueError, match='failure evidence file changed'):
        reclaim(archive, cloud)
    assert (root/'partial-prefix/layer.kv').exists()
    (tmp_path/'log').write_text('log')
    assert reclaim(archive, cloud)['classification'] == 'FAILED'
    assert not root.exists()


@pytest.mark.parametrize('mutation', ['document', 'evidence', 'success', 'identity'])
def test_failed_pack_rejects_bad_or_changed_provenance(tmp_path, mutation):
    root, failure, archive = failed_payload(tmp_path)
    doc = json.loads(failure.read_text())
    if mutation == 'document':
        doc['classification'] = 'COMPLETE'
    elif mutation == 'evidence':
        (tmp_path/'rows').write_text('changed')
    elif mutation == 'success':
        doc['weights_unchanged'] = True
    else:
        doc['worker']['returncode'] = 0
    failure.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        pack(root, failure, archive, failed=True)
    assert (root/'partial-prefix/layer.kv').exists()


def test_failed_receipt_cannot_be_reclassified_as_success(tmp_path):
    root, failure, archive = failed_payload(tmp_path)
    receipt = pack(root, failure, archive, failed=True)
    receipt['classification'] = 'COMPLETE'
    archive.with_suffix('.tar.receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='identity differs'):
        verify(archive)
    assert (root/'partial-prefix/layer.kv').exists()


def test_failed_document_mutation_blocks_reclaim(tmp_path):
    root, failure, archive = failed_payload(tmp_path)
    receipt = pack(root, failure, archive, failed=True)
    cloud = tmp_path/'cloud.json'
    cloud.write_text(json.dumps({'id': 'verified', 'name': archive.name,
                                 'size': receipt['size'], 'md5Checksum': receipt['md5']}))
    failure.write_text(failure.read_text() + ' ')
    # Offline verification remains valid even when live evidence has changed.
    assert verify(archive)['classification'] == 'FAILED'
    with pytest.raises(ValueError, match='failure evidence document changed'):
        reclaim(archive, cloud)
    assert (root/'partial-prefix/layer.kv').exists()
    assert not archive.with_suffix('.tar.reclaimed.json').exists()


@pytest.mark.parametrize('digest', [None, 'bad'])
def test_missing_or_invalid_identity_digest_in_both_inventory_and_receipt(tmp_path, digest):
    root, archive, _ = payload(tmp_path)
    receipt_path = archive.with_suffix('.tar.receipt.json')
    receipt = json.loads(receipt_path.read_text())
    with tarfile.open(archive) as source:
        inventory = json.load(source.extractfile('ARCHIVE_INVENTORY.json'))
    if digest is None:
        del inventory['completion_sha256'], receipt['completion_sha256']
    else:
        inventory['completion_sha256'] = receipt['completion_sha256'] = digest
    with tarfile.open(archive, 'w') as output:
        output.add(root/'data', arcname='data')
        data = json.dumps(inventory).encode()
        info = tarfile.TarInfo('ARCHIVE_INVENTORY.json')
        info.size = len(data)
        output.addfile(info, io.BytesIO(data))
    raw = archive.read_bytes()
    receipt.update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest(), md5=hashlib.md5(raw).hexdigest())
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='archive identity differs'):
        verify(archive)
