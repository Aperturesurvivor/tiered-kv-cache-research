"""Preserve a completed or explicitly failed run's scratch in a restorable tar.

Run only while GPU measurements are idle. Pack and verify never remove sources.
Reclaim requires separately fetched cloud metadata, verifies the archive and
every source file, then removes only those preserved local duplicates. Provider
checksums are required by default; a recorded size-only fallback is explicit.
An explicit same-host option accepts unchanged pack-time identity/ctime metadata
instead of rereading source bytes; every archived byte is still hashed, and
missing or changed source signatures fall back to a full source hash.
Repeated --entry arguments restrict packing and reclamation to named immediate
children, allowing separate parts without changing absolute restore paths.
Other source entries are preserved. All parts are required for a complete run.
Restore
to original_root because KV checkpoints contain absolute prefix paths.
For a failed run, pass --failed and a failure evidence JSON as the second
positional argument. It must identify the run, source revision, scratch root,
failed terminal worker and supervisor, and SHA256-bound rows, state and log.
No completion or weights-unchanged claim is made for failed archives.
Verify checks the archive's bytes and embedded evidence offline; it does not
establish that the referenced live files or processes still match. Reclaim
separately rechecks the live failure document and referenced files before
removing anything. The caller must independently establish terminal process
and GPU state before packing.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
import re


class HashStream:
    def __init__(self, stream):
        self.stream = stream
        self.sha256 = hashlib.sha256()
        self.md5 = hashlib.md5()
        self.size = 0

    def write(self, data):
        written = self.stream.write(data)
        if written != len(data):
            raise OSError('short archive write')
        self.sha256.update(data)
        self.md5.update(data)
        self.size += written
        return written

    def read(self, size=-1):
        data = self.stream.read(size)
        self.sha256.update(data)
        self.md5.update(data)
        self.size += len(data)
        return data


def source_signature(info):
    """Same-host identity/change metadata; reads may change atime, not ctime."""
    return {key: getattr(info, 'st_' + key) for key in
            ('dev', 'ino', 'mode', 'size', 'mtime_ns', 'ctime_ns')}


def selected_entries(entries):
    if entries is None:
        return None
    if (not entries or any(not isinstance(name, str) or not name or
            name in ('.', '..') or Path(name).name != name for name in entries)
            or len(set(entries)) != len(entries)):
        raise ValueError('entries must be unique immediate child names')
    return sorted(entries)


def source_paths(root, entries=None):
    if entries is None:
        return sorted(root.rglob('*'))
    paths = []
    for name in selected_entries(entries):
        entry = root/name
        mode = entry.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f'non-regular source entry: {entry}')
        paths.append(entry)
        if stat.S_ISDIR(mode):
            paths.extend(entry.rglob('*'))
    return sorted(paths)


def failure_evidence(raw, root, *, check_files):
    """Validate explicit terminal-failure provenance, never a completion marker."""
    evidence = json.loads(raw)
    if (evidence.get('classification') != 'FAILED' or evidence.get('run') != root.name
            or evidence.get('original_root') != str(root.resolve())
            or not re.fullmatch(r'[0-9a-f]{40}', evidence.get('source_revision', ''))
            or evidence.get('weights_unchanged') is not None or evidence.get('rows') is not None):
        raise ValueError('invalid FAILED run identity or success classification')
    for role in ('worker', 'supervisor'):
        identity = evidence.get(role, {})
        if (type(identity.get('pid')) is not int or identity['pid'] <= 0
                or type(identity.get('create_time')) not in (int, float)
                or identity['create_time'] <= 0 or type(identity.get('returncode')) is not int
                or identity['returncode'] == 0):
            raise ValueError('terminal failed process identities required')
    refs = evidence.get('evidence_files', {})
    if set(refs) != {'rows', 'state', 'log'}:
        raise ValueError('rows, state and log evidence required')
    for ref in refs.values():
        if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
            raise ValueError('invalid evidence file binding')
        source = Path(ref['path'])
        if (not source.is_absolute() or source.resolve().is_relative_to(root.resolve())
                or not re.fullmatch(r'[0-9a-f]{64}', ref['sha256'])):
            raise ValueError('invalid evidence file binding')
        if check_files and (source.is_symlink() or not source.is_file()
                or hashlib.sha256(source.read_bytes()).hexdigest() != ref['sha256']):
            raise ValueError('failure evidence file changed')
    return evidence


def pack(root, complete, destination, *, entries=None, failed=False):
    root, complete, destination = map(Path, (root, complete, destination))
    if root.is_symlink() or not root.is_dir():
        raise ValueError('source must be an ordinary directory')
    marker = complete.read_bytes()
    if failed:
        if complete.is_symlink() or complete.resolve().is_relative_to(root.resolve()):
            raise ValueError('failure evidence must be outside scratch')
        done = failure_evidence(marker, root, check_files=True)
    else:
        done = json.loads(marker)
        if done.get('classification') == 'FAILED' or not done.get('weights_unchanged') or not done.get('rows'):
            raise ValueError('successful completed model run required')
        if root.name != complete.parent.name:
            raise ValueError('completion marker and scratch run names differ')
    if destination.resolve().is_relative_to(root.resolve()):
        raise ValueError('archive destination must be outside source')
    entries = selected_entries(entries)
    paths = source_paths(root, entries)
    for entry in paths:
        mode = entry.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f'non-regular source entry: {entry}')
    files = [entry for entry in paths if entry.is_file()]
    if not files:
        raise ValueError('empty source')
    destination.parent.mkdir(parents=True, exist_ok=True)
    inventory = {'original_root': str(root.resolve()), 'files': []}
    if failed:
        inventory.update(classification='FAILED', failure_evidence=done,
                         failure_evidence_text=marker.decode('utf-8'),
                         failure_evidence_path=str(complete.resolve()),
                         failure_evidence_sha256=hashlib.sha256(marker).hexdigest())
    else:
        inventory.update(completion=done, completion_sha256=hashlib.sha256(marker).hexdigest())
    if entries is not None:
        inventory['entries'] = entries
    with destination.open('xb') as output:
        writer = HashStream(output)
        # Match TarFile's source copy size to its stream buffer. The default
        # 16 KiB copies repeatedly concatenate the growing 8 MiB bytes buffer.
        with tarfile.open(fileobj=writer, mode='w|', bufsize=8 * 1024 * 1024,
                          copybufsize=8 * 1024 * 1024) as archive:
            for entry in files:
                before = entry.stat()
                relative = entry.relative_to(root).as_posix()
                if relative == 'ARCHIVE_INVENTORY.json':
                    raise ValueError('reserved inventory filename already exists')
                info = archive.gettarinfo(str(entry), arcname=relative)
                with entry.open('rb') as source:
                    reader = HashStream(source)
                    archive.addfile(info, reader)
                after = entry.stat()
                if source_signature(before) != source_signature(after):
                    raise RuntimeError(f'source changed while archiving: {entry}')
                if reader.size != before.st_size:
                    raise RuntimeError(f'incomplete source read: {entry}')
                inventory['files'].append({'path': relative, 'size': reader.size,
                                           'sha256': reader.sha256.hexdigest(),
                                           'source_signature': source_signature(after)})
            if source_paths(root, entries) != paths:
                raise RuntimeError('source tree changed while archiving')
            if complete.read_bytes() != marker:
                raise RuntimeError('completion marker changed while archiving')
            if failed:
                failure_evidence(marker, root, check_files=True)
            data = json.dumps(inventory, sort_keys=True, indent=2).encode()
            info = tarfile.TarInfo('ARCHIVE_INVENTORY.json')
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        output.flush()
    identity_keys = ('original_root', 'classification', 'failure_evidence_sha256') if failed else ('original_root', 'completion_sha256')
    receipt = {**{key: inventory[key] for key in identity_keys},
               'archive': str(destination), 'size': writer.size,
               'sha256': writer.sha256.hexdigest(), 'md5': writer.md5.hexdigest(),
               'files': len(files), 'payload_bytes': sum(row['size'] for row in inventory['files'])}
    if entries is not None:
        receipt['entries'] = entries
    destination.with_suffix(destination.suffix + '.receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def verify(archive_path, *, with_inventory=False):
    archive_path = Path(archive_path)
    receipt = json.loads(archive_path.with_suffix(archive_path.suffix + '.receipt.json').read_text())
    observed = []
    inventory = None
    with archive_path.open('rb') as source:
        reader = HashStream(source)
        with tarfile.open(fileobj=reader, mode='r|', bufsize=8 * 1024 * 1024) as archive:
            for member in archive:
                if not member.isfile() or member.name.startswith('/') or '..' in Path(member.name).parts:
                    raise ValueError('unexpected archive member')
                extracted = archive.extractfile(member)
                if member.name == 'ARCHIVE_INVENTORY.json':
                    if inventory is not None:
                        raise ValueError('duplicate inventory')
                    inventory = json.load(extracted)
                else:
                    item = HashStream(extracted)
                    while item.read(8 * 1024 * 1024):
                        pass
                    observed.append({'path': member.name, 'size': item.size, 'sha256': item.sha256.hexdigest()})
        # Tar iteration stops at its end marker. Hash any unread trailing bytes
        # too; bytes prefetched by the stream were already hashed exactly once.
        while reader.read(8 * 1024 * 1024):
            pass
    if (reader.size != receipt['size'] or reader.sha256.hexdigest() != receipt['sha256']
            or reader.md5.hexdigest() != receipt['md5']):
        raise ValueError('whole-archive digest mismatch')
    expected = ([{key: row[key] for key in ('path', 'size', 'sha256')}
                 for row in inventory['files']] if inventory is not None else None)
    if observed != expected:
        raise ValueError('archive content differs from inventory')
    if len(observed) != receipt['files'] or len({row['path'] for row in observed}) != len(observed):
        raise ValueError('archive file count mismatch')
    failed = inventory.get('classification') == 'FAILED'
    identity_keys = ('original_root', 'classification', 'failure_evidence_sha256') if failed else ('original_root', 'completion_sha256')
    digest_key = 'failure_evidence_sha256' if failed else 'completion_sha256'
    if (any(key not in inventory or key not in receipt or inventory[key] != receipt[key]
            for key in identity_keys)
            or not isinstance(inventory[digest_key], str)
            or not re.fullmatch(r'[0-9a-f]{64}', inventory[digest_key])):
        raise ValueError('archive identity differs from receipt')
    if failed:
        raw = inventory['failure_evidence_text'].encode('utf-8')
        if (hashlib.sha256(raw).hexdigest() != inventory['failure_evidence_sha256']
                or json.loads(raw) != inventory['failure_evidence']
                or 'completion' in inventory or 'completion_sha256' in inventory
                or 'completion_sha256' in receipt
                or not Path(inventory['failure_evidence_path']).is_absolute()
                or Path(inventory['failure_evidence_path']).resolve().is_relative_to(Path(inventory['original_root']).resolve())):
            raise ValueError('failure evidence identity differs')
        failure_evidence(raw, Path(inventory['original_root']), check_files=False)
    elif (inventory.get('classification') is not None or receipt.get('classification') is not None
          or any(key.startswith('failure_evidence') for key in inventory)
          or 'failure_evidence_sha256' in receipt
          or not inventory.get('completion', {}).get('weights_unchanged')
          or not inventory.get('completion', {}).get('rows')
          or inventory.get('completion', {}).get('classification') == 'FAILED'):
        raise ValueError('archive classification differs')
    entries = selected_entries(inventory.get('entries'))
    if entries != inventory.get('entries') or entries != receipt.get('entries'):
        raise ValueError('archive entry scope differs from receipt')
    if entries is not None and any(Path(row['path']).parts[0] not in entries for row in observed):
        raise ValueError('archive payload lies outside selected entries')
    if sum(row['size'] for row in observed) != receipt['payload_bytes']:
        raise ValueError('archive payload size mismatch')
    return (receipt, inventory) if with_inventory else receipt


def reclaim(archive_path, cloud_metadata, *, allow_provider_size_only=False,
            trust_unchanged_source_signatures=False):
    """Release verified duplicates, retaining the archive and verification scope."""
    archive_path = Path(archive_path)
    metadata = json.loads(Path(cloud_metadata).read_text())
    receipt = json.loads(archive_path.with_suffix(archive_path.suffix + '.receipt.json').read_text())
    if not metadata.get('id') or metadata.get('name') != archive_path.name:
        raise ValueError('cloud identity/name required')
    if int(metadata['size']) != receipt['size']:
        raise ValueError('cloud size/digest differs from local archive receipt')
    if metadata.get('md5Checksum'):
        if metadata['md5Checksum'] != receipt['md5']:
            raise ValueError('cloud size/digest differs from local archive receipt')
        cloud_verification = 'provider_size_and_md5'
    elif allow_provider_size_only:
        cloud_verification = 'provider_identity_and_size_only; checksum unavailable'
    else:
        raise ValueError('provider checksum required unless size-only fallback is explicit')
    if metadata.get('sha256Checksum') and metadata['sha256Checksum'] != receipt['sha256']:
        raise ValueError('cloud SHA256 differs from archive receipt')
    receipt, inventory = verify(archive_path, with_inventory=True)
    root = Path(inventory['original_root'])
    if inventory.get('classification') == 'FAILED':
        evidence_path = Path(inventory['failure_evidence_path'])
        if (evidence_path.is_symlink() or not evidence_path.is_file()
                or evidence_path.read_bytes() != inventory['failure_evidence_text'].encode('utf-8')):
            raise ValueError('failure evidence document changed')
        failure_evidence(evidence_path.read_bytes(), root, check_files=True)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError('original source directory unavailable')
    if archive_path.resolve().is_relative_to(root.resolve()):
        raise ValueError('archive must be outside source')
    entries = inventory.get('entries')
    paths = source_paths(root, entries)
    if any(not (stat.S_ISREG(p.lstat().st_mode) or stat.S_ISDIR(p.lstat().st_mode)) for p in paths):
        raise ValueError('non-regular source entry')
    files = [p for p in paths if p.is_file()]
    if [p.relative_to(root).as_posix() for p in files] != [row['path'] for row in inventory['files']]:
        raise ValueError('source inventory changed; preserve all local files')
    signatures = {}
    source_checks = {'sha256_files': 0, 'unchanged_signature_files': 0}
    for entry, row in zip(files, inventory['files']):
        before = entry.stat()
        unchanged = (trust_unchanged_source_signatures and
                     row.get('source_signature') == source_signature(before))
        if unchanged:
            source_checks['unchanged_signature_files'] += 1
        else:
            with entry.open('rb') as source:
                reader = HashStream(source)
                while reader.read(8 * 1024 * 1024):
                    pass
            if reader.size != row['size'] or reader.sha256.hexdigest() != row['sha256']:
                raise ValueError('source content changed; preserve all local files')
            source_checks['sha256_files'] += 1
        after = entry.stat()
        signature = source_signature(after)
        if source_signature(before) != signature:
            raise ValueError('source changed during verification')
        signatures[entry] = signature
    if source_paths(root, entries) != paths:
        raise ValueError('source tree changed during verification')
    # All verification precedes mutation. A partial failure remains recoverable
    # from the verified cloud archive; never recursively remove an unknown tree.
    record = {**receipt, 'cloud_metadata': metadata, 'cloud_verification': cloud_verification,
              'local_verification': ('whole archive SHA256/MD5 and internal per-file SHA256; '
                                     'source verification counts and scope recorded separately'),
              'source_verification': {**source_checks,
                  'signature_scope': ('explicit same-host unchanged dev/inode/mode/size/mtime_ns/ctime_ns; '
                                      'missing or changed signatures require full source SHA256'),
                  'trust_unchanged_source_signatures': trust_unchanged_source_signatures},
              'status': 'reclaim_started'}
    record_path = archive_path.with_suffix(archive_path.suffix + '.reclaimed.json')
    with record_path.open('x') as output:
        output.write(json.dumps(record, indent=2)+'\n')
    for entry in files:
        current = entry.lstat()
        if not stat.S_ISREG(current.st_mode) or source_signature(current) != signatures[entry]:
            raise ValueError('source changed before release; stop')
        entry.unlink()
    for directory in sorted((p for p in paths if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        directory.rmdir()
    if entries is None or not any(root.iterdir()):
        root.rmdir()
    record['status'] = 'local_duplicates_reclaimed'
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('pack')
    create.add_argument('root', type=Path)
    create.add_argument('complete', type=Path)
    create.add_argument('destination', type=Path)
    create.add_argument('--entry', action='append', dest='entries',
                        help='archive only this immediate child; repeat for a bounded group, retaining original restore paths')
    create.add_argument('--failed', action='store_true', help='second positional argument is an explicit FAILED evidence document')
    check = commands.add_parser('verify')
    check.add_argument('archive', type=Path)
    release = commands.add_parser('reclaim')
    release.add_argument('archive', type=Path)
    release.add_argument('cloud_metadata', type=Path, help='independently fetched Drive file metadata JSON')
    release.add_argument('--allow-provider-size-only', action='store_true',
                         help='record weaker provider verification when connector omits checksums; local full hashes remain mandatory')
    release.add_argument('--trust-unchanged-source-signatures', action='store_true',
                         help='explicit same-host fast path for unchanged pack-time identity/ctime; archive hashes remain mandatory')
    args = parser.parse_args()
    if args.command == 'pack':
        result = pack(args.root, args.complete, args.destination, entries=args.entries, failed=args.failed)
    elif args.command == 'verify':
        result = verify(args.archive)
    else:
        result = reclaim(args.archive, args.cloud_metadata,
                         allow_provider_size_only=args.allow_provider_size_only,
                         trust_unchanged_source_signatures=args.trust_unchanged_source_signatures)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
