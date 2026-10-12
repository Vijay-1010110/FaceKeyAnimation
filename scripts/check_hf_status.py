import sys
import io
import re
from collections import defaultdict
from datetime import datetime, timezone
from huggingface_hub import HfApi

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

api = HfApi()
repo_id = 'VijayTheOne/facekey-gen3-dataset'

commits = api.list_repo_commits(repo_id=repo_id, repo_type='dataset')
files = api.list_repo_files(repo_id=repo_id, repo_type='dataset')

tar_files = [f for f in files if f.endswith('.tar.gz') or f.endswith('.npz')]

print(f"Total dataset files on HF: {len(files)}")
print(f"Total chunk archives: {len(tar_files)}")
print(f"Total commits logged: {len(commits)}")

# Analyze source patterns from filenames/commit titles
source_stats = defaultdict(lambda: {'count': 0, 'latest_time': None, 'latest_file': '', 'workers': set()})

for c in commits:
    title = c.title
    created = c.created_at
    match = re.search(r'dataset_chunk_([a-zA-Z0-9\-_]+)_w(\d+)_([0-9_]+)\.tar\.gz', title)
    if match:
        src = match.group(1)
        worker = match.group(2)
        stat = source_stats[src]
        stat['count'] += 1
        stat['workers'].add(worker)
        if stat['latest_time'] is None or created > stat['latest_time']:
            stat['latest_time'] = created
            stat['latest_file'] = title
    else:
        m2 = re.search(r'dataset_chunk_([a-zA-Z0-9\-_]+)', title)
        src = m2.group(1) if m2 else 'initial_setup'
        stat = source_stats[src]
        stat['count'] += 1
        if stat['latest_time'] is None or created > stat['latest_time']:
            stat['latest_time'] = created
            stat['latest_file'] = title

print('\n' + '='*80)
print('SUMMARY BY SOURCE / NODE:')
print('='*80)
for src, data in sorted(source_stats.items(), key=lambda x: str(x[1]['latest_time']), reverse=True):
    workers_str = ', '.join(f'w{w}' for w in sorted(data['workers'])) if data['workers'] else 'N/A'
    print(f"Source: {src:<22} | Chunks: {data['count']:>3} | Active Workers: [{workers_str}]")
    print(f"  |-- Latest Commit: {data['latest_time']}")
    print(f"  |-- Last Chunk:    {data['latest_file']}\n")

print('\n' + '='*80)
print('STORAGE VOLUME BY SOURCE / NODE:')
print('='*80)
tree = list(api.list_repo_tree(repo_id=repo_id, repo_type='dataset', path_in_repo='chunks'))
vol_stats = defaultdict(lambda: {'count': 0, 'bytes': 0})
for item in tree:
    match = re.search(r'dataset_chunk_([a-zA-Z0-9\-_]+)_w\d+', item.path)
    src = match.group(1) if match else 'other'
    vol_stats[src]['count'] += 1
    vol_stats[src]['bytes'] += item.size if hasattr(item, 'size') and item.size else 0

total_all_gb = sum(v['bytes'] for v in vol_stats.values()) / (1024**3)
for src, data in vol_stats.items():
    gb = data['bytes'] / (1024**3)
    mb = data['bytes'] / (1024**2)
    pct = (gb / total_all_gb * 100) if total_all_gb > 0 else 0
    print(f"Source: {src:<22} | Chunks: {data['count']:>3} | Total Size: {gb:5.2f} GB ({mb:8.1f} MB) | {pct:4.1f}%")

print(f"\nTOTAL DATA COLLECTED ON GEN 3 HUB: {total_all_gb:.2f} GB ({len(tree)} chunks)")

