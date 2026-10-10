"""Download the pinned small OmniJev model and its Qwen backbone."""

import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from huggingface_hub import snapshot_download

MODELS = [
    ('Qwen/Qwen3.5-0.8B', '2fc06364715b967f1860aea9cf38778875588b17', 'models/Qwen3.5-0.8B'),
    ('tinnel123/OmniJev-0.8B', '3c9aa908f3c9d3f3ccf16bb281ee705d9e98252f', 'models/OmniJev-0.8B-v1.1'),
]

if __name__ == '__main__':
    for repo, revision, output in MODELS:
        snapshot_download(repo, revision=revision, local_dir=output, max_workers=3,
                          ignore_patterns=['results*', 'README.md', '.gitattributes'])
        Path(output, 'source_revision.json').write_text(
            json.dumps({'repo': repo, 'revision': revision}, indent=2), encoding='utf-8',
        )
        print('Ready:', output, flush=True)
