#!../venv/bin/python3
import os
from huggingface_hub import snapshot_download

dest = os.getenv('HF_HOME', default='../.hf_models')
os.makedirs(dest, exist_ok=True)

revisions = {
	'bert-base-cased': 'cd5ef92a9fb2f889e972770a36d4ed042daf221e',
	'distilbert-base-cased': '6ea81172465e8b0ad3fddeed32b986cdcdcffcf0',
	'roberta-base': 'e2da8e2f811d1448a5b465c236feacd80ffbac7b',
	'microsoft/deberta-v3-base': '8ccc9b6f36199bec6961081d44eb72fb3f7353f3'
}

for model, rev in revisions.items():
    snapshot_download(repo_id=model, revision=rev, local_dir=dest, cache_dir=dest)
