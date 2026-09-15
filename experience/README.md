# Released Experience Library

The curated ExpGeoLoc experience library is hosted on
[Baidu Netdisk (code: `fz6i`)](https://pan.baidu.com/s/1TKP4gepjlfJ1DzFTF12GTg?pwd=fz6i)
because the SQLite database exceeds GitHub's single-file size limit.

The release contains 89 merged and deduplicated cross-task experience entries.
After downloading and extracting it, point `--experience-dir` to the directory
that directly contains:

```text
geoexp7k/
├── memory.sqlite
└── chroma/
```

Install Chroma support and run the frozen-library evaluation with:

```bash
pip install ".[memory]"
python scripts/run_dataset_eval.py \
  --datasets geoexp7k-test \
  --experience-mode retrieve_only \
  --experience-dir path/to/geoexp7k \
  --workers 4 \
  --resume
```

SQLite is the authoritative experience store. The bundled Chroma directory is
its persistent semantic-retrieval index; keep the two together when moving the
library.
