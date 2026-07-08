#!/bin/bash
set -e

python -c "
from taxonerd import TaxoNERD
import os

ner = TaxoNERD(prefer_gpu=True)
model = os.environ.get('TAXONERD_MODEL', 'en_ner_eco_md')
ner.load(
    model=model,
    linker='gbif_backbone',
    threshold=0.7,
    exclude=['tagger','parser','taxo_abbrev_detector','taxon_linker','pysbd_sentencizer']
)
ner.find_in_text('Homo sapiens')
print('READY')
"

exec uvicorn duui_taxonerd:app --host 0.0.0.0 --port 9714 --workers 1
