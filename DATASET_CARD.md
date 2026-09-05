# Dataset Card

This project is designed for the TCIA **Bone Marrow Cytomorphology_MLL_Helmholtz_Fraunhofer** dataset, DOI [10.7937/TCIA.AXH3-T579](https://doi.org/10.7937/TCIA.AXH3-T579). See the [official collection page](https://www.cancerimagingarchive.net/collection/bone-marrow-cytomorphology_mll_helmholtz_fraunhofer/) for current access and usage terms.

## Class abbreviations

The dataset uses the following expert-defined cell categories:

| Code | Category |
| --- | --- |
| ABE | Abnormal eosinophil |
| ART | Artefact |
| BAS | Basophil |
| BLA | Blast |
| EBO | Erythroblast |
| EOS | Eosinophil |
| FGC | Faggott cell |
| HAC | Hairy cell |
| KSC | Smudge cell |
| LYI | Immature lymphocyte |
| LYT | Lymphocyte |
| MMZ | Metamyelocyte |
| MON | Monocyte |
| MYB | Myelocyte |
| NGB | Band neutrophil |
| NGS | Segmented neutrophil |
| NIF | Not identifiable |
| OTH | Other cell |
| PEB | Proerythroblast |
| PLM | Plasma cell |
| PMO | Promyelocyte |

## Data license

The dataset is identified by its publisher as available under the [Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), subject to the dataset publisher's access conditions and any applicable TCIA terms. Users must provide appropriate attribution, link the license, and indicate changes when redistributing or adapting dataset material.

The dataset is not included in this repository and is not relicensed under the repository's MIT license. Verify the current license and access terms at the source before downloading, sharing, or publishing derived material.

## Reproducibility

Obtain the dataset from TCIA and place extracted files under `data/raw/`. Run `inspect_dataset.py` and review `reports/dataset_summary.json` before preparation. The code infers class labels only from the recognized class-directory layout; it does not invent patient, slide, or source groups. Any metadata mapping and group key must be documented and validated after inspection.

Do not commit raw images, processed splits, patient identifiers, credentials, uploads, model weights, or prediction images. Only reviewed, de-identified aggregate reports may be retained.

## Current download inspection

The official TCIA page lists 171,375 JPG images from 945 subjects and describes the cells as de-identified and expert annotated. The historical local inspection counted 171,374 JPG files, so that download was not a byte-for-byte complete copy of the official image count. Its file layout exposed no patient, slide, case, or source mapping usable by this project. The recorded experiment therefore used a stratified image-level split, not a patient-level split. This can make performance optimistic if related cells occur across splits.

The expert annotation validates the cell-category labels; it does not provide the patient grouping information needed for an independent patient-level test split.

The historical report found 20 unreadable or incomplete EBO files, while the recorded prepared set contains 21 fewer images than the local JPG count. It also found duplicate content under MON and PMO class folders. Because the historical report predates full-decode validation, it must be regenerated after repairing or redownloading the data. The current preparation path fails closed on invalid images and any duplicate content by default; a conflicting label requires domain review before a new scientific claim.

The committed `reports/dataset_summary.json` is sanitized historical evidence and contains no personal filesystem path. A new report and split manifest must identify the code/configuration context used for a new experiment.
