# Repository Title / Paper Title

**Additional figures, animations, data, and code associated with:**
**“Full Paper Title”**

Authors: **Author 1**, **Author 2**, **Author 3**
Affiliations: **Institution 1**, **Institution 2**
Corresponding author: **[name@email.com](mailto:name@email.com)**

---

## Paper Information

* **Title:** *Full Paper Title*
* **Authors:** Author 1, Author 2, Author 3
* **Journal / Conference:** Journal or Conference Name
* **Status:** Under review / Accepted / Published
* **DOI:** `TBD`
* **Preprint:** `TBD`
* **Publication date:** `TBD`

Suggested citation:

```bibtex
@article{AuthorYearPaper,
  title   = {Full Paper Title},
  author  = {Author, A. and Author, B. and Author, C.},
  journal = {Journal Name},
  year    = {YYYY},
  doi     = {DOI}
}
```

---

## Overview

This repository contains supplementary material associated with the paper:

> *Full Paper Title*

The repository includes additional figures, animations, scripts, numerical results, notebooks, and supporting material that complement the results presented in the manuscript.

The main purpose of this repository is to provide:

* Additional visual material not included in the main manuscript.
* Extended figures and animations supporting the interpretation of the results.
* Scripts and notebooks used to generate selected figures.
* Supplementary technical information to improve reproducibility.
* A structured archive of paper-related material.

---

## Abstract

Paste the paper abstract here.

```text
Abstract text...
```

---

## Repository Structure

```text
.
├── README.md
├── figures/
│   ├── main/
│   ├── supplementary/
│   └── extended/
├── animations/
├── scripts/
├── notebooks/
├── data/
├── results/
├── docs/
├── environment.yml
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

### Folder Description

| Folder                   | Description                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------------------- |
| `figures/`               | Figures associated with the paper, including supplementary and extended versions.                   |
| `figures/main/`          | Final figures appearing in the main manuscript.                                                     |
| `figures/supplementary/` | Figures included as supplementary material.                                                         |
| `figures/extended/`      | Additional figures not included in the final manuscript but useful for interpretation.              |
| `animations/`            | Videos, GIFs, or animations illustrating temporal evolution, parameter effects, or model behaviour. |
| `scripts/`               | Source code used to process data, run calculations, or generate figures.                            |
| `notebooks/`             | Jupyter notebooks for exploratory analysis or figure generation.                                    |
| `data/`                  | Input data, processed datasets, or references to external datasets.                                 |
| `results/`               | Numerical outputs, processed results, or intermediate files.                                        |
| `docs/`                  | Additional documentation, notes, or technical explanations.                                         |

---

## Figures

This section lists the figures available in the repository.

| Figure                  | File                                  | Description                                  | Related paper section  |
| ----------------------- | ------------------------------------- | -------------------------------------------- | ---------------------- |
| Figure 1                | `figures/main/figure_01.png`          | Short description of the figure.             | Section X              |
| Figure 2                | `figures/main/figure_02.png`          | Short description of the figure.             | Section X              |
| Supplementary Figure S1 | `figures/supplementary/figure_S1.png` | Short description.                           | Supplementary Material |
| Extended Figure E1      | `figures/extended/figure_E1.png`      | Additional diagnostic or explanatory figure. | Additional analysis    |

Recommended naming convention:

```text
figure_01_short-description.png
figure_02_short-description.pdf
figure_S1_short-description.png
figure_E1_short-description.png
```

For publication-quality material, vector formats such as `.pdf`, `.svg`, or `.eps` are recommended when possible.

---

## Animations

This section lists the animations or videos included in the repository.

| Animation   | File                          | Description                         | Format |
| ----------- | ----------------------------- | ----------------------------------- | ------ |
| Animation 1 | `animations/animation_01.gif` | Short description of what is shown. | GIF    |
| Animation 2 | `animations/animation_02.mp4` | Short description of what is shown. | MP4    |

Recommended naming convention:

```text
animation_01_short-description.gif
animation_02_short-description.mp4
```

Each animation should include a brief explanation of:

* What physical, numerical, or conceptual process is represented.
* Which variables are shown.
* The meaning of the colour scale, if applicable.
* The parameter values or simulation case used.
* The corresponding figure, section, or result in the paper.

---

## Code

The repository may include scripts and notebooks used for post-processing, figure generation, or additional analysis.

### Requirements

The code was developed using:

* Python: `X.Y`
* NumPy: `X.Y`
* SciPy: `X.Y`
* Matplotlib: `X.Y`
* Pandas: `X.Y`
* Other relevant packages: `TBD`

Install dependencies with:

```bash
pip install -r requirements.txt
```

or, if using Conda:

```bash
conda env create -f environment.yml
conda activate environment-name
```

### Running the Code

Example command:

```bash
python scripts/generate_figure_01.py
```

Expected output:

```text
figures/main/figure_01.png
```

If notebooks are provided, they can be launched with:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

---

## Data

Describe the data included in the repository.

| Dataset   | Location             | Description                         | Source                     |
| --------- | -------------------- | ----------------------------------- | -------------------------- |
| Dataset 1 | `data/raw/...`       | Raw input data.                     | Original / External source |
| Dataset 2 | `data/processed/...` | Processed data used for figures.    | Generated from scripts     |
| Dataset 3 | `results/...`        | Numerical results from simulations. | Generated internally       |

If the data are too large to be hosted directly on GitHub, provide the external location here:

* Data repository: `TBD`
* DOI: `TBD`
* Access conditions: Public / Restricted / Available upon request

---

## Reproducibility

To reproduce the main figures:

1. Install the required dependencies.
2. Download or generate the required data.
3. Run the scripts in `scripts/`.
4. Generated figures will be saved in `figures/` or `results/`.

Example workflow:

```bash
python scripts/preprocess_data.py
python scripts/generate_results.py
python scripts/generate_figures.py
```

For full reproducibility, document:

* Software versions.
* Input data.
* Random seeds, if applicable.
* Parameter values.
* Hardware requirements, if relevant.
* Any non-public data dependencies.

---

## Versioning

Recommended version tags:

| Repository version | Paper status | Description                                        |
| ------------------ | ------------ | -------------------------------------------------- |
| `v0.1`             | Draft        | Initial repository structure.                      |
| `v1.0`             | Submitted    | Version associated with manuscript submission.     |
| `v1.1`             | Revised      | Updated version after peer review.                 |
| `v2.0`             | Published    | Final version associated with the published paper. |

The repository version associated with the published paper should be archived using Zenodo or a similar service to obtain a permanent DOI.

---

## How to Cite

If you use this repository, please cite the associated paper:

```bibtex
@article{AuthorYearPaper,
  title   = {Full Paper Title},
  author  = {Author, A. and Author, B. and Author, C.},
  journal = {Journal Name},
  year    = {YYYY},
  doi     = {DOI}
}
```

If citing the repository directly:

```bibtex
@misc{AuthorYearRepository,
  author       = {Author, A. and Author, B. and Author, C.},
  title        = {Repository Title},
  year         = {YYYY},
  publisher    = {GitHub},
  version      = {vX.Y},
  doi          = {Repository DOI},
  url          = {Repository URL}
}
```

---

## Contact

For questions, comments, or requests, please contact:

**Corresponding author:**
Name Surname
Institution
Email: `name@email.com`

Additional contacts:

* Author 2: `email@example.com`
* Author 3: `email@example.com`

---

## License

Specify the license for the repository.

Recommended options:

* **Code:** MIT License / BSD-3-Clause / GPL-3.0
* **Figures and text:** Creative Commons Attribution 4.0 International, CC BY 4.0
* **Data:** CC BY 4.0 / CC0 / Custom license

Example:

```text
Code in this repository is released under the MIT License.
Figures, documentation, and supplementary material are released under the CC BY 4.0 License unless otherwise stated.
```

See the `LICENSE` file for details.

---

## Notes

* Some figures may differ slightly from the final published version due to journal formatting or post-processing.
* Large files may be stored externally and linked from this repository.
* Files marked as `draft`, `old`, or `deprecated` should not be used for citation or reproduction unless explicitly stated.
* The DOI and citation information will be updated after publication.

