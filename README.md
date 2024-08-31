# REM
Source code for paper “REM: An Integrative Rule Extraction Methodology for Explainable Data Analysis in Healthcare”. The pre-print is available at: https://www.biorxiv.org/content/10.1101/2021.01.22.427799v2.


# Overview
Deep learning models are receiving increasing attention in clinical decision-making, however the lack of explainability impedes their deployment in day-to-day clinical practice. We propose REM, an explainable methodology for extracting rules from deep neural networks and combining them with rules from non-deep learning models. This allows integrating machine learning and reasoning for investigating basic and applied biological research questions. We evaluate the utility of REM in two case studies for the predictive tasks of classifying histological and immunohistochemical breast cancer subtypes from genotype and phenotype data. We demonstrate that REM efficiently extracts accurate, comprehensible rulesets from deep neural networks that can be readily integrated with rulesets obtained from tree-based approaches. REM provides explanation facilities for predictions and enables the clinicians to validate and calibrate the extracted rulesets with their domain knowledge. With these functionalities, REM caters for a novel and direct human-in-the-loop approach in clinical decision-making.

# Requirements
REM requires Python and R installation. Versions the code is tested on are Python 3.7 and R 3.6.

Python dependencies are listed in "code/requirements.txt".

The following packages need to be installed in R:
- `C50`
- `Cubist`
- `Formula`
- `inum`
- `libcoin`
- `magrittr`
- `mvtnorm`
- `partykit`
- `plyr`
- `Rcpp`
- `reshape2`
- `stringr`
- `stringi`

# Demo
A quick demonstration of rule extraction from neural networks using Breast Cancer Wisconsin dataset is provided in demo.ipynb.

# Instruction for use
For how to run the software on your data, see demo.ipynb.
Reproduction instruction are provided in "code/reproduction_instruction.txt".

# License
This project is covered under the MIT License.
