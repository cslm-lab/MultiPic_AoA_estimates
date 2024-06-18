# Dataset of German age-of-acquisition norms for all 750 items of the MultiPic database

This repository contains the final German age-of-acquisition (AoA) norms for all 750 images of the [Multilingual Picture (MultiPic) database](https://www.bcbl.eu/databases/multipic), as well as all code and background information for the creation of our study questionnaires and the development and analysis of the AoA norms themselves. 
If you are interested in the final survey set-up, please check out the corresponding [OSF Project](https://doi.org/10.17605/OSF.IO/3ETJY).

[Klick here to get to the final AoA estimates directly.](estimates/data/)

## Structure of the repository
- **estimates:** Contains the AoA norms for all 750 items of the MultiPic corpus, as well as all raw study data and code used to derive those norms.
- **external_resources:** Contains a script (`download_corpora.py`) for downloading all already existing databases which are required for code in the [estimates](estimates/) and [study_setup](study_setup/) directories. Also contains a convenient word information overview document and the code to create it.
- **study_setup:** Contains all data, exploration and final code which serve as the base for creating our AoA questionnaire. 

## Installing requirements
To ensure that all code from this repository runs smoothly, you can use the provided environment file (`aoa_environment.yaml`) to replicate our working environment.

If you are using conda or mamba for managing virtual environments, simply copy the fitting line for execution in your terminal:

`$ conda env create --file aoa_environment.yaml`

`$ mamba env create --file aoa_environment.yaml`

By default, the new environment will be called `aoa`.

=================================================================================
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
