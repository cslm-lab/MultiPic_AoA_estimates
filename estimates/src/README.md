# AoA_estimates_for_MultiPic.Rmd
R Markdown file used for the creation + analysis of our age-of-acquisition (AoA) norms.

**Required data (and their structure):**
- estimates:
	- data:
		- raw:
			- derivatives:
				- `item_based_data_add_info.csv`
	- src:
		- `helper_functions.R`
- external_resources:
	- norms:
		- `Birchenough_2017.csv`
		- `Schröder_2012.xls`
		- `Kuperman_2012.xlsx`
- study_setup:
	- data:
		- items_lists:
		    - `list_A_repeated.csv`
		    - `list_B_repeated.csv`
		    - `list_C_repeated.csv`
		    - `control_items.csv`

# data_wrangling.py
Script that takes the downloaded survey data and brings it into an item-based database format.

**Required data (and their structure):**
- estimates:
	- data:
		- raw:
			- survey:
				- `data_lists.csv`
				- `data_survey_start.csv`
				- `values_survey_start.csv`
- study_setup:
	- data:
		- items_lists:
		    - `list_A.csv`
		    - `list_A_repeated.csv`
		    - `list_B.csv`
		    - `list_B_repeated.csv`
		    - `list_C.csv`
		    - `list_C_repeated.csv`
		    - `control_items.csv`
		- `example sentences.ods`
    
**Creates `item_based_data.csv` in data/raw/derivatives directory.**

# fill_in_info_for_duplicates.py
Script that takes the final group-averaged AoA estimates for the unique MultiPic items calculated in `AoA_estimates_for_MultiPic.Rmd` and fills in the corresponding values for the duplicate items.

**Required data (and their structure):**
- estimates:
	- data:
		- `aoa_estimates_unique.csv`
- study_setup:
	- data:
		- `example_sentences.ods`
- external_resources:
	- `MultiPic_with_frequencies.csv`

**Saves a CSV with AoA + additional information for all 750 MultiPic items to `../data/aoa_estimates_complete.csv`.**

# helper_functions.R
Collection of custom functions used in `AoA_estimates_for_MultiPic.Rmd` so that the Rmd is cleaner.

# merge_database_infos.Rmd
Takes raw item-based data (output of `data_wrangling.py`) and combines it with information from existing databases: AoA estimates from Schröder (2012) and Birchenough (2017), and frequency information from SUBTLEX-DE (2011).
**It saves the result as `item_based_data_add_info.csv` in ../data/raw/derivatives.**

**Required data (and their structure):**
- estimates:
	- data:
		- raw:
			- derivatives:
				- `item_based_data.csv` --> output from `data_wrangling.py`
- external_resources:
	- norms:
		- `Birchenough_2017.csv`
		- `Schröder_2012.xls`
	- `MultiPic_with_frequencies.csv`
- study_setup:
	- data:
		- items_lists:
			- `familiarisation_items_overview.csv`


