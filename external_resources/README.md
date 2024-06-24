# External resources
For all code to function in the [study_setup](../study_setup/) and [estimates](../estimates/) directories,
the following structure in external_resources is expected:
- frequencies:
    - *SUBTLEX-DE_cleaned_with_Google00.txt*
- multipic:
    - *German_MultiPic_version1.csv*
    - *MultiPic_version5.csv*
- norms:
    - *Birchenough_2016.csv*
    - *Schröder_2012.xls*
    - *Kuperman_2012.xlsx*
- *MultiPic_with_frequencies.csv*

**The SUBTLEX-DE website + data storage is currently under construction and unavailable in its usual address (as of April 18, 2024)**. The former website can still be accessed through a web archive (e.g. here: [website Aug 20, 2022](https://web.archive.org/web/20220820044707/http://crr.ugent.be/); [data storage Aug 15, 2022](https://web.archive.org/web/20220815051558/http://crr.ugent.be/subtlex-de/)).

For convenience, the SUBTLEX-DE frequencies are provided in this repository already (until such a time where the corpus can be easily downloaded again automatically).
The rest of **the expected structure + databases can be automatically created and downloaded by running the [download_corpora script](download_corpora.py)**.

## download_corpora.py
This script downloads all existing corpora necessary for the survey's stimuli creation
as well as the estimate validation (currently except for SUBTLEX-DE, see comment above):
- Schröder (2012)
- Birchenough (2016)
- MultiPic (2018, 2022)
- Kuperman (2012)
- (SUBTLEX-DE; Marc Brysbaert's website currently under construction)

The script can be run by opening a terminal to the location of the script and using the command `$ python3 download_corpora.py`

If you already have those databases saved locally, feel free to change the paths in the 
specific files to point to your copies instead of downloading them anew.


## merge_multipic_subtlex.py
This script combines information from the [German MultiPic database (version 1)](https://www.bcbl.eu/databases/multipic/) and the [German SUBTLEX (SUBTLEX-DE) corpus](http://crr.ugent.be/archives/534) and saves it in a new CSV file: [MultiPic_with_frequencies.csv](MultiPic_with_frequencies.csv).

The script can be run by opening a terminal to the location of the script and using the command `$ python3 merge_multipic_subtlex.py`.

When running the script, it first asks the user whether the necessary corpora are already present in the expected format (see description [here](#external-resources)). **If yes**, the script continues with the expected paths to the corpora.
**If no,** the script asks the user to enter the path to where the specific file is saved, so it can access it in the later steps.

In the actual information combination step, the script will collect the following information for each item of the MultiPic corpus:
- from MultiPic itself:
    - ITEM
    - PICTURE
    - NAME1
    - H_INDEX
    - PERCENTAGE_MODAL_NAME
    - VISUAL_COMPLEXITY
- from SUBTLEX-DE:
    - SUTBLEX
    - lgSUBTLEX
    - Google00pm
    - lgGoogle00
    
If there is only one orthographic variant of the current MultiPic item in the SUBTLEX-DE corpus, the frequency information of this entry will be picked. If there are several orthographic variants in SUBTLEX-DE for one MultiPic item, the variant that officially has the correct spelling will be picked. For more background on this reasoning step see [the notebook exploring the frequencies](../study_setup/notebooks/exploring_frequencies.ipynb)

The finished dataframe is then saved locally as *MultiPic_with_frequencies.csv*.

The file with merged information (and the corpora, if they had to be downloaded) can be found in [the data directory](../data/).

## MultiPic_with_frequencies.csv
Output of `merge_multipic_subtlex.py`.
Combines information from the German MultiPic (version 1) and SUBTLEX-DE for convenient word information retrieval.
For more information on the contents see [the merging script description](#merge_multipic_subtlexpy).