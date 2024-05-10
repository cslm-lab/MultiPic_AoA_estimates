# items_lists.py
This script automatically determins familiarisation, target and control items for the AoA questionnaire, taking into account that we want to split the target items from the MultiPic corpus across 3 weighted lists.

Taking the German MultiPic, the Birchenough et al. (2016) AoA database, and the frequency information from SUBTLEX-DE, the script:

- saves a list of potentially duplicate items
- combines the MultiPic database with our example sentences
- removes duplicate entries (so there is only one of each)
- selects 31 control items (aka items shared across lists)
- assigns all other items to 3 lists (á 228 items)
- selects 25 repeating items per list
- selects 10 familiarisation items

The collections of control items, list-specific target and reated items, and familiarisation items are saved as individual CSV files in [items_lists](../data/items_lists/), where the information about dublicate items is also stored.

**PREREQUISITE**: This scipt expects that the script 
[`merge_multipic_subtlex.py`](../../external_resources/merge_multipic_subtlex.py) has already run.
It also expects the Birchenough et al. (2016) data, SUBTLEX-DE, and our MultiPic example sentences to be present (which can be achieved by running the script [`download_corpora.py`](../../external_resources/download_corpora.py), see [external_resources](../../external_resources/)).

The script can be run by opening the script location in a terminal and typing:
`$ python3 items_lists.py`
