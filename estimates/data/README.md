# Raw data from the study + derived age-of-acquisition estimates

- **raw:**
    - **derivatives:**
        - `item_based_data.csv`: Output from [`data_wrangling.py`](../src/README.md#data_wranglingpy). Wrangled survey data (from the survey directory) that has been reshaped into an item-based format in order to make analyses easier.
        - `item_based_data_add_info.csv`: Output from [`merge_database_infos.Rmd`](../src/README.md#merge_database_infosrmd). It is an extension of `item_based_data.csv`, with additional information from other existing databases.
    - **survey:** Contains the raw data from the survey. The data are split into the introduction pages which were the same for all participants (`data_survey_start.csv`) and the individual lists (`data_lists.csv`). Additionally, each data csv comes with 3 information files: the `codebook`, and an information file each regarding the used `values` and `variables`.
- `aoa_estimates_complete.csv`: Final age-of-acquisition estimates for all 750 items of the MultiPic corpus.
- `aoa_estimates_unique.csv`: Final age-of-acquisition estimates -- contains only the 715 unique word items from the MultiPic corpus with the highest name agreement.

## Structure of the final AoA files
- **item_number**: MultiPic item numbers.
- **item**: MultiPic word items corresponding to the item numbers.
- **estimate_mean**: Group mean age-of-acquisition estimates (ratings collected on a continuous scale).
- **estimate_sd**: Standard deviation of `estimate_mean`.
- **min**: Smallest estimated acquisition age for each item.
- **max**: Highest estimated acquisition age for each item.
- **estimateLikert_mean**: Group mean age-of-acquisition estimates on a 7-point Likert scale (each continuous rating transformed into Likert rating, then group mean taken from all transformed Likert ratings).
- **estimate_Likert_sd**: Standard deviation of the Likert-transformed age-of-acquisition estimates (`estimateLikert_mean`).
- **minLikert**: Smallest estimated Likert-transformed acquisition age for each item.
- **maxLikert**: Highest estimated Likert-transformed acquisition age for each item.
- **example_sentence**: Disambiguating example sentences that were presented together with the items in the questionnaire.
- **H_INDEX**: The value of the H index (name agreement) for each item's drawing, as provided by MultiPic.
- **VISUAL_COMPLEXITY**: The mean rating of visual complexity given to each item's drawing in a 1 (very simple) to 5 (very complex) scale, as provided by MultiPic.
- **lgSUBTLEX**: The log word frequency as provided by SUBTLEX-DE (log10(number of times the word is encountered in the corpus independent of letter case + 1)).
- **B: AoA mean**: Birchenough et al. (2017): group mean age-of-acquisition estimates (ratings collected on a continuous scale).
- **B: AoA SD**: Birchenough et al. (2017): standard deviation of `B: AoA mean`.
- **B: min**: Birchenough et al. (2017): lowest individual rating received.
- **B: max**: Birchenough et al. (2017): largest individual rating received.
- **B: AoALikert mean**: Birchenough et al. (2017): group mean age-of-acquisition estimates converted into a score on a 7-point Likert scale.
- **B: AoALikert SD**: Birchenough et al. (2017): standard deviation of `B: AoALikert mean`.
- **B: minLikert**: Birchenough et al. (2017): lowest score received on Likert scale.
- **B: maxLikert**: Birchenough et al. (2017): highest score received on Likert scale.
- **S: AoALikert mean**: Schröder et al. (2012): group mean age-of-acquisition estimates (ratings collected on a 7-point Likert scale).
- **S: AoALikert SD**: Schröder et al. (2012): standard deviation of `S: AoALikert mean`.