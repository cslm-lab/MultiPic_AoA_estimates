"""
This script takes the calculated unique age-of-acquisition estimates and
fills in the correct information for the duplicate MultiPic items.
It saves a file `aoa_estimates_complete.csv` to the data directory.

Run in terminal with: $ python3 fill_in_info_for_duplicates.py
"""

import pandas as pd

# define paths
aoa_path = '../data/aoa_estimates_unique.csv'
sentences_path = '../../study_setup/data/example_sentences.ods'
mp_freq_path = '../../external_resources/MultiPic_with_frequencies.csv'

# load databases
aoa_df = pd.read_csv(aoa_path)
sentences_df = pd.read_excel(sentences_path, engine='odf', sheet_name='MultiPic') 
rename_dict = {'ITEM': 'item_number', 'NAME1': 'item'}
sentences_df = sentences_df.rename(columns=rename_dict)
mp_freq_df = pd.read_csv(mp_freq_path)

# create final df with all items
merged_df = pd.merge(sentences_df, aoa_df, on=['item', 'item_number'], how='left')

# find duplicate rows
# truly duplicate items share the same item name and example sentence
duplicate_rows = merged_df[merged_df.duplicated(subset=['item', 'EXAMPLE'],keep=False)]
duplicate_values = set(duplicate_rows['item'].values)

# iterate through duplicate values
for i in duplicate_values:
    # get corresponding dataframe entries
    entries = duplicate_rows[duplicate_rows['item']==i]
    # find out which entry is still missing info + which contains the info
    na_idx = entries[entries['estimate_mean'].isna()].index[0]
    info_idx = entries[entries['estimate_mean'].notna()].index[0]
    # copy info 
    merged_df.loc[na_idx,'estimate_mean':'S: AoALikert SD'] = merged_df.loc[info_idx,'estimate_mean':'S: AoALikert SD']
    # correct MultiPic info
    merged_df.loc[na_idx,'H_INDEX'] = mp_freq_df.loc[na_idx,'H_INDEX']
    merged_df.loc[na_idx,'VISUAL_COMPLEXITY'] = mp_freq_df.loc[na_idx,'VISUAL_COMPLEXITY']

# drop column with example sentence
merged_df.drop(columns='EXAMPLE', inplace=True)

# save estimates
merged_df.to_csv('../data/aoa_estimates_complete.csv', index=False)