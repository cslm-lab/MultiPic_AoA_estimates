"""
- save list of potentially duplicate items
- combine MultiPic database with our example sentences
- remove duplicate entries (so there is only one of each)
- select 31 control items (aka items shared across lists)
- assign all other items to 3 lists (á 228 items)
- select 25 repeating items per list
- select 10 familiarisation items

PREREQUISITE: this scipt expects that the script 
`merge_multipic_subtlex.py` has already run.
It also expects the Birchenough et al. (2017) data, our 
MultiPic example sentences, and SUBTLEX-DE to be present.

TO RUN THE SCRIPT: open script location in terminal and type:
$ python3 items_lists.py
"""

# import relevant packages
import pandas as pd
import numpy as np
import csv
import random
import os

# define paths
mp_freq_path = '../../external_resources/MultiPic_with_frequencies.csv'
aoa_path = '../../external_resources/norms/Birchenough_2017.csv'
sentences_path = '../data/example_sentences.ods'
# this is still needed because we want to find good familiarisation
# items that are NOT present in MultiPic already, and we need 
# frequency information for those items, too
subtlex_path = '../../external_resources/frequencies/SUBTLEX-DE_cleaned_with_Google00.txt'

# saving path
save_path = '../data/items_lists/'
os.makedirs(save_path, exist_ok=True)

# define functions for easier use
def remove_umlauts(string):
    """ Removes umlauts and ß and lowercases strings.
    Input:
        string: A string.
    Output:
        new_string: Same string in lowercase and without umlauts.
    """
    umlauts = {'ä':'ae','ö':'oe','ü':'ue','ß':'ss'}
    new_string = string.lower()
    for umlaut in umlauts:
        new_string = new_string.replace(umlaut, umlauts[umlaut])
    return new_string

def select_repreated_items(df, items_list):
    """
    For a given list of items and a fitting dataframe, this function
    first draws one random item from each of the AoA bins (expected: 5) 
    to ensure that the supported range is represented at all. Then,
    another 20 items will be drawn randomly from the entire pool of 
    remaining items per list.

    Input:
        df: dataframe of combined info from MultiPic + Birchenough (2017);
            column of AoA bins expected additionally.
        items_list: list of item numbers from list A, B or C.
    Output:
        selection: list of 25 items that shall be repeated in the experiment.
    """
    random.seed(43)
    # prepare empty list for collection of repeated items
    selection = []
    # filter dataframe to items present in given list
    list_df = df[df['ITEM'].isin(items_list)]
    # set of items present in list
    all_items = set(list_df['ITEM'])

    # select one item from each bin
    for i in range(5):
        bin_items = set(list_df[list_df['AoA bins'] == i]['ITEM'])
        item = random.sample(bin_items,k=1)
        # append item to list
        selection += item
        # remove drawn item from list item pool
        all_items -= set(item)
    
    # draw further 24 items from remaining list item pool
    items = random.sample(all_items,k=20)
    # append items to list
    selection += items
    return selection

######################################################################################
# load databases
print('>> Load databases...')
# MultiPic with frequencies
mp_freq_df = pd.read_csv(mp_freq_path)

# Birchenough et al. (2017)
aoa_df = pd.read_csv(aoa_path, encoding='latin_1', usecols=[0,4,5,6,7,8,9,10,11,12,13])
# lowercase words + remove umlauts to make it comparable to MultiPic vers. 1
for i in aoa_df.index:
    cleaned_string = remove_umlauts(aoa_df.loc[i,'Word'])
    aoa_df.loc[i,'Word'] = cleaned_string

# example senteces
sentences_df = pd.read_excel(sentences_path, engine='odf', usecols=[0,2], sheet_name='MultiPic')
print('Done.')

# load cleaned SUBTLEX-DE dataset as dataframe, use word + spellcheck + SUBTLEX + lgSUBTLEX + Google00pm + lgGoogle00
subtlex_df = pd.read_csv(subtlex_path, sep='\t', decimal=',', encoding='latin_1', usecols=[0,2,4,5,8,9])

####################################
# save a list of item names that occur several times
print('>> \nSave list of items in MultiPic that occur more than once...')
rows = mp_freq_df[mp_freq_df.duplicated(subset=['NAME1'],keep=False)]
values = list(set(rows['NAME1'].values))
with open(save_path+'item names occurring several times', 'w') as f: # !! adapt path!!!
    write = csv.writer(f, delimiter='\n')
    write.writerow(values)
print('Done.')

# combine MultiPic with our example sentences
print('>> \nCombine MultiPic with created example sentences...')
mp_freq_df = mp_freq_df.merge(sentences_df,how='outer', on='ITEM')
print('Done.')

# find duplicate rows, remove one of each
print('>> \nRemove duplicate items...')
# truly duplicate items share the same item name and example sentence
duplicate_rows = mp_freq_df[mp_freq_df.duplicated(subset=['NAME1', 'EXAMPLE'],keep=False)]
duplicate_values = set(duplicate_rows['NAME1'].values)

# iterate through duplicate values
for i in duplicate_values:
    # get corresponding dataframe entries
    entries = duplicate_rows[duplicate_rows['NAME1']==i]
    # find minimum within entries
    h_min = entries['H_INDEX'].min()
    # drop rows from dataframe if they don't have the minimum H index
    for i in entries.index:
        if entries.loc[i, 'H_INDEX'] != h_min:
            mp_freq_df.drop(index=i, inplace=True)

# reset index
mp_freq_df.reset_index(drop=True,inplace=True)
print('Done.')
print(f'There are {len(duplicate_values)} truly duplicate values in the original dataframe.')
print('Amount of unique items:',len(mp_freq_df))

# split unique items into control items and 3 unique lists
print('\n>> Assign items to control items and 3 lists...')

# determine frequency bins; add to dataframe
print('Divide total word list into 10 equally sized frequency bins')
freq_col = mp_freq_df['lgSUBTLEX']
mp_freq_df['freq bins'] = pd.qcut(freq_col,q=10,labels=False, precision=10)

print('Iterate through frequency bins, randomly assign items to lists')
# set up lists
shared_items_list = []
list_A = []
list_B = []
list_C = []

# set random seed
random.seed(43)

# set beginning of "leftover" iteration
add_to = 'A'

# iterate through the 10 frequency bins
for i in range(10): 
    freq_bin = mp_freq_df[mp_freq_df['freq bins'] == i]
    item_set = set(freq_bin['ITEM'])

    # in each bin, pick appropriate items for each category 
    # for shared items: draw 3 random items
    shared_items = random.sample(item_set, k=3)
    # append items to shared items list  
    shared_items_list += shared_items
    # remove drawn items from item pool
    item_set -= set(shared_items)

    # for different lists: divide remaining items from frequency bin
    # into 3 lists
    div_items = round(len(item_set)/3, 1)
    # if cleanly divisible
    if str(div_items).endswith('0'):
        k = int(len(item_set)/3)
        # add same amount of items to all lists
        for l in [list_A, list_B, list_C]:
            # draw k random items
            list_items = random.sample(item_set, k=k)
            # append items to list
            l += list_items
            # remove drawn items from item pool
            item_set -= set(list_items)
    # if not cleanly divisible: make sure distribution 
    # will add up to 3 equally long lists in the end
    else:
        # only distribute same amount of items for now 
        k = len(item_set)//3
        for l in [list_A, list_B, list_C]:
            # draw k random items
            list_items = random.sample(item_set, k=k)
            # append items to list
            l += list_items
            # remove drawn items from item pool
            item_set -= set(list_items)
        # assign "leftovers"
        while item_set:
            # draw one random item
            item = random.sample(item_set, k=1)
            # add item to one list in turn
            if add_to == 'A':
                list_A += item
                item_set -= set(item)
                add_to = 'B'
            elif add_to == 'B':
                list_B += item
                item_set -= set(item)
                add_to = 'C'
            elif add_to == 'C':
                list_C += item
                item_set -= set(item)
                add_to = 'A'
    
    # sanity check:
    print(f'Finished frequency bin {i}')
    if len(item_set) != 0:
        print(f'Something went wrong in frequency bin {i}, items still unassigned!')
        print(item_set)

# distribute items without frequency information, too!
nan_bin = mp_freq_df[freq_col.isna()==True]
nan_item_set = set(nan_bin['ITEM'])

# pick one item for the shared list, append
shared_item = random.sample(nan_item_set, k=1)
shared_items_list += shared_item
# remove drawn item from item pool
nan_item_set -= set(shared_item)

# pick items for individual lists, append
k = len(nan_item_set)//3
# add same amount of items to all lists
for l in [list_A, list_B, list_C]:
    # draw k random items
    list_items = random.sample(nan_item_set, k=k)
    # append items to list
    l += list_items
    # remove drawn items from item pool
    nan_item_set -= set(list_items)
# assign "leftovers" equally to lists
while nan_item_set:
    # draw one random item
    item = random.sample(nan_item_set, k=1)
    # add item to one list in turn
    if add_to == 'A':
        list_A += item
        nan_item_set -= set(item)
        add_to = 'B'
    elif add_to == 'B':
        list_B += item
        nan_item_set -= set(item)
        add_to = 'C'
    elif add_to == 'C':
        list_C += item
        nan_item_set -= set(item)
        add_to = 'A'

# save lists to csv
print('Save lists to csv')
for l in [list_A, list_B, list_C, shared_items_list]:
    l.sort()
np.savetxt(save_path+'list_A.csv', list_A, delimiter=', ', fmt='% i')
np.savetxt(save_path+'list_B.csv', list_B, delimiter=', ', fmt='% i')
np.savetxt(save_path+'list_C.csv', list_C, delimiter=', ', fmt='% i')
np.savetxt(save_path+'control_items.csv', shared_items_list, delimiter=', ', fmt='% i')
print('Done.')
print(f'Final list lengths:\nControls: {len(shared_items_list)}, A: {len(list_A)}, B: {len(list_B)}, C: {len(list_C)}')

####################################
# select repeated items for each list
print('\n>> Select repeated items per list (not for control items)...')

# find items in Birchenough (2017) that are in MultiPic
print('Find items in Birchenough (2017) that are in MultiPic')
rep_df = aoa_df.merge(mp_freq_df.rename(columns={'NAME1':'Word'}), on='Word')
rep_df.reset_index(drop=True, inplace=True)

# limit df to items with relatively low SD
print('Limit df to items with relatively low SD (< mean+std)')
mean = rep_df['SD'].describe()['mean']
std = rep_df['SD'].describe()['std']
std_rep_df = rep_df[rep_df['SD'] < mean+std].copy()

# add a column of AoA bins to df
std_rep_df['AoA bins'] = pd.cut(std_rep_df['AoAestimate'],5,labels=False)

# select items that shall be reapeated in the experiment
rep_A = select_repreated_items(std_rep_df, list_A)
rep_B = select_repreated_items(std_rep_df, list_B)
rep_C = select_repreated_items(std_rep_df, list_C)

# sort lists
for l in [rep_A, rep_B, rep_C]:
    l.sort()
    
# save lists to csv
np.savetxt(save_path+'list_A_repeated.csv', rep_A, delimiter=', ', fmt='% i')
np.savetxt(save_path+'list_B_repeated.csv', rep_B, delimiter=', ', fmt='% i')
np.savetxt(save_path+'list_C_repeated.csv', rep_C, delimiter=', ', fmt='% i')

####################################
# select familiarisation items
print('\n>> Select items for familiarisation phase from Birchenough et al. (2017)...')
print('Combine database with frequency information from SUBTLEX-DE')

# keep rows from Birchenough (2017) that are not in MultiPic
fam_df = aoa_df[~aoa_df['Word'].isin(mp_freq_df['NAME1'].values)]
fam_df.reset_index(drop=True, inplace=True)

# get AoA values
fam_vals = set(fam_df['Word'].values)

# get SUBTLEX values
subtlex_vals = set(subtlex_df['Word'])
# make tokens comparable to MultiPic 
# by lowercasing + removing umlauts and ß
# create dict to map original to cleaned token
subtlex_vals_dict = dict()
for token in subtlex_vals:
    cleaned = remove_umlauts(token)
    # create new cleaned dict key if it doesn't exist yet
    if not cleaned in subtlex_vals_dict.keys():
        subtlex_vals_dict[cleaned] = []
    # map original to cleaned token
    subtlex_vals_dict[cleaned].append(token)
# create a set variable for easier calling
subtlex_vals_cleaned = set(subtlex_vals_dict.keys())

# combine AoA + frequency information
# prepare relevant SUBTLEX columns
fam_df = fam_df.reindex(columns=fam_df.columns.tolist()+['SUBTLEX','lgSUBTLEX','Google00pm','lgGoogle00'])
# fill frequency cells
for token in fam_vals:
    # check if token is present in SUBTLEX-DE
    if token in subtlex_vals_cleaned:
        # get correct row in SUBTLEX-DE
        index = 0
        # if there is more than one orthographic variant in SUBTLEX-DE
        if len(subtlex_vals_dict[token]) > 1:
            # check if entry 2 has correct spelling
            variant = subtlex_vals_dict[token][1]
            row = subtlex_df[(subtlex_df['Word'] == variant) & (subtlex_df['spell-check OK (1/0)'] == 1)]
            if len(row) > 0:
                # if yes, take note of index
                index = row.index[0]
        # else take row index from first variant
        variant = subtlex_vals_dict[token][0]
        index = subtlex_df[subtlex_df['Word'] == variant].index[0]

        # account for the fact that some names might occur more than once
        # add frequency information to combined_df
        indices = fam_df[fam_df['Word'] == token].index
        for i in range(len(indices)):
            fam_df.loc[indices[i], 'SUBTLEX'] = subtlex_df.loc[index, 'SUBTLEX']
            fam_df.loc[indices[i], 'lgSUBTLEX'] = subtlex_df.loc[index, 'lgSUBTLEX']
            fam_df.loc[indices[i], 'Google00pm'] = subtlex_df.loc[index, 'Google00pm']
            fam_df.loc[indices[i], 'lgGoogle00'] = subtlex_df.loc[index, 'lgGoogle00']


# NOTE: Manual selection was necessary!
# for selection process and reasoning see selecting_items.ipynb
fam_items = ['becher','reis','zeugnis','komma','kloster','solo','seuche','reaktor','hypothek','dozent']
fam_filtered_df = fam_df[fam_df['Word'].isin(fam_items)].sort_values(by=['AoAestimate'])

# save info to csv
print('Save familiarisation items with infos from Birchenough + SUBTLEX-DE')
fam_filtered_df.to_csv(save_path+'familiarisation_items_overview.csv', index=False)
print('Done.')

print('\nEnd of script!')