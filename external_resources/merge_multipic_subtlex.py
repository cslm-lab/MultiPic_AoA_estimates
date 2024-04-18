"""
This script combines information from the MultiPic database (version 1) 
and the SUBTLEX-DE corpus, and returns a new csv file.

The script first checks whether the corpora are saved in the format
of `download_corpora.py`.
If yes, it proceeds with the expected paths,
if no, it asks you to provide the abolute paths to your saved copies.

As information it takes the following columns:
> From MultiPic:
1. ITEM
2. PICTURE
3. NAME1
5. H_INDEX
6. PERCENTAGE_MODAL_NAME
10. VISUAL_COMPLEXITY

> From SUBTLEX-DE:
5. SUBTLEX
6. lgSUBTLEX
9. Google00pm
10. lgGoogle00
"""

# import relevant packages
import os
import numpy as np
import pandas as pd
from sys import exit
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

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

###########################################################################
###########################################################################
print('SCRIPT IS RUNNING')
# initiate path variables
multipic_path = ''
subtlex_path = ''

# check for databases 
for corpus in ['MultiPic corpus (version 1)', 'SUBTLEX-DE corpus']:
    print(f'\nIs the {corpus} in the format expected from `download_corpora.py`?')
    print('Type y for yes or n for no and press ENTER.')
    presence = input('>>> ')

    if presence == 'y':
        print(f'\nContinuing with the expected path for {corpus}.')
        # use expected relative paths
        if corpus == 'MultiPic corpus (version 1)':
            multipic_path = 'multipic/German_MultiPic_version1.csv'
        else: 
            subtlex_path = 'frequencies/SUBTLEX-DE_cleaned_with_Google00.txt'
    elif presence == 'n':
        if corpus == 'MultiPic corpus (version 1)':
            print('\nPlease enter the absolute path of the CSV file.')
            print('e.g.: home/user/German_MultiPic/German_MultiPic_CSV.csv')
            # take input as path
            multipic_path = input('>>> ')
            print('Path saved.')
        else:
            print('\nPlease enter the absolute path of the TXT file.')
            print('e.g.: home/user/SUBTLEX-DE_txt_cleaned_with_Google00/SUBTLEX-DE_cleaned_with_Google00.txt')
            # take input as path
            subtlex_path = input('>>> ')
            print('Path saved.')
    else:
        exit('There was a typo. Please restart the script.')

###########################################################################
# MultiPic
print('\n Extract relevant information from MultiPic...')
# load MultiPic database as dataframe
combined_df = pd.read_csv(multipic_path,sep=';', decimal=',', usecols=['ITEM','PICTURE','NAME1','H_INDEX','PERCENTAGE_MODAL_NAME','VISUAL_COMPLEXITY'])
# get relevant tokens
multipic_vals = set(combined_df['NAME1'])
print('Done.')

# SUBTLEX-DE
print('Add lexical information from SUBTLEX-DE...')
# prepare relevant SUBTLEX columns
combined_df = combined_df.reindex(columns=combined_df.columns.tolist()+['SUBTLEX','lgSUBTLEX','Google00pm','lgGoogle00'])

# load cleaned SUBTLEX-DE dataset as dataframe
subtlex_df = pd.read_csv(subtlex_path, sep='\t', decimal=',', encoding='latin_1',)
# drop redundant column
subtlex_df.drop(columns=['Unnamed: 10'], inplace=True)

# get SUBTLEX-DE tokens
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


# iterate through tokens to fill frequency cells
for token in multipic_vals:
    # remove any '-' (concerns u-boot and t-shirt)
    token = token.replace('-','')
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
        
        # get frequency information
        subt = subtlex_df.loc[index, 'SUBTLEX']
        lgsubt = subtlex_df.loc[index, 'lgSUBTLEX']
        google = subtlex_df.loc[index, 'Google00pm']
        lggoogle =subtlex_df.loc[index, 'lgGoogle00']

        # account for the fact that some names occur more than once
        # add frequency information to combined_df
        indices = combined_df[combined_df['NAME1'] == token].index
        for i in range(len(indices)):
            combined_df.loc[indices[i], 'SUBTLEX'] = subt
            combined_df.loc[indices[i], 'lgSUBTLEX'] = lgsubt
            combined_df.loc[indices[i], 'Google00pm'] = google
            combined_df.loc[indices[i], 'lgGoogle00'] = lggoogle

print('Done.')

# save dataframe as CSV file
print('\nSave combined information as new CSV file...')
combined_df.to_csv('MultiPic_with_frequencies.csv', index=False)
print('All done! \nEND OF SCRIPT')
