"""
This script downloads all existing corpora necessary for the survey's stimuli creation
as well as the estimate validation:
    - Schröder (2012)
    - Birchenough (2017)
    - MultiPic (2018, 2022)
    - Kuperman (2012)
    - (SUBTLEX-DE; Marc Brysbaert's website currently under construction)

Run with: $ python3 download_corpora.py

If you already have those databases saved locally, feel free to change the paths in the 
specific files to point to your copies instead of downloading them anew.
"""

# import relevant packages
import os
import numpy as np
import pandas as pd
from sys import exit
from io import BytesIO
from urllib.request import urlretrieve
from urllib.request import urlopen
from zipfile import ZipFile

def download_corpus(zipurl, filename, save_path):
    """ Downloads and unpacks a corpus zip-file.
    Input: 
        zipurl: zip-file url
        filename: name of file to be extracted
        save_path: path/directory in which to save the extracted file
    Output: 
        -- 
    """
    with urlopen(zipurl) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extract(filename, path=save_path)

###########################################################################
###########################################################################
print('SCRIPT IS RUNNING')

# create directories
print('Creating directories...')
directories = ['multipic', 'norms', 'frequencies']
for directory in directories:
    os.makedirs(directory, exist_ok=True)
print('Done.')

# download corpora
print('\nDownloading corpora...')
print('> MultiPic version 1...')
multipic_url = 'https://www.bcbl.eu/bcbl-corporativa/wp-content/uploads/2016/10/German_MultiPic.zip'
download_corpus(multipic_url, 'German_MultiPic_CSV.csv', 'multipic/')
os.rename('multipic/German_MultiPic_CSV.csv', 'multipic/German_MultiPic_version1.csv')
print('> Done.')

print('> MultiPic version 5...')
multipic_url_5 = 'https://figshare.com/ndownloader/files/34462247'
multipic_df = pd.read_csv(multipic_url_5, sep=';', decimal=',')
multipic_df.to_csv('multipic/MultiPic_version5.csv', sep=';', decimal=',',index=False)
print('> Done.')

# print('> SUBTLEX-DE...')
# subtlex_url = 'https://crr.ugent.be/subtlex-de/SUBTLEX-DE_txt_cleaned_with_Google00.zip'
# download_corpus(subtlex_url, 'SUBTLEX-DE_cleaned_with_Google00.txt', 'frequencies/')
# print('> Done.')

print('> Schröder (2012)...')
schröder_url = 'https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-011-0164-y/MediaObjects/13428_2011_164_MOESM1_ESM.xls'
urlretrieve(schröder_url, "norms/Schröder_2012.xls")
print('> Done.')

print('> Birchenough (2017)...')
birchenough_url = 'https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-016-0718-0/MediaObjects/13428_2016_718_MOESM1_ESM.csv'
urlretrieve(birchenough_url, 'norms/Birchenough_2017.csv')
print('> Done.')

print('> Kuperman (2012)...')
kuperman_url = 'https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-013-0348-8/MediaObjects/13428_2013_348_MOESM1_ESM.xlsx'
urlretrieve(kuperman_url, 'norms/Kuperman_2012.xlsx')
print('> Done.')

print('\nSCRIPT IS FINISHED')