import pandas as pd
import numpy as np
import os

# LOAD IN DATA
print('>> Load data...')
# ratings etc from survey
survey_start_df = pd.read_csv('../data/raw/survey/data_survey_start.csv', encoding='latin')
lists_df = pd.read_csv('../data/raw/survey/data_lists.csv', encoding='latin')
# MultiPic + familiarisation item information
example_multipic_df = pd.read_excel('../../study_setup/data/example_sentences.ods', sheet_name='MultiPic')
example_fam_df = pd.read_excel('../../study_setup/data/example_sentences.ods', sheet_name='Familiarisation')
survey_values_df = pd.read_csv('../data/raw/survey/values_survey_start.csv', encoding='latin')
# list information
list_a = np.loadtxt('../../study_setup/data/items_lists/list_A.csv', dtype=int)
list_a_repeated = np.loadtxt('../../study_setup/data/items_lists/list_A_repeated.csv', dtype=int)
list_b= np.loadtxt('../../study_setup/data/items_lists/list_B.csv', dtype=int)
list_b_repeated = np.loadtxt('../../study_setup/data/items_lists/list_B_repeated.csv', dtype=int)
list_c = np.loadtxt('../../study_setup/data/items_lists/list_C.csv', dtype=int)
list_c_repeated = np.loadtxt('../../study_setup/data/items_lists/list_C_repeated.csv', dtype=int)
list_control = np.loadtxt('../../study_setup/data/items_lists/control_items.csv', dtype=int)
print('Done.')

# HELPER FUNCTIONS FOR INFORMATION RETRIEVAL
def create_lookup_df(item_list, column_list):
    """
    Creates lookup dataframes for easier matching between item number, 
    item word + example sentence, and item survey variable.
    Input:
        item_list: lists from the data/items_lists/ directory
        column_list: list of relevant variable names (column names)
    Output:
        lookup_df: dataframe with columns ITEM, NAME1, EXAMPLE, variable_name
    """
    lookup_df = example_multipic_df.loc[example_multipic_df['ITEM'].isin(item_list)].copy()
    lookup_df.reset_index(inplace=True, drop=True)
    lookup_df['variable_name'] = column_list
    return lookup_df

def get_order_dict(id, rotation_info):
    """
    Finds out the order items were presented to a participant.
    Note: count starts at 1.
    Input: 
        id: participant ID
        rotation_info: a, b, or c rotation
    Output: 
        dictionaries of the form {page_number: order of appearance}
        fam_order: order of appearance of familiarisation items
        list_order: order of appearance of list items
    """
    # first familiarisation items
    fam_order = dict()
    for count, rotation_column in enumerate(fam_rotation, start=1):
        item_number = int(survey_start_df.loc[survey_start_df['CASE'] == id, rotation_column])
        item_idx = survey_start_df.loc[survey_start_df['CASE'] == id, rotation_column] 
        fam_order[item_number] = count
    # then list specific rotation (a/b/c)
    list_order = dict()
    for count, rotation_column in enumerate(rotation_info, start=11):
        page_number = int(survey_start_df.loc[survey_start_df['CASE'] == id, rotation_column])
        item_idx = survey_values_df.loc[(survey_values_df['VAR'] == rotation_column) & (survey_values_df['RESPONSE'] == page_number)].index[0] 
        item_number = survey_values_df.loc[item_idx, 'MEANING'].strip()
        if item_number.startswith('s'):
            shared_idx = int(item_number.replace('s',''))-1
            item_number = str(list_control[shared_idx])
        list_order[item_number] = count
    return fam_order, list_order

def get_rating(df, idx, item_column):
    """
    Function that returns the AoA rating of a specific item.
    Input:
        df: dataframe the rating is stored in
        idx: global df index of the participant
        item_column: an item's survey variable 
    Output:
        rating: an item's AoA rating (as int)
    """
    rating = df.loc[idx, item_column]
    if not pd.isna(rating):
        rating = int(rating)
    return rating

def get_fam_word(item_column):
    """
    Retrieves the presented familiarisation word item and its example sentence.
    Input:
        fam_item_number: number of the familiarisation item page
    Output:
        fam_page_number: page number item appeared on, for easier lookup of order of presentation
        fam_word: familiarisation word corresponding to the item in question
        fam_sentence: example sentence presented with familiarisation item
        fam_time_column: variable name of the time column for that familiarisation item
    """
    item_idx = example_fam_df.loc[example_fam_df['variable_name'] == item_column].index[0]
    fam_page_number = item_idx+1
    fam_word = example_fam_df.loc[item_idx, 'NAME']
    fam_sentence = example_fam_df.loc[item_idx, 'EXAMPLE']
    fam_time_column = example_fam_df.loc[item_idx, 'time_name']
    return fam_page_number, fam_word, fam_sentence, fam_time_column

def get_list_word(lookup_df, item_column):
    """
    Retrieves the presented word item, its example sentence and the corresponding 
    MultiPic item number based on the survey variable.
    Input:
        lookup_df: the reference lookup dataframe to use
        item_column: an item's survey variable
    Output:
        item_number: item number corresponding to the item in question
        item_word: item word corresponding to the item in question
        item_sentence: example sentence presented with the item
        item_time_column: variable name of the time column for that item
    """
    item_idx = lookup_df.loc[lookup_df['variable_name'] == item_column].index[0]
    item_number = lookup_df.loc[item_idx, 'ITEM']
    item_word = lookup_df.loc[item_idx, 'NAME1']
    item_sentence = lookup_df.loc[item_idx, 'EXAMPLE']
    item_time_column = lookup_df.loc[item_idx, 'time_name']
    return item_number, item_word, item_sentence, item_time_column

def create_normal_item_dict(lookup_df, item_column, list_order, list_idx):
    """
    Collects all item relevant information in a dictionary.
    Input: 
        lookup_df: the reference lookup dataframe to use
        item_column: an item's survey variable
        list_order: dictionary of containing order of appearance of list items
        list_idx: participant datapoint index in lists_df
    Output:
        item_dict: dictionary containing all item-specific information
    """
    item_dict = dict()
    item_number, item_word, item_sentence, item_time_column = get_list_word(lookup_df, item_column)
    item_estimate = get_rating(lists_df, list_idx, item_column)        
    # collect item information in dict
    item_dict['item'] = item_word
    item_dict['item_number'] = item_number
    item_dict['estimate'] = item_estimate
    item_dict['example_sentence'] = item_sentence
    item_dict['repetition'] = 0
    item_dict['order'] = list_order[str(item_number)]
    item_dict['time'] = lists_df.loc[list_idx, item_time_column]
    return item_dict

# DEFINE COLUMNS FOR ITEM-BASED DATASET
columns = [
    'ID', # participant ID
    'item', # word asked to estimate AoA of
    'item_number', # index of MultiPic entry
    'estimate', # AoA estimate of item
    'example_sentence', # example sentence presented with item
    'repetition', # 0 for original, 1 for repeated item
    'order', # presented order aka when was the item presented?
    'platform', # where was participant recruited? SONA or Prolific
    'list', # which list did they fill out?
    'gender', # participant's gender
    'age', # participant's age
    'country', # participant's country of residence
    'education', # participant's highest level of formal education
    'L1', # participant's first language
    'monoling', # monolingual upbringing until age 6?
    'lang_dis', # language disorder?
    'read_dis', # reading disorder?
    'sight', # eyesight: normal or corrected?
    'children', # does participant have children?
    'child_age', # what age do the children have?
    'time', # time taken to estimate item's AoA
    'time_sum', # total time taken (without outliers)
    'finished', # survey complete (1) or incomplete (0)
    'violation', # is one of our criterions violated? (no 0/yes 1)
]

###################################################################################################

# INFORMATION PREPARATIONS
print('>> Prepare information for easier lookup...')
# create dict for information that stay the same
# dict structure: {ID 1: {information}, ID 2: {information}, ...}
# 'platform': where was participant recruited? SONA or Prolific
# 'list': which list did they fill out?
# 'gender': participant's gender
# 'age': participant's age
# 'country': participant's country of residence
# 'education': participant's highest level of formal education
# 'L1': participant's first language
# 'monoling': monolingual upbringing until age 6?
# 'lang_dis': language disorder?
# 'read_dis': reading disorder?
# 'sight': eyesight: normal or corrected?
# 'children': does participant have children?
# 'child_age': what age do the children have?
# 'time_sum': total time taken (without outliers)
# 'finished': survey complete (1) or incomplete (0)
# 'violation': is one of our criterions violated?
meta_info = dict()

for i in survey_start_df.index:
    finished = 1
    violation = 0
    # anonymised ID
    id = survey_start_df.loc[i, 'CASE']
    # platform
    if survey_start_df.loc[i, 'IN16'] == 1:
        platform = 'Prolific'
    elif survey_start_df.loc[i, 'IN16'] == 2:
        platform = 'SONA'
    elif survey_start_df.loc[i, 'IN16'] == -1:
        platform = 'other'
    # list
    if survey_start_df.loc[i, 'RA01'] == 1:
        list_id = 'A'
    elif survey_start_df.loc[i, 'RA01'] == 2:
        list_id = 'B'
    elif survey_start_df.loc[i, 'RA01'] == 3:
        list_id = 'C'
    else:
        list_id = np.nan
        finished = 0
        violation = 1
    # gender
    if survey_start_df.loc[i, 'SD01'] == 1:
        gender = 'female'
    elif survey_start_df.loc[i, 'SD01'] == 2:
        gender = 'male'
    elif survey_start_df.loc[i, 'SD01'] == 3:
        gender = 'diverse'
    else: 
        gender = np.nan
        violation = 1
    # age 
    age = survey_start_df.loc[i, 'SD02_01']
    # country
    if survey_start_df.loc[i, 'SD07'] == 1:
        country = 'Germany'
    elif survey_start_df.loc[i, 'SD07'] == 2:
        country = 'Austria'
        violation = 1
    elif survey_start_df.loc[i, 'SD07'] == 3:
        country = 'Switzerland'
        violation = 1
    elif survey_start_df.loc[i, 'SD07'] == 4:
        country = 'other'
        violation = 1
    else: 
        country = np.nan
        violation = 1
    # education
    # for clarification of values see codebook or values csv
    education = survey_start_df.loc[i, 'SD10']
    # L1
    if survey_start_df.loc[i, 'SD19'] == 1:
        l1 = 1
    elif survey_start_df.loc[i, 'SD19'] == 2:
        l1 = 0
        violation = 1
    else: 
        l1 = np.nan
        violation = 1
    # monoling
    if survey_start_df.loc[i, 'SD20'] == 1:
        monoling = 0
        violation = 1
    elif survey_start_df.loc[i, 'SD20'] == 2:
        monoling = 1
    else: 
        monoling = np.nan
        violation = 1
    # lang_dis
    if survey_start_df.loc[i, 'SD21'] == 1:
        lang_dis = 1
        violation = 1
    elif survey_start_df.loc[i, 'SD21'] == 2:
        lang_dis = 0
    else: 
        lang_dis = np.nan
        violation = 1
    # read_dis
    if survey_start_df.loc[i, 'SD25'] == 1:
        read_dis = 1
        violation = 1
    elif survey_start_df.loc[i, 'SD25'] == 2:
        read_dis = 0
    else: 
        read_dis = np.nan
        violation = 1
    # sight
    if survey_start_df.loc[i, 'SD22'] == 1:
        sight = 'normal'
    elif survey_start_df.loc[i, 'SD22'] == 2:
        sight = 'corrected'
    else: 
        sight = np.nan
    # children
    if survey_start_df.loc[i, 'SD24'] == 1:
        children = 1
    elif survey_start_df.loc[i, 'SD24'] == 2:
        children = 0
    else: 
        children = np.nan
    # child_age
    child_age = []
    if children:
        # find out how many kids; append age of each kid to list
        for kid in range(1,int(survey_start_df.loc[i,'SD23'])+1):
            child_age.append(survey_start_df.loc[i, 'SD23x0'+str(kid)])
    # time_sum
    if id in set(lists_df['REF']):
        time_sum = int(survey_start_df.loc[i,'TIME_SUM']) + int(lists_df[lists_df['REF'] == id]['TIME_SUM'])
    else:
        time_sum = int(survey_start_df.loc[i,'TIME_SUM'])
        finished = 0
        violation = 1

    # create dict entry
    meta_info[id] = {
        'platform': platform,
        'list': list_id,
        'gender': gender,
        'age': age,
        'country': country,
        'education': education,
        'L1': l1,
        'monoling': monoling,
        'lang_dis': lang_dis,
        'read_dis': read_dis,
        'sight': sight,
        'children': children,
        'child_age': child_age,
        'time_sum': time_sum,
        'finished': finished,
        'violation': violation,
    }

# filter relevant column names
survey_columns = list(survey_start_df.columns)
list_columns = list(lists_df.columns)
# RA02: page rotation of list A
# RA03: page rotation of familiarisation items
# RA04: page rotation of list B
# RA05: page rotation of list C
a_rotation = [x for x in survey_columns if x.startswith('RA02') and not x.endswith('CP')]
b_rotation = [x for x in survey_columns if x.startswith('RA04') and not x.endswith('CP')]
c_rotation = [x for x in survey_columns if x.startswith('RA05') and not x.endswith('CP')]
fam_rotation = [x for x in survey_columns if x.startswith('RA03') and not x.endswith('CP')]
# fam items: WF
fam_columns = [x for x in survey_columns if x.startswith('WF') and not x.endswith('a')]
# shared/control items: WS
control_columns = [x for x in list_columns if x.startswith('WS') and not x.endswith('a')]
# list A: W1-W3; repeated items: WR01-WR25
a_columns = [x for x in list_columns if (x.startswith('W1') or x.startswith('W2') or x.startswith('W3')) and not x.endswith('a')] 
a_rep_columns = ['WR0'+str(x)+'_01' for x in range(1,10)] + ['WR'+str(x)+'_01' for x in range(10,26)]
# list B: W4-W6; repeated items: WR26-WR50
b_columns = [x for x in list_columns if (x.startswith('W4') or x.startswith('W5') or x.startswith('W6')) and not x.endswith('a')] 
b_rep_columns = ['WR'+str(x)+'_01' for x in range(26,51)]
# list C: W7-W9; repeated items: WR51-WR75
c_columns = [x for x in list_columns if (x.startswith('W7') or x.startswith('W8') or x.startswith('W9')) and not x.endswith('a')] 
c_rep_columns = ['WR'+str(x)+'_01' for x in range(51,76)]
# time columns
fam_time_columns = [x for x in survey_columns if x.startswith('TIME') and not x.endswith('SUM') and not x.endswith('RSI')]
del fam_time_columns[:5] # removes time for demographic pages
item_time_columns = [x for x in list_columns if x.startswith('TIME') and not x.endswith('SUM') and not x.endswith('RSI')]

# create lookup dataframes for easier matching
a_lookup_df = create_lookup_df(list_a, a_columns)
a_rep_lookup_df = create_lookup_df(list_a_repeated, a_rep_columns)
b_lookup_df = create_lookup_df(list_b, b_columns)
b_rep_lookup_df = create_lookup_df(list_b_repeated, b_rep_columns)
c_lookup_df = create_lookup_df(list_c, c_columns)
c_rep_lookup_df = create_lookup_df(list_c_repeated, c_rep_columns)
control_lookup_df = create_lookup_df(list_control, control_columns)
# add time names to lookup
# shared/control items are the first pages in all lists
control_lookup_df['time_name'] = item_time_columns[:31]
# list a has shared/control items -> list items -> repeated items
a_lookup_df['time_name'] = item_time_columns[31:259]
a_rep_lookup_df['time_name'] = item_time_columns[259:]
# lists b and c have shared/control items -> repeated items -> list items
for df in [b_rep_lookup_df, c_rep_lookup_df]:    
    df['time_name'] = item_time_columns[31:56]
for df in [b_lookup_df, c_lookup_df]:
    df['time_name'] = item_time_columns[56:]
# also add variable column name information to familiarisation items df 
example_fam_df['variable_name'] = fam_columns
example_fam_df['time_name'] = fam_time_columns

print('Done.')
###################################################################################################

# CREATE ITEM-BASED DATASET
print('>> Create item-based dataset...')
data_df = pd.DataFrame(columns=columns)

# iterate through participants, add all information to big df
for id in meta_info.keys(): 
    print('... ID', id)
    participant_df = pd.DataFrame(columns=columns)

    # get list ID (A/B/C)
    list_id = meta_info[id]['list']
    if list_id == 'A':
        rotation = a_rotation
        lookup_df = a_lookup_df
        rep_lookup_df = a_rep_lookup_df
        item_columns = a_columns
        item_rep_columns = a_rep_columns
    elif list_id == 'B':
        rotation = b_rotation
        lookup_df = b_lookup_df
        rep_lookup_df = b_rep_lookup_df
        item_columns = b_columns
        item_rep_columns = b_rep_columns
    elif list_id == 'C':
        rotation = c_rotation
        lookup_df = c_lookup_df
        rep_lookup_df = c_rep_lookup_df
        item_columns = c_columns
        item_rep_columns = c_rep_columns
    else:
        # if the participant did not get to being assigned a list, 
        # skip participant
        continue
    
    # get idx
    survey_idx = survey_start_df.loc[survey_start_df['CASE'] == id].index[0]
    list_idx = lists_df.loc[lists_df['REF'] == id].index[0]
    # get order
    fam_order, list_order = get_order_dict(id, rotation)

    # fam items
    for fam_item in fam_columns:
        item_dict = dict()
        fam_page_number, fam_word, fam_sentence, fam_time_column = get_fam_word(fam_item)
        fam_estimate = get_rating(survey_start_df, survey_idx, fam_item)
        # collect item information in dict
        item_dict['item'] = fam_word
        item_dict['estimate'] = fam_estimate
        item_dict['example_sentence'] = fam_sentence
        item_dict['repetition'] = 0
        item_dict['order'] = fam_order[fam_page_number]
        item_dict['time'] = survey_start_df.loc[survey_idx, fam_time_column]
        # append row to intermediate dataframe
        participant_df = pd.concat([participant_df, pd.DataFrame(item_dict, index=[0])], ignore_index=True) 

    # list + shared/control items
    for item in item_columns:
        item_dict = create_normal_item_dict(lookup_df, item, list_order, list_idx)
        # append row to intermediate dataframe
        participant_df = pd.concat([participant_df, pd.DataFrame(item_dict, index=[0])], ignore_index=True) 
    for item in control_columns:
        item_dict = create_normal_item_dict(control_lookup_df, item, list_order, list_idx)
        participant_df = pd.concat([participant_df, pd.DataFrame(item_dict, index=[0])], ignore_index=True) 
        
    # repeated items 
    for item in item_rep_columns:
        item_dict = dict()
        item_number, item_word, item_sentence, item_time_column = get_list_word(rep_lookup_df, item)
        item_estimate = get_rating(lists_df, list_idx, item)        
        # collect item information in dict
        item_dict['item'] = item_word
        item_dict['item_number'] = item_number
        item_dict['estimate'] = item_estimate
        item_dict['example_sentence'] = item_sentence
        item_dict['repetition'] = 1
        item_dict['order'] = list_order['r'+str(item_number)]
        item_dict['time'] = lists_df.loc[list_idx, item_time_column]
        # append row to intermediate dataframe
        participant_df = pd.concat([participant_df, pd.DataFrame(item_dict, index=[0])], ignore_index=True)

    # insert participant informatino that is same for all items
    participant_df['ID'] = id
    participant_df['platform'] = meta_info[id]['platform']
    participant_df['list'] = list_id
    participant_df['gender'] = meta_info[id]['gender']
    participant_df['age'] = meta_info[id]['age']
    participant_df['country'] = meta_info[id]['country']
    participant_df['education'] = meta_info[id]['education']
    participant_df['L1'] = meta_info[id]['L1']
    participant_df['monoling'] = meta_info[id]['monoling']
    participant_df['lang_dis'] = meta_info[id]['lang_dis']
    participant_df['read_dis'] = meta_info[id]['read_dis']
    participant_df['sight'] = meta_info[id]['sight']
    participant_df['children'] = meta_info[id]['children']
    participant_df['child_age'] = str(meta_info[id]['child_age'])
    participant_df['time_sum'] = meta_info[id]['time_sum']
    participant_df['finished'] = meta_info[id]['finished']
    participant_df['violation'] = meta_info[id]['violation']

    # append participant_df to overall data dataframe
    data_df = pd.concat([data_df, participant_df], ignore_index=True)

# SAVE ITEM-BASED DATASET
print('... save item-based dataset...')
save_directory = '../data/raw/derivatives'
os.makedirs(save_directory, exist_ok=True)
data_df.to_csv(save_directory+'/item_based_data.csv', index=False)
print('All done!')