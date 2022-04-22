import re
import unicodedata
import string
from itertools import chain
from collections import Counter

import numpy as np
import pandas as pd
from iso3166 import countries


all_letters = string.ascii_letters + " .,;'-"

author_name_column_map = {
    'authors': ['article_author1', 'article_author2'],
    'pref_a': ['preface_author1', 'preface_author2', ],
    'intro_a': ['intro_author1', 'intro_author2'],
    'editor': ['editor'],
    'funder': ['funder'],
    'executor_org': ['executor_org']
}

t_authors = ['article_author1',
             'article_author2',
             'preface_author1',
             'preface_author2',
             'intro_author1',
             'intro_author2',
             'editor']


def add_fullname_columns(relationship_records):
    namecolumns = ['{author}_surname', '{author}_infix', '{author}_initials']
    relrecs = pd.DataFrame(relationship_records)
    for c in ['year']:
        relrecs[c] = relrecs[c].astype('int')

    for a in t_authors:
        clst = [c.format(author=a) for c in namecolumns]
        colnm = '{a}'.format(a=a)
        relrecs[colnm] = relrecs[clst].apply(lambda x: aut_to_fn(x), axis=1)
    return relrecs


def clean_relation_records(relation_records):
    keepcolumns = ['series', 'volume', 'volume_title', 'year', 'funder', 'client', 'executor_org'] + t_authors
    cleanrecs = relation_records[keepcolumns].fillna('')
    return cleanrecs


def get_fullname_relations(relationship_records):
    return add_fullname_columns(relationship_records)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def acronym(text_string):
    if text_string == 'International Migration':
        return 'IM'
    else:
        return 'IMR'


def normalise_name(author_name):
    author_name = unicode_to_ascii(author_name)
    author_name = author_name.title()
    author_name = author_name.replace('Abandan-Unat', 'Abadan-Unat')
    author_name = author_name.replace('Bastos De Avila', 'Avila')
    # Jr. is dropped from the index name
    author_name = author_name.replace('Purcell Jr.', 'Purcell')
    # Titles like Father
    titles = ['Father ', 'Ambassador ']
    for title in titles:
        if title in author_name:
            author_name = author_name.replace(title, ' ')
    # Prefix in Spanish and French: Por and Par
    if ' Por ' in author_name:
        author_name = author_name.replace(' Por ', ' ')
    if ' Par ' in author_name:
        author_name = author_name.replace(' Par ', ' ')
    author_name = re.sub(r' +', ' ', author_name)
    return author_name.strip()


def parse_surname(author_name: str):
    author_name = normalise_name(author_name)
    return ','.join(author_name.split(',')[:-1]).replace('ij', 'y').title()


def parse_surname_initial(author_name: str):
    author_name = normalise_name(author_name)
    if ',' not in author_name:
        return author_name
    surname = ','.join(author_name.split(',')[:-1]).replace('ij', 'y').title()
    initial = author_name.split(', ')[-1][0]
    return f'{surname}, {initial}'


def parse_author_index_name(row):
    if row['prs_infix'] != '':
        return ', '.join([row['prs_surname'], row['prs_infix'], row['prs_initials']])
    else:
        return ', '.join([row['prs_surname'], row['prs_initials']])


#########################
# Generic Data Handling #
#########################


def map_dataset(publisher, article_type):
    # all REMP and IM (published by Wiley) articles are bunlded in a single dataset
    if publisher == 'Staatsdrukkerij' or publisher == 'Wiley':
        return 'REMP_IM'
    # The IMR articles are separated in review articles and research articles
    return 'IMR_research' if article_type == 'main' else 'IMR_review'


def yr2cat(x):
    s = x['period_start']
    e = x['last_known_date']
    try:
        start = int(s)
    except ValueError:
        start = 0
    try:
        end = int(e)
    except ValueError:
        end = start
    return pd.Interval(start, end, closed='both')


def map_bool(value):
    if isinstance(value, bool):
        return 1 if value else 0
    else:
        return value


def is_decade(value):
    return isinstance(value, int)


def highlight_decade(row):
    color_cat = {
        'Dutch Government': '#ffbbbb',
        'ICEM': '#ff6666',
        'REMP': '#ff0000',
        'REMP_IM': '#0000ff',
        'IMR_research': '#8888ff',
        'IMR_review': '#bbbbff',
    }
    color = color_cat[row['cat']]
    return [f'background-color: {color}' if is_decade(col) and row[col] else '' for col in row.keys()]


def lowercase_headers(records: list):
    return [dict((k.lower(), v) for k, v in record.items()) for record in records]


def get_entity_category(entity: dict, entity_category: dict):
    if entity.get('entity_name') in entity_category:
        return entity_category[entity['entity_name']]
    else:
        return 'unknown'


def get_entity_name(entity: dict, ):
    if entity['prs_infix'] != '':
        return f"{entity['prs_surname'].strip()}, {entity['prs_infix'].strip()}, {entity['prs_initials'].strip()}"
    else:
        return f"{entity['prs_surname'].strip()}, {entity['prs_initials'].strip()}"


def get_entity_country(entity: dict, entity_category):
    if entity.get('entity_name') in entity_category:
        return entity_category[entity['entity_name']]
    else:
        return 'unknown'


def make_nodes(entity):
    return [n.get('entity_name') for n in entity if n.get('entity_name')]


def make_link_from_entity(entity, revnodelist, rempgraph):
    counter = []
    authors = [n.get('entity_name') for n in entity if n.get('entity_role') == 'article_author']
    links = []
    for aut in authors:
        autnr = revnodelist[aut]
        counter.append(autnr)
        for node in entity:
            if node.get('entity_role') != 'article_author':
                if node.get('entity_role'):  # we don't include titles
                    category = node.get('entity_role') or "unknown"
                    target = revnodelist[node.get('entity_name')]
                    graphnode = rempgraph.nodes()[target]
                    if graphnode.get('category'):
                        graphnode['category'].append(category)
                    else:
                        graphnode['category'] = [category]
                    link = (autnr, target, {"link_type": node.get('entity_role') or 'unknown'})
                    links.append(link)
                    counter.append(target)

    return links, counter


def aut_to_fn(cols):
    if cols[0].strip() != '':
        return f'{cols[0] + ","} {cols[1]} {cols[2] or ""}'.strip()
    else:
        return None


def read_publication_decades():
    publications = get_publication_author_administrator_overlap()
    publication_decade = publications[(publications.in_admin == 1) & (publications.in_pub == 1)].drop(
        ['in_admin', 'in_pub'], axis=1)
    publication_decade.sort_values(by='author_surname_initial').style.apply(highlight_decade, axis=1)
    return publication_decade
    #
    # publications['total'] = publications[['1950', '190']].sum(axis=1).groupby('author_surname_initial').agg('sum')


def get_administrator_decades(category_records):
    admin_df = read_administrators_dataframe(category_records)

    decade_cols = [1950, 1960, 1970, 1980, 1990]
    org_cols = ['author_surname_initial', 'organisation']
    decade_admin_df = admin_df[org_cols + ['prs_country']].merge(admin_df[decade_cols].astype(int),
                                                                 left_index=True,
                                                                 right_index=True)
    decade_admin_df = decade_admin_df.rename(columns={'dataset': 'cat'})
    decade_admin_df['in_admin'] = 1
    return decade_admin_df


def get_canonic_administrators(category_records):
    decade_administrator_df = get_administrator_decades(category_records)

    canonic_countries = {'prs_country': {'UK': 'GB',
                                         'USA': 'US',
                                         'UK /AU': 'AU',
                                         '': 'US', }
                         }
    canonic_administrator_df = decade_administrator_df.replace(canonic_countries)
    canonic_administrator_df = canonic_administrator_df.loc[canonic_administrator_df.prs_country != 'unknown']
    return canonic_administrator_df


def get_author_country_counts(cat_p_df, relationship_records):
    author_list = get_author_list(relationship_records)

    # cat_p_df = dataset_api.read_person_categories()
    c_nrs = cat_p_df.loc[cat_p_df.fullname.isin(author_list)][
        ['fullname', 'prs_country']].prs_country.value_counts().rename_axis('country').reset_index(name='number')
    c_nrs.loc[c_nrs.country == 'UK', 'country'] = 'GB'
    c_nrs['numeric'] = c_nrs.country.map(lambda x: int(countries.get(x).numeric))
    return c_nrs


def get_per_decade_administrators():
    canonic_admin_df = get_canonic_administrators()
    per_decade_df = {}

    for decade in range(1950, 2000, 10):
        decade_df = canonic_admin_df.loc[canonic_admin_df[decade] > 0].prs_country.value_counts().rename_axis(
            'country').reset_index(name='number')
        decade_df['numeric'] = decade_df.country.map(lambda x: int(countries.get(x).numeric))
        per_decade_df[decade] = decade_df
    return per_decade_df


def chainer(s):
    return list(chain.from_iterable(s.fillna('').str.split(' && ')))


def split_publication_per_author(pub_df):
    # calculate lengths of splits
    lens = pub_df['article_author'].fillna('').str.split(' && ').map(len)

    # create new dataframe, repeating or chaining as appropriate
    split_pub_df = pd.DataFrame({
        'journal': np.repeat(pub_df['journal'], lens),
        'issue_pub_year': np.repeat(pub_df['issue_pub_year'], lens),
        'publisher': np.repeat(pub_df['publisher'], lens),
        'dataset': np.repeat(pub_df['dataset'], lens),
        'article_author': chainer(pub_df['article_author']),
        'article_author_index_name': chainer(pub_df['article_author_index_name']),
        'article_author_affiliation': chainer(pub_df['article_author_affiliation'])
    })

    # Make sure title case is used consistently in the author index name column
    split_pub_df['article_author_index_name'] = split_pub_df['article_author_index_name'].str.title()
    # add a column with surname and first name initial extracted from the author index name
    split_pub_df['author_surname_initial'] = split_pub_df.article_author_index_name.apply(parse_surname_initial)
    # add a column with surname only
    split_pub_df['author_surname'] = split_pub_df.article_author_index_name.apply(parse_surname)
    # add a column with the decade in which the issue was published that contains an article
    split_pub_df['issue_pub_decade'] = split_pub_df.issue_pub_year.apply(lambda x: int(x / 10) * 10)
    # map journal names to their acronyms
    split_pub_df.journal = split_pub_df.journal.apply(acronym)

    # remove articles with no authors
    split_pub_df = split_pub_df[split_pub_df.article_author != '']

    return split_pub_df.reset_index(drop=True)


def get_per_author_publications(pub_df):
    # pub_df = dataset_api.read_publications()
    # Code adapted from https://stackoverflow.com/questions/50731229/split-cell-into-multiple-rows-in-pandas-dataframe
    pub_df['dataset'] = pub_df.apply(lambda x: map_dataset(x['publisher'], x['article_type']), axis=1)
    # split articles on authors, one row per author per article
    split_pub_df = split_publication_per_author(pub_df)
    # make a dataframe for publications per decade
    decade_pub_df = pd.get_dummies(split_pub_df.issue_pub_decade)

    pub_df = split_pub_df[['author_surname_initial', 'dataset']].merge(decade_pub_df, left_index=True, right_index=True)
    pub_df = pub_df.rename(columns={'dataset': 'cat'})
    pub_df = pub_df.groupby(['author_surname_initial', 'cat']).sum().reset_index()
    pub_df['in_pub'] = 1
    return pub_df


def cutdecade(x, decade):
    result = False
    if x.right < decade[0]:
        return False
    if x.left > decade[1]:
        return False
    if x.left > decade[0] or x.right >= decade[0]:
        return True


def read_administrators_dataframe(category_records):
    admin_df = pd.DataFrame(category_records)
    admin_df['article_author_index_name'] = admin_df.apply(parse_author_index_name, axis=1)
    admin_df['author_surname_initial'] = admin_df.article_author_index_name.apply(parse_surname_initial)
    admin_df['period'] = admin_df.apply(lambda x: yr2cat(x), axis=1)

    decades = {
        1950: (1950, 1960),
        1960: (1960, 1970),
        1970: (1970, 1980),
        1980: (1980, 1990),
        1990: (1990, 2000),
        2000: (2000, 2010)
    }

    for key in decades:
        decade = decades[key]
        admin_df[key] = admin_df.period.apply(lambda x: cutdecade(x, decade))

    decade_cols = [1950, 1960, 1970, 1980, 1990]
    org_cols = ['author_surname_initial', 'organisation']

    temp_admin_df = admin_df[org_cols].merge(admin_df[decade_cols].astype(int), left_index=True, right_index=True)
    temp_admin_df = temp_admin_df.rename(columns={'dataset': 'cat'})
    temp_admin_df['in_admin'] = 1
    admin_df['in_admin'] = temp_admin_df.in_admin
    admin_df[decade_cols] = admin_df[decade_cols].astype(int)

    return admin_df


def get_publication_author_administrator_overlap(pub_df, category_records):
    # load the csv data into a data frame
    pub_df = get_per_author_publications(pub_df)

    admin_df = read_administrators_dataframe(category_records)
    # Hernoem temp_df naar iets inhoudelijks
    publications = pd.concat([admin_df.rename(columns={'organisation': 'cat'}).set_index('author_surname_initial'),
                              pub_df.rename(columns={'dataset': 'cat'}).set_index('author_surname_initial')])

    for name in publications.index:
        publications.loc[name, 'in_pub'] = publications.loc[name, 'in_pub'].max()
        publications.loc[name, 'in_admin'] = publications.loc[name, 'in_admin'].max()
    publications = publications.reset_index()
    return publications


def generate_overview(relationship_records) -> pd.DataFrame:
    relationship_records = add_fullname_columns(relationship_records)
    cleanrecs = clean_relation_records(relationship_records)
    cnted = {}
    for key in author_name_column_map:
        cnted[key] = Counter()
        for f in author_name_column_map[key]:
            cnted[key].update(cleanrecs[f].value_counts().to_dict())
    overview = pd.DataFrame(cnted).fillna(0)
    overview.drop(index='', inplace=True)
    overview['total'] = overview.agg('sum', axis=1)

    for column in overview.columns:
        overview[column] = overview[column].astype('int')
    return overview


def get_author_list(relationship_records):
    overview = generate_overview(relationship_records)
    auts = overview.loc[overview.authors > 0]
    autlst = list([', '.join(aut.split(',  ')) for aut in auts.index])
    return autlst


def junkyard():
    aut_category = {}
    for aut in auts.index:
        n = ', '.join(aut.split(',  '))
        aut_category[n] = entity_category.get(n) or 'unknown'

    cat_p_df.loc[cat_p_df.fullname.isin(autlst)][['fullname', 'prs_country']].prs_country.value_counts()
    aut_category = {}
    for aut in auts.index:
        n = ', '.join(aut.split(',  '))
        aut_category[n] = entity_category.get(n) or 'unknown'
