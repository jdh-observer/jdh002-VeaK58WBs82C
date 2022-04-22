#!/usr/bin/env python
# coding: utf-8

import sys
from itertools import chain
from collections import defaultdict, Counter

import pandas as pd

sys.path.append('/data/home/jupyter-jdh-artikel/.local/lib/python3.7/site-packages')

import scripts.datasets as dataset_api
from scripts.data_wrangling import get_fullname_relations, clean_relation_records, get_entity_name
from scripts.data_wrangling import t_authors


def make_overal_results():
    periods = dataset_api.periods
    relationship_records = get_fullname_relations()
    cleanrecs = clean_relation_records(relationship_records)
    period_results = defaultdict(dict)
    for period in periods:
        recs = cleanrecs.loc[relationship_records.year.isin(range(periods[period]['start'], periods[period]['end']))]
        recnrs = len(recs)
        period_results[period]['nr of titles'] = recnrs
        relationfields = t_authors + ['executor_org', 'funder', 'client']
        for c in relationfields:
            period_results[period][c] = len(recs[c].unique())

    overall_results = {}
    for c in t_authors + ['funder', 'client', 'executor_org']:
        overall_results[c] = list(relationship_records[c].unique())


def get_surnames():
    entity_records = dataset_api.read_person_entity_records()
    return [record['prs_surname'] for record in entity_records]


def generate_country_frame(overview):
    entity_records = dataset_api.read_person_entity_records()
    cat_p_df = dataset_api.read_person_categories()
    freq_auts = overview.loc[overview.authors > 1]
    auts = overview.loc[overview.authors > 0]

    entity_nationality = {get_entity_name(entity=record): record.get('prs_country') or 'unknown' for record in
                          entity_records}

    surnms = get_surnames()
    entities = [', '.join(n.split(',  ')) for n in list(overview.index)]
    entity_nationality2 = cat_p_df.loc[cat_p_df.fullname.isin(entities)][['fullname', 'prs_country']]
    entity_nationality_sn = cat_p_df.loc[cat_p_df.prs_surname.isin(surnms)]

    # = cat_p_df.loc[cat_p_df.fullname.isin(list(overview.index))]

    aut_country = {}

    for aut in auts.index:
        n = ', '.join(aut.split(',  '))
        aut_country[n] = entity_nationality.get(n) or 'unknown'
    autcountry = pd.DataFrame().from_dict(aut_country, orient="index")
    sups = overview.loc[(overview.funder > 0) | (overview.pref_a > 0)].sort_values(by='total', ascending=False)
    sup_country = {}
    for sup in sups.index:
        n = ', '.join(aut.split(',  '))
        sup_country[n] = entity_nationality.get(n) or 'unknown'
    supcountry = pd.DataFrame().from_dict(sup_country, orient='index')
    supcountry.loc[supcountry[0] != 'unknown'].value_counts()
    overview.loc[(overview.authors > 0) & (overview.total > overview.authors)]
    subgraphs = {}
