from typing import Dict, List, Tuple
import requests
from collections import defaultdict

import pandas as pd

from scripts.data_wrangling import lowercase_headers, get_entity_name


periods = {
    1950: {'start': 1950, 'end': 1959},
    1960: {'start': 1960, 'end': 1969},
    1970: {'start': 1970, 'end': 1985},
}


def read_person_entity_records():
    with open('data/person_entities.tsv', 'rt') as fh:
        entity_records = parse_tsv_records(fh.read())
        return lowercase_headers(entity_records)


def read_person_relationship_records():
    with open('data/person_relationships.tsv', 'rt') as fh:
        records = parse_tsv_records(fh.read())
        for record in records:
            record['year'] = int(record['year'])
        return records


def read_person_category_records():
    with open('data/person_categories.tsv', 'rt') as fh:
        return parse_tsv_records(fh.read())


def read_person_organisation_records():
    with open('data/person_organisation.tsv', 'rt') as fh:
        category_records = parse_tsv_records(fh.read())
        return lowercase_headers(category_records)


def read_person_categories():
    categorized_persons = read_person_category_records()
    categorized_persons = lowercase_headers(categorized_persons)
    # we convert these into a dataframe for easier selection
    cat_p_df = pd.DataFrame(categorized_persons)
    cat_p_df['fullname'] = cat_p_df.apply(lambda row: get_entity_name(entity=row), axis=1)
    return cat_p_df


def read_relationships():
    return retrieve_spreadsheet_records(record_type='relationships')


def read_entity_roles():
    entity_records = read_person_entity_records()
    entity_roles = {get_entity_name(record): [record['prs_role1'], record['prs_role2'], record['prs_role3']]
                    for record in entity_records}
    for k in entity_roles:
        nr = [e for e in entity_roles[k] if e != '']
        entity_roles[k] = nr
    return entity_roles


def read_entity_categories():
    entity_records = read_person_entity_records()
    entity_category = {get_entity_name(entity = record): record.get('prs_category') or 'unknown' for record in entity_records}
    return entity_category


def download_spreadsheet_datasets():
    ent_url, rel_url, cat_url, org_url = get_spreadsheet_urls()
    ent_data = download_spreadsheet_data(ent_url)
    with open('data/person_entities.tsv', 'wt') as fh:
        fh.write(ent_data)
    rel_data = download_spreadsheet_data(rel_url)
    with open('data/person_relationships.tsv', 'wt') as fh:
        fh.write(rel_data)
    cat_data = download_spreadsheet_data(cat_url)
    with open('data/person_categories.tsv', 'wt') as fh:
        fh.write(cat_data)
    org_data = download_spreadsheet_data(org_url)
    with open('data/person_organisation.tsv', 'wt') as fh:
        fh.write(org_data)


def download_spreadsheet_data(spreadsheet_url: str) -> str:
    response = requests.get(spreadsheet_url)
    if response.status_code == 200:
        return response.text
    else:
        raise ValueError(response.text)


def row_has_content(row: List[str]) -> bool:
    for cell in row:
        if cell != '':
            return True
    return False


def parse_tsv_records(tsv_string: str) -> List[Dict[str, any]]:
    if '\r\n' in tsv_string:
        tsv_string = tsv_string.replace('\r\n', '\n')
    rows = [row_string.split('\t') for row_string in tsv_string.split('\n')]
    headers = rows[0]
    data_rows = [row for row in rows[1:] if row_has_content(row)]
    records = [{header: row[hi] for hi, header in enumerate(headers)} for row in data_rows]
    for record in records:
        for field in record:
            if field == 'year':
                record[field] = int(record[field])
    return records


def get_spreadsheet_urls() -> Tuple[any, any, any, any]:
    spreadsheet_key = '1u691b_EcRfwZ-ipQobFvZZeBJlA0fATErfuymQx_rM8'
    rel_gid = '1337791397'
    ent_gid = '1301599057'
    cat_gid = '1771542502'
    org_gid = '2120744831'
    base_url = 'https://docs.google.com/spreadsheets/d/'
    spreadsheet_url_relationships = f'{base_url}{spreadsheet_key}/export?gid={rel_gid}&format=tsv'
    spreadsheet_url_entities = f'{base_url}{spreadsheet_key}/export?gid={ent_gid}&format=tsv'
    spreadsheet_url_categories = f'{base_url}{spreadsheet_key}/export?gid={cat_gid}&format=tsv'
    spreadsheet_url_organisation = f'{base_url}{spreadsheet_key}/export?gid={org_gid}&format=tsv'
    return spreadsheet_url_entities, spreadsheet_url_relationships, spreadsheet_url_categories, \
           spreadsheet_url_organisation


def retrieve_spreadsheet_records(record_type: str = 'relationships'):
    url_entities, url_relationships, url_categories, url_overlap = get_spreadsheet_urls()
    if record_type == 'entities':
        spreadsheet_string = download_spreadsheet_data(url_entities)
    elif record_type == 'relationships':
        spreadsheet_string = download_spreadsheet_data(url_relationships)
    elif record_type == 'categories':
        spreadsheet_string = download_spreadsheet_data(url_categories)
    elif record_type == 'overlap':
        spreadsheet_string = download_spreadsheet_data(url_overlap)
    else:
        raise ValueError('Unknown record type, must be "entities", "categories", "overlap" or "relationships"')
    return parse_tsv_records(spreadsheet_string)


def extract_record_entities(record):
    name_map = defaultdict(list)
    publication = {'entity_type': 'publication'}
    record_entities = []
    for header in record:
        if header in ['volume_title', 'series', 'volume', 'year']:
            publication[header] = record[header]
            if header == 'year':
                publication[header] = int(record[header])
        elif header in ['executor_org', 'client', 'funder'] and record[header]:
            entity = {
                'entity_name': record[header],
                'entity_role': header,
                'entity_type': 'organisation'
            }
            record_entities.append(entity)
        elif '_' in header and record[header]:
            field = '_'.join(header.split('_')[:-1])
            name_map[field].append(record[header])
    for field in name_map:
        entity = {
            'entity_name': ', '.join(name_map[field]),
            'entity_role': field[:-1] if field != 'editor' else field,
            'entity_type': 'person'
        }
        if entity['entity_name'] == 'Beijer':
            entity['entity_name'] = 'Beijer, G.'
        record_entities.append(entity)
    record_entities.append(publication)
    return record_entities


def make_bibliographic_record(volume, authors):
    record = {
        'article_title': None,
        'article_doi': None,
        'article_author': None,
        'article_author_index_name': None,
        'article_author_affiliation': None,
        'article_page_range': None,
        'article_pub_date': str(volume['year']),
        'article_pub_year': volume['year'],
        'issue_section': None,
        'issue_number': None,
        'issue_title': None,
        'issue_page_range': None,
        'issue_pub_date': str(volume['year']),
        'issue_pub_year': volume['year'],
        'volume': volume['volume'],
        'journal': volume['series'],
        'publisher': 'REMP'
    }
    if authors[0]['entity_role'] == 'article_author':
        record['issue_section'] = 'article'
        record['article_title'] = volume['volume_title']
    elif authors[0]['entity_role'] == 'preface_author':
        record['issue_section'] = 'front_matter'
        record['article_title'] = 'Preface'
    elif authors[0]['entity_role'] == 'intro_author':
        record['issue_section'] = 'front_matter'
        record['article_title'] = 'Introduction'
    else:
        raise ValueError('unknown entity role!')
    record['article_author'] = ' && '.join([author['entity_name'] for author in authors])
    record['article_author_index_name'] = ' && '.join([author['entity_name'] for author in authors])
    record['article_author_affiliation'] = ' && '.join(['' for _author in authors])
    return record


def make_bibliographic_records(relationship_records: List[Dict[str, str]]):
    bib_records = []
    for ri, record in enumerate(relationship_records):
        entities = extract_record_entities(record)
        volume = None
        main_article = []
        preface = []
        introduction = []
        for entity in entities:
            if entity['entity_type'] == 'publication':
                volume = entity
            elif entity['entity_role'] == 'article_author':
                main_article.append(entity)
            elif entity['entity_role'] == 'intro_author':
                introduction.append(entity)
            elif entity['entity_role'] == 'preface_author':
                preface.append(entity)
        bib_record = make_bibliographic_record(volume, main_article)
        bib_records.append(bib_record)
        if len(introduction) > 0:
            bib_record = make_bibliographic_record(volume, introduction)
            bib_records.append(bib_record)
        if len(preface) > 0:
            bib_record = make_bibliographic_record(volume, preface)
            bib_records.append(bib_record)
    return bib_records


def read_publications():
    # load the csv data into a data frame
    records_file = 'scripts/data/main-review-article-records.csv'
    return pd.read_csv(records_file)
