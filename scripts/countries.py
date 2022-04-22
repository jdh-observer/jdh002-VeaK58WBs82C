from typing import List, Tuple
import re
from collections import Counter


def clean_line(line):
    """remove footnotes and parentheses."""
    line = line.strip()
    line = re.sub(r' \(.*?\)', '', line)
    line = re.sub(r'\[.*?\]', '', line)
    return line


def split_value(value):
    """split values on commas and slashes."""
    if '/' in value:
        value = value.replace('/', ', ')
    if ',' in value:
        return value.split(', ')
    else:
        return [value]


def map_country_acronym(title: str) -> str:
    """Map acronym of country to full country name."""
    title = title.replace('U.S.A.', 'United States of America')
    title = title.replace('U.S.', 'United States')
    title = title.replace('US', 'United States')
    title = title.replace('U.K.', 'United Kingdom')
    title = title.replace('UK', 'United Kingdom')
    return title


def read_countries_continents():
    """Read country - continent data."""
    country_continent_map = {
        # Most of these are not countries but regions mentioned in titles
        # others are former countries (Soviet Union) or non/partially-recognised countries (Kosovo)
        'Western Sahara': 'Africa',
        'Korea': 'Asia',
        'Middle East': 'Asia',
        'Palestine': 'Asia',
        'Soviet Union': 'Asia',
        'England': 'Europe',
        'Jersey': 'Europe',
        'Kosovo': 'Europe',
        'Northern Ireland': 'Europe',
        'Scotland': 'Europe',
        'Wales': 'Europe',
        'Caribbean': 'North America',
        'Saba': 'North America',
        'Virgin Islands': 'North America',
        'West Indies': 'North America',
        'French Polynesia': 'Oceania',
        'Latin America': 'South America',
    }
    countries = {'Soviet Union', 'Middle East', 'Latin America', 'Caribbean', 'West Indies'}
    continents = set()
    # list from https://www.worldatlas.com/cntycont.htm
    countries_file = 'data/countries.csv'
    with open(countries_file, 'rt') as fh:
        for line in fh:
            country, continent = line.strip().split('\t')
            continents.add(continent)
            countries.add(country)
            country_continent_map[country] = continent
    return countries, continents, country_continent_map


def show_counts(country_count: Counter, continent_count: Counter) -> None:
    print('Countries\n----------------------------------')
    for country, count in country_count.most_common(20):
        print(f'{country: <30}{count: >4}')

    print('\n\nContinents\n----------------------------------')
    for continent, count in continent_count.most_common(20):
        print(f'{continent: <30}{count: >4}')


class CountryLookup:

    def __init__(self):
        countries, continents, country_continent_map = read_countries_continents()
        self.countries = countries
        self.continents = continents
        self.country_continent_map = country_continent_map
        self.nationality_country_map = {}
        self.citizen_country_map = {}
        self.nationality_contintent_map = {}
        self.citizen_contintent_map = {}
        self.read_nationalities_citizens()

    def count_countries_continents(self, titles: List[str],
                                   include_nationalities: bool = False) -> Tuple[Counter, Counter]:
        country_count = Counter()
        continent_count = Counter()
        for country in self.countries:
            country_count[country] = 0
        for continent in self.continents:
            continent_count[continent] = 0

        for title in titles:
            title = map_country_acronym(title.title())
            countries, continents = self.extract_countries_continents(title, include_nationalities)
            country_count.update(countries)
            continent_count.update(continents)
        return country_count, continent_count

    def read_nationalities_citizens(self):
        # list from https://en.wikipedia.org/wiki/List_of_adjectival_and_demonymic_forms_for_countries_and_nations
        nationalities_file = 'data/nationalities.csv'
        with open(nationalities_file, 'rt') as fh:
            _headers = next(fh)
            for line in fh:
                line = clean_line(line)
                country, nationality, citizens = line.split('\t')
                country = re.sub(r',.*', '', country)
                nationality_list = split_value(nationality)
                citizen_list = split_value(citizens)
                if country not in self.countries and country not in self.continents:
                    self.countries.add(country)
                for nationality in nationality_list:
                    if country in self.countries:
                        self.nationality_country_map[nationality] = country
                    elif country in self.continents:
                        self.nationality_contintent_map[nationality] = country
                    if country in self.country_continent_map:
                        self.nationality_contintent_map[nationality] = self.country_continent_map[country]
                for citizen in citizen_list:
                    if country in self.countries:
                        self.citizen_country_map[citizen] = country
                    elif country in self.continents:
                        self.citizen_contintent_map[citizen] = country
                    if country in self.country_continent_map:
                        self.citizen_contintent_map[citizen] = self.country_continent_map[country]

    def extract_countries_continents(self, title: str, include_nationalities: bool = False):
        continents_mentioned = set()
        countries_mentioned = set()
        title = map_country_acronym(title.title())
        for continent in self.continents:
            if continent in title:
                continents_mentioned.add(continent)
        for country in self.countries:
            if country in title:
                continents_mentioned.add(self.country_continent_map[country])
                countries_mentioned.add(country)
        if include_nationalities:
            for nationality in self.nationality_country_map:
                if nationality in title:
                    continents_mentioned.add(self.nationality_contintent_map[nationality])
                    countries_mentioned.add(self.nationality_country_map[nationality])
            for citizens in self.citizen_country_map:
                if citizens in title:
                    continents_mentioned.add(self.citizen_contintent_map[citizens])
                    countries_mentioned.add(self.citizen_country_map[citizens])
        return list(countries_mentioned), list(continents_mentioned)


