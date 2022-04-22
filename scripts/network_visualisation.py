from collections import defaultdict, Counter

import altair as alt
import nx_altair as nxa
from community import community_louvain
import hvplot.networkx as hvnx
import holoviews as hv

from scripts.data_wrangling import *
from scripts.network_analysis import *
from scripts.graph_community import community_layout
import scripts.datasets as dataset_api
from scripts.graphs_and_tables import get_surnames


def get_records():
    periods = dataset_api.periods
    entity_roles = dataset_api.read_entity_roles()
    entity_category = dataset_api.read_entity_categories()
    entity_records = dataset_api.read_person_entity_records()
    cat_p_df = dataset_api.read_person_categories()
    relationship_records = dataset_api.read_person_relationship_records()


def get_remp_entities(entity_records, cat_p_df):
    roled = {'author': ['article_author1_surname', 'article_author2_surname'],
             'preface_author': ['preface_author1_surname', 'preface_author2_surname'],
             'intro_author': ['intro_author1_surname', 'intro_author2_surname'],
             'executor': ['executor_org'],
             'funder': ['funder'],
             'client': ['client'],
             'editor': ['editor_surname'],
             'unknown': ['']}
    surnms = get_surnames()
    nms = [get_entity_name(entity=record) for record in entity_records]
    remp_entities = cat_p_df.loc[cat_p_df.fullname.isin(nms)]
    remp_entities_comp = cat_p_df.loc[cat_p_df.prs_surname.isin(surnms)]


def make_remp_graph(relationship_records, entity_category):
    record_entities = defaultdict(list)
    entity_count = Counter()
    entity_role_count = Counter()
    nodelist = []

    for ri, record in enumerate(relationship_records):
        entities = dataset_api.extract_record_entities(record)
        record_entities[ri].append(entities)
        entity_count.update([entity['entity_name'] for entity in entities if 'entity_name' in entity])
        entity_role_count.update(
            [entity['entity_role'] + ' ' + entity['entity_name'] for entity in entities if 'entity_name' in entity])
        for entity in entities:
            if entity['entity_type'] == 'person':
                entity['entity_type'] = get_entity_category(entity, entity_category)

    for rentity in record_entities:
        nodelist.extend(make_nodes(record_entities[rentity][0]))
    nodelist = list(set(nodelist))
    revnodelist = {}
    rempgraph = nx.Graph()
    for node in enumerate(nodelist):
        rempgraph.add_node(node[0], id=node[0], name=node[1])
        revnodelist[node[1]] = node[0]
    linklist = []
    counter = Counter()
    for rentity in record_entities:
        links, cntr = make_link_from_entity(record_entities[rentity][0], revnodelist, rempgraph)
        linklist.extend(links)
        counter.update(cntr)
    return linklist, counter


def get_communities(cat_p_df):
    communities = {}
    for i in cat_p_df.organisation.unique():
        communities[i] = list(cat_p_df.loc[cat_p_df.organisation == i].fullname)
    return communities


def make_period_graph(entity_categories, period_relationships) -> Graph:
    period_graph = generate_graph()
    for record in sorted(period_relationships, key=lambda x: x['year']):
        entities = dataset_api.extract_record_entities(record)
        for entity in entities:
            if entity['entity_type'] == 'person':
                entity['entity_type'] = get_entity_category(entity, entity_categories)
        named_entities = [entity for entity in entities if 'entity_name' in entity]
        add_entities(period_graph, named_entities)
        add_record_links(period_graph, named_entities)
    return period_graph


def make_graph_dict(relationship_records, entity_category, periods):
    grphdict = {}

    for period in periods:
        period = periods[period]
        pn = period['start']
        periodgraph = generate_graph()
        for record in relationship_records:
            record['year'] = int(record['year'])
        for ri, record in enumerate(sorted(relationship_records, key=lambda x: x['year'])):
            record['year'] = int(record['year'])
            if record['year'] < period['start'] or record['year'] > period['end']:
                continue
            entities = dataset_api.extract_record_entities(record)
            for entity in entities:
                if entity['entity_type'] == 'person':
                    entity['entity_type'] = get_entity_category(entity, entity_category)
            named_entities = [entity for entity in entities if 'entity_name' in entity]
            add_entities(periodgraph, named_entities)
            add_record_links(periodgraph, named_entities)
        grphdict[pn] = periodgraph
    return grphdict


def make_period_centralities(grphdict):
    periodcentralities = {}
    for pn in grphdict:
        periodcentralities[pn] = nx.eigenvector_centrality(grphdict[pn])
    return periodcentralities


def get_common_names(period_graphs: Dict[int, Graph]):
    commonnames = {}
    window_size = 2

    nodelists = {}
    for pn in period_graphs:
        nodelists[pn] = [n for n in period_graphs[pn].nodes()]
    seq = list(nodelists.keys())
    for i in range(len(seq) - window_size + 1):
        w = seq[i: i + window_size]
        #     {e:set(nodelists[e]) for e in }
        for item in w:
            if item not in commonnames.keys():
                commonnames[item] = list(set(nodelists[w[0]]) & set(nodelists[w[1]]))

    for i in commonnames:
        commonnames[i].append('Haveman, B.W.')
    return commonnames


# this method factors out commonalities for the graphs below
def add_period_graph_commonalities(period_graph: Graph,
                 communities: dict,
                 commonnames: list):
    period_centralities = nx.eigenvector_centrality(period_graph)
    for f in period_graph.nodes():
        if f in commonnames:
            label = f
        else:
            label = ''
        period_graph.nodes[f].update({  # "community" : comty,
            # "edgecolor": colors.get(comty) or 'purple',
            "centrality": period_centralities[f] * 4,
            "name": f,
            "label": label
        })

    for f in period_graph.edges():
        for i, c in communities.items():
            if f[0] and f[1] in c:
                comty = period_graph.edges[f].get('community') or ''
                comty = ', '.join([comty, i])
                # comty = ''.join([c[0] for c in comty])
                period_graph.edges[f].update({"community": comty,
                                             })

    for f in period_graph.edges():
        comty = period_graph.edges[f].get('community') or []
        comty = ''.join([c[0] for c in comty])
        period_graph.edges[f].update({"community": comty})
    partition = community_louvain.best_partition(period_graph)
    pos = community_layout(period_graph, partition)

    return period_graph, pos


def make_period_chart(period_graph: Graph, communities, commonnames, period: int):
    period_graph, pos = add_period_graph_commonalities(period_graph,
                                                      communities, commonnames[period])

    chart = nxa.draw_networkx(
        G=period_graph,
        pos=pos,
        node_size='centrality',
        node_color='entity_type',
        edge_color='community',
        cmap='category20',
        # edge_c
        # map='category10',
        node_tooltip=['name'],
        node_label='label',
        font_color="black",
        font_size=11,
    )
    start = dataset_api.periods[period]['start']
    end = dataset_api.periods[period]['end']
    chart.title = f"REMP network {start}-{end}"
    chart.properties(
        height=600,
        width=800
    )
    return chart


def make_chart_dict(person_relationships, entity_categories,
                    person_categories, periods):
    chartdict = {}
    period_graphs = {}
    for period in periods:
        period_start, period_end = periods[period]
        period_relationships = [r for r in person_relationships if period_start <= r['year'] < period_end]
        period_graphs[period] = make_period_graph(entity_categories, period_relationships)

    commonnames = get_common_names(period_graphs)
    communities = get_communities(person_categories)
    for period in period_graphs:
        chart = make_period_chart(period_graphs[period], communities, commonnames, period)
        chartdict[period] = chart
    return chartdict


def concatenate_charts(chartdict):
    vconcat = alt.vconcat(chartdict[0], chartdict[1], chartdict[2])
    return vconcat


def make_chart_dict_alternative(grphdict, periods, commonnames):
    chartdict2 = {}
    charts = None
    for period in enumerate(grphdict.keys()):
        n = period[0]
        pn = period[1]
        periodgraph, pos = add_period_graph_commonalities(pn)
        labels = {node: node for node in periodgraph.nodes() if node in commonnames[pn]}
        start = periods[pn]['start']
        end = periods[pn]['end']
        chart = hvnx.draw(G=periodgraph,
                          pos=pos,
                          with_labels=True,
                          labels=labels,
                          node_size=hv.dim('centrality') * 200,
                          node_color='entity_type',
                          edge_color='community',
                          cmap='accent',
                          edge_cmap='category20',
                          node_tooltip=['name'],
                          # node_label='name',
                          font_color="black",
                          # font_size='11',
                          width=800,
                          height=600,
                          )
        chart.title = f"REMP network {start}-{end}"

        # chart.configure_view(width=800, height=600,)
        chart.Overlay.opts(title=f"REMP network {start}-{end}")
        if not charts:
            charts = chart
        else:
            charts = charts + chart

    # N.B. this is an alternative visualisation

    charts.Overlay.opts(title="REMP networks 1950s-1970s")
