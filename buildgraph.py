import json
import io
import networkx as nx
from utility import remove_dead_nodes
from networkx.readwrite.gpickle import write_gpickle

graphs = [
    (
        "r500",
        "records500.json",
        [48, 55, 54] + [1182, 1333, 1431, 1507],
    ),
    (
        "r1000",
        "records1000.json",
        [1554, 1561, 1560] + [516, 919, 1062, 1469],
    ),
    (
        "r2000",
        "records2000.json",
        [
            14166,
            14172,
            14173,
        ]
        + [18729, 19276, 19374, 19892],
    ),
    (
        "r4000",
        "records4000.json",
        [
            8407,
            8414,
            8413,
        ]
        + [18986, 19698, 19708, 20080],
    ),
    (
        "ads",
        "records.json",
        [
            128,
            288,
            134,
            233,
            329,
            282,
            105,
            234,
            139,
            81,
            277,
            121,
            122,
            127,
            61,
            126,
            31,
        ],
    ),
    (
        "ads5",
        "records5.json",
        [
            2822,
            2569,
            2186,
            32401,
            32403,
            32407,
            32409,
            32410,
            2459,
            2970,
            32423,
            2984,
            32428,
            2888,
            2761,
            2381,
            2382,
            2400,
            3047,
            32360,
            3178,
            32362,
            32364,
            32369,
            2546,
            3061,
            2171,
            2556,
        ],
    ),
    (
        "ads10",
        "records10.json",
        [
            1408,
            1921,
            1158,
            1159,
            1806,
            1554,
            1299,
            26,
            27,
            1691,
            1822,
            31,
            32,
            417,
            33,
            2081,
            419,
            42,
            1708,
            1846,
            2102,
            1977,
            1466,
            63,
            1472,
            2112,
            1730,
            2118,
            1607,
            2121,
            1225,
            1741,
            1358,
            1871,
            215,
            218,
            1373,
            1120,
            1121,
            1761,
            1132,
            1140,
            1398,
        ],
    ),
]


for name, json_location, DA_set in graphs:
    DA = DA_set[0]
    DG = nx.DiGraph(DA=DA)
    edges = json.load(io.open(f"data/{json_location}", "r", encoding="utf-8-sig"))

    for e in edges:
        access_type = e["p"]["segments"][0]["relationship"]["type"]
        if access_type in [
            "AdminTo",
            "MemberOf",
            "HasSession",
        ]:
            start = e["p"]["start"]["identity"]
            end = e["p"]["end"]["identity"]
            if start in DA_set:
                start = DA
            if end in DA_set:
                end = DA
            if start == end:
                continue
            if start != DA and not DG.has_edge(start, end):
                DG.add_edge(start, end, access_type=access_type)

    remove_dead_nodes(DG)
    write_gpickle(DG, f"data/{name}.gpickle")
