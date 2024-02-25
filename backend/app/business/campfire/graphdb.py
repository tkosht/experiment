import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


def to_cypher_string(props: dict, suffix="") -> str:
    if not props:
        return ""
    return "{" + ", ".join([f"{k}: ${k}{suffix}" for k in props.keys()]) + "}"


def to_cypher_params(props: dict, suffix="") -> dict:
    if not props:
        return {}
    return {k + suffix: v for k, v in props.items()}


class GraphDb(object):
    def __init__(self):
        self.user = os.environ["neo4j_user"]
        self.pswd = os.environ["neo4j_pswd"]
        self.driver = GraphDatabase.driver(
            uri="bolt://neo4j", auth=(self.user, self.pswd)
        )

    def add_node(self, label: str, **props):
        def _add_node(tx, label, **props):
            properties = to_cypher_string(props)
            query = f"CREATE (n: {label} {properties})"
            tx.run(query, **props)

        with self.driver.session() as session:
            session.execute_write(_add_node, label, **props)

    def add_edge(
        self, label: str = "Edge", node_keys1: dict = {}, node_keys2: dict = {}
    ):
        """
        Adds an edge between two nodes in the graph.

        Args:
            label (str): The label of the edge. defaults to "Edge".
            node_keys1 (dict): The properties of the first node. if no label is provided, it defaults to "Node".
            node_keys2 (dict): The properties of the second node. if no label is provided, it defaults to "Node".
        """

        def _add_edge(tx, label: str, node_keys1: dict, node_keys2: dict):
            node_label1 = node_keys1.pop("label", "Node")
            node_label2 = node_keys2.pop("label", "Node")
            node_cond1 = to_cypher_string(node_keys1, suffix="1")
            node_cond2 = to_cypher_string(node_keys2, suffix="2")
            query = f"""
            MATCH (n1:{node_label1} {node_cond1}), (n2:{node_label2} {node_cond2})
            CREATE (n1)-[r:{label}]->(n2)
            """
            _node_keys1 = to_cypher_params(node_keys1, suffix="1")
            _node_keys2 = to_cypher_params(node_keys2, suffix="2")
            tx.run(query, **_node_keys1, **_node_keys2)

        with self.driver.session() as session:
            session.execute_write(_add_edge, label, node_keys1, node_keys2)

    def close(self):
        self.driver.close()


if __name__ == "__main__":
    g = GraphDb()
    g.add_node(
        label="TestLabel",
        url="http://localhost:8397/hello",
        name="test1",
        area="area1",
        value="value1",
    )
    g.add_node(
        label="TestLabel",
        url="http://localhost2:8397/world",
        name="test2",
        area="area2",
        value="value2",
    )

    g.add_edge(
        label="RELATION",
        node_keys1={"label": "TestLabel", "name": "test1"},
        node_keys2={"label": "TestLabel", "name": "test2"},
    )
    g.close()
    print("done!")
