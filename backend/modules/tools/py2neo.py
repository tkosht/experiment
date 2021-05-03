if __name__ == "__main__":
    from py2neo import Graph

    g = Graph("bolt://neo4j:7687", name="testdb", auth=("neo4j", "test"))
    result = g.run("MATCH (n:Greeting) RETURN n.message")
    print(result)
    for r in result:
        print(r["n.message"])
