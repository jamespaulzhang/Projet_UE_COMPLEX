# 图的存储: 使用 邻接表 (liste d’adjacence)，用 dict[int, list[int]] 实现

import random

def read_graph(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    idx_sommets = lines.index("Sommets")
    idx_aretes = lines.index("Aretes")
    
    sommets = [int(x) for x in lines[idx_sommets + 1 : lines.index("Nombre d aretes")]]
    adj = {s: [] for s in sommets}
    
    for line in lines[idx_aretes + 1 :]:
        u, v = map(int, line.split())
        adj[u].append(v)
        adj[v].append(u)
    
    return adj


class Graph:
    def __init__(self, adj):
        self.adj = adj

    def sommets(self):
        return list(self.adj.keys())

    def aretes(self):
        edges = set()
        for u in self.adj:
            for v in self.adj[u]:
                if (v, u) not in edges:
                    edges.add((u, v))
        return list(edges)

    def degree(self, v):
        return len(self.adj[v])
    
    def degrees_dict(self):
        return {u: len(neigh) for u, neigh in self.adj.items()}

    def degrees_list(self):
        verts = self.sommets()
        return [self.degree(v) for v in verts]

    def max_degree_vertex(self, return_all=True):
        verts = self.sommets()
        if not verts:
            return ([], 0) if return_all else (None, 0)
        
        degs = self.degrees_dict()
        max_deg = max(degs.values())
        max_vertices = [v for v in verts if degs.get(v, 0) == max_deg]

        if return_all:
            return (max_vertices, max_deg)
        else:
            return (max_vertices[0], max_deg)
    
    def copy(self):
        return Graph({u: list(neigh) for u, neigh in self.adj.items()})
    
    def remove_vertex(self, v):
        if v not in self.adj:
            return self.copy()
        
        new_adj = {u: [x for x in neigh if x != v] for u, neigh in self.adj.items() if u != v}
        return Graph(new_adj)
    
    def remove_vertices(self, vertices):
        to_remove = set(vertices)

        if not to_remove:
            return self.copy()

        new_adj = {}
        for u, neigh in self.adj.items():
            if u in to_remove:
                continue
            filtered = [w for w in neigh if w not in to_remove]
            new_adj[u] = filtered

        return Graph(new_adj)
    
    def __repr__(self):
        return f"Graph({self.adj})"

def generate_random_graph(n: int, p: float):
    if n <= 0:
        raise ValueError("n doit être > 0")
    if not (0 < p < 1):
        raise ValueError("p doit être dans l'intervalle (0, 1)")

    adj = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i].append(j)
                adj[j].append(i)

    return Graph(adj)

# ======= Test =======
G = Graph(read_graph("exempleinstance.txt"))
print("Sommets:", G.sommets())
print("Aretes:", G.aretes())

sommets_de_G = G.sommets()
for v in sommets_de_G:
    print(f"Degree de sommet {v}:", G.degree(v))

print("Degree dictionnaire de G", G.degrees_dict())
print("Degree list de G", G.degrees_list())
print("Max degree vertex", G.max_degree_vertex(return_all=True))

nouveau_G = G.remove_vertex(1)
print("Nouveau graph:", nouveau_G)
print("Sommets:", nouveau_G.sommets())
print("Aretes:", nouveau_G.aretes())

nouveau_G2 = G.remove_vertices([1,2])
print("Nouveau graph:", nouveau_G2)
print("Sommets:", nouveau_G2.sommets())
print("Aretes:", nouveau_G2.aretes())

print(generate_random_graph(10,0.5))