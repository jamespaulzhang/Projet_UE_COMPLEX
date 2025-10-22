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

print("Une couverture par algo couplage: ",G.algo_couplage())
print("Une couverture par algo glouton: ",G.algo_glouton())

# tests_algos(p=0.3, num_instances=10)

# Nmax estimé = 547
# G = generate_random_graph(10,0.5)

# Exemples simples à tester
Gpath = Graph({0:[1], 1:[0,2], 2:[1,3], 3:[2,4], 4:[3]})  # chemin 5 sommets
brute = bruteforce_vertex_cover(Gpath.adj)
print("Bruteforce (path n=5):", brute)
res = Gpath.branchement_simple()
print("Branchement result:", res)
print("Valid?", Gpath.est_couverture_valide(res[0] if isinstance(res, tuple) else res))