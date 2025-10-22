import random
import time
import matplotlib.pyplot as plt
import math
import statistics
import itertools

def read_graph(filename):
    """
    Lit un graphe à partir d'un fichier texte.
    
    Args:
        filename: Chemin vers le fichier contenant la définition du graphe
        
    Returns:
        dict: Dictionnaire représentant la liste d'adjacence du graphe
    """
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
    """Classe représentant un graphe non orienté avec des listes d'adjacence."""
    
    def __init__(self, adj):
        """
        Initialise le graphe avec une liste d'adjacence.
        
        Args:
            adj: Dictionnaire des listes d'adjacence
        """
        self.adj = adj

    def sommets(self):
        """Retourne la liste des sommets du graphe."""
        return list(self.adj.keys())

    def aretes(self):
        """
        Retourne la liste des arêtes du graphe.
        
        Returns:
            list: Liste de tuples (u, v) représentant les arêtes
        """
        edges = set()
        for u in self.adj:
            for v in self.adj[u]:
                if (v, u) not in edges:
                    edges.add((u, v))
        return list(edges)

    def degree(self, v):
        """Retourne le degré du sommet v."""
        return len(self.adj[v])
    
    def degrees_dict(self):
        """
        Retourne un dictionnaire des degrés de tous les sommets.
        
        Returns:
            dict: Dictionnaire {sommet: degré}
        """
        return {u: len(neigh) for u, neigh in self.adj.items()}

    def degrees_list(self):
        """
        Retourne une liste des degrés de tous les sommets.
        
        Returns:
            list: Liste des degrés dans l'ordre des sommets
        """
        verts = self.sommets()
        return [self.degree(v) for v in verts]

    def max_degree_vertex(self, return_all=True):
        """
        Trouve le(s) sommet(s) de degré maximal.
        
        Args:
            return_all: Si True, retourne tous les sommets de degré maximal
                       Si False, retourne un seul sommet arbitraire
                       
        Returns:
            tuple: (liste des sommets, degré maximal) ou (sommet, degré maximal)
        """
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
        """Retourne une copie profonde du graphe."""
        return Graph({u: list(neigh) for u, neigh in self.adj.items()})
    
    def remove_vertex(self, v):
        """
        Retire un sommet du graphe et retourne un nouveau graphe.
        
        Args:
            v: Sommet à retirer
            
        Returns:
            Graph: Nouveau graphe sans le sommet v
        """
        if v not in self.adj:
            return self.copy()
        
        new_adj = {u: [x for x in neigh if x != v] for u, neigh in self.adj.items() if u != v}
        return Graph(new_adj)
    
    def remove_vertices(self, vertices):
        """
        Retire plusieurs sommets du graphe et retourne un nouveau graphe.
        
        Args:
            vertices: Ensemble ou liste de sommets à retirer
            
        Returns:
            Graph: Nouveau graphe sans les sommets spécifiés
        """
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
    
    def remove_vertices_inplace(self, vertices):
        """
        Retire plusieurs sommets du graphe en modifiant le graphe actuel.
        
        Args:
            vertices: Ensemble ou liste de sommets à retirer
        """
        to_remove = set(vertices)
        for v in list(to_remove):
            if v in self.adj:
                del self.adj[v]
        for u, neigh in self.adj.items():
            self.adj[u] = [w for w in neigh if w not in to_remove]
    
    def algo_couplage(self):
        """
        Algorithme de couplage pour le problème de couverture de sommets.
        
        Returns:
            set: Ensemble de sommets formant une couverture
        """
        edges = self.aretes()

        C = set()
        for (u, v) in edges:
            if (u not in C) and (v not in C):
                C.add(u)
                C.add(v)
        return C
    
    def algo_glouton(self):
        """
        Algorithme glouton pour le problème de couverture de sommets.
        
        Returns:
            set: Ensemble de sommets formant une couverture
        """
        Gc = self.copy()
        C = set()
        while True:
            edges = Gc.aretes()
            if not edges:
                break
            v, _ = Gc.max_degree_vertex(return_all=False)
            C.add(v)
            Gc.remove_vertices_inplace({v})
        return C
    
    def branchement_simple(self):
        """
        Algorithme de branchement simple pour la couverture de sommets.
        Évite de stocker des copies complètes du graphe.
        
        Returns:
            tuple: (meilleure couverture, nombre de nœuds générés)
        """
        best_C = set(self.sommets())
        
        # Pile : chaque élément est (arêtes_restantes, solution_courante)
        stack = []
        initial_edges = self.aretes()
        stack.append((initial_edges, set()))
        
        nodes_generated = 0  # Compteur de nœuds générés

        while stack:
            remaining_edges, current_solution = stack.pop()
            nodes_generated += 1

            # S'il n'y a plus d'arêtes, mettre à jour la meilleure solution
            if not remaining_edges:
                if len(current_solution) < len(best_C):
                    best_C = current_solution
                continue

            # Choisir une arête pour le branchement
            u, v = remaining_edges[0]

            # Branche 1 : choisir le sommet u
            new_edges1 = [(x, y) for (x, y) in remaining_edges if x != u and y != u]
            new_solution1 = current_solution | {u}
            stack.append((new_edges1, new_solution1))

            # Branche 2 : choisir le sommet v
            new_edges2 = [(x, y) for (x, y) in remaining_edges if x != v and y != v]
            new_solution2 = current_solution | {v}
            stack.append((new_edges2, new_solution2))

        return best_C, nodes_generated
    
    def est_couverture_valide(self, C):
        """
        Vérifie si un ensemble de sommets est une couverture valide.
        
        Args:
            C: Ensemble de sommets à vérifier
            
        Returns:
            bool: True si C couvre toutes les arêtes, False sinon
        """
        for u, v in self.aretes():
            if u not in C and v not in C:
                return False
        return True
    
    def __repr__(self):
        """Représentation textuelle du graphe."""
        return f"Graph({self.adj})"

def generate_random_graph(n: int, p: float):
    """
    Génère un graphe aléatoire selon le modèle G(n, p).
    
    Args:
        n: Nombre de sommets
        p: Probabilité qu'une arête existe entre deux sommets
        
    Returns:
        Graph: Graphe aléatoire généré
        
    Raises:
        ValueError: Si n <= 0 ou p n'est pas dans (0, 1)
    """
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

# ===========================
# Fonctions utilitaires
# ===========================

def mesure_algo(G, algo_name):
    """
    Mesure le temps d'exécution d'un algorithme sur un graphe.
    
    Args:
        G: Graphe d'entrée
        algo_name: Nom de l'algorithme ("glouton" ou "couplage")
        
    Returns:
        tuple: (temps d'exécution, taille de la couverture)
    """
    start = time.time()
    if algo_name == "glouton":
        C = G.algo_glouton()
    elif algo_name == "couplage":
        C = G.algo_couplage()
    else:
        raise ValueError("Algo inconnu")
    t = time.time() - start
    return t, len(C)

def mesure_temps_et_qualite(algo_name, n, p, num_instances=10):
    """
    Retourne le temps moyen et la taille moyenne de la couverture sur plusieurs instances.
    
    Args:
        algo_name: Nom de l'algorithme
        n: Nombre de sommets
        p: Probabilité des arêtes
        num_instances: Nombre d'instances à tester
        
    Returns:
        tuple: (temps moyen, taille moyenne de couverture)
    """
    temps_list = []
    taille_list = []
    for _ in range(num_instances):
        G = generate_random_graph(n, p)
        t, taille = mesure_algo(G, algo_name)
        temps_list.append(t)
        taille_list.append(taille)
    return sum(temps_list)/num_instances, sum(taille_list)/num_instances

# ===========================
# Recherche de Nmax
# ===========================

def trouver_Nmax(algo_name, p=0.3, seuil_sec=3):
    """
    Trouve la taille Nmax pour laquelle l'algorithme s'exécute en moins de seuil_sec secondes.
    
    Args:
        algo_name: Nom de l'algorithme
        p: Probabilité des arêtes
        seuil_sec: Seuil de temps en secondes
        
    Returns:
        int: Taille Nmax estimée
    """
    n = 1
    while True:
        t, _ = mesure_temps_et_qualite(algo_name, n, p, num_instances=3)
        if t > seuil_sec:
            return n
        n += 1

# ===========================
# Mesure et tracé
# ===========================

def tests_algos(p=0.3, num_instances=10):
    """
    Teste et compare les algorithmes glouton et de couplage.
    
    Args:
        p: Probabilité des arêtes
        num_instances: Nombre d'instances par test
        
    Returns:
        tuple: Résultats des tests
    """
    # 1. Identifier Nmax
    Nmax_glouton = trouver_Nmax("glouton", p)
    Nmax_couplage = trouver_Nmax("couplage", p)
    Nmax = min(Nmax_glouton, Nmax_couplage)  # prendre la taille commune pour comparaison
    print(f"Nmax estimé = {Nmax}")

    # 2. Définir les 10 points de test
    ns = [int(Nmax * i / 10) for i in range(1, 11)]

    temps_glouton, taille_glouton = [], []
    temps_couplage, taille_couplage = [], []

    # 3. Mesurer temps moyen et couverture moyenne
    for n in ns:
        t_g, s_g = mesure_temps_et_qualite("glouton", n, p, num_instances)
        t_c, s_c = mesure_temps_et_qualite("couplage", n, p, num_instances)
        temps_glouton.append(t_g)
        taille_glouton.append(s_g)
        temps_couplage.append(t_c)
        taille_couplage.append(s_c)

    # 4. Tracer les courbes temps (échelle log-log)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(
        [math.log2(n) for n in ns],
        [math.log2(t) if t > 0 else 0 for t in temps_glouton],
        'r-o', label='Glouton'
    )
    plt.plot(
        [math.log2(n) for n in ns],
        [math.log2(t) if t > 0 else 0 for t in temps_couplage],
        'g-o', label='Couplage'
    )
    plt.xlabel("log₂(n)")
    plt.ylabel("log₂(Temps moyen en s)")
    plt.title(f"Échelle log-log (p={p})")
    plt.grid(True)
    plt.legend()

    # 5. Tracer les courbes qualité (taille de couverture)
    plt.subplot(1,2,2)
    plt.plot(ns, taille_glouton, 'r-o', label='Glouton')
    plt.plot(ns, taille_couplage, 'g-o', label='Couplage')
    plt.xlabel("Nombre de sommets n")
    plt.ylabel("Taille moyenne couverture")
    plt.title("Qualité des solutions")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return ns, temps_glouton, temps_couplage, taille_glouton, taille_couplage

# Vérification par force brute pour petits graphes (n <= 20 à utiliser avec prudence)
def bruteforce_vertex_cover(adj):
    """
    Algorithme de force brute pour la couverture de sommets.
    À n'utiliser que pour de petits graphes (n <= 20).
    
    Args:
        adj: Liste d'adjacence du graphe
        
    Returns:
        set: Couverture de sommets optimale
    """
    # adj: dict[int,list[int]]
    V = sorted(adj.keys())
    m = len(V)
    edges = [(u,v) for u in adj for v in adj[u] if u < v]
    best = None
    for r in range(0, m+1):
        for subset in itertools.combinations(V, r):
            S = set(subset)
            ok = True
            for (u,v) in edges:
                if u not in S and v not in S:
                    ok = False
                    break
            if ok:
                best = S
                return best  # première couverture minimale trouvée -> optimale
    return best

# ===========================
# Tests pour la méthode de branchement
# ===========================

def mesurer_branchement_instance_return_nodes(n, p, max_time_par_instance=60):
    """
    Génère une instance G(n,p), exécute G.branchement_simple() et renvoie les résultats.
    
    Args:
        n: Nombre de sommets
        p: Probabilité des arêtes
        max_time_par_instance: Temps maximum par instance (secondes)
        
    Returns:
        tuple: (temps, taille_couverture, valide_flag, timeout_flag, nodes_generated)
    """
    G = generate_random_graph(n, p)
    t0 = time.time()
    result = G.branchement_simple()
    t = time.time() - t0

    # Compatibilité : si retourne (C, nodes_generated) on extrait, sinon nodes_generated=None
    if isinstance(result, tuple) and len(result) == 2:
        C, nodes_generated = result
    else:
        C = result
        nodes_generated = None

    valide = G.est_couverture_valide(C)
    timeout_flag = (t > max_time_par_instance)
    return t, len(C), valide, timeout_flag, nodes_generated

def tester_branchement_sur_une_valeur_p(n_values, p_values_labels, num_instances=5, max_time_par_instance=30):
    """
    Exécute des tests pour une liste de n et différentes valeurs de p.
    
    Args:
        n_values: Liste des tailles de graphes à tester
        p_values_labels: Liste des probabilités ou '1/sqrt' pour p=1/√n
        num_instances: Nombre d'instances par configuration
        max_time_par_instance: Temps maximum par instance
        
    Returns:
        dict: Résultats agrégés pour chaque valeur de p
    """
    all_results = {}

    for p_label in p_values_labels:
        res = {'n': [], 'temps_moyen': [], 'nodes_moyen': [], 'timeouts_rate': [], 'valid_rate': []}

        for n in n_values:
            # Déterminer p effectif pour ce n
            if isinstance(p_label, str) and p_label == '1/sqrt':
                p_eff = 1.0 / math.sqrt(n)
            else:
                p_eff = float(p_label)

            temps_list = []
            nodes_list = []
            valides = 0
            timeouts = 0

            for _ in range(num_instances):
                t, taille, valide, timeout_flag, nodes_generated = mesurer_branchement_instance_return_nodes(n, p_eff, max_time_par_instance)
                temps_list.append(t)
                if nodes_generated is not None:
                    nodes_list.append(nodes_generated)
                if valide:
                    valides += 1
                if timeout_flag:
                    timeouts += 1

            res['n'].append(n)
            # Moyenne des temps (en secondes)
            res['temps_moyen'].append(statistics.mean(temps_list))
            # nodes_generated peut être vide si la méthode ne rapporte pas ce nombre
            res['nodes_moyen'].append(statistics.mean(nodes_list) if nodes_list else None)
            res['timeouts_rate'].append(timeouts / num_instances)
            res['valid_rate'].append(valides / num_instances)

            # Affichage console intermédiaire pour suivre la progression
            p_show = f"1/sqrt(n)" if (isinstance(p_label, str) and p_label == '1/sqrt') else f"{p_eff:.3f}"
            print(f"[p={p_show}] n={n} -> t_mean={res['temps_moyen'][-1]:.3f}s, "
                  f"timeouts={res['timeouts_rate'][-1]*100:.0f}%, valid={res['valid_rate'][-1]*100:.0f}%, "
                  f"nodes_mean={res['nodes_moyen'][-1]}")

        all_results[p_label] = res

    return all_results

def tracer_resultats_branchement(all_results, title_suffix=""):
    """
    Trace deux sous-figures :
      - log2(temps_moyen) vs n (semi-log: y en log2)
      - nodes_moyen vs n (échelle linéaire)
      
    Args:
        all_results: Dictionnaire de résultats retourné par tester_branchement_sur_une_valeur_p
        title_suffix: Suffixe à ajouter au titre des graphiques
    """
    plt.figure(figsize=(12,5))

    # Sous-graphique 1 : temps (y en log2, x en n)
    plt.subplot(1,2,1)
    for p_label, res in all_results.items():
        ns = res['n']
        temps = res['temps_moyen']
        # Éviter log(0) en remplaçant par petite valeur
        ys = [math.log2(max(t, 1e-9)) for t in temps]
        legend = "1/sqrt(n)" if (isinstance(p_label, str) and p_label == '1/sqrt') else f"p={p_label}"
        plt.plot(ns, ys, marker='o', label=legend)
    plt.xlabel("n (nombre de sommets)")
    plt.ylabel("log₂(Temps moyen (s))")
    plt.title("Branchement simple — temps (semi-log) " + title_suffix)
    plt.grid(True)
    plt.legend()

    # Sous-graphique 2 : nodes_generated (échelle linéaire)
    plt.subplot(1,2,2)
    any_nodes = False
    for p_label, res in all_results.items():
        ns = res['n']
        nodes = res['nodes_moyen']
        if any(v is not None for v in nodes):
            # Utiliser l'échelle linéaire, tracer directement le nombre de nœuds
            xs_plot = []
            ys_plot = []
            for x, y in zip(ns, nodes):
                if y is not None:
                    xs_plot.append(x)
                    ys_plot.append(y)
            legend = "1/sqrt(n)" if (isinstance(p_label, str) and p_label == '1/sqrt') else f"p={p_label}"
            plt.plot(xs_plot, ys_plot, marker='o', label=legend)
            any_nodes = True

    if any_nodes:
        plt.xlabel("n (nombre de sommets)")
        plt.ylabel("Nombre moyen de noeuds générés")
        plt.title("Branchement simple — taille de l'arbre de recherche " + title_suffix)
        plt.grid(True)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "nodes_generated non fourni par branchement_simple", ha='center')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    """
    Programme principal avec menu interactif pour tester les fonctions de graphes.
    """
    current_graph = None
    
    while True:
        print("\n" + "="*60)
        print("MENU PRINCIPAL - ALGORITHMES DE GRAPHES")
        print("="*60)
        print("1. Charger un graphe depuis un fichier")
        print("2. Générer un graphe aléatoire")
        print("3. Afficher les informations du graphe courant")
        print("4. Tester les algorithmes de couverture")
        print("5. Tester l'algorithme de branchement simple")
        print("6. Comparaison des algorithmes glouton et couplage")
        print("7. Tests de performance du branchement")
        print("8. Vérification par force brute (petits graphes)")
        print("9. Quitter")
        print("-"*60)
        
        choix = input("Votre choix (1-9): ").strip()
        
        if choix == "1":
            # Charger un graphe depuis un fichier
            filename = input("Entrez le nom du fichier (ex: exempleinstance.txt): ").strip()
            try:
                current_graph = Graph(read_graph(filename))
                print(f"✓ Graphe chargé depuis {filename}")
                print(f"  Sommets: {len(current_graph.sommets())}, Arêtes: {len(current_graph.aretes())}")
            except Exception as e:
                print(f"✗ Erreur lors du chargement: {e}")
                
        elif choix == "2":
            # Générer un graphe aléatoire
            try:
                n = int(input("Nombre de sommets (n): "))
                p = float(input("Probabilité des arêtes (p, 0-1): "))
                current_graph = generate_random_graph(n, p)
                print(f"✓ Graphe aléatoire G({n}, {p}) généré")
                print(f"  Sommets: {len(current_graph.sommets())}, Arêtes: {len(current_graph.aretes())}")
            except ValueError as e:
                print(f"✗ Entrée invalide: {e}")
            except Exception as e:
                print(f"✗ Erreur: {e}")
                
        elif choix == "3":
            # Afficher les informations du graphe
            if current_graph is None:
                print("✗ Aucun graphe chargé. Veuillez d'abord charger ou générer un graphe.")
                continue
                
            print("\n--- INFORMATIONS DU GRAPHE ---")
            print(f"Sommets: {current_graph.sommets()}")
            print(f"Arêtes: {current_graph.aretes()}")
            print(f"Degrés par sommet: {current_graph.degrees_dict()}")
            max_verts, max_deg = current_graph.max_degree_vertex(return_all=True)
            print(f"Sommet(s) de degré maximal: {max_verts} (degré {max_deg})")
            
        elif choix == "4":
            # Tester les algorithmes de couverture
            if current_graph is None:
                print("✗ Aucun graphe chargé.")
                continue
                
            print("\n--- ALGORITHMES DE COUVERTURE ---")
            try:
                couverture_couplage = current_graph.algo_couplage()
                couverture_glouton = current_graph.algo_glouton()
                
                print(f"Couverture par couplage: {couverture_couplage}")
                print(f"  Taille: {len(couverture_couplage)}, Valide: {current_graph.est_couverture_valide(couverture_couplage)}")
                
                print(f"Couverture par glouton: {couverture_glouton}")
                print(f"  Taille: {len(couverture_glouton)}, Valide: {current_graph.est_couverture_valide(couverture_glouton)}")
                
            except Exception as e:
                print(f"✗ Erreur lors de l'exécution des algorithmes: {e}")
                
        elif choix == "5":
            # Tester l'algorithme de branchement simple
            if current_graph is None:
                print("✗ Aucun graphe chargé.")
                continue
                
            print("\n--- ALGORITHME DE BRANCHEMENT SIMPLE ---")
            try:
                if len(current_graph.sommets()) > 20:
                    print("⚠️  Attention: le graphe a plus de 20 sommets, le calcul peut être long.")
                    reponse = input("Voulez-vous continuer? (o/n): ").strip().lower()
                    if reponse != 'o':
                        continue
                
                start_time = time.time()
                resultat = current_graph.branchement_simple()
                temps_ecoule = time.time() - start_time
                
                if isinstance(resultat, tuple):
                    couverture, noeuds_generes = resultat
                    print(f"Couverture optimale: {couverture}")
                    print(f"  Taille: {len(couverture)}")
                    print(f"  Valide: {current_graph.est_couverture_valide(couverture)}")
                    print(f"  Noeuds générés: {noeuds_generes}")
                else:
                    couverture = resultat
                    print(f"Couverture optimale: {couverture}")
                    print(f"  Taille: {len(couverture)}")
                    print(f"  Valide: {current_graph.est_couverture_valide(couverture)}")
                    
                print(f"  Temps d'exécution: {temps_ecoule:.3f} secondes")
                
            except Exception as e:
                print(f"✗ Erreur lors de l'exécution du branchement: {e}")
                
        elif choix == "6":
            # Comparaison des algorithmes glouton et couplage
            print("\n--- COMPARAISON GLouton vs COUPLAGE ---")
            try:
                p = float(input("Probabilité p (défaut 0.3): ") or "0.3")
                num_instances = int(input("Nombre d'instances (défaut 10): ") or "10")
                
                print("Lancement des tests... (cela peut prendre quelques secondes)")
                tests_algos(p=p, num_instances=num_instances)
                
            except ValueError as e:
                print(f"✗ Entrée invalide: {e}")
            except Exception as e:
                print(f"✗ Erreur: {e}")
                
        elif choix == "7":
            # Tests de performance du branchement
            print("\n--- TESTS DE PERFORMANCE DU BRANCHEMENT ---")
            try:
                print("Valeurs recommandées:")
                print("  n_values = [8, 10, 12, 14, 16]")
                print("  p_values = [0.1, 0.3, 0.5, '1/sqrt']")
                print("  instances = 3-5, timeout = 30s")
                
                # Configuration par défaut
                n_values = [8, 10, 12, 14, 16]
                p_values_labels = [0.1, 0.3, 0.5, '1/sqrt']
                num_instances = 3
                max_time_par_instance = 30
                
                print(f"\nConfiguration actuelle:")
                print(f"  n_values: {n_values}")
                print(f"  p_values: {p_values_labels}")
                print(f"  instances: {num_instances}")
                print(f"  timeout: {max_time_par_instance}s")
                
                modifier = input("Voulez-vous modifier la configuration? (o/n): ").strip().lower()
                if modifier == 'o':
                    try:
                        n_input = input("n_values (séparés par des virgules, ex: 8,10,12,14,16): ")
                        if n_input:
                            n_values = [int(x.strip()) for x in n_input.split(',')]
                        
                        p_input = input("p_values (séparés par des virgules, ex: 0.1,0.3,0.5,1/sqrt): ")
                        if p_input:
                            p_values_labels = []
                            for p in p_input.split(','):
                                p = p.strip()
                                if p == '1/sqrt':
                                    p_values_labels.append(p)
                                else:
                                    p_values_labels.append(float(p))
                        
                        num_instances = int(input(f"Nombre d'instances (défaut {num_instances}): ") or num_instances)
                        max_time_par_instance = int(input(f"Timeout (défaut {max_time_par_instance}s): ") or max_time_par_instance)
                        
                    except ValueError as e:
                        print(f"✗ Configuration invalide, utilisation des valeurs par défaut: {e}")
                
                print(f"\nLancement des tests avec:")
                print(f"  n_values: {n_values}")
                print(f"  p_values: {p_values_labels}")
                print(f"  instances: {num_instances}")
                print(f"  timeout: {max_time_par_instance}s")
                print("Cela peut prendre plusieurs minutes...")
                
                all_results = tester_branchement_sur_une_valeur_p(
                    n_values, p_values_labels, 
                    num_instances=num_instances, 
                    max_time_par_instance=max_time_par_instance
                )
                
                tracer_resultats_branchement(all_results, title_suffix=f"(instances={num_instances})")
                
            except Exception as e:
                print(f"✗ Erreur lors des tests de performance: {e}")
                
        elif choix == "8":
            # Vérification par force brute
            if current_graph is None:
                print("✗ Aucun graphe chargé.")
                continue
                
            print("\n--- VÉRIFICATION PAR FORCE BRUTE ---")
            try:
                if len(current_graph.sommets()) > 20:
                    print("✗ La force brute n'est recommandée que pour n ≤ 20")
                    continue
                    
                start_time = time.time()
                couverture_brute = bruteforce_vertex_cover(current_graph.adj)
                temps_ecoule = time.time() - start_time
                
                print(f"Couverture optimale (force brute): {couverture_brute}")
                print(f"  Taille: {len(couverture_brute)}")
                print(f"  Valide: {current_graph.est_couverture_valide(couverture_brute)}")
                print(f"  Temps d'exécution: {temps_ecoule:.3f} secondes")
                
            except Exception as e:
                print(f"✗ Erreur lors de la force brute: {e}")
                
        elif choix == "9":
            # Quitter
            print("Au revoir!")
            break
            
        else:
            print("✗ Choix invalide. Veuillez choisir un nombre entre 1 et 9.")
        
        # Pause avant de revenir au menu
        if choix not in ["6", "7"]:  # Pas de pause après les tests automatiques
            input("\nAppuyez sur Entrée pour continuer...")

# Point d'entrée du programme
if __name__ == "__main__":
    # Exemple d'utilisation rapide (décommentez pour tester)
    # Ces tests seront exécutés avant d'afficher le menu
    
    print("Initialisation...")
    
    # Test avec un graphe chemin
    print("\nTest rapide avec un graphe chemin (0-1-2-3-4):")
    Gpath = Graph({0:[1], 1:[0,2], 2:[1,3], 3:[2,4], 4:[3]})
    
    print("Algorithme de couplage:", Gpath.algo_couplage())
    print("Algorithme glouton:", Gpath.algo_glouton())
    
    resultat_branchement = Gpath.branchement_simple()
    if isinstance(resultat_branchement, tuple):
        couverture, noeuds = resultat_branchement
        print(f"Branchement simple: {couverture} (noeuds: {noeuds})")
    else:
        print(f"Branchement simple: {resultat_branchement}")
    
    print("Validation:", Gpath.est_couverture_valide(resultat_branchement[0] if isinstance(resultat_branchement, tuple) else resultat_branchement))
    
    # Lancement du menu principal
    main()