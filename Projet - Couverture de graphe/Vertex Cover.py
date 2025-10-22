import random
import time
import matplotlib.pyplot as plt
import math
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
    
    def _borne_inf(self, remaining_edges):
        """
        Calcule une borne inférieure pour le nombre de sommets nécessaires
        pour couvrir les arêtes restantes.
        """
        if not remaining_edges:
            return 0
            
        # Construire le sous-graphe
        sub_vertices = set()
        for (x, y) in remaining_edges:
            sub_vertices.add(x)
            sub_vertices.add(y)
        sub_adj = {v: [] for v in sub_vertices}
        for (x, y) in remaining_edges:
            sub_adj[x].append(y)
            sub_adj[y].append(x)

        # Paramètres du sous-graphe
        n_sub = len(sub_vertices)
        m_sub = len(remaining_edges)
        
        # Calcul du degré max Δ dans le sous-graphe
        if n_sub == 0:
            delta_sub = 0
        else:
            delta_sub = max((len(neigh) for neigh in sub_adj.values()), default=0)

        # b1 = ceil(m / Δ) si Δ > 0, sinon 0
        if delta_sub > 0:
            b1 = math.ceil(m_sub / delta_sub)
        else:
            b1 = 0

        # b2 = taille du couplage maximal (approché gloutonnement)
        matched = set()
        matching_size = 0
        for (a, b) in remaining_edges:
            if a not in matched and b not in matched:
                matched.add(a)
                matched.add(b)
                matching_size += 1
        b2 = matching_size

        # b3 = borne basée sur la formule quadratique
        if n_sub > 0:
            inner = (2 * n_sub - 1) ** 2 - 8 * m_sub
            inner = max(inner, 0.0)
            b3_val = (2 * n_sub - 1 - math.sqrt(inner)) / 2.0
            b3 = math.ceil(b3_val)
            if b3 < 0:
                b3 = 0
        else:
            b3 = 0

        # Retourne le maximum des bornes
        return max(b1, b2, b3)
    
    def branchement_couplage_avec_borne(self):
        """
        Algorithme de branchement simple pour la couverture de sommets.
        À chaque nœud, on calcule :
          - une solution réalisable par l'algorithme de couplage sur le sous-graphe
          - des bornes inférieures b1, b2, b3 et b = max(b1,b2,b3)
        On utilise lower_bound = len(current_solution) + b pour le pruning.

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

            # Si plus d'arêtes, on a une solution réelle (courante)
            if not remaining_edges:
                if len(current_solution) < len(best_C):
                    best_C = set(current_solution)
                continue

            # --- Construire le sous-graphe induit par remaining_edges ---
            sub_vertices = set()
            for (x, y) in remaining_edges:
                sub_vertices.add(x)
                sub_vertices.add(y)
            sub_adj = {v: [] for v in sub_vertices}
            for (x, y) in remaining_edges:
                sub_adj[x].append(y)
                sub_adj[y].append(x)

            # Paramètres du sous-graphe
            n_sub = len(sub_vertices)
            m_sub = len(remaining_edges)
            # Calcul du degré max Δ dans le sous-graphe
            if n_sub == 0:
                delta_sub = 0
            else:
                delta_sub = max((len(neigh) for neigh in sub_adj.values()), default=0)

            # --- Calcul d'un matching (greedy) sur remaining_edges to get b2 = |M| ---
            # On va construire un matching M_edges en marquant les sommets utilisés
            matched = set()
            matching_size = 0
            for (a, b) in remaining_edges:
                if a not in matched and b not in matched:
                    matched.add(a)
                    matched.add(b)
                    matching_size += 1
            b2 = matching_size

            # --- Calcul de b1, b3 ---
            # b1 = ceil(m / Δ) si Δ > 0, sinon 0
            if delta_sub > 0:
                b1 = math.ceil(m_sub / delta_sub)
            else:
                b1 = 0

            # b3 = ceil( (2n-1 - sqrt((2n-1)^2 - 8m)) / 2 )
            # protéger l'intérieur de la racine contre une petite négativité numérique
            if n_sub > 0:
                inner = (2 * n_sub - 1) ** 2 - 8 * m_sub
                inner = max(inner, 0.0)
                b3_val = (2 * n_sub - 1 - math.sqrt(inner)) / 2.0
                b3 = math.ceil(b3_val)
                if b3 < 0:
                    b3 = 0
            else:
                b3 = 0

            # borne b = max(b1, b2, b3)
            b_lower = max(b1, b2, b3)

            # --- Calculer une solution réalisable via algo_couplage sur le sous-graphe ---
            subG = Graph(sub_adj)
            C_couplage_sub = subG.algo_couplage()  # ensemble de sommets couvrant toutes les remaining_edges

            # Solution réalisable en ce nœud: current_solution ∪ C_couplage_sub
            feasible_solution = current_solution | C_couplage_sub
            # Si cette solution réalisable est meilleure que best_C, on la conserve
            if len(feasible_solution) < len(best_C):
                best_C = set(feasible_solution)

            # Borne inférieure combinée : current_solution + b_lower
            lower_bound = len(current_solution) + b_lower

            # Si la borne n'améliore pas best_C, on coupe ce nœud (pruning)
            if lower_bound >= len(best_C):
                continue

            # Choisir une arête pour le branchement (heuristique simple: la première)
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
    
    def branchement_avec_glouton_seulement(self):
        """
        Branchement utilisant l'algorithme glouton pour générer des solutions réalisables
        """
        best_C = set(self.sommets())
        initial_edges = self.aretes()
        stack = [(initial_edges, set())]
        nodes_generated = 0

        while stack:
            remaining_edges, current_solution = stack.pop()
            nodes_generated += 1

            # Élagage si solution courante déjà pire
            if len(current_solution) >= len(best_C):
                continue

            # Solution par algorithme glouton pour le sous-graphe restant
            if remaining_edges:
                temp_adj = {}
                for u, v in remaining_edges:
                    temp_adj.setdefault(u, []).append(v)
                    temp_adj.setdefault(v, []).append(u)
                
                # Utiliser l'algorithme glouton
                couverture_glouton = Graph(temp_adj).algo_glouton()
                candidate = current_solution.union(couverture_glouton)
                
                if len(candidate) < len(best_C):
                    best_C = candidate

            # Si plus d'arêtes, solution courante est valide
            if not remaining_edges:
                if len(current_solution) < len(best_C):
                    best_C = current_solution
                continue

            # Branchement standard
            u, v = remaining_edges[0]
            
            # Branche 1
            edges1 = [(x, y) for x, y in remaining_edges if x != u and y != u]
            sol1 = current_solution | {u}
            if len(sol1) < len(best_C):
                stack.append((edges1, sol1))
            
            # Branche 2
            edges2 = [(x, y) for x, y in remaining_edges if x != v and y != v]
            sol2 = current_solution | {v}
            if len(sol2) < len(best_C):
                stack.append((edges2, sol2))

        return best_C, nodes_generated
    
    def branchement_glouton_avec_borne(self):
        """
        Branchement utilisant l'algorithme glouton pour générer des solutions réalisables
        ET les bornes inférieures pour le pruning
        """
        best_C = set(self.sommets())
        initial_edges = self.aretes()
        stack = [(initial_edges, set())]
        nodes_generated = 0

        while stack:
            remaining_edges, current_solution = stack.pop()
            nodes_generated += 1

            # Élagage si solution courante déjà pire
            if len(current_solution) >= len(best_C):
                continue

            # Calcul de la borne inférieure
            LB = self._borne_inf(remaining_edges)
            
            # Élagage par borne inférieure
            if len(current_solution) + LB >= len(best_C):
                continue

            # Solution par algorithme glouton pour le sous-graphe restant
            if remaining_edges:
                temp_adj = {}
                for u, v in remaining_edges:
                    temp_adj.setdefault(u, []).append(v)
                    temp_adj.setdefault(v, []).append(u)
                
                # Utiliser l'algorithme glouton
                couverture_glouton = Graph(temp_adj).algo_glouton()
                candidate = current_solution.union(couverture_glouton)
                
                if len(candidate) < len(best_C):
                    best_C = candidate

            # Si plus d'arêtes, solution courante est valide
            if not remaining_edges:
                if len(current_solution) < len(best_C):
                    best_C = current_solution
                continue

            # Branchement standard
            u, v = remaining_edges[0]
            
            # Branche 1
            edges1 = [(x, y) for x, y in remaining_edges if x != u and y != u]
            sol1 = current_solution | {u}
            if len(sol1) < len(best_C):
                stack.append((edges1, sol1))
            
            # Branche 2
            edges2 = [(x, y) for x, y in remaining_edges if x != v and y != v]
            sol2 = current_solution | {v}
            if len(sol2) < len(best_C):
                stack.append((edges2, sol2))

        return best_C, nodes_generated

    def branchement_avec_bornes_seulement(self):
        """
        Branchement utilisant seulement les bornes inférieures (sans solutions réalisables)
        """
        best_C = set(self.sommets())
        initial_edges = self.aretes()
        stack = [(initial_edges, set())]
        nodes_generated = 0

        while stack:
            remaining_edges, current_solution = stack.pop()
            nodes_generated += 1

            # Élagage si solution courante déjà pire
            if len(current_solution) >= len(best_C):
                continue

            # Calcul de la borne inférieure
            LB = self._borne_inf(remaining_edges)
            
            # Élagage par borne inférieure
            if len(current_solution) + LB >= len(best_C):
                continue

            # Si plus d'arêtes, solution courante est valide
            if not remaining_edges:
                if len(current_solution) < len(best_C):
                    best_C = current_solution
                continue

            # Branchement standard
            u, v = remaining_edges[0]
            
            # Branche 1
            edges1 = [(x, y) for x, y in remaining_edges if x != u and y != u]
            sol1 = current_solution | {u}
            if len(sol1) < len(best_C):
                stack.append((edges1, sol1))
            
            # Branche 2
            edges2 = [(x, y) for x, y in remaining_edges if x != v and y != v]
            sol2 = current_solution | {v}
            if len(sol2) < len(best_C):
                stack.append((edges2, sol2))

        return best_C, nodes_generated

    def branchement_avec_couplage_seulement(self):
        """
        Branchement utilisant seulement le couplage (borne inférieure triviale = 0)
        """
        best_C = set(self.sommets())
        initial_edges = self.aretes()
        stack = [(initial_edges, set())]
        nodes_generated = 0

        while stack:
            remaining_edges, current_solution = stack.pop()
            nodes_generated += 1

            # Élagage si solution courante déjà pire
            if len(current_solution) >= len(best_C):
                continue

            # Solution par couplage pour le sous-graphe restant
            if remaining_edges:
                temp_adj = {}
                for u, v in remaining_edges:
                    temp_adj.setdefault(u, []).append(v)
                    temp_adj.setdefault(v, []).append(u)
                
                couverture_couplage = Graph(temp_adj).algo_couplage()
                candidate = current_solution.union(couverture_couplage)
                
                if len(candidate) < len(best_C):
                    best_C = candidate

            # Si plus d'arêtes, solution courante est valide
            if not remaining_edges:
                if len(current_solution) < len(best_C):
                    best_C = current_solution
                continue

            # Branchement standard
            u, v = remaining_edges[0]
            
            # Branche 1
            edges1 = [(x, y) for x, y in remaining_edges if x != u and y != u]
            sol1 = current_solution | {u}
            if len(sol1) < len(best_C):
                stack.append((edges1, sol1))
            
            # Branche 2
            edges2 = [(x, y) for x, y in remaining_edges if x != v and y != v]
            sol2 = current_solution | {v}
            if len(sol2) < len(best_C):
                stack.append((edges2, sol2))

        return best_C, nodes_generated
    
    def branchement_ameliore_v1(self):
        """
        Branchement amélioré version 1 : dans la deuxième branche, exclure les cas déjà traités
        """
        best_C = set(self.sommets())
        initial_edges = self.aretes()
        stack = [(initial_edges, set())]
        nodes_generated = 0

        while stack:
            remaining_edges, current_solution = stack.pop()
            nodes_generated += 1

            # Élagage
            if len(current_solution) >= len(best_C):
                continue

            # Si pas d'arêtes restantes, mettre à jour la meilleure solution
            if not remaining_edges:
                if len(current_solution) < len(best_C):
                    best_C = current_solution
                continue

            # Choisir une arête pour le branchement
            u, v = remaining_edges[0]

            # Branche 1 : choisir le sommet u
            edges1 = [(x, y) for x, y in remaining_edges if x != u and y != u]
            sol1 = current_solution | {u}
            if len(sol1) < len(best_C):
                stack.append((edges1, sol1))

            # Branche 2 : choisir le sommet v, et ne pas choisir u (donc doit choisir tous les voisins de u)
            # Obtenir tous les voisins de u (sauf v, car v est déjà choisi)
            neighbors_u = set()
            for x, y in remaining_edges:
                if x == u and y != v:
                    neighbors_u.add(y)
                elif y == u and x != v:
                    neighbors_u.add(x)
            
            # Dans la deuxième branche, choisir v et tous les voisins de u
            sol2 = current_solution | {v} | neighbors_u
            
            # Supprimer toutes les arêtes liées à u, v et les voisins de u
            vertices_to_remove = {u, v} | neighbors_u
            edges2 = [(x, y) for x, y in remaining_edges 
                    if x not in vertices_to_remove and y not in vertices_to_remove]
            
            if len(sol2) < len(best_C):
                stack.append((edges2, sol2))

        return best_C, nodes_generated
    
    def branchement_ameliore_v2(self):
        """
        Branchement amélioré version 2 : choisir le sommet de degré maximal pour le branchement
        """
        best_C = set(self.sommets())
        initial_edges = self.aretes()
        stack = [(initial_edges, set())]
        nodes_generated = 0

        while stack:
            remaining_edges, current_solution = stack.pop()
            nodes_generated += 1

            # Élagage
            if len(current_solution) >= len(best_C):
                continue

            # Si pas d'arêtes restantes, mettre à jour la meilleure solution
            if not remaining_edges:
                if len(current_solution) < len(best_C):
                    best_C = current_solution
                continue

            # Calculer le degré de chaque sommet dans le graphe actuel
            degree = {}
            for u, v in remaining_edges:
                degree[u] = degree.get(u, 0) + 1
                degree[v] = degree.get(v, 0) + 1

            # Trouver l'arête avec le sommet de degré maximal
            max_degree = -1
            best_edge = None
            for u, v in remaining_edges:
                current_max = max(degree.get(u, 0), degree.get(v, 0))
                if current_max > max_degree:
                    max_degree = current_max
                    best_edge = (u, v)

            u, v = best_edge
            
            # S'assurer que u est le sommet avec le degré le plus élevé
            if degree.get(v, 0) > degree.get(u, 0):
                u, v = v, u

            # Branche 1 : choisir le sommet de degré maximal u
            edges1 = [(x, y) for x, y in remaining_edges if x != u and y != u]
            sol1 = current_solution | {u}
            if len(sol1) < len(best_C):
                stack.append((edges1, sol1))

            # Branche 2 : choisir v, et ne pas choisir u (donc doit choisir tous les voisins de u)
            neighbors_u = set()
            for x, y in remaining_edges:
                if x == u and y != v:
                    neighbors_u.add(y)
                elif y == u and x != v:
                    neighbors_u.add(x)
            
            sol2 = current_solution | {v} | neighbors_u
            vertices_to_remove = {u, v} | neighbors_u
            edges2 = [(x, y) for x, y in remaining_edges 
                    if x not in vertices_to_remove and y not in vertices_to_remove]
            
            if len(sol2) < len(best_C):
                stack.append((edges2, sol2))

        return best_C, nodes_generated
    
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
def tester_strategies_branchement(
    n_values, p_values_labels, 
    num_instances=3, max_time_par_instance=30, 
    strategies_to_run=None
):
    """
    Teste différentes stratégies de branchement sur plusieurs instances
    """
    if strategies_to_run is None:
        strategies_to_run = ['simple', 'couplage_borne', 'glouton', 'bornes', 'couplage', 'glouton_borne']

    all_results = {}
    
    # Mapping des noms de stratégies vers les méthodes
    strategy_methods = {
        'simple': 'branchement_simple',
        'couplage_borne': 'branchement_couplage_avec_borne',  # 修改这里
        'glouton': 'branchement_avec_glouton_seulement', 
        'bornes': 'branchement_avec_bornes_seulement',
        'couplage': 'branchement_avec_couplage_seulement',  # 修改这里
        'glouton_borne': 'branchement_glouton_avec_borne'
    }

    for n in n_values:
        for p_label in p_values_labels:
            if p_label == '1/sqrt':
                p = 1.0 / math.sqrt(n)
            else:
                p = float(p_label)
            
            key = (n, p)
            all_results[key] = {s: [] for s in strategies_to_run}

            for inst in range(num_instances):
                G = generate_random_graph(n, p)
                print(f"n={n}, p={p:.4f}, inst={inst+1}/{num_instances}")

                for strategy in strategies_to_run:
                    method_name = strategy_methods[strategy]
                    method = getattr(G, method_name)
                    
                    t0 = time.time()
                    try:
                        result = method()
                        t1 = time.time()
                        
                        if isinstance(result, tuple):
                            coverage, nodes = result
                        else:
                            coverage = result
                            nodes = None
                        
                        valid = G.est_couverture_valide(coverage)
                        timeout = (t1 - t0) > max_time_par_instance
                        
                        all_results[key][strategy].append({
                            'time': t1 - t0,
                            'nodes': nodes,
                            'size': len(coverage),
                            'valid': valid,
                            'timeout': timeout
                        })
                        
                    except Exception as e:
                        all_results[key][strategy].append({
                            'time': None, 'nodes': None, 'size': None, 
                            'valid': False, 'error': str(e), 'timeout': False
                        })

    return all_results

def mesurer_branchement_instance_return_nodes(n, p, methode='simple', max_time_par_instance=60):
    """
    Génère une instance G(n,p), exécute la méthode spécifiée et renvoie les résultats.
    
    Args:
        n: Nombre de sommets
        p: Probabilité des arêtes
        methode: 'simple', 'couplage_borne' ou 'test'
        max_time_par_instance: Temps maximum par instance (secondes)
        
    Returns:
        tuple: (temps, taille_couverture, valide_flag, timeout_flag, nodes_generated)
    """
    G = generate_random_graph(n, p)
    t0 = time.time()
    
    if methode == 'simple':
        result = G.branchement_simple()
    elif methode == 'couplage_borne':  # 修改这里
        result = G.branchement_couplage_avec_borne()  # 修改这里
    elif methode == 'test':
        result = G.branchement_avec_glouton_seulement()
    else:
        raise ValueError(f"Méthode inconnue : {methode}")

    t = time.time() - t0

    if isinstance(result, tuple) and len(result) == 2:
        C, nodes_generated = result
    else:
        C = result
        nodes_generated = None

    valide = G.est_couverture_valide(C)
    timeout_flag = (t > max_time_par_instance)
    return t, len(C), valide, timeout_flag, nodes_generated

def tester_branchement_sur_une_valeur_p_complet(
    n_values, p_values_labels, 
    num_instances=3, max_time_par_instance=30, 
    methods_to_run=None):
    """
    Pour chaque n in n_values et chaque p in p_values_labels (float ou '1/sqrt'),
    génère num_instances graphes aléatoires et exécute uniquement les méthodes spécifiées.
    methods_to_run: liste parmi ['simple', 'couplage_borne', 'test'], par défaut toutes.
    Retourne un dict structuré similaire à avant.
    """
    if methods_to_run is None:
        methods_to_run = ['simple', 'couplage_borne', 'test']  # 修改这里

    all_results = {}
    for n in n_values:
        for p_label in p_values_labels:
            if p_label == '1/sqrt':
                p = 1.0 / math.sqrt(n)
            else:
                p = float(p_label)
            key = (n, p)
            all_results[key] = {m: [] for m in methods_to_run}

            for inst in range(num_instances):
                G = generate_random_graph(n, p)

                if 'simple' in methods_to_run:
                    t0 = time.time()
                    try:
                        res_s = G.branchement_simple()
                        t1 = time.time()
                        cov_s, nodes_s = (res_s if isinstance(res_s, tuple) else (res_s, None))
                        valid_s = G.est_couverture_valide(cov_s)
                        all_results[key]['simple'].append({
                            'time': t1 - t0, 'nodes': nodes_s, 'size': len(cov_s), 'valid': valid_s
                        })
                    except Exception as e:
                        all_results[key]['simple'].append({'time': None, 'nodes': None, 'size': None, 'valid': False, 'error': str(e)})

                if 'couplage_borne' in methods_to_run:  # 修改这里
                    t0 = time.time()
                    try:
                        res_c = G.branchement_couplage_avec_borne()  # 修改这里
                        t1 = time.time()
                        cov_c, nodes_c = (res_c if isinstance(res_c, tuple) else (res_c, None))
                        valid_c = G.est_couverture_valide(cov_c)
                        all_results[key]['couplage_borne'].append({  # 修改这里
                            'time': t1 - t0, 'nodes': nodes_c, 'size': len(cov_c), 'valid': valid_c
                        })
                    except Exception as e:
                        all_results[key]['couplage_borne'].append({'time': None, 'nodes': None, 'size': None, 'valid': False, 'error': str(e)})  # 修改这里

                if 'test' in methods_to_run:
                    t0 = time.time()
                    try:
                        res_t = G.branchement_avec_glouton_seulement()
                        t1 = time.time()
                        cov_t, nodes_t = (res_t if isinstance(res_t, tuple) else (res_t, None))
                        valid_t = G.est_couverture_valide(cov_t)
                        all_results[key]['test'].append({
                            'time': t1 - t0, 'nodes': nodes_t, 'size': len(cov_t), 'valid': valid_t
                        })
                    except Exception as e:
                        all_results[key]['test'].append({'time': None, 'nodes': None, 'size': None, 'valid': False, 'error': str(e)})

                print(f"n={n}, p={p:.4f}, inst={inst+1}/{num_instances} done.")

    return all_results

def tracer_comparaison_strategies(all_results, strategies_to_plot=None, title_suffix=""):
    """
    Trace la comparaison des différentes stratégies de branchement
    """
    # Extraire les clés et déterminer les stratégies à tracer
    keys = list(all_results.keys())
    ns = sorted({k[0] for k in keys})
    ps = sorted({k[1] for k in keys})
    
    if strategies_to_plot is None:
        example_key = keys[0]
        strategies_to_plot = list(all_results[example_key].keys())

    # Noms complets pour la légende
    strategy_names = {
        'simple': 'Branchement simple',
        'couplage_borne': 'Avec couplage et bornes',
        'glouton': 'Avec algorithme glouton', 
        'bornes': 'Avec bornes inférieures',
        'couplage': 'Avec couplage seulement',
        'glouton_borne': 'Avec glouton et bornes'
    }

    strategy_colors = {
        'simple': 'red', 
        'couplage_borne': 'blue', 
        'glouton': 'green', 
        'bornes': 'orange',
        'couplage': 'purple',
        'glouton_borne': 'brown'
    }
    
    p_markers = {
        0.1: 'o',
        0.3: 's',
        0.5: '^',
        '1/sqrt': 'D'
    }
    
    p_linestyles = {
        0.1: '-',
        0.3: '--',
        0.5: '-.',
        '1/sqrt': ':'
    }

    # Créer les figures
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_time, ax_nodes, ax_size, ax_ratio = axes.flatten()

    for p in ps:
        xs = []
        data = {s: {'time': [], 'nodes': [], 'size': []} for s in strategies_to_plot}

        for n in ns:
            key = (n, p)
            if key not in all_results:
                continue

            for strategy in strategies_to_plot:
                records = all_results[key].get(strategy, [])
                
                def mean_or_nan(lst, field):
                    vals = [r[field] for r in lst if r.get(field) is not None]
                    return float(sum(vals)) / len(vals) if vals else float('nan')
                
                data[strategy]['time'].append(mean_or_nan(records, 'time'))
                data[strategy]['nodes'].append(mean_or_nan(records, 'nodes'))
                data[strategy]['size'].append(mean_or_nan(records, 'size'))

            xs.append(n)

        for strategy in strategies_to_plot:
            if strategy not in strategy_colors:
                continue
                
            color = strategy_colors[strategy]
            
            marker = p_markers.get(p, 'o')
            linestyle = p_linestyles.get(p, '-')
            
            if p == '1/sqrt':
                p_label = '1/√n'
            else:
                p_label = f'{p:.3f}'
            
            label = f"{strategy_names.get(strategy, strategy)} (p={p_label})"
            
            # Temps
            ax_time.plot(xs, data[strategy]['time'], color=color, marker=marker, 
                        linestyle=linestyle, label=label)
            
            # Nœuds (échelle log)
            log_nodes = [math.log(x) if x and x > 0 else float('nan') for x in data[strategy]['nodes']]
            ax_nodes.plot(xs, log_nodes, color=color, marker=marker, 
                         linestyle=linestyle, label=label)
            
            # Tailles
            ax_size.plot(xs, data[strategy]['size'], color=color, marker=marker, 
                        linestyle=linestyle, label=label)
            
            # Ratio par rapport à la méthode simple (si disponible)
            if strategy != 'simple' and 'simple' in strategies_to_plot:
                ratio_times = []
                for i, t_simple in enumerate(data['simple']['time']):
                    t_current = data[strategy]['time'][i]
                    if t_simple > 0 and not math.isnan(t_simple) and not math.isnan(t_current):
                        ratio_times.append(t_current / t_simple)
                    else:
                        ratio_times.append(float('nan'))
                ax_ratio.plot(xs, ratio_times, color=color, marker=marker, 
                             linestyle=linestyle, label=label)

    # Configurer les axes
    ax_time.set_xlabel('n (nombre de sommets)')
    ax_time.set_ylabel('Temps moyen (s)')
    ax_time.set_title(f'Temps d\'exécution {title_suffix}')
    ax_time.grid(True)
    ax_time.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax_nodes.set_xlabel('n (nombre de sommets)')
    ax_nodes.set_ylabel('log(Nœuds générés)')
    ax_nodes.set_title(f'Nœuds générés (échelle log) {title_suffix}')
    ax_nodes.grid(True)
    ax_nodes.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax_size.set_xlabel('n (nombre de sommets)')
    ax_size.set_ylabel('Taille moyenne de la couverture')
    ax_size.set_title(f'Qualité des solutions {title_suffix}')
    ax_size.grid(True)
    ax_size.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax_ratio.set_xlabel('n (nombre de sommets)')
    ax_ratio.set_ylabel('Ratio de temps')
    ax_ratio.set_title(f'Ratio de temps vs branchement simple {title_suffix}')
    ax_ratio.grid(True)
    ax_ratio.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    # Afficher un résumé statistique
    print("\n=== RÉSUMÉ STATISTIQUE ===")
    for strategy in strategies_to_plot:
        print(f"\n{strategy_names.get(strategy, strategy)}:")
        total_time = 0
        total_nodes = 0
        count = 0
        
        for key in keys:
            records = all_results[key].get(strategy, [])
            for record in records:
                if record.get('time') is not None:
                    total_time += record['time']
                    count += 1
                if record.get('nodes') is not None:
                    total_nodes += record['nodes']
        
        if count > 0:
            avg_time = total_time / count
            avg_nodes = total_nodes / len(keys) if keys else 0
            print(f"  Temps moyen: {avg_time:.4f}s")
            print(f"  Nœuds moyens: {avg_nodes:.0f}")

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
        print("4. Test des algorithmes GLOUTON et COUPLAGE sur le graphe courant")
        print("5. Test interactif de branchement sur le graphe courant")
        print("6. Comparaison statistique de l'algo GLOUTON vs COUPLAGE sur plusieurs graphes aléatoires")
        print("7. Tests de performance sur plusieurs graphes (benchmark)")
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
            # Tester l'algorithme de branchement : simple vs couplage
            if current_graph is None:
                print("✗ Aucun graphe chargé.")
                continue

            print("\n--- ALGORITHME DE BRANCHEMENT : SIMPLE vs COUPLAGE ---")
            print("1. Branchement simple (branchement_simple)")
            print("2. Branchement avec couplage et bornes (branchement_couplage_avec_borne)")
            print("3. Les deux et comparer")
            sous = input("Choix (1-3): ").strip()

            try:
                if len(current_graph.sommets()) > 22:
                    print("⚠️  Attention: le graphe a plus de 22 sommets, le calcul peut être long.")
                    reponse = input("Voulez-vous continuer? (o/n): ").strip().lower()
                    if reponse != 'o':
                        continue

                if sous == "1" or sous == "3":
                    start_time = time.time()
                    res_simple = current_graph.branchement_simple()
                    t_simple = time.time() - start_time
                    cov_simple, nodes_simple = (res_simple if isinstance(res_simple, tuple) else (res_simple, None))

                if sous == "2" or sous == "3":
                    start_time = time.time()
                    res_coupl = current_graph.branchement_couplage_avec_borne()
                    t_coupl = time.time() - start_time
                    cov_coupl, nodes_coupl = (res_coupl if isinstance(res_coupl, tuple) else (res_coupl, None))

                if sous == "1":
                    print(f"\n[Branchement simple] Couverture: {cov_simple}")
                    print(f" Taille: {len(cov_simple)}, Valide: {current_graph.est_couverture_valide(cov_simple)}")
                    print(f" Noeuds générés: {nodes_simple}, Temps: {t_simple:.3f}s")
                elif sous == "2":
                    print(f"\n[Branchement couplage] Couverture: {cov_coupl}")
                    print(f" Taille: {len(cov_coupl)}, Valide: {current_graph.est_couverture_valide(cov_coupl)}")
                    print(f" Noeuds générés: {nodes_coupl}, Temps: {t_coupl:.3f}s")
                else:
                    print("\n--- Résultats comparés ---")
                    print(f"Simple: taille={len(cov_simple)}, noeuds={nodes_simple}, temps={t_simple:.3f}s")
                    print(f"Couplage: taille={len(cov_coupl)}, noeuds={nodes_coupl}, temps={t_coupl:.3f}s")
                    # Comparaison de qualité
                    if len(cov_coupl) < len(cov_simple):
                        print(" => Couplage a trouvé une meilleure solution.")
                    elif len(cov_coupl) > len(cov_simple):
                        print(" => Simple a trouvé une meilleure solution (surprenant).")
                    else:
                        print(" => Même taille de couverture.")
                    # Comparaison de l'efficacité
                    if nodes_simple is not None and nodes_coupl is not None:
                        print(f" => Ratio noeuds (couplage/simple) = {nodes_coupl}/{nodes_simple} = {nodes_coupl/nodes_simple:.3f}")
                    print(f" => Ratio temps (couplage/simple) = {t_coupl:.3f}/{t_simple:.3f} = {t_coupl/t_simple:.3f}")

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
            print("\n--- TESTS DE PERFORMANCE DU BRANCHEMENT AVEC DIFFÉRENTES STRATÉGIES ---")
            try:
                # 1. Choix des stratégies à tester
                print("Stratégies disponibles:")
                print("1. Branchement simple (baseline)")
                print("2. Avec couplage et bornes (couplage_borne)")
                print("3. Avec algorithme glouton pour solutions réalisables")
                print("4. Avec bornes inférieures seulement")
                print("5. Avec couplage seulement (couplage)")
                print("6. Avec glouton et bornes (glouton_borne)")
                print("7. Comparaison: Avec couplage et bornes vs Avec glouton et bornes")
                print("8. Comparaison: Avec algorithme glouton vs Avec couplage seulement")
                print("9. Toutes les stratégies")
                
                strat_choice = input("Choisissez les stratégies à tester (ex: 1,2,3,4,5,6,7,8 ou 9 pour toutes): ").strip()
                if strat_choice == '1':
                    methods_to_test = ['simple']
                elif strat_choice == '2':
                    methods_to_test = ['couplage_borne']
                elif strat_choice == '3':
                    methods_to_test = ['glouton']
                elif strat_choice == '4':
                    methods_to_test = ['bornes']
                elif strat_choice == '5':
                    methods_to_test = ['couplage']
                elif strat_choice == '6':
                    methods_to_test = ['glouton_borne']
                elif strat_choice == '7':
                    methods_to_test = ['couplage_borne', 'glouton_borne']
                    print("Comparaison sélectionnée: Avec couplage et bornes vs Avec glouton et bornes")
                elif strat_choice == '8':
                    methods_to_test = ['glouton', 'couplage']
                    print("Comparaison sélectionnée: Avec algorithme glouton vs Avec couplage seulement")
                else:
                    methods_to_test = ['simple', 'couplage_borne', 'glouton', 'bornes', 'couplage', 'glouton_borne']

                # 2. Configuration des paramètres
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
                        n_input = input("n_values (séparés par des virgules): ")
                        if n_input:
                            n_values = [int(x.strip()) for x in n_input.split(',') if x.strip()]
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
                        print(f"✗ Entrée invalide, utilisation des valeurs par défaut: {e}")

                # 3. Lancer les tests
                print("\nLancement des tests... cela peut prendre plusieurs minutes.")
                all_results = tester_strategies_branchement(
                    n_values, p_values_labels,
                    num_instances=num_instances,
                    max_time_par_instance=max_time_par_instance,
                    strategies_to_run=methods_to_test
                )

                # 4. Tracer les résultats
                tracer_comparaison_strategies(all_results, strategies_to_plot=methods_to_test,
                                            title_suffix=f"(instances={num_instances})")

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
