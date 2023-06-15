import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import mplcursors


def hierarchy_pos(G, root, levels=None, width=1., depth=1.):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       depth: vertical space allocated for drawing'''
    TOTAL = "total"
    CURRENT = "current"
    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL : 0, CURRENT : 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels =  make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1/levels[currentLevel][TOTAL]
        left = dx/2
        pos[node] = ((left + dx*levels[currentLevel][CURRENT])*width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc-vert_gap)
        return pos
    if levels is None:
        levels = make_levels({})
    else:
        levels = {l:{TOTAL: levels[l], CURRENT:0} for l in levels}
    vert_gap = depth / (max([l for l in levels])+1)
    return make_pos({})

def get_node_color(k, tree):
    if k < len(tree.nodes):
        return '#1b78bf' if tree.nodes[k].is_leaf else '#8a5c0c'
    
    return '#1b78bf'

def format_attr(x):
    if isinstance(x, float):
        return '{:.2f}'.format(x)
    else: return x

def draw_tree(tree, ax = None):
    stack = []
    show = ax is None
    G = nx.DiGraph()
    for i, n in enumerate(tree.nodes):
        if n.arity:
            children = []
            for _ in range(n.arity):
                
                children.append(stack.pop())
            while len(children):
                idx = children.pop()
                G.add_edge(i, idx, weight = tree.nodes[idx].value)
        
        stack.append(i)
    G.add_edge(len(tree.nodes), len(tree.nodes) - 1, weight = tree.nodes[len(tree.nodes) - 1].value)
    
    if show:
        fig, ax = plt.subplots(1,1, figsize=(max(np.sqrt(len(tree.nodes)).item(), 5) , max(np.sqrt(len(tree.nodes)).item(), 6)))

    d = dict(G.degree)
    
    node_color=[  get_node_color(k, tree) for k in d]
    node_label = {idx: str(n) for idx, n in enumerate(tree.nodes)}
    node_label[len(tree.nodes)] = 'out'
    
    pos = hierarchy_pos(G, len(tree.nodes))
    nodes = nx.draw_networkx_nodes(G, 
            pos = pos,
            node_color= node_color,
            node_size = 1000,
            ax = ax
            )
    
    nx.draw_networkx_edges(
        G,
        ax = ax, 
        edge_color = '#8a5c0c',
        pos = pos, 
    )
    
    nx.draw_networkx_labels(G, 
                            pos, 
                            font_color='white',
                            alpha = 0.9,
                            labels = node_label,
                            ax = ax,
                            )

    edge_weight = {}
    
    attr_list = ['__class__','id', 'parent', 'depth', 'length', 'arity']
    
    node_attr = {}
    for u in G.nodes:
        if u < len(tree.nodes):
            node_attr[u] = {attr : format_attr(getattr(tree.nodes[u], attr)) for attr in attr_list}
        else:
            node_attr[u] = {}
    
    
    for e, w in nx.get_edge_attributes(G,'weight').items():
        edge_weight[e] = {'edge_weight': '{:.2f}'.format(w)}
    
    # for e, w in nx.get_edge_attributes(G,'bias').items():
    #     edge_weight[e] = {'edge_weight': edge_weight[e]['edge_weight'] +  ', {:.2f}'.format(w)}
        
    # for e, w in nx.get_edge_attributes(G,'mean').items():
    #     edge_weight[e] = {'edge_weight': edge_weight[e]['edge_weight'] +  ', {:.2f}'.format(w)}

    # for e, w in nx.get_edge_attributes(G,'var').items():
    #     edge_weight[e] = {'edge_weight': edge_weight[e]['edge_weight'] +  ', {:.2f}'.format(w)}    
    
    nx.set_edge_attributes(G, edge_weight)
    nx.set_node_attributes(G, node_attr)
    
    nx.draw_networkx_edge_labels(G,
            pos = pos,
            edge_labels = nx.get_edge_attributes(G,'edge_weight'),
            font_color='#8a5c0c', font_size=6, label_pos=0.4,
            rotate= False,
            ax = ax,
            )
    


    def update_annot(sel):
        nonlocal nodes
        node_index = sel.index
        node_name = list(G.nodes)[node_index]
        node_attr = G.nodes[node_name]
        
        text = '\n'.join(f'{k}: {v}' for k, v in node_attr.items())
        sel.annotation.set_text(text)
        
    cursor = mplcursors.cursor(nodes, hover=True)
    cursor.connect('add', update_annot)
    if show:
        plt.show()
    return ax