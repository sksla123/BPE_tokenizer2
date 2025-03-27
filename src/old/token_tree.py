## 잘못 만든 코드 (삭제 예정정)

from collections import defaultdict
from src.token.token import Token
from src.vocab.merge import MergeRule

class TokenNode:
    def __init__(self, id:int, start_idx:int, end_idx:int, token: Token, is_root: bool = False, is_end: bool = False):
        self.id = id
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.token = token
        self.is_root = is_root
        self.is_end = is_end
        self.parent = []
        self.children = []

class TokenTree:
    def __init__(self, word: str):
        self.word = word
        self.root_node = TokenNode(-1, -1, -1, Token('[ ROOT ]', False), is_root=True)
        self.end_node = TokenNode(-2, -1, -1, Token('[ END ]', True), is_end=True)

        self.subwords = set()
        
        self.subword_to_node_id = {}
        self.node_id_to_subword = {-1: '[ ROOT ]', -2: '[ END ]'}
        self.node_id_to_node = {-1: self.root_node, -2: self.end_node}

        self.edge_map = {}
        self.edge_to_node_id_pairs = {}

        nodes_by_start_idx = defaultdict(list)

        id_counter = 0
        for i in range(len(word)):
            for j in range(i + 1, len(word) + 1):
                subword = word[i:j]
                self.subwords.add(subword)

                token = Token(subword, i != 0)
                subword_node = TokenNode(id_counter, i, j, token)

                self.subword_to_node_id.setdefault(subword, []).append(id_counter)
                self.node_id_to_subword[id_counter] = subword
                self.node_id_to_node[id_counter] = subword_node
                nodes_by_start_idx[i].append(subword_node)

                id_counter += 1

        for node in self.node_id_to_node.values():
            if node.start_idx == 0:
                self.connect_nodes(self.root_node, node, weight=1 if node.end_idx == 1 else 0)

            for next_node in nodes_by_start_idx[node.end_idx]:
                weight = 1 if self.is_one_char_distance(node, next_node) else 0
                self.connect_nodes(node, next_node, weight)

            if node.end_idx == len(self.word):
                weight = 1 if (node.end_idx - node.start_idx) == 1 else 0
                self.connect_nodes(node, self.end_node, weight)

    def is_one_char_distance(self, node_a, node_b):
        return node_b.end_idx - node_a.end_idx == 1

    def connect_nodes(self, parent_node, child_node, weight):
        parent_node.children.append(child_node.id)
        child_node.parent.append(parent_node.id)
        self.edge_map[(parent_node.id, child_node.id)] = weight

    def build_max_weight_subtree(self, node_id=None, path=None):
        if node_id is None:
            node_id = self.root_node.id
            path = [node_id]

        if node_id == self.end_node.id:
            return path

        children = [(child_id, self.edge_map[(node_id, child_id)]) for child_id in self.node_id_to_node[node_id].children]
        if not children:
            return path

        max_weight_child = max(children, key=lambda x: (x[1], -x[0]))[0]
        path.append(max_weight_child)
        return self.build_max_weight_subtree(max_weight_child, path)

if __name__ == '__main__':
    test_word = "test"
    tree = TokenTree(test_word)

    print("Nodes and their connections:")
    for node_id, node in tree.node_id_to_node.items():
        print(f"Node {node_id}: '{node.token}', children: {node.children}, parents: {node.parent}")
    print("Edge Map:")
    for edge, weight in tree.edge_map.items():
        print(f"{edge}({tree.node_id_to_subword[edge[0]]} -> {tree.node_id_to_subword[edge[1]]}): weight {weight}")

    max_weight_subtree = tree.build_max_weight_subtree()
    print("Max weight subtree path:", max_weight_subtree, f"({[tree.node_id_to_subword[node_id] for node_id in max_weight_subtree]})")

    t_node_ids = tree.subword_to_node_id.get('t', [])
    es_node_ids = tree.subword_to_node_id.get('es', [])
    for t_node_id in t_node_ids:
        for es_node_id in es_node_ids:
            if (t_node_id, es_node_id) in tree.edge_map:
                tree.edge_map[(t_node_id, es_node_id)] = 100

    print("After update edge map---------------------")

    print("Nodes and their connections:")
    for node_id, node in tree.node_id_to_node.items():
        print(f"Node {node_id}: '{node.token}', children: {node.children}, parents: {node.parent}")
    print("Edge Map:")
    for edge, weight in tree.edge_map.items():
        print(f"{edge}({tree.node_id_to_subword[edge[0]]} -> {tree.node_id_to_subword[edge[1]]}): weight {weight}")

    max_weight_subtree = tree.build_max_weight_subtree()
    print("Max weight subtree path:", max_weight_subtree, f"({[tree.node_id_to_subword[node_id] for node_id in max_weight_subtree]})")
