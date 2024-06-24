class MVDNode:
    def __init__(self, attribute=None, parent=None):
        self.attribute = attribute
        self.parent = parent
        self.children = {}
        self.dependencies = []

    def add_child(self, attribute):
        if attribute not in self.children:
            self.children[attribute] = MVDNode(attribute, self)
        return self.children[attribute]

    def add_dependency(self, X, Y):
        dep = (tuple(sorted(X)), tuple(sorted(Y)))
        if dep not in self.dependencies:
            self.dependencies.append(dep)

class MVDTree:
    def __init__(self):
        self.root = MVDNode("Root")

    def add_path(self, attributes, dep_bases=set()):
        current_node = self.root
        for attr in attributes:
            current_node = current_node.add_child(attr)
        for dep in dep_bases:
            current_node.add_dependency(attributes, [dep])

