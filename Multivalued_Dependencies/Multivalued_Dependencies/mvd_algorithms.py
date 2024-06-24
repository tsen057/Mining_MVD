from itertools import combinations, chain
import itertools
import logging
import os
import time
import pandas as pd

from Multivalued_Dependencies.mvd_node import MVDNode, MVDTree

def q(df, X, Y):
    if not X:
        return True
    # For checking MVD X ->> Y, we need to check if all Y combinations appear with X combinations independently
    x_groups = df.groupby(X)
    all_y_combinations = set(tuple(row) for row in df[Y].drop_duplicates().to_numpy())
    for _, group in x_groups:
        y_combinations_in_group = set(tuple(row) for row in group[Y].drop_duplicates().to_numpy())
        if y_combinations_in_group != all_y_combinations:
            return False
    return True

def enumerate_sentences(attributes):
    # Generate hypotheses from the most general to more specific
    for r in range(1, len(attributes) + 1):
        for X in combinations(attributes, r):
            Y_candidates = [y for y in attributes if y not in X]
            yield from ((list(X), list(Y)) for Y in chain.from_iterable(combinations(Y_candidates, i) for i in range(len(Y_candidates)+1)))

def top_down_algorithm(df, attributes):
    mvd_tree = MVDTree()
    for X, Y in enumerate_sentences(attributes):
         # Ensure Y is not empty
        if Y: 
            if q(df, X, Y):  
                current_node = mvd_tree.root
                for attr in X:  
                    current_node = current_node.add_child(attr)
                current_node.add_dependency(X, Y)  
    return mvd_tree


def q_bottom_up(df, X, Y):
    if not X:
        return False
    grouped = df.groupby(list(X))
    all_y_combinations = set(tuple(row) for row in df[list(Y)].drop_duplicates().to_numpy())
    for _, group in grouped:
        y_combinations_in_group = set(tuple(row) for row in group[list(Y)].drop_duplicates().to_numpy())
        if y_combinations_in_group != all_y_combinations:
            return False
    return True

def bottom_up_algorithm(df, attributes):
    mvd_tree = MVDTree()
    nodes = {tuple([attr]): mvd_tree.root.add_child(attr) for attr in attributes}

    # Consider combinations of attributes for X
    for r in range(1, len(attributes) + 1):
        for X in itertools.combinations(attributes, r):
            Y_candidates = [y for y in attributes if y not in X]
            # Test dependencies from X to each subset of Y
            for s in range(1, len(Y_candidates) + 1):
                for Y in itertools.combinations(Y_candidates, s):
                    if q_bottom_up(df, list(X), list(Y)):
                        # Find or create the path for X in the tree
                        current_node = mvd_tree.root
                        for attr in X:
                            current_node = current_node.add_child(attr)
                        current_node.add_dependency(X, Y)
    return mvd_tree
 
def print_mvd_tree(node, indent=0):
    result = ' ' * indent + f"Node: {node.attribute}\n"
    for dep in node.dependencies:
        X, Y = dep
        result += ' ' * indent + f"  Dependency: {X} ->> {Y}\n"
    for child in node.children.values():
        result += print_mvd_tree(child, indent + 4)
    return result
        
    
def analyze_mvd(df, attributes, top_down=True):
    # Define the maximum execution time in seconds (1 hour 30 minutes)
    max_time = 5400  
    start_time = time.time()  
    # Condition check for large data
    row_condition = len(df) > 100
    col_condition = len(attributes) > 4
    if row_condition and col_condition:
        print("Using chunk-based processing for both rows and columns due to large data size.")
        try:
            if(top_down):
                tree = process_data_in_chunks(df, attributes, chunk_size=300, top_down=True, start_time=start_time, max_time=max_time)
            else:
                tree = process_data_in_chunks(df, attributes, chunk_size=300, top_down=False, start_time=start_time, max_time=max_time)
        except TimeoutError:
            print("Processing terminated due to timeout.")
            return None, None
    else:
        print("Using row-wise chunk-based processing.")
        try:
            if(top_down):
                tree = top_down_algorithm(df, attributes) 
            else:    
                tree = bottom_up_algorithm(df, attributes)
        except TimeoutError:
            print("Processing terminated due to timeout.")
            return None, None
    return tree
    


def process_data_in_chunks(df, attributes, chunk_size, top_down=True, start_time=None, max_time=5400):
    main_tree = MVDTree()
    # Calculate number of chunks
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    # Divide attributes into manageable chunks
    attribute_chunks = [attributes[i:i + 7] for i in range(0, len(attributes), 7)] 
    # Iterate over each data chunk
    for i in range(num_chunks):
        current_time = time.time()
        if (current_time - start_time) > max_time:
            logging.warning("Processing terminated due to exceeding time limit.")
            break  
        # Process each chunk of the dataframe
        df_chunk = df.iloc[i * chunk_size:(i + 1) * chunk_size]
        for attr_chunk in attribute_chunks:
            try:
                if top_down:
                    chunk_tree = top_down_algorithm(df_chunk, attr_chunk)
                else:
                    chunk_tree = bottom_up_algorithm(df_chunk, attr_chunk)
                merge_trees(main_tree, chunk_tree)
            except Exception as e:
                logging.error(f"Error processing chunk {i} with attributes {attr_chunk}: {str(e)}")
    return main_tree

 # Merge the chunk_tree into the main_tree starting from the root
def merge_trees(main_tree, chunk_tree):
    result = merge_nodes(main_tree.root, chunk_tree.root)
    return result

def merge_nodes(main_node, chunk_node):
    merge_details = []
    # Merge the dependencies of the nodes
    existing_deps = set((tuple(dep[0]), tuple(dep[1])) for dep in main_node.dependencies)
    for dep in chunk_node.dependencies:
        if (tuple(dep[0]), tuple(dep[1])) not in existing_deps:
            main_node.dependencies.append(dep)
            merge_details.append(f"Dependency added: {dep}")
    # Iterate over the children of the chunk_node to merge them
    for child_attr, chunk_child in chunk_node.children.items():
        if child_attr in main_node.children:
            details = merge_nodes(main_node.children[child_attr], chunk_child)
            merge_details.append(f"Merged existing child {child_attr}: {details}")
        else:
            main_node.children[child_attr] = chunk_child
            merge_details.append(f"New child added: {child_attr}")
    return "\n".join(merge_details)

def setup_logging():
    log_filename = 'mvd_log.txt'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s', filemode='a')
    if os.path.exists(log_filename):
        print("Log file already exists. New logs will be appended.")
    else:
        print("Log file does not exist. It will be created.")
    logging.info("Starting a new logging session.")
