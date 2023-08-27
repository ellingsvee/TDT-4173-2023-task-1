import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


# class DecisionTree:
    
#     def __init__():
#         # NOTE: Feel free add any hyperparameters 
#         # (with defaults) as you see fit
#         pass
    
#     def fit(self, X, y):
#         """
#         Generates a decision tree for classification
        
#         Args:
#             X (pd.DataFrame): a matrix with discrete value where
#                 each row is a sample and the columns correspond
#                 to the features.
#             y (pd.Series): a vector of discrete ground-truth labels
#         """
#         # TODO: Implement 
#         raise NotImplementedError()
    
#     def predict(self, X):
#         """
#         Generates predictions
        
#         Note: should be called after .fit()
        
#         Args:
#             X (pd.DataFrame): an mxn discrete matrix where
#                 each row is a sample and the columns correspond
#                 to the features.
            
#         Returns:
#             A length m vector with predictions
#         """
#         # TODO: Implement 
#         raise NotImplementedError()
    
#     def get_rules(self):
#         """
#         Returns the decision tree as a list of rules
        
#         Each rule is given as an implication "x => y" where
#         the antecedent is given by a conjuction of attribute
#         values and the consequent is the predicted label
        
#             attr1=val1 ^ attr2=val2 ^ ... => label
        
#         Example output:
#         >>> model.get_rules()
#         [
#             ([('Outlook', 'Overcast')], 'Yes'),
#             ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
#             ...
#         ]
#         """
#         # TODO: Implement
#         raise NotImplementedError()


# # --- Some utility functions 
    
# def accuracy(y_true, y_pred):
#     """
#     Computes discrete classification accuracy
    
#     Args:
#         y_true (array<m>): a length m vector of ground truth labels
#         y_pred (array<m>): a length m vector of predicted labels
        
#     Returns:
#         The average number of correct predictions
#     """
#     assert y_true.shape == y_pred.shape
#     return (y_true == y_pred).mean()


# def entropy(counts):
#     """
#     Computes the entropy of a partitioning
    
#     Args:
#         counts (array<k>): a lenth k int array >= 0. For instance,
#             an array [3, 4, 1] implies that you have a total of 8
#             datapoints where 3 are in the first group, 4 in the second,
#             and 1 one in the last. This will result in entropy > 0.
#             In contrast, a perfect partitioning like [8, 0, 0] will
#             result in a (minimal) entropy of 0.0
            
#     Returns:
#         A positive float scalar corresponding to the (log2) entropy
#         of the partitioning.
    
#     """
#     assert (counts >= 0).all()
#     probs = counts / counts.sum()
#     probs = probs[probs > 0]  # Avoid log(0)
#     return - np.sum(probs * np.log2(probs))





# class TreeNode:
#     def __init__(self, value):
#         self.value = value
#         self.children = []

#     def add_child(self, child_node):
#         self.children.append(child_node)

class TreeNode:
    def __init__(self, label=None, value=None):
        self.label = label
        self.value = value
        self.children = []
    def add_child(self, label=None, value=None):
        print("Ok so far")
        new_node = TreeNode(label=label, value=value)
        self.children.append(new_node)
    def add_label(self, label):
        self.label = label
    def add_value(self, value):
        self.value = value
    def retrieve_children(self):
        return self.children

class DecisionTree:
    
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        self.tree = self.build_tree(X, y)
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        # TODO: Implement 
        raise NotImplementedError()
    
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        # TODO: Implement
        raise NotImplementedError()

    def build_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return TreeNode(label=y.iloc[0])

        if len(X.columns) == 0:
            most_common_label = self.most_common_value(y)
            return TreeNode(label=most_common_label)
        
        root = TreeNode()
        attr = self.find_best_attribute(X, y)

        print(f"attr = {attr}")

        values = X[attr].unique()
        print(f"values = {values}")
        for value in values:
            if X[X[attr] == value].empty:
                root.add_child(label=self.most_common_value(y), value=value)
            else:

                print(f"y[y != attr] = {y[y != attr]}")
                print(f"X[X[attr] == value] = {X[X[attr] == value]}")
                root.add_child(label=self.build_tree(X[X[attr] == value], y[y != attr]))



    def most_common_value(self, y):
        unique_labels, counts = np.unique(y, return_counts=True)
        most_common_index = np.argmax(counts)
        return unique_labels[most_common_index]    

    def find_best_attribute(self, X, y):
        tot_entropy = entropy(y.value_counts())
        gains = {}
        for attribute in X.columns:
            gains[attribute] = self.gain(X, y, attribute, tot_entropy)

        print(f"gains = {gains}")
        return max(gains, key=lambda k: gains[k])

    def gain(self, X, y, attribute, tot_entropy):
        values = X[attribute].unique()
        return tot_entropy - np.sum([(len(y[X[attribute] == value])/len(y))*entropy(y[X[attribute] == value].value_counts()) for value in values])

        

# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))



