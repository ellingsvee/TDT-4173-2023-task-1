import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)
import random
from icecream import ic # Remove 

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


class DecisionTree:
    
    def __init__(self, target_attribute = 'Play Tennis'):
        self.tree = None
        self.target_attribute = target_attribute

    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        self.booleans = y.unique().tolist()
        self.tree = self.build_tree(X, y, X.columns.tolist())

    def build_tree(self, X, y, attributes):
        root = []

        unique_y_vals = y.unique()
        if len(unique_y_vals) == 1:
            if unique_y_vals[0] in self.booleans:
                root.append(([], unique_y_vals[0]))
            return root

        if len(attributes) == 0:
            most_common = self.most_common_value(y)
            root.append(([], most_common))
            return root

        attr = self.find_best_attribute(X, y, attributes)
        for val in X[attr].unique().tolist():
            condition_mask = X[attr] == val
            if X[condition_mask].empty:
                leaf_label = self.most_common_value(y)
                root.append(([(attr, val)], leaf_label))
            else:
                new_attributes = np.delete(attributes, np.argwhere(attributes == attr))
                sub_tree = self.build_tree(X[condition_mask], y[condition_mask], new_attributes)
                for sub_case in sub_tree:
                    root.append(([(attr, val)] + sub_case[0], sub_case[1]))
        return root

    def most_common_value(self, y):
        return y.value_counts().idxmax() 

    def find_best_attribute(self, X, y, attributes):
        tot_entropy = entropy(y.value_counts())
        gains = {}
        for attribute in attributes:
            gains[attribute] = self.gain(X, y, attribute, tot_entropy)
        return max(gains, key=lambda k: gains[k])

    def gain(self, X, y, attribute, tot_entropy):
        values = X[attribute].unique()
        return tot_entropy - np.sum([(len(y[X[attribute] == value])/len(y))*entropy(y[X[attribute] == value].value_counts()) for value in values])


    def predict(self, X, tree=None):
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
        if tree == None:
            tree = self.tree
        return np.array([self.predict_sample(row.to_dict(), tree) for _, row in X.iterrows()])

    def predict_sample(self, sample, tree):
        # for conditions, label in tree:
        #     conditions_satisfied = all(sample[attr] == val for attr, val in conditions)
        #     if conditions_satisfied == True:
        #         return label
        # # If no "direct" path was found, remove one of the rules and base the prediction on that
        
        rules = []
        for conditions, label in tree:
            conditions_satisfied = all(sample[attr] == val for attr, val in conditions)
            if conditions_satisfied == True:
                return label
            
            rules.append([[sample[attr] == val for attr, val in conditions], label, conditions]) # If no perfect fit was found
        
        return self.find_most_similar(rules) # If no perfect fit was founc

    def find_most_similar(self, rules):
        # If no "direct" path was found, find the other rule that if the most "similar" to the sample we want to predict.
        max_true_count = 0
        best_index = -1
        for i, rule in enumerate(rules):
            lst = rule[0]
            true_count = lst.count(True)

            if true_count > max_true_count:
                max_true_count = true_count
                best_index = i
        return rules[best_index][1]

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
        return self.tree
    
    def prune_tree(self, X, y):
        """
        Prunes the decision tree using the MDL criterion
        
        Args:
            X (pd.DataFrame): Validation dataset features
            y (pd.Series): Validation dataset labels
        """
        best_accuracy = accuracy(y, self.predict(X, self.tree))
        pruned_tree = self.tree

        for i, (conditions, label) in enumerate(self.tree):
            if conditions:
                # Temporarily remove the subtree rooted at this node
                pruned_subtree = pruned_tree[:i] + pruned_tree[i+1:]
                accuracy_without_subtree = accuracy(y, self.predict(X, pruned_subtree))

                if accuracy_without_subtree > best_accuracy:
                    best_accuracy = accuracy_without_subtree
                    pruned_tree = pruned_subtree
        
        print(f"Performed tree pruning: {len(self.tree) - len(pruned_tree)} rule(s) removed.")
        self.tree = pruned_tree

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



# Testing
# data_1 = pd.read_csv('decision_tree/data_1.csv')
# data_1

# Separate independent (X) and dependent (y) variables
# X = data_1.drop(columns=['Play Tennis'])
# y = data_1['Play Tennis']

# # Create and fit a Decrision Tree classifier
# model_1 = DecisionTree('Play Tennis')  # <-- Should work with default constructor
# model_1.fit(X, y)

# # Verify that it perfectly fits the training set
# print(f'Accuracy: {accuracy(y_true=y, y_pred=model_1.predict(X)) * 100 :.1f}%')

# for rules, label in model_1.get_rules():
#     conjunction = ' ∩ '.join(f'{attr}={value}' for attr, value in rules)
#     print(f'{"✅" if label == "Yes" else "❌"} {conjunction} => {label}')

# data_2 = pd.read_csv('decision_tree/data_2.csv')

# data_2_train = data_2.query('Split == "train"')
# data_2_valid = data_2.query('Split == "valid"')
# data_2_test = data_2.query('Split == "test"')
# X_train, y_train = data_2_train.drop(columns=['Outcome', 'Split']), data_2_train.Outcome
# X_valid, y_valid = data_2_valid.drop(columns=['Outcome', 'Split']), data_2_valid.Outcome
# X_test, y_test = data_2_test.drop(columns=['Outcome', 'Split']), data_2_test.Outcome
# data_2.Split.value_counts()

# X_train = X_train.drop('Birth Month', axis=1)
# X_test = X_test.drop('Birth Month', axis=1)
# X_valid = X_valid.drop('Birth Month', axis=1)

# # Fit model (TO TRAIN SET ONLY)
# model_2 = DecisionTree(target_attribute='Outcome')  # <-- Feel free to add hyperparameters 
# model_2.fit(X_train, y_train)
# print(f'Train: {accuracy(y_train, model_2.predict(X_train)) * 100 :.1f}%')
# print(f'Valid: {accuracy(y_valid, model_2.predict(X_valid)) * 100 :.1f}%')
# print(f'Test: {accuracy(y_test, model_2.predict(X_test)) * 100 :.1f}%')

# model_2.prune_tree(X_valid, y_valid)
# print(f'Train: {accuracy(y_train, model_2.predict(X_train)) * 100 :.1f}%')
# print(f'Valid: {accuracy(y_valid, model_2.predict(X_valid)) * 100 :.1f}%')
# print(f'Test: {accuracy(y_test, model_2.predict(X_test)) * 100 :.1f}%')