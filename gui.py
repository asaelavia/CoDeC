import re
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import threading
import queue
import pandas as pd
import numpy as np
import time
import random
from sklearn.model_selection import train_test_split
import dice_ml
from class_models import Mlp, pretrain
from dice_ml.utils.helpers import DataTransfomer
import torch
from torch.utils.data import DataLoader, TensorDataset
import itertools
from contextlib import contextmanager
from z3 import *

# ============= FONT SCALING CONFIGURATION =============
FONT_SCALE = 1.4  # Adjust this to scale all fonts (e.g., 1.2 for 20% larger, 0.8 for 20% smaller)
def scaled_size(size):
    """Return a scaled dimension (for widths, heights, etc.)"""
    return int(size * FONT_SCALE)

def scaled_font(size, weight="normal"):
    """Return a scaled font tuple"""
    scaled_size = int(size * FONT_SCALE)
    if weight == "normal":
        return "SF Pro Display", scaled_size
    else:
        return "SF Pro Display", scaled_size, weight
# ======================================================
VERBOSE = True
LR_NY = 0.001 # GOOD NY
LR_ADULT = 0.03 # GOOD Adult
# MIN_ITER_ADULT = 5
MIN_ITER_ADULT = 250
MIN_ITER_NY = 500

MAX_ITER_ADULT = 500
# MAX_ITER_ADULT = 15
MAX_ITER_NY = 1500
_solver_cache = {}

def reset_solver_cache():
    """Reset the solver cache to empty."""
    _solver_cache.clear()
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)
diversity_seed = seed
set_param("random_seed", seed)
def extract_names_and_conditions(line):
    constraints = []
    pattern = r'([\w.-]+)\s*([<>=!]=?)\s*([\w."]+)'
    matches = re.findall(pattern, line.rstrip())
    for match in matches:
        lhs, op, rhs = match
        constraints.append((lhs, op, rhs))
    return constraints


def load_constraints(path):
    f = open(path, 'r')
    constraints_txt = []
    for line in f:
        constraint = extract_names_and_conditions(line.rstrip())
        if 't1.' in line:
            if any(op in line for op in ['>=', '<=', '< ', '> ']):
                rev_constraint = []
                for pred in constraint:
                    if 't0.' in pred[0]:
                        rev_constraint.append((pred[0].replace('t0.', 't1.'), pred[1], (pred[2].replace('t1.', 't0.'))))
                    else:
                        rev_constraint.append((pred[0].replace('t1.', 't0.'), pred[1], (pred[2].replace('t1.', 't0.'))))
                rev_constraint_fixed = []
                for pred in rev_constraint:
                    rev_constraint_fixed.append(
                        f'{pred[0].replace("t0.", "cf_row.").replace("t1.", "df.")} {pred[1]} {pred[2].replace("t0.", "cf_row.").replace("t1.", "df.")}')
                constraints_txt.append(rev_constraint_fixed)
        constraint_fixed = []
        for pred in constraint:
            constraint_fixed.append(
                f'{pred[0].replace("t0.", "cf_row.").replace("t1.", "df.")} {pred[1]} {pred[2].replace("t0.", "cf_row.").replace("t1.", "df.")}')
        constraints_txt.append(constraint_fixed)

    def cons_func(df, cf_row, cons_id, exclude=None):
        if exclude is None:
            exclude = []
        df_mask = '('
        for pred in constraints_txt[cons_id]:
            if (pred.split('.')[1].split(' ')[0] in exclude) and ('cf_row' in pred):
                continue
            df_mask += f'({pred}) & '
        if len(df_mask[:-1]) == 0:
            if len(exclude) == 0:
                return pd.Series([False] * len(df), index=df.index)
            else:
                return pd.Series([True] * len(df), index=df.index)
        res = eval(df_mask[:-3] + ')')
        if type(res) != bool:
            return res
        if not res:
            return pd.Series([False] * len(df), index=df.index)
        else:
            return pd.Series([True] * len(df), index=df.index)

    dic_cols = {}
    unary_cons_lst = []
    unary_cons_lst_single = []
    bin_cons = []
    cons_feat = [[] for _ in range(len(constraints_txt))]
    for index, cons in enumerate(constraints_txt):
        unary = True
        for pred in cons:
            col = pred.split('.')[1].split(' ')[0]
            if index not in dic_cols.get(col, []):
                dic_cols[col] = dic_cols.get(col, []) + [index]
            cons_feat[index].append(col)
            if unary:
                if 'df.' in pred:
                    unary = False
        if unary:
            if len(cons) == 1:
                unary_cons_lst_single.append(index)
            else:
                unary_cons_lst.append(index)
        else:
            bin_cons.append(index)
    f.close()
    return constraints_txt, dic_cols, cons_func, cons_feat, unary_cons_lst, unary_cons_lst_single, bin_cons


# Add these functions after the load_constraints() function:

def classify_columns(dataset):
    """
    Classify all columns in dataset
    Returns: dict with 'integer', 'float', 'categorical' keys containing lists of column names
    """

    def get_column_type(dataset, col):
        try:
            values = dataset[col].dropna()
            if len(values) == 0:
                return 'categorical'

            numeric_values = pd.to_numeric(values, errors='coerce')

            if numeric_values.isna().any():
                return 'categorical'

            if np.allclose(numeric_values, numeric_values.astype(int)):
                return 'integer'
            else:
                return 'float'
        except:
            return 'categorical'

    col_types = {}
    for col in dataset.columns:
        col_type = get_column_type(dataset, col)
        col_types[col] = col_type

    return col_types


def bounds_builder(row_val, comb, dataset, constraints, cons_feat, dic_cols, cons_function):
    """Build constraint bounds for Z3 solver"""
    val_cons_dfs = {}
    comb_set = set(comb)

    for cons in range(len(constraints)):
        val_cons_set = dataset[cons_function(dataset, row_val, cons, comb)]
        if len(val_cons_set) != 0 and len(comb_set | set(cons_feat[cons])) == 0:
            return None
        val_cons_dfs[cons] = val_cons_set

    column_types = classify_columns(dataset)

    full_bounds = []
    attribute_to_dataframes = {}

    ops = {'==': 'point', '!=': 'not_point', '<': 'right_exclusive',
           '<=': 'right_inclusive', '>': 'left_exclusive', '>=': 'left_inclusive'}
    ops_rev = {'==': 'point', '!=': 'not_point', '<': 'left_exclusive',
               '<=': 'left_inclusive', '>': 'right_exclusive', '>=': 'right_inclusive'}

    for cons in range(len(constraints)):
        if cons not in set([val for col in comb for val in dic_cols.get(col, [])]):
            full_bounds.append(pd.DataFrame())
            continue

        if len(val_cons_dfs[cons]) == 0:
            full_bounds.append(pd.DataFrame())
            continue

        comb_cons = []
        count_preds = 0

        for pred in constraints[cons]:
            if 'cf_row' not in pred:
                continue

            parts = pred.split(' ')
            first_clause, op, second_clause = parts[0], parts[1], parts[2]
            cf_pred = first_clause if 'cf_row' in first_clause else second_clause
            cf_col = cf_pred.split('.')[1]

            if cf_col not in comb_set:
                continue

            df_pred = first_clause if cf_pred == second_clause else second_clause
            unary = 1
            col_type = column_types.get(cf_col, 'unknown')

            if 'df.' in df_pred:
                values = val_cons_dfs[cons][cf_col]
                cf_is_second = (cf_pred == second_clause)
            else:
                if '"' in df_pred:
                    if col_type == 'float':
                        const_value = float(df_pred[1:-1])
                    elif col_type == 'integer':
                        const_value = int(df_pred[1:-1])
                    else:
                        const_value = df_pred[1:-1]
                else:
                    if col_type == 'float':
                        const_value = float(df_pred)
                    elif col_type == 'integer':
                        const_value = int(df_pred)
                    else:
                        const_value = df_pred
                val = const_value
                cf_is_second = False
                unary = len(val_cons_dfs[cons])

            if unary == 1:
                comb_cons.append([
                    (cf_col, val, ops_rev[op] if cf_is_second else ops[op])
                    for val in values
                ])
            else:
                comb_cons.append([
                                     (cf_col, val, ops_rev[op] if cf_is_second else ops[op])
                                 ] * unary)

            count_preds += 1
            if count_preds >= len(comb):
                break

        constraint_bounds = []
        if comb_cons:
            for i in range(len(comb_cons[0])):
                full_bound = []
                for j in range(len(comb_cons)):
                    col, val, op_type = comb_cons[j][i]
                    full_bound += [col, val, op_type]
                constraint_bounds.append(full_bound)

        if constraint_bounds:
            first_bound = constraint_bounds[0]
            num_cols = len(first_bound)

            column_headers = []
            constraint_attributes = set()
            for i in range(0, num_cols, 3):
                col_name = first_bound[i]
                constraint_attributes.add(col_name)
                column_headers.extend([col_name, f"{col_name}_value", f"{col_name}_op"])

            if len(set(len(bound) for bound in constraint_bounds)) == 1:
                constraint_df = pd.DataFrame(constraint_bounds, columns=column_headers).drop_duplicates()
            else:
                max_cols = len(first_bound)
                padded_array = np.full((len(constraint_bounds), max_cols), None, dtype=object)
                for i, bound in enumerate(constraint_bounds):
                    padded_array[i, :len(bound)] = bound
                constraint_df = pd.DataFrame(padded_array, columns=column_headers).drop_duplicates()

            full_bounds.append(constraint_df)

            current_df_index = len(full_bounds) - 1
            for attr in constraint_attributes:
                if attr not in attribute_to_dataframes:
                    attribute_to_dataframes[attr] = []
                attribute_to_dataframes[attr].append(current_df_index)
        else:
            full_bounds.append(pd.DataFrame())

    return column_types, full_bounds, attribute_to_dataframes


def create_normalization_params(x_train, non_cat_cols, cat_cols):
    """Create normalization parameters for continuous features"""
    norm_params = {}
    normalized_medians = {}

    for col in non_cat_cols:
        col_min = x_train[col].min()
        col_max = x_train[col].max()
        col_range = col_max - col_min

        if col_range <= 1e-8:
            col_range = 1

        norm_params[col] = {
            'min': col_min,
            'range': col_range
        }

        normalized_col = (x_train[col] - col_min) / col_range
        median = normalized_col.median()
        mad = np.median(np.abs(normalized_col - median))

        normalized_medians[col] = mad

    return norm_params, normalized_medians


# Add this class after the helper functions:

class ProjectionConfig:
    """Configuration object to hold all projection parameters."""

    def __init__(self, df, constraints, dic_cols, cons_function, cons_feat,
                 categorical_column_names, fixed_feat, cont_feat,
                 norm_params=None, normalized_medians=None,
                 unary_cons_lst=None, unary_cons_lst_single=None, bin_cons=None,
                 solver_timeout=10000, delta=0.0, found_points=None, gamma=2):  # ADD gamma parameter
        self.df = df
        self.constraints = constraints
        self.dic_cols = dic_cols
        self.cons_function = cons_function
        self.cons_feat = cons_feat
        self.categorical_column_names = categorical_column_names
        self.fixed_feat = fixed_feat
        self.cont_feat = cont_feat
        self.norm_params = norm_params or {}
        self.normalized_medians = normalized_medians or {}
        self.unary_cons_lst = unary_cons_lst or []
        self.unary_cons_lst_single = unary_cons_lst_single or []
        self.bin_cons = bin_cons or []
        self.solver_timeout = solver_timeout
        self.delta = delta
        self.mode = 'solver'
        self.found_points = found_points
        self.gamma = gamma  # ADD gamma attribute

def dpp_style(cfs, dice_inst):
    num_cfs = len(cfs)
    """Computes the DPP of a matrix."""
    det_entries = torch.ones((num_cfs, num_cfs))
    for i in range(num_cfs):
        for j in range(num_cfs):
            det_entries[(i, j)] = 1.0 / (
                    1.0 + compute_dist(torch.tensor(cfs.iloc[i].values.astype('float32')),
                                       torch.tensor(cfs.iloc[j].values.astype('float32')),
                                       dice_inst))
            if i == j:
                det_entries[(i, j)] += 0.0001
    diversity_loss = torch.det(det_entries)
    return diversity_loss


def compute_dist(x1, x2, dice_inst):
    return torch.sum(torch.mul((torch.abs(x1 - x2)), dice_inst.feature_weights_list), dim=0)


def n_cfs_score(comb_cfs, origin_instance, transformer, exp_random):
    dpp_score = dpp_style(transformer.transform(comb_cfs), exp_random).item()
    prox_distances = []
    for index, row in comb_cfs.iterrows():
        proximity_dist = compute_dist(
            torch.tensor(transformer.transform(origin_instance.to_frame().T).values.astype('float32')).flatten(),
            torch.tensor(transformer.transform(row.to_frame().T).values.astype('float32')).flatten(), exp_random)
        proximity_dist /= len(exp_random.minx[0])
        prox_distances.append(proximity_dist)
    proximity_distance = np.mean(prox_distances)
    return dpp_score, proximity_distance


def n_best_cfs_heuristic(cfs_pool, origin_instance, k, transformer, exp_random):
    cfs_pool = cfs_pool.copy()
    curr_best = cfs_pool[1:0]
    comb = ()
    dist_dic = {}
    best_indices = []
    for _ in range(min(len(cfs_pool),k)):
        dic = {}
        for i, cf in cfs_pool.iterrows():
            dpp_score, proximity_distance = n_cfs_score(
                pd.concat([curr_best, pd.DataFrame([cf])], axis=0, ignore_index=True), origin_instance, transformer,
                exp_random)
            if len(curr_best) == 0:
                dist_dic[i] = proximity_distance
            dic[i] = (dpp_score, proximity_distance, dpp_score - 0.1 * proximity_distance)
        best_index = max(dic, key=lambda x: dic[x][-1])
        best_cf = cfs_pool.loc[best_index]
        cfs_pool = cfs_pool.drop(best_index, axis=0)
        curr_best = pd.concat([curr_best, pd.DataFrame([best_cf])], axis=0)
        best_indices.append(best_index)
    return curr_best, dic[best_index][0], [dist_dic[i] for i in best_indices]

timeout_occurred = False
def timeout_handler():
    global timeout_occurred
    timeout_occurred = True


# def project_solver(row, projection_config):
#     """Project a single row using Z3 solver"""
#
#     dataset = projection_config.df
#     orig_dataset = dataset.copy()
#     orig_row = row.copy()
#
#     # Setup viable columns
#     fixed_feat1 = projection_config.fixed_feat
#     viable_cols = [col for col in dataset.columns
#                    if col not in fixed_feat1 and col in projection_config.dic_cols]
#
#     # Prepare combined dataset for categorical encoding
#     combined = pd.concat([dataset, row.to_frame().T], ignore_index=True)
#     combined[projection_config.categorical_column_names] = \
#         combined[projection_config.categorical_column_names].astype('category')
#
#     cat_cols = combined.select_dtypes(include=['category']).columns
#     category_mappings = {
#         col: {v: k for k, v in enumerate(combined[col].cat.categories)}
#         for col in cat_cols
#     }
#
#     # Convert to integer codes
#     combined[cat_cols] = combined[cat_cols].apply(lambda x: x.cat.codes)
#     dataset = combined.iloc[:-1]
#     row = combined.iloc[-1]
#     dataset[projection_config.categorical_column_names] = \
#         dataset[projection_config.categorical_column_names].astype('category')
#
#     # Check which constraints are violated
#     must_cons = set()
#     for cons in range(len(projection_config.constraints)):
#         non_follow_cons = projection_config.cons_function(orig_dataset, orig_row, cons)
#         if non_follow_cons.sum() > 0:
#             must_cons.add(cons)
#
#     if len(must_cons) == 0:
#         print('No constraints to project')
#         return orig_row.copy()
#
#     random.shuffle(viable_cols)
#
#     # Try projection with all viable columns
#     for i in range(len(viable_cols), len(viable_cols) + 1):
#         combs = list(itertools.combinations(viable_cols, i))
#         for comb in combs:
#             result = _try_projection_with_z3(
#                 comb, row, orig_row, dataset, orig_dataset,
#                 projection_config, must_cons, category_mappings
#             )
#             if result is not None:
#                 return result
#
#     return None

def project_solver(row, projection_config):
    """Project a single row using Z3 solver"""

    dataset = projection_config.df
    orig_dataset = dataset.copy()
    orig_row = row.copy()

    # Setup viable columns
    fixed_feat1 = projection_config.fixed_feat
    viable_cols = [col for col in dataset.columns
                   if col not in fixed_feat1 and col in projection_config.dic_cols]

    # Prepare combined dataset for categorical encoding
    combined = pd.concat([dataset, row.to_frame().T], ignore_index=True)
    combined[projection_config.categorical_column_names] = \
        combined[projection_config.categorical_column_names].astype('category')

    cat_cols = combined.select_dtypes(include=['category']).columns
    category_mappings = {
        col: {v: k for k, v in enumerate(combined[col].cat.categories)}
        for col in cat_cols
    }

    # Convert to integer codes
    combined[cat_cols] = combined[cat_cols].apply(lambda x: x.cat.codes)
    dataset = combined.iloc[:-1]
    row = combined.iloc[-1]
    dataset[projection_config.categorical_column_names] = \
        dataset[projection_config.categorical_column_names].astype('category')

    # Check which constraints are violated
    must_cons = set()
    for cons in range(len(projection_config.constraints)):
        non_follow_cons = projection_config.cons_function(orig_dataset, orig_row, cons)
        if non_follow_cons.sum() > 0:
            must_cons.add(cons)

    if len(must_cons) == 0:
        print('No constraints to project')
        return orig_row.copy()

    random.shuffle(viable_cols)

    # Try projection with all viable columns
    for i in range(len(viable_cols), len(viable_cols) + 1):
        combs = list(itertools.combinations(viable_cols, i))
        for comb in combs:
            result = _try_projection_with_z3(
                comb, row, orig_row, dataset, orig_dataset,
                projection_config, must_cons, category_mappings
            )
            if result is not None:
                return result  # Can be either single value or tuple (orig, coded)

    return None

# def _try_projection_with_z3(comb, row, orig_row, dataset, orig_dataset,
#                             projection_config, must_cons, category_mappings):
#     """Try projection for a specific combination using Z3"""
#
#     # Check constraint coverage
#     if not all(any(feat in comb for feat in projection_config.cons_feat[cons])
#                for cons in must_cons):
#         return None
#
#     # Create cache key
#     cache_key = (tuple(sorted(comb)), tuple(orig_row[projection_config.fixed_feat]))
#     # cache_key = (tuple(sorted(comb)))
#
#     if cache_key not in _solver_cache:
#         # Build solver
#         column_types = classify_columns(orig_dataset)
#         s = Optimize()
#         vars = {}
#
#         # Create Z3 variables
#         for col in dataset.columns:
#             if col == 'label':
#                 continue
#             if col in projection_config.cont_feat:
#                 if column_types[col] == 'float':
#                     var = Real(col)
#                 else:
#                     var = Int(col)
#                 vars[col] = var
#                 s.add(var >= dataset[col].min(), var <= dataset[col].max())
#             else:
#                 var = Int(col)
#                 vars[col] = var
#                 s.add(var >= 0, var <= len(dataset[col].cat.categories) - 1)
#
#         # Build bounds and add constraints
#         column_types, full_bounds, attribute_to_dataframes = bounds_builder(
#             orig_row, comb, orig_dataset, projection_config.constraints,
#             projection_config.cons_feat, projection_config.dic_cols,
#             projection_config.cons_function
#         )
#
#         # Map categorical values to codes
#         for attr in attribute_to_dataframes:
#             if attr not in projection_config.categorical_column_names:
#                 continue
#             for bound in attribute_to_dataframes[attr]:
#                 if len(full_bounds[bound]) > 0:
#                     full_bounds[bound][f'{attr}_value'] = \
#                         full_bounds[bound][f'{attr}_value'].map(category_mappings[attr])
#
#         # Add constraint bounds
#         for bound in full_bounds:
#             if len(bound) == 0:
#                 continue
#
#             bound_values = bound.values  # Convert once
#             for row_idx in range(len(bound_values)):
#                 row_bound = bound_values[row_idx]
#                 row_constraints = []
#                 for i in range(0, len(row_bound), 3):
#                     col = row_bound[i + 0]
#                     val = row_bound[i + 1]
#                     op = row_bound[i + 2]
#
#                     if col not in vars:
#                         continue
#
#                     # Create constraint based on operator
#                     if op == 'point':
#                         constraint = vars[col] == val
#                     elif op == 'not_point':
#                         constraint = vars[col] != val
#                     elif op == 'left_exclusive':
#                         constraint = vars[col] > val
#                     elif op == 'left_inclusive':
#                         constraint = vars[col] >= val
#                     elif op == 'right_exclusive':
#                         constraint = vars[col] < val
#                     elif op == 'right_inclusive':
#                         constraint = vars[col] <= val
#                     else:
#                         continue
#
#                     row_constraints.append(constraint)
#
#                 if row_constraints:
#                     s.add(Not(And(row_constraints)))
#
#         _solver_cache[cache_key] = (s, copy.deepcopy(vars), column_types)
#         print('Bounds stored in cache')
#     else:
#         print('Bounds loaded from cache')
#         s, vars, column_types = _solver_cache[cache_key]
#         vars = copy.deepcopy(vars)
#
#     s.push()
#
#     # Add fixed feature constraints
#     for col in projection_config.fixed_feat:
#         if col in vars:
#             s.add(vars[col] == row[col])
#
#     # Build objective function
#     distance = 0
#     for var in vars:
#         if var in projection_config.cont_feat:
#             med_scaled = projection_config.normalized_medians.get(var, 1.0) * \
#                          projection_config.norm_params.get(var, {'range': 1.0})['range']
#             if med_scaled <= 1e-10:
#                 med_scaled = 1.0
#             distance += Abs(vars[var] - row[var]) / med_scaled
#         else:
#             distance += If(vars[var] == row[var], 0, 1)
#
#     opt = s.minimize(distance)
#
#     # Solve
#     s.set(timeout=projection_config.solver_timeout)
#     res = s.check().r
#
#     if res != -1:
#         m = s.model()
#         row_val = row.copy()
#
#         # Extract solution
#         for col in comb:
#             model_val = m[vars[col]]
#             if model_val is None:
#                 continue
#
#             if col in projection_config.cont_feat:
#                 if column_types[col] == 'float':
#                     value = model_val.as_decimal(10).replace('?', '')
#                     row_val[col] = float(value)
#                 else:
#                     value = model_val.as_long()
#                     row_val[col] = value
#             else:
#                 row_val[col] = dataset[col].cat.categories[model_val.as_long()]
#
#         # Convert back to original categories
#         row_val_orig = row_val.copy()
#         for col, mapping in category_mappings.items():
#             if col in row_val_orig.index:
#                 reverse_mapping = {v: k for k, v in mapping.items()}
#                 if col in projection_config.categorical_column_names:
#                     row_val_orig[col] = reverse_mapping.get(int(row_val[col]), row_val[col])
#
#         print('Projection successful')
#         s.pop()
#         return row_val_orig
#     else:
#         s.pop()
#         return None


def diversity_slack_variable(s, vars, found_points, norm_params, normalized_medians, non_cat_cols, cat_cols):
    if len(found_points) == 0:
        return 0

    # Create a slack variable for minimum distance
    min_dist = Real('min_diversity_distance')

    # Add constraints: min_dist must be <= distance to each point
    for i in range(len(found_points)):
        distance_to_point = 0

        for col in non_cat_cols:
            if col in vars:
                med_scaled = normalized_medians[col] * norm_params[col]['range']
                if med_scaled <= 1e-10:
                    med_scaled = 1.0
                distance_to_point += Abs(vars[col] - found_points.iloc[i][col]) / med_scaled

        for col in cat_cols:
            if col in vars:
                distance_to_point += If(vars[col] == found_points.iloc[i][col], 0, 1)

        # Key constraint: min_dist must be less than or equal to this distance
        s.add(min_dist <= distance_to_point)

    # Add bounds to help solver
    s.add(min_dist >= 0)
    # Optional: add upper bound based on problem knowledge
    # s.add(min_dist <= len(non_cat_cols) + len(cat_cols))

    return min_dist


def _try_projection_with_z3(comb, row, orig_row, dataset, orig_dataset,
                            projection_config, must_cons, category_mappings):
    """Try projection for a specific combination using Z3"""

    # Check constraint coverage
    if not all(any(feat in comb for feat in projection_config.cons_feat[cons])
               for cons in must_cons):
        return None

    # Create cache key
    cache_key = (tuple(sorted(comb)), tuple(orig_row[projection_config.fixed_feat]))

    if cache_key not in _solver_cache:
        # Build solver
        column_types = classify_columns(orig_dataset)
        s = Optimize()
        vars = {}

        # Create Z3 variables
        for col in dataset.columns:
            if col == 'label':
                continue
            if col in projection_config.cont_feat:
                if column_types[col] == 'float':
                    var = Real(col)
                else:
                    var = Int(col)
                vars[col] = var
                s.add(var >= dataset[col].min(), var <= dataset[col].max())
            else:
                var = Int(col)
                vars[col] = var
                s.add(var >= 0, var <= len(dataset[col].cat.categories) - 1)

        # Build bounds and add constraints
        column_types, full_bounds, attribute_to_dataframes = bounds_builder(
            orig_row, comb, orig_dataset, projection_config.constraints,
            projection_config.cons_feat, projection_config.dic_cols,
            projection_config.cons_function
        )

        # Map categorical values to codes
        for attr in attribute_to_dataframes:
            if attr not in projection_config.categorical_column_names:
                continue
            for bound in attribute_to_dataframes[attr]:
                if len(full_bounds[bound]) > 0:
                    full_bounds[bound][f'{attr}_value'] = \
                        full_bounds[bound][f'{attr}_value'].map(category_mappings[attr])

        # Add constraint bounds
        for bound in full_bounds:
            if len(bound) == 0:
                continue

            bound_values = bound.values
            for row_idx in range(len(bound_values)):
                row_bound = bound_values[row_idx]
                row_constraints = []
                for i in range(0, len(row_bound), 3):
                    col = row_bound[i + 0]
                    val = row_bound[i + 1]
                    op = row_bound[i + 2]

                    if col not in vars:
                        continue

                    if op == 'point':
                        constraint = vars[col] == val
                    elif op == 'not_point':
                        constraint = vars[col] != val
                    elif op == 'left_exclusive':
                        constraint = vars[col] > val
                    elif op == 'left_inclusive':
                        constraint = vars[col] >= val
                    elif op == 'right_exclusive':
                        constraint = vars[col] < val
                    elif op == 'right_inclusive':
                        constraint = vars[col] <= val
                    else:
                        continue

                    row_constraints.append(constraint)

                if row_constraints:
                    s.add(Not(And(row_constraints)))

        # _solver_cache[cache_key] = (s, copy.deepcopy(vars), column_types)
        _solver_cache[cache_key] = (s, vars, column_types)
        print('Bounds stored in cache')
    else:
        print('Bounds loaded from cache')
        s, vars, column_types = _solver_cache[cache_key]
        # vars = copy.deepcopy(vars)

    s.push()

    # Add fixed feature constraints
    for col in projection_config.fixed_feat:
        if col in vars:
            s.add(vars[col] == row[col])

    # Add diversity constraints if found_points exist
    found_points = projection_config.found_points
    if found_points is not None and len(found_points) > 0:
        for i in range(len(found_points)):
            s.add(Or([vars[col] != found_points.iloc[i][col]
                      for col in vars if col != 'label']))

    # Build objective function
    distance = 0
    for var in vars:
        if var in projection_config.cont_feat:
            med_scaled = projection_config.normalized_medians.get(var, 1.0) * \
                         projection_config.norm_params.get(var, {'range': 1.0})['range']
            if med_scaled <= 1e-10:
                med_scaled = 1.0
            distance += Abs(vars[var] - row[var]) / med_scaled
        else:
            distance += If(vars[var] == row[var], 0, 1)

    # Add diversity term to objective if found_points exist
    if found_points is not None and len(found_points) > 1e3:
        print(f'Found {len(found_points)} points')
        diversity_term = diversity_slack_variable(
            s, vars, found_points,
            projection_config.norm_params,
            projection_config.normalized_medians,
            projection_config.cont_feat,
            projection_config.categorical_column_names
        )
        total_dist = distance - projection_config.delta * diversity_term
    else:
        total_dist = distance

    opt = s.minimize(total_dist)

    # Solve
    s.set(timeout=projection_config.solver_timeout)
    res = s.check().r

    if res != -1:
        m = s.model()
        row_val = row.copy()

        # Extract solution
        for col in comb:
            model_val = m[vars[col]]
            if model_val is None:
                continue

            if col in projection_config.cont_feat:
                if column_types[col] == 'float':
                    value = model_val.as_decimal(10).replace('?', '')
                    row_val[col] = float(value)
                else:
                    value = model_val.as_long()
                    row_val[col] = value
            else:
                row_val[col] = dataset[col].cat.categories[model_val.as_long()]

        # Convert back to original categories
        row_val_orig = row_val.copy()
        for col, mapping in category_mappings.items():
            if col in row_val_orig.index:
                reverse_mapping = {v: k for k, v in mapping.items()}
                if col in projection_config.categorical_column_names:
                    row_val_orig[col] = reverse_mapping.get(int(row_val[col]), row_val[col])

        print('Projection successful')
        s.pop()

        if projection_config.gamma > 0:
            print(f'Adding gamma constraint: at least {int(projection_config.gamma)} features must differ')

            # Build coded values dict from model - ONLY for comb
            row_val_coded = {}
            for var in comb:
                model_val = m[vars[var]]
                if model_val is None:
                    continue

                if var in projection_config.cont_feat:
                    if column_types[var] == 'float':
                        row_val_coded[var] = float(model_val.as_decimal(10).replace('?', ''))
                    else:
                        row_val_coded[var] = model_val.as_long()
                else:
                    row_val_coded[var] = model_val.as_long()  # Store integer code

            feature_differences = []

            for var in comb:
                if var not in row_val_coded:
                    continue

                if var in projection_config.cont_feat:
                    # For continuous: consider "different" if change exceeds a threshold
                    med_scaled = projection_config.normalized_medians.get(var, 1.0) * \
                                 projection_config.norm_params.get(var, {'range': 1.0})['range']
                    if med_scaled <= 1e-10:
                        med_scaled = 1.0

                    # Is the change significant? (at least 1 normalized unit)
                    diff = Abs(vars[var] - row_val_coded[var])
                    is_different = diff > med_scaled
                    feature_differences.append(If(is_different, 1, 0))
                else:
                    # For categorical: simple equality check
                    is_different = vars[var] != row_val_coded[var]
                    feature_differences.append(If(is_different, 1, 0))

            # Count total number of different features
            num_different_features = Sum(feature_differences)

            # Require at least gamma features to be different
            gamma_constraint = num_different_features >= int(projection_config.gamma)

            s.add(gamma_constraint)
            _solver_cache[cache_key] = (s, vars, column_types)
            print(f'Gamma constraint added: {int(projection_config.gamma)} features must differ')

        return row_val_orig, row_val
    else:
        s.pop()
        return None

# def project_instances(cf_example, df, cont_feat, d, fixed_feat, dic_cols, constraints, cons_function, cons_feat,
#                       unary_cons_lst_single, bin_cons, norm_params, normalized_medians):
#     """Project instances using Z3 solver"""
#     project_cfs_df = cf_example.final_cfs_df_sparse.copy()
#
#     for col in df.columns:
#         if col in cont_feat + ['label']:
#             continue
#         project_cfs_df[col] = pd.Categorical(project_cfs_df[col],
#                                              categories=d.data_df[col].cat.categories)
#
#     # Create projection config
#     categorical_column_names = [col for col in df.columns
#                                 if col not in cont_feat + ['label']]
#
#     projection_config = ProjectionConfig(
#         df=d.data_df,
#         constraints=constraints,
#         dic_cols=dic_cols,
#         cons_function=cons_function,
#         cons_feat=cons_feat,
#         categorical_column_names=categorical_column_names,
#         fixed_feat=fixed_feat,
#         cont_feat=cont_feat,
#         norm_params=norm_params,
#         normalized_medians=normalized_medians,
#         unary_cons_lst=[],
#         unary_cons_lst_single=unary_cons_lst_single,
#         bin_cons=bin_cons
#     )
#
#     projected_cfs_df = project_cfs_df.copy()
#     projected_cfs_df = projected_cfs_df[1:0]
#
#     for index, row in project_cfs_df.iterrows():
#         print(f'Projecting instance {index} with Z3 solver')
#         proj_row = project_solver(row, projection_config)
#         if proj_row is not None:
#             projected_cfs_df.loc[len(projected_cfs_df)] = proj_row
#
#     return projected_cfs_df


def project_instances(cf_example, df, cont_feat, d, fixed_feat, dic_cols, constraints, cons_function, cons_feat,
                      unary_cons_lst_single, bin_cons, norm_params, normalized_medians, delta=50.0,
                      model=None, transformer=None):
    """Project instances with diversity only from accepted samples"""

    project_cfs_df = cf_example.final_cfs_df_sparse.copy()

    for col in df.columns:
        if col in cont_feat + ['label']:
            continue
        project_cfs_df[col] = pd.Categorical(project_cfs_df[col],
                                             categories=d.data_df[col].cat.categories)

    categorical_column_names = [col for col in df.columns
                                if col not in cont_feat + ['label']]

    projection_config = ProjectionConfig(
        df=d.data_df,
        constraints=constraints,
        dic_cols=dic_cols,
        cons_function=cons_function,
        cons_feat=cons_feat,
        categorical_column_names=categorical_column_names,
        fixed_feat=fixed_feat,
        cont_feat=cont_feat,
        norm_params=norm_params,
        normalized_medians=normalized_medians,
        unary_cons_lst=[],
        unary_cons_lst_single=unary_cons_lst_single,
        bin_cons=bin_cons,  # Start with previously accepted
        delta=delta
    )

    projected_cfs_df = project_cfs_df.copy()
    projected_cfs_df = projected_cfs_df[1:0]


    for index, row in project_cfs_df.iterrows():
        print(f'Projecting instance {index} with Z3 solver')

        # Log current diversity set size
        if projection_config.found_points is not None:
            print(f'  Enforcing diversity from {len(projection_config.found_points)} accepted samples')

        # Project with current found_points
        result = project_solver(row, projection_config)

        if result is not None:
            if isinstance(result, tuple):
                proj_row, proj_row_coded = result
            else:
                proj_row = result
                proj_row_coded = None

            projected_cfs_df.loc[len(projected_cfs_df)] = proj_row

            # # **Check if accepted immediately AFTER projection**
            # if model is not None and transformer is not None and proj_row_coded is not None:
            #     proj_row_for_model = proj_row.drop('label', errors='ignore')
            #     try:
            #         prob = model(
            #             torch.tensor(
            #                 transformer.transform(proj_row_for_model.to_frame().T).values.astype('float32')
            #             ).float()
            #         ).detach().numpy()
            #         is_accepted = np.round(prob)[0][0] == 1
            #
            #         print(f'  Model prediction: {prob[0][0]:.3f}, Accepted: {is_accepted}')
            #
            #         if is_accepted:
            #             # Add to batch accepted samples
            #             if accepted_coded_in_batch is None:
            #                 accepted_coded_in_batch = proj_row_coded.to_frame().T
            #             else:
            #                 accepted_coded_in_batch = pd.concat(
            #                     [accepted_coded_in_batch, proj_row_coded.to_frame().T],
            #                     ignore_index=True
            #                 )
            #
            #             # **UPDATE projection_config.found_points AFTER acceptance**
            #             # This ensures the NEXT projection is diverse from this accepted sample
            #             if found_points_coded is not None:
            #                 projection_config.found_points = pd.concat(
            #                     [found_points_coded, accepted_coded_in_batch],
            #                     ignore_index=True
            #                 )
            #             else:
            #                 projection_config.found_points = accepted_coded_in_batch
            #
            #             print(f'  ✓ Added to diversity set (total: {len(projection_config.found_points)})')
            #         else:
            #             print(f'  ✗ Rejected by model, NOT added to diversity')
            #
            #     except Exception as e:
            #         print(f'  Warning: Could not check model: {e}')
            #         import traceback
            #         traceback.print_exc()

    return projected_cfs_df

# def project_counterfactuals(dice_exp, df, cont_feat, d, fixed_feat, dic_cols, constraints, cons_function, cons_feat,
#                             unary_cons_lst_single, bin_cons, norm_params, normalized_medians):
#     all_instances_cfs = []
#     for cf_example in dice_exp.cf_examples_list:
#         all_instances_cfs.append(
#             project_instances(cf_example, df, cont_feat, d, fixed_feat, dic_cols, constraints, cons_function,
#                               cons_feat, unary_cons_lst_single, bin_cons, norm_params, normalized_medians))
#     return all_instances_cfs

def project_counterfactuals(dice_exp, df, cont_feat, d, fixed_feat, dic_cols, constraints, cons_function, cons_feat,
                            unary_cons_lst_single, bin_cons, norm_params, normalized_medians, delta=50.0,
                            model=None, transformer=None):  # **NEW: add model and transformer**
    """Project counterfactuals with diversity from accepted samples only

    Args:
        model: Neural network model to check acceptance
        transformer: Data transformer for model input
    """
    all_instances_cfs = []

    for cf_example in dice_exp.cf_examples_list:
        projected_cfs = project_instances(
            cf_example, df, cont_feat, d, fixed_feat, dic_cols, constraints, cons_function,
            cons_feat, unary_cons_lst_single, bin_cons, norm_params, normalized_medians, delta=delta,
            model=model, transformer=transformer  # **NEW**
        )
        all_instances_cfs.append(projected_cfs)
    return all_instances_cfs


# Timing utility
@contextmanager
def timed_section(name, queue=None):
    """Context manager for timing code sections"""
    start = time.time()
    yield
    duration = time.time() - start
    msg = f"{name} took {duration:.2f} seconds"
    print(msg)
    if queue:
        queue.put({'type': 'progress', 'text': msg})

# def bfs_counterfactuals(exp, threshold, model, fixed_feat, exp_random, df, cont_feat, d, dic_cols, constraints,
#                         cons_function, cons_feat, unary_cons_lst_single, bin_cons, transformer,
#                         norm_params, normalized_medians, progress_queue=None, max_iter=50):
#     with timed_section("Projecting counterfactuals", progress_queue):
#         projected_cfs = project_counterfactuals(exp, df, cont_feat, d, fixed_feat, dic_cols, constraints, cons_function,
#                                                 cons_feat, unary_cons_lst_single, bin_cons, norm_params, normalized_medians)
#
#     all_accepted_instances = []
#     for i, cfs in enumerate(projected_cfs):
#         accepted_final_cfs = []
#         cfs = cfs.drop('label', axis=1)
#         probs = model(torch.tensor(transformer.transform(cfs).values.astype('float32')).float()).detach().numpy()
#         labels = np.round(probs)
#         accepted = cfs[labels == 1]
#         print('###########ACCEPTED#############')
#         print(accepted)
#
#         not_accepted = cfs[labels == 0]
#         print('###########NOT ACCEPTED#############')
#         print(not_accepted)
#         if len(not_accepted) == 0:
#             return accepted
#         features_to_vary = list(df.columns)
#         for feat in fixed_feat:
#             features_to_vary.remove(feat)
#
#         not_accepted_final_cfs = []
#         iteration = 0
#         while len(accepted) < threshold and iteration < 3:
#             iteration += 1
#             if progress_queue:
#                 progress_queue.put({'type': 'progress',
#                                     'text': f'BFS iteration {iteration}, found {len(accepted)} valid counterfactuals'})
#
#             try:
#                 dice_exp_random = exp_random.generate_counterfactuals(not_accepted, total_CFs=threshold,
#                                                                       desired_class=1,
#                                                                       verbose=False,
#                                                                       learning_rate=0.1,  # Increased
#                                                                       min_iter=5,  # Reduced
#                                                                       max_iter=max_iter,  # Reduced
#                                                                       features_to_vary=features_to_vary)
#             except:
#                 break
#             print(f'dice_exp_random:\n {print(dice_exp_random.cf_examples_list[0].final_cfs_df_sparse)}')
#             projected_cfs_not_accepted = project_counterfactuals(dice_exp_random, df, cont_feat, d, fixed_feat,
#                                                                  dic_cols, constraints, cons_function, cons_feat,
#                                                                  unary_cons_lst_single, bin_cons,
#                                                                  norm_params, normalized_medians)
#             print(f'projected_cfs_not_accepted:\n {projected_cfs_not_accepted}')
#             not_accepted = None
#             for cfs in projected_cfs_not_accepted:
#                 cfs = cfs.drop('label', axis=1)
#                 probs = model(
#                     torch.tensor(transformer.transform(cfs).values.astype('float32')).float()).detach().numpy()
#                 labels = np.round(probs)
#                 accepted_2 = cfs[labels == 1]
#                 not_accepted_final_cfs.append(accepted_2)
#                 if not_accepted is None:
#                     not_accepted = cfs[labels == 0]
#                 else:
#                     print(f'not_accepted:\n {not_accepted}')
#                     not_accepted = pd.concat([not_accepted, cfs[labels == 0]])
#                 if len(accepted_2) != 0:
#                     print(f'accepted:\n {accepted}')
#                     accepted = pd.concat([accepted, accepted_2], ignore_index=True)
#                     accepted_final_cfs.append(accepted_2)
#
#         print('###########ACCEPTED NEW#############')
#         print(accepted)
#         return accepted
#     return None
#

def bfs_counterfactuals(exp, threshold, model, fixed_feat, exp_random, df, cont_feat, d, dic_cols, constraints,
                        cons_function, cons_feat, unary_cons_lst_single, bin_cons, transformer,
                        norm_params, normalized_medians, progress_queue=None,min_iter=5, max_iter=50,lr = 0.01, delta=50.0):
    """Generate counterfactuals with diversity from accepted samples only"""

    # Track ALL accepted samples across entire BFS run
    accepted_final_cfs = None

    # Prepare category mappings
    categorical_column_names = [col for col in df.columns if col not in cont_feat + ['label']]
    combined_temp = pd.concat([d.data_df, df.iloc[:1]], ignore_index=True)
    combined_temp[categorical_column_names] = combined_temp[categorical_column_names].astype('category')
    cat_cols = combined_temp.select_dtypes(include=['category']).columns
    category_mappings = {col: {v: k for k, v in enumerate(combined_temp[col].cat.categories)}
                         for col in cat_cols}

    with timed_section("Projecting counterfactuals", progress_queue):
        # Initial projection - passes model and transformer for immediate checking
        projected_cfs = project_counterfactuals(
            exp, df, cont_feat, d, fixed_feat, dic_cols, constraints, cons_function,
            cons_feat, unary_cons_lst_single, bin_cons, norm_params, normalized_medians,
            delta=delta,
            model=model,
            transformer=transformer
        )

    for i, cfs in enumerate(projected_cfs):
        cfs = cfs.drop('label', axis=1, errors='ignore')

        probs = model(torch.tensor(transformer.transform(cfs).values.astype('float32')).float()).detach().numpy()
        labels = np.round(probs)

        accepted = cfs[labels == 1]
        print('###########ACCEPTED (Initial)#############')
        print(accepted)
        print(f'Projected: {len(cfs)}, Accepted: {len(accepted)}')

        # Track in original format for returning
        if len(accepted) > 0:
            if accepted_final_cfs is None:
                accepted_final_cfs = accepted
            else:
                accepted_final_cfs = pd.concat([accepted_final_cfs, accepted], ignore_index=True)

        not_accepted = cfs[labels == 0]
        print('###########NOT ACCEPTED (Initial)#############')
        print(not_accepted)

        if len(not_accepted) == 0:
            return accepted_final_cfs if accepted_final_cfs is not None else accepted

        # Setup features to vary
        features_to_vary = list(df.columns)
        for feat in fixed_feat:
            features_to_vary.remove(feat)

        iteration = 0
        while (accepted_final_cfs is None or len(accepted_final_cfs) < threshold) and iteration < 3:
            iteration += 1

            if progress_queue:
                current_count = len(accepted_final_cfs) if accepted_final_cfs is not None else 0
                progress_queue.put({'type': 'progress',
                                    'text': f'BFS iteration {iteration}, found {current_count} valid counterfactuals'})

            try:
                dice_exp_random = exp_random.generate_counterfactuals(
                    not_accepted,
                    total_CFs=threshold,
                    desired_class=1,
                    verbose=VERBOSE,
                    learning_rate=lr,
                    min_iter=min_iter,
                    max_iter=max_iter,
                    features_to_vary=features_to_vary
                )
            except:
                break
            for cf_examples_list in dice_exp_random.cf_examples_list:
                print(f'dice_exp_random:\n {cf_examples_list.final_cfs_df_sparse}')

            # **Pass model for immediate acceptance checking**
            projected_cfs_not_accepted= project_counterfactuals(
                dice_exp_random, df, cont_feat, d, fixed_feat,
                dic_cols, constraints, cons_function, cons_feat,
                unary_cons_lst_single, bin_cons,
                norm_params, normalized_medians,
                delta=delta,
                model=model,
                transformer=transformer
            )

            print(f'projected_cfs_not_accepted:\n {projected_cfs_not_accepted}')

            not_accepted = None
            for cfs in projected_cfs_not_accepted:
                cfs = cfs.drop('label', axis=1, errors='ignore')

                # Double-check acceptance
                probs = model(
                    torch.tensor(transformer.transform(cfs).values.astype('float32')).float()
                ).detach().numpy()
                labels = np.round(probs)

                accepted_2 = cfs[labels == 1]
                print(f'Iteration {iteration} - Projected: {len(cfs)}, Accepted: {len(accepted_2)}')

                # Track in original format
                if len(accepted_2) > 0:
                    if accepted_final_cfs is None:
                        accepted_final_cfs = accepted_2
                    else:
                        accepted_final_cfs = pd.concat([accepted_final_cfs, accepted_2], ignore_index=True)

                # Track rejected
                if not_accepted is None:
                    not_accepted = cfs[labels == 0]
                else:
                    not_accepted = pd.concat([not_accepted, cfs[labels == 0]])

        print('###########ACCEPTED (Final)#############')
        print(accepted_final_cfs)
        return accepted_final_cfs if accepted_final_cfs is not None else pd.DataFrame()
    return None

def code_counterfactuals(query_instances, constraints_path, dataset_path, fixed_feat, k,
                         model_cache, transformer_cache, constraints_cache, progress_queue=None,min_iter = 5, max_iter = 50,lr=0.01, delta = 0.0):
    """Modified to use caches and progress queue and return both DiCE and CoDeC results"""

    def send_progress(message):
        if progress_queue:
            progress_queue.put({'type': 'progress', 'text': message})

    # Load constraints with caching
    if constraints_path in constraints_cache:
        constraint_data = constraints_cache[constraints_path]
        constraints, dic_cols, cons_function, cons_feat, unary_cons_lst, unary_cons_lst_single, bin_cons = constraint_data
    else:
        send_progress("Loading constraints")
        constraint_data = load_constraints(constraints_path)
        constraints_cache[constraints_path] = constraint_data
        constraints, dic_cols, cons_function, cons_feat, unary_cons_lst, unary_cons_lst_single, bin_cons = constraint_data

    # Load dataset
    send_progress("Loading dataset")
    df = pd.read_csv(dataset_path)

    cont_feat = []
    for col in df.columns:
        if df[col].dtype != 'object' and col != 'label':
            cont_feat.append(col)
    for col in df.columns:
        if col in cont_feat + ['label']:
            continue
        df[col] = df[col].astype('object')

    y = df['label']
    train_dataset, test_dataset, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0, stratify=y)
    x_train = train_dataset.drop('label', axis=1)

    cache_key = dataset_path
    dataset_filename = dataset_path.split('/')[-1].split('\\')[-1].split('.')[0]
    model_state_dict_path = f'models/{dataset_filename}_model_state_dict.dict'

    # Check if model is already in cache
    if cache_key in model_cache:
        model = model_cache[cache_key]
        transformer = transformer_cache[cache_key]
        d = dice_ml.Data(dataframe=train_dataset, continuous_features=cont_feat, outcome_name='label')
        print("Model loaded from cache")
    else:
        send_progress("Preparing model and transformer")
        d = dice_ml.Data(dataframe=train_dataset, continuous_features=cont_feat, outcome_name='label')
        transformer = DataTransfomer('ohe-min-max')
        transformer.feed_data_params(d)
        transformer.initialize_transform_func()
        X_train = transformer.transform(x_train)

        model = Mlp(X_train.shape[1], [100, 1])

        # Check if model exists on disk BEFORE setting to training mode
        if os.path.exists(model_state_dict_path):
            model.load_state_dict(torch.load(model_state_dict_path))
            print("Model loaded from disk")
        else:
            # Only train if model doesn't exist on disk
            print("Training new model...")
            model.train()
            X_train_fixed = X_train.values.astype(np.float32)
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_train_fixed), torch.Tensor(y_train.values.astype('int'))),
                64, shuffle=True)
            model = pretrain(model, 'cpu', train_loader, lr=1e-4, epochs=100)
            torch.save(model.state_dict(), model_state_dict_path)
            print('Model trained and saved to disk')

        # Cache for future use
        model_cache[cache_key] = model
        transformer_cache[cache_key] = transformer
    model.eval()
    m = dice_ml.Model(model=model, backend='PYT', func="ohe-min-max")
    exp_random = dice_ml.Dice(d, m, method="gradient", constraints=False)

    features_to_vary = list(df.columns)
    for feat in fixed_feat:
        features_to_vary.remove(feat)

    send_progress("Generating initial counterfactuals")
    dice_exp_random = exp_random.generate_counterfactuals(
        query_instances,
        total_CFs=k,
        desired_class=1,
        verbose=VERBOSE,
        features_to_vary=features_to_vary,
        min_iter=min_iter,  # Reduced
        max_iter=max_iter,  # Reduced
        learning_rate=lr  # Increased
        # learning_rate=5e-2  # Increased
    )
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None,
                           'display.max_colwidth', None):
        print(dice_exp_random.cf_examples_list[0].final_cfs_df_sparse)

    # STORE DICE RESULTS
    dice_cfs = dice_exp_random.cf_examples_list[0].final_cfs_df_sparse.drop('label', axis=1)
    dice_cfs_processed, dice_dpp_score, dice_distances = n_best_cfs_heuristic(dice_cfs, query_instances.iloc[0], k, transformer, exp_random)

    # with open('dice_gui_metrics.pkl', 'wb') as f:
    #     pickle.dump((dice_cfs_processed, dice_dpp_score, dice_distances), f)

    send_progress("Running BFS for constraint satisfaction")
    # Create normalization parameters
    norm_params, normalized_medians = create_normalization_params(
        x_train, cont_feat,
        [col for col in x_train.columns if col not in cont_feat]
    )

    # Then pass these to bfs_counterfactuals:
    codec_cfs = bfs_counterfactuals(
        dice_exp_random, k, model, fixed_feat, exp_random, df, cont_feat, d,
        dic_cols, constraints, cons_function, cons_feat, unary_cons_lst_single, bin_cons,
        transformer, norm_params, normalized_medians, progress_queue,min_iter = min_iter, max_iter=max_iter,lr = lr, delta=delta
    )
    reset_solver_cache()
    print(codec_cfs)
    codec_cfs_processed, codec_dpp_score, codec_distances = n_best_cfs_heuristic(codec_cfs, query_instances.iloc[0], k, transformer, exp_random)
    # codec_cfs_processed, codec_dpp_score, codec_distances = dice_cfs_processed,dice_dpp_score,dice_distances
    # with open('codec_gui_metrics.pkl', 'wb') as f:
    #     pickle.dump((codec_cfs_processed, codec_dpp_score, codec_distances), f)

    # Return both DiCE and CoDeC results
    return {
        'codec_results': (codec_cfs_processed, codec_dpp_score, codec_distances),
        'dice_results': (dice_cfs_processed, dice_dpp_score, dice_distances),
        'exp_random': exp_random,
        'constraint_data': {  # ADD: Return constraint-related data
            'cons_function': cons_function,
            'constraints': constraints,
            'cons_feat': cons_feat,
            'dataset': d.data_df  # The training dataset for constraint checking
        }
    }

# Set CustomTkinter appearance

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class AnimatedButton(ctk.CTkButton):
    """Custom button with hover animations"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        self.configure(font=scaled_font(16, "bold"))

    def on_leave(self, event):
        self.configure(font=scaled_font(15, "bold"))


class StaticButton(ctk.CTkButton):
    """Button without hover animations (for delete buttons)"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DatasetPreviewDialog(ctk.CTkToplevel):
    """Modern dataset preview dialog"""

    def __init__(self, parent, df, colors):
        super().__init__(parent)
        self.title("Dataset Preview")
        self.geometry("1200x700")
        self.colors = colors

        # Configure window
        self.configure(fg_color=self.colors['bg_primary'])

        # Main frame
        main_frame = ctk.CTkFrame(self, fg_color=self.colors['bg_secondary'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="Dataset Preview (10 Random Samples)",
            font=scaled_font(32, "bold"),
            text_color=self.colors['text_primary']
        )
        title_label.pack(pady=20)

        # Get random samples
        n_samples = min(10, len(df))
        random_df = df.sample(n=n_samples).reset_index(drop=True)

        # Create canvas frame for scrollable table
        canvas_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        canvas_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        # Canvas with scrollbars
        canvas = tk.Canvas(
            canvas_frame,
            bg=self.colors['bg_primary'],
            highlightthickness=0
        )

        # Horizontal scrollbar
        h_scrollbar = ctk.CTkScrollbar(
            canvas_frame,
            orientation='horizontal',
            command=canvas.xview,
            fg_color=self.colors['bg_tertiary'],
            button_color=self.colors['accent'],
            button_hover_color=self.colors['accent_hover']
        )
        h_scrollbar.pack(side='bottom', fill='x')

        # Vertical scrollbar
        v_scrollbar = ctk.CTkScrollbar(
            canvas_frame,
            command=canvas.yview,
            fg_color=self.colors['bg_tertiary'],
            button_color=self.colors['accent'],
            button_hover_color=self.colors['accent_hover']
        )
        v_scrollbar.pack(side='right', fill='y')

        canvas.pack(side='left', fill='both', expand=True)
        canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

        # Create scrollable frame inside canvas
        table_frame = ctk.CTkFrame(canvas, fg_color="transparent")
        canvas_window = canvas.create_window((0, 0), window=table_frame, anchor='nw')

        # Headers
        headers = list(random_df.columns)
        header_frame = ctk.CTkFrame(table_frame, fg_color=self.colors['bg_tertiary'], corner_radius=10)
        header_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))

        for col, header in enumerate(headers):
            header_label = ctk.CTkLabel(
                header_frame,
                text=header,
                font=scaled_font(18, "bold"),
                text_color=self.colors['accent'],
                width=150
            )
            header_label.grid(row=0, column=col, padx=8, pady=15, sticky='ew')

        # Data rows
        for row_idx in range(len(random_df)):
            row_frame = ctk.CTkFrame(table_frame, fg_color=self.colors['bg_primary'], corner_radius=8)
            row_frame.grid(row=row_idx + 1, column=0, sticky='ew', pady=3)

            for col_idx, header in enumerate(headers):
                value = random_df.iloc[row_idx][header]
                value_str = str(value)[:25] + ("..." if len(str(value)) > 25 else "")

                value_label = ctk.CTkLabel(
                    row_frame,
                    text=value_str,
                    font=scaled_font(16),
                    text_color=self.colors['text_primary'],
                    width=150
                )
                value_label.grid(row=0, column=col_idx, padx=8, pady=12, sticky='ew')

        # Update scroll region
        table_frame.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

        # Bind mousewheel events
        def on_mousewheel(event):
            shift = (event.state & 0x1) != 0
            if shift:
                canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # Bind to canvas size changes
        def configure_canvas(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        table_frame.bind("<Configure>", configure_canvas)

        # Close button
        close_btn = AnimatedButton(
            main_frame,
            text="Close",
            width=150,
            height=40,
            font=scaled_font(15, "bold"),
            fg_color=self.colors['accent'],
            hover_color=self.colors['accent_hover'],
            command=self.destroy
        )
        close_btn.pack(pady=10)

        # Make modal
        self.transient(parent)
        self.grab_set()


class ModernCounterfactualGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CODEC • Constraints-Guided Diverse Counterfactuals")

        # Set window to start at full screen size
        self.root.geometry("1400x900")  # Start with a large size
        self.root.state('zoomed')  # Windows - maximize
        # self.root.attributes('-zoomed', True)  # Linux
        # For Mac, you might want to use:
        # self.root.attributes('-fullscreen', False)
        # self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")

        # Modern color scheme - Cyberpunk inspired
        self.colors = {
            'bg_primary': '#0a0a0a',
            'bg_secondary': '#111111',
            'bg_tertiary': '#1a1a1a',
            'bg_card': '#161616',
            'accent': '#00d4ff',
            'accent_secondary': '#ff006e',
            'accent_hover': '#00a8cc',
            'success': '#00ff88',
            'warning': '#ffaa00',
            'error': '#ff0055',
            'text_primary': '#ffffff',
            'text_secondary': '#b0b0b0',
            'text_dim': '#666666',
            'border': '#2a2a2a',
            'glow': '#00d4ff'
        }

        # Configure root window
        self.root.configure(bg=self.colors['bg_primary'])

        # Initialize variables
        self.dataset_path = tk.StringVar()
        self.constraints_path = tk.StringVar()
        self.num_counterfactuals = tk.StringVar(value="2")
        self.fixed_features = set()
        self.animation_running = False
        self.current_screen = "input"
        self.current_results_mode = "codec"  # Track current results mode
        self.stored_results = {}  # Store both DiCE and CoDeC results

        # Initialize caches
        self.cached_models = {}
        self.cached_transformers = {}
        self.cached_constraints = {}
        self.initial_point_widgets = {}

        # Initialize queue
        self.computation_queue = queue.Queue()
        self.check_computation_queue()

        # Create main layout
        self.setup_ui()

    def setup_ui(self):
        # Create main container
        self.main_container = ctk.CTkFrame(self.root, fg_color=self.colors['bg_primary'])
        self.main_container.pack(fill='both', expand=True)

        # Create screens
        self.create_input_screen()
        self.create_results_screen()

        # Show input screen by default
        self.show_input_screen()

    def create_animated_header(self, parent, subtitle="Constraints-Guided Diverse Counterfactuals"):
        """Create an animated header with gradient and particles"""
        header_frame = ctk.CTkFrame(parent, fg_color=self.colors['bg_secondary'], height=scaled_size(100))
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)

        # Create content frame to prevent text overlap
        content_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        content_frame.pack(fill='both', expand=True, padx=80)

        # Title and subtitle in vertical layout
        title_label = ctk.CTkLabel(
            content_frame,
            text="CODEC",
            font=scaled_font(54, "bold"),
            text_color=self.colors['accent']
        )
        title_label.pack(anchor='w', pady=(15, 0))

        subtitle_label = ctk.CTkLabel(
            content_frame,
            text=subtitle,
            font=scaled_font(18),
            text_color=self.colors['text_secondary']
        )
        subtitle_label.pack(anchor='w')

        # Status indicator
        self.status_frame = ctk.CTkFrame(
            header_frame,
            fg_color=self.colors['bg_tertiary'],
            corner_radius=20,
            width=200,
            height=40
        )
        self.status_frame.place(relx=0.81, rely=0.5, anchor='center')

        self.status_indicator = ctk.CTkLabel(
            self.status_frame,
            text="● Ready",
            font=scaled_font(16),
            text_color=self.colors['success']
        )
        self.status_indicator.pack(expand=True)

    def create_input_screen(self):
        """Create input configuration screen"""
        self.input_screen = ctk.CTkFrame(self.main_container, fg_color=self.colors['bg_primary'])

        # Header
        self.create_animated_header(self.input_screen)

        # Main content area
        content_area = ctk.CTkFrame(self.input_screen, fg_color="transparent")
        content_area.pack(fill='both', expand=True, padx=40, pady=20)

        # Left panel - File inputs
        left_panel = ctk.CTkFrame(content_area, fg_color=self.colors['bg_secondary'], corner_radius=15)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 20))

        # Right panel - Configuration
        right_panel = ctk.CTkFrame(content_area, fg_color=self.colors['bg_secondary'], corner_radius=15)
        right_panel.pack(side='right', fill='both', expand=True)

        # Setup panels
        self.setup_file_inputs_panel(left_panel)
        self.setup_configuration_panel(right_panel)

    def setup_file_inputs_panel(self, parent):
        """Setup file inputs panel"""
        # Title
        title = ctk.CTkLabel(
            parent,
            text="Data Sources",
            font=scaled_font(32, "bold"),
            text_color=self.colors['text_primary']
        )
        title.pack(pady=30)

        # Dataset section
        dataset_frame = ctk.CTkFrame(parent, fg_color=self.colors['bg_tertiary'], corner_radius=15)
        dataset_frame.pack(fill='x', padx=30, pady=10)

        dataset_content = ctk.CTkFrame(dataset_frame, fg_color="transparent")
        dataset_content.pack(fill='x', padx=20, pady=20)

        ctk.CTkLabel(
            dataset_content,
            text="Dataset File",
            font=scaled_font(20),
            text_color=self.colors['text_secondary']
        ).pack(anchor='w')

        input_frame = ctk.CTkFrame(dataset_content, fg_color="transparent")
        input_frame.pack(fill='x', pady=10)

        self.dataset_entry = ctk.CTkEntry(
            input_frame,
            textvariable=self.dataset_path,
            height=45,
            font=scaled_font(18),
            fg_color=self.colors['bg_primary'],
            border_color=self.colors['border'],
            border_width=2
        )
        self.dataset_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))

        AnimatedButton(
            input_frame,
            text="Browse",
            width=120,
            height=45,
            font=scaled_font(19, "bold"),
            fg_color=self.colors['accent'],
            hover_color=self.colors['accent_hover'],
            text_color="black",
            command=self.browse_dataset
        ).pack(side='left', padx=(0, 10))

        self.preview_btn = AnimatedButton(
            input_frame,
            text="Preview",
            width=120,
            height=45,
            font=scaled_font(19, "bold"),
            fg_color=self.colors['accent_secondary'],
            hover_color="#cc0056",
            command=self.preview_dataset,
            state="disabled"
        )
        self.preview_btn.pack(side='left')

        # Constraints section
        constraints_frame = ctk.CTkFrame(parent, fg_color=self.colors['bg_tertiary'], corner_radius=15)
        constraints_frame.pack(fill='x', padx=30, pady=10)

        constraints_content = ctk.CTkFrame(constraints_frame, fg_color="transparent")
        constraints_content.pack(fill='x', padx=20, pady=20)

        ctk.CTkLabel(
            constraints_content,
            text="Constraints File",
            font=scaled_font(20),
            text_color=self.colors['text_secondary']
        ).pack(anchor='w')

        input_frame2 = ctk.CTkFrame(constraints_content, fg_color="transparent")
        input_frame2.pack(fill='x', pady=10)

        self.constraints_entry = ctk.CTkEntry(
            input_frame2,
            textvariable=self.constraints_path,
            height=45,
            font=scaled_font(18),
            fg_color=self.colors['bg_primary'],
            border_color=self.colors['border'],
            border_width=2
        )
        self.constraints_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))

        AnimatedButton(
            input_frame2,
            text="Browse",
            width=120,
            height=45,
            font=scaled_font(19, "bold"),
            fg_color=self.colors['accent'],
            hover_color=self.colors['accent_hover'],
            text_color="black",
            command=self.browse_constraints
        ).pack(side='left', padx=(0, 10))  # Changed to side='left' with right padding

        self.constraints_preview_btn = AnimatedButton(
            input_frame2,
            text="Preview",
            width=120,
            height=45,
            font=scaled_font(19, "bold"),
            fg_color=self.colors['accent_secondary'],
            hover_color="#cc0056",
            command=self.preview_constraints,
            state="disabled"
        )
        self.constraints_preview_btn.pack(side='left')  # Changed to side='left'

        # Parameters section
        params_frame = ctk.CTkFrame(parent, fg_color=self.colors['bg_tertiary'], corner_radius=15)
        params_frame.pack(fill='x', padx=30, pady=10)

        params_content = ctk.CTkFrame(params_frame, fg_color="transparent")
        params_content.pack(fill='x', padx=20, pady=20)

        ctk.CTkLabel(
            params_content,
            text="Number of Counterfactuals",
            font=scaled_font(20),
            text_color=self.colors['text_secondary']
        ).pack(anchor='w')

        slider_frame = ctk.CTkFrame(params_content, fg_color="transparent")
        slider_frame.pack(fill='x', pady=10)

        self.num_cf_slider = ctk.CTkSlider(
            slider_frame,
            from_=1,
            to=10,
            number_of_steps=9,
            height=25,
            progress_color=self.colors['accent'],
            button_color=self.colors['accent'],
            button_hover_color=self.colors['accent_hover'],
            command=self.update_cf_count
        )
        self.num_cf_slider.pack(side='left', fill='x', expand=True, padx=(0, 20))
        self.num_cf_slider.set(2)

        self.cf_count_label = ctk.CTkLabel(
            slider_frame,
            text="2",
            font=scaled_font(32, "bold"),
            text_color=self.colors['accent'],
            width=40
        )
        self.cf_count_label.pack(side='right')

        # Generate button
        self.generate_btn = AnimatedButton(
            parent,
            text="🚀 Generate Counterfactuals",
            height=70,
            font=scaled_font(24, "bold"),
            fg_color=self.colors['accent'],
            hover_color=self.colors['accent_hover'],
            text_color="black",
            command=self.generate_counterfactuals
        )
        self.generate_btn.pack(fill='x', padx=30, pady=30)

    def setup_configuration_panel(self, parent):
        """Setup configuration panel"""
        # Scrollable content
        scroll_frame = ctk.CTkScrollableFrame(
            parent,
            fg_color="transparent",
            scrollbar_button_color=self.colors['bg_tertiary'],
            scrollbar_button_hover_color=self.colors['accent']
        )
        scroll_frame.pack(fill='both', expand=True, padx=30, pady=30)

        # Initial instance section
        instance_frame = ctk.CTkFrame(scroll_frame, fg_color=self.colors['bg_tertiary'], corner_radius=15)
        instance_frame.pack(fill='x', pady=(0, 20))

        instance_header = ctk.CTkLabel(
            instance_frame,
            text="Input Tuple",
            font=scaled_font(26, "bold"),
            text_color=self.colors['text_primary']
        )
        instance_header.pack(pady=(20, 10))

        self.instance_content_frame = ctk.CTkFrame(instance_frame, fg_color="transparent")
        self.instance_content_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        # Placeholder
        placeholder_label = ctk.CTkLabel(
            self.instance_content_frame,
            text="Load a dataset to configure initial instance",
            font=scaled_font(18),
            text_color=self.colors['text_dim']
        )
        placeholder_label.pack(pady=40)

        # Immutable features section
        features_frame = ctk.CTkFrame(scroll_frame, fg_color=self.colors['bg_tertiary'], corner_radius=15)
        features_frame.pack(fill='x')

        features_header = ctk.CTkLabel(
            features_frame,
            text="Immutable Attributes",
            font=scaled_font(26, "bold"),
            text_color=self.colors['text_primary']
        )
        features_header.pack(pady=(20, 10))

        self.features_content_frame = ctk.CTkFrame(features_frame, fg_color="transparent")
        self.features_content_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        # Placeholder
        features_placeholder = ctk.CTkLabel(
            self.features_content_frame,
            text="Load a dataset to select immutable features",
            font=scaled_font(18),
            text_color=self.colors['text_dim']
        )
        features_placeholder.pack(pady=40)

    def reorder_codec_by_closest_dice(self, dice_data, codec_data, exp_random, transformer):
        """Reorder CoDeC counterfactuals to match closest DiCE counterfactuals using weighted distance"""

        # Get counterfactuals as DataFrames
        dice_cfs = pd.DataFrame(dice_data['counterfactuals'])
        codec_cfs = pd.DataFrame(codec_data['counterfactuals'])

        # Transform to normalized space
        dice_transformed = torch.tensor(transformer.transform(dice_cfs).values.astype('float32'))
        codec_transformed = torch.tensor(transformer.transform(codec_cfs).values.astype('float32'))

        # Compute pairwise distances using the SAME weighted distance as in original code
        distances = np.zeros((len(dice_cfs), len(codec_cfs)))
        for i in range(len(dice_cfs)):
            for j in range(len(codec_cfs)):
                # Use the same compute_dist function from your code
                dist = compute_dist(dice_transformed[i], codec_transformed[j], exp_random).item()
                distances[i, j] = dist

        # Greedy matching: for each DiCE CF, find closest unused CoDeC CF
        used_codec_indices = set()
        codec_reorder = []
        codec_distances_reorder = []

        for i in range(len(dice_cfs)):
            # Find closest unused CoDeC CF
            min_dist = float('inf')
            min_idx = -1
            for j in range(len(codec_cfs)):
                if j not in used_codec_indices and distances[i, j] < min_dist:
                    min_dist = distances[i, j]
                    min_idx = j

            if min_idx >= 0:
                codec_reorder.append(codec_data['counterfactuals'][min_idx])
                codec_distances_reorder.append(codec_data['distances'][min_idx])
                used_codec_indices.add(min_idx)

        # Add any remaining CoDeC CFs
        for j in range(len(codec_cfs)):
            if j not in used_codec_indices:
                codec_reorder.append(codec_data['counterfactuals'][j])
                codec_distances_reorder.append(codec_data['distances'][j])

        return codec_reorder, codec_distances_reorder

    def create_results_screen(self):
        """Create results visualization screen"""
        self.results_screen = ctk.CTkFrame(self.main_container, fg_color=self.colors['bg_primary'])

        # Header with back button and toggle buttons
        header_frame = ctk.CTkFrame(self.results_screen, fg_color=self.colors['bg_secondary'], height=100)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)

        # Back button
        back_btn = AnimatedButton(
            header_frame,
            text="← Back to Input",
            width=150,
            height=40,
            font=scaled_font(16, "bold"),
            fg_color=self.colors['bg_tertiary'],
            hover_color=self.colors['accent_hover'],
            command=self.show_input_screen
        )
        back_btn.place(x=30, rely=0.5, anchor='w')

        # Title with current mode indicator
        self.title_label = ctk.CTkLabel(
            header_frame,
            text="CoDeC Results",
            font=scaled_font(40, "bold"),
            text_color=self.colors['text_primary']
        )
        self.title_label.place(relx=0.5, rely=0.5, anchor='center')

        # Toggle buttons
        # In the create_results_screen() method, I modified the toggle buttons section:

        # Toggle buttons
        toggle_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        toggle_frame.place(relx=0.79, rely=0.5, anchor='center')

        self.codec_btn = AnimatedButton(
            toggle_frame,
            text="CoDeC",
            width=scaled_size(80),
            height=scaled_size(35),
            font=scaled_font(14, "bold"),
            fg_color=self.colors['accent'],
            hover_color=self.colors['accent_hover'],
            command=lambda: self.switch_results_mode("codec")
        )
        self.codec_btn.pack(side='left', padx=(0, scaled_size(15)))  # ✅ Increased padding

        self.dice_btn = AnimatedButton(
            toggle_frame,
            text="DiCE",
            width=scaled_size(80),
            height=scaled_size(35),
            font=scaled_font(14, "bold"),
            fg_color=self.colors['bg_tertiary'],
            hover_color=self.colors['accent_hover'],
            command=lambda: self.switch_results_mode("dice")
        )
        self.dice_btn.pack(side='left', padx=(0, scaled_size(15)))

        # NEW: Compare button
        self.compare_btn = AnimatedButton(
            header_frame,
            text="⚖️ Compare",
            width=scaled_size(100),  # Reduced width since text is shorter
            height=scaled_size(40),
            font=scaled_font(14, "bold"),
            fg_color=self.colors['accent_secondary'],
            hover_color="#cc0056",
            command=self.show_comparison_dialog
        )
        self.compare_btn.place(relx=0.98, rely=0.5, anchor='e')

        # Metrics panel
        metrics_frame = ctk.CTkFrame(self.results_screen, fg_color="transparent", height=scaled_size(140))
        metrics_frame.pack(fill='x', padx=scaled_size(40), pady=scaled_size(25))
        metrics_frame.pack_propagate(False)

        # Diversity score card
        diversity_card = ctk.CTkFrame(
            metrics_frame,
            fg_color=self.colors['bg_tertiary'],
            corner_radius=15
        )
        diversity_card.place(relx=0.35, rely=0.5, anchor='center', relwidth=0.25, relheight=1)

        ctk.CTkLabel(
            diversity_card,
            text="Diversity Score",
            font=scaled_font(22),
            text_color=self.colors['text_secondary']
        ).pack(pady=(20, 10))

        self.diversity_label = ctk.CTkLabel(
            diversity_card,
            text="--",
            font=scaled_font(56, "bold"),
            text_color=self.colors['accent']
        )
        self.diversity_label.pack()

        # Distance score card
        distance_card = ctk.CTkFrame(
            metrics_frame,
            fg_color=self.colors['bg_tertiary'],
            corner_radius=15
        )
        distance_card.place(relx=0.65, rely=0.5, anchor='center', relwidth=0.25, relheight=1)

        ctk.CTkLabel(
            distance_card,
            text="Avg. Distance",
            font=scaled_font(22),
            text_color=self.colors['text_secondary']
        ).pack(pady=(20, 10))

        self.distance_label = ctk.CTkLabel(
            distance_card,
            text="--",
            font=scaled_font(56, "bold"),
            text_color=self.colors['success']
        )
        self.distance_label.pack()

        # Results table frame
        self.results_table_frame = ctk.CTkFrame(
            self.results_screen,
            fg_color=self.colors['bg_secondary'],
            corner_radius=15
        )
        self.results_table_frame.pack(fill='both', expand=True, padx=40, pady=(0, 40))

        # Placeholder
        self.results_placeholder = ctk.CTkLabel(
            self.results_table_frame,
            text="Generate counterfactuals to see results",
            font=scaled_font(22),
            text_color=self.colors['text_dim']
        )
        self.results_placeholder.pack(expand=True)

    def switch_results_mode(self, mode):
        """Switch between CoDeC and DiCE results"""
        self.current_results_mode = mode

        # Update button states
        if mode == "codec":
            self.codec_btn.configure(fg_color=self.colors['accent'])
            self.dice_btn.configure(fg_color=self.colors['bg_tertiary'])
            self.title_label.configure(text="CoDeC Results")
        else:
            self.dice_btn.configure(fg_color=self.colors['accent'])
            self.codec_btn.configure(fg_color=self.colors['bg_tertiary'])
            self.title_label.configure(text="DiCE Results")

        # Show the appropriate results if they exist
        if self.stored_results:
            self.display_current_results()

    def display_current_results(self):
        """Display results based on current mode"""
        if self.current_results_mode in self.stored_results:
            data = self.stored_results[self.current_results_mode]

            # If displaying CoDeC results, reorder them to match DiCE
            if self.current_results_mode == 'codec' and 'dice' in self.stored_results:
                exp_random = self.stored_results.get('exp_random')
                dataset_key = self.dataset_path.get()

                if exp_random is not None and dataset_key in self.cached_transformers:
                    transformer = self.cached_transformers[dataset_key]

                    # Reorder CoDeC CFs to match DiCE
                    codec_reordered, codec_distances_reordered = self.reorder_codec_by_closest_dice(
                        self.stored_results['dice'],
                        self.stored_results['codec'],
                        exp_random,
                        transformer
                    )

                    # Create reordered data for display
                    data = {
                        **data,
                        'counterfactuals': codec_reordered,
                        'distances': codec_distances_reordered
                    }

            # Update metrics
            self.diversity_label.configure(text=f"{data['diversity_score']:.3f}")
            avg_distance = sum(data['distances']) / len(data['distances']) if data['distances'] else 0
            self.distance_label.configure(text=f"{avg_distance:.3f}")

            # Display results table
            self.display_results(data)

    def show_input_screen(self):
        """Show input screen"""
        self.results_screen.pack_forget()
        self.input_screen.pack(fill='both', expand=True)
        self.current_screen = "input"

    def show_results_screen(self):
        """Show results screen"""
        self.input_screen.pack_forget()
        self.results_screen.pack(fill='both', expand=True)
        self.current_screen = "results"

    def browse_dataset(self):
        """Browse for dataset file"""
        initial_dir = os.path.join(os.getcwd(), 'data')

        filename = filedialog.askopenfilename(
            title="Select Dataset CSV",
            initialdir=initial_dir,
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filename:
            try:
                self.dataset_path.set(filename)
                self.update_status("Loading dataset...", self.colors['warning'])
                self.preview_btn.configure(state="normal")

                # Load the dataset and update UI
                self.load_dataset_preview()

            except Exception as e:
                print(f"Browse dataset error: {e}")  # Debug print
                self.update_status(f"Error loading dataset: {str(e)}", self.colors['error'])

    def browse_constraints(self):
        """Browse for constraints file"""
        initial_dir = os.path.join(os.getcwd(), 'constraints')

        filename = filedialog.askopenfilename(
            title="Select Constraints File",
            initialdir=initial_dir,
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )
        if filename:
            self.constraints_path.set(filename)
            self.update_status("Constraints loaded", self.colors['success'])
            self.constraints_preview_btn.configure(state="normal")

    def preview_dataset(self):
        """Show dataset preview dialog"""
        if hasattr(self, 'df'):
            DatasetPreviewDialog(self.root, self.df, self.colors)

    def update_status(self, text, color):
        """Update status indicator"""
        self.status_indicator.configure(text=f"● {text}", text_color=color)

    def update_cf_count(self, value):
        """Update counterfactual count label"""
        self.cf_count_label.configure(text=str(int(value)))
        self.num_counterfactuals.set(str(int(value)))

    def set_initial_instance_defaults(self, default_values):
        """Set default values for initial instance inputs"""
        if not hasattr(self, 'initial_point_widgets') or not self.initial_point_widgets:
            return

        for column, value in default_values.items():
            if column in self.initial_point_widgets:
                widget = self.initial_point_widgets[column]
                if isinstance(widget, ctk.CTkComboBox):
                    widget.set(value)
                else:  # CTkEntry
                    widget.delete(0, 'end')
                    widget.insert(0, value)

    def load_dataset_preview(self):
        """Load and display dataset preview"""
        try:
            # Load dataset
            self.df = pd.read_csv(self.dataset_path.get())

            # Clear any existing error status
            self.update_status("Dataset loaded", self.colors['success'])

            # Update initial instance configuration
            self.setup_initial_instance_inputs()

            # Update features list
            self.setup_features_selection()

            dataset_name = self.dataset_path.get()

            if "nyhouse" in dataset_name:
                # Set defaults for nyhouse dataset
                default_values = {
                    'type': 'Condo_for_sale',
                    'beds': '0',
                    'bath': '0',
                    'propertysqft': '600',
                    'locality': 'New_York',
                    'sublocality': 'Manhattan'
                }
                self.set_initial_instance_defaults(default_values)
            elif "adult_small" in dataset_name:
                # Set defaults for adult dataset
                default_values = {
                    'Education': 'Bachelors',
                    'Edu_Num': '13',
                    'Occupation': 'Tech_support',
                    'Hours': '27',
                }
                self.set_initial_instance_defaults(default_values)
            elif "adult" in dataset_name:
                # Set defaults for adult dataset
                default_values = {
                    'age': '28',
                    'workclass': 'State_gov',
                    'education': 'Bachelors',
                    'education_num': '13',
                    # 'marital_status': 'Divorced',
                    'marital_status': 'Married_civ_spouse',
                    'occupation': 'Machine_op_inspct',
                    # 'relationship': 'Husband',
                    'relationship': 'Wife',
                    'race': 'Black',
                    'sex': 'Female',
                    # 'hours_per_week': '1',
                    'hours_per_week': '45',
                    'native_country': 'United_States'
                }
                self.set_initial_instance_defaults(default_values)

        except Exception as e:
            print(f"Dataset loading error: {e}")  # Debug print
            self.update_status(f"Error: {str(e)}", self.colors['error'])

    def setup_initial_instance_inputs(self):
        """Setup initial instance configuration inputs"""
        try:
            # Ensure initial_point_widgets exists
            if not hasattr(self, 'initial_point_widgets'):
                self.initial_point_widgets = {}

            # Clear existing widgets and references
            for widget in self.instance_content_frame.winfo_children():
                widget.destroy()

            # Clear the widgets dictionary to prevent stale references
            self.initial_point_widgets.clear()

            # Ensure we have a valid dataframe
            if not hasattr(self, 'df') or self.df is None:
                print("No dataframe available for initial instance setup")
                return

            # Create inputs for each feature
            for i, column in enumerate(self.df.columns):
                if column == 'label':
                    continue

                # Feature frame
                feature_frame = ctk.CTkFrame(self.instance_content_frame, fg_color="transparent")
                feature_frame.pack(fill='x', pady=8)

                # Feature label
                ctk.CTkLabel(
                    feature_frame,
                    text=column,
                    font=scaled_font(22, "bold"),
                    text_color=self.colors['text_secondary'],
                    width=150,
                    anchor='w'
                ).pack(side='left')

                if self.df[column].dtype == 'object':  # Categorical
                    unique_values = sorted(self.df[column].unique())
                    widget = ctk.CTkComboBox(
                        feature_frame,
                        values=unique_values,
                        font=scaled_font(18),
                        fg_color=self.colors['bg_primary'],
                        button_color=self.colors['accent'],
                        button_hover_color=self.colors['accent_hover'],
                        dropdown_fg_color=self.colors['bg_primary'],
                        dropdown_hover_color=self.colors['bg_tertiary'],
                        width=200
                    )
                    widget.set(unique_values[0])
                else:  # Continuous
                    min_val = int(self.df[column].min())
                    max_val = int(self.df[column].max())

                    input_container = ctk.CTkFrame(feature_frame, fg_color="transparent")
                    input_container.pack(side='right', padx=10)

                    widget = ctk.CTkEntry(
                        input_container,
                        font=scaled_font(18),
                        fg_color=self.colors['bg_primary'],
                        border_color=self.colors['border'],
                        border_width=2,
                        width=100
                    )
                    widget.insert(0, str(min_val))
                    widget.pack(side='left')

                    # Range label
                    range_label = ctk.CTkLabel(
                        input_container,
                        text=f"[{min_val}-{max_val}]",
                        font=scaled_font(16),
                        text_color=self.colors['text_dim']
                    )
                    range_label.pack(side='left', padx=(10, 0))

                widget.pack(side='right', padx=10)
                self.initial_point_widgets[column] = widget

        except Exception as e:
            print(f"Error in setup_initial_instance_inputs: {e}")
            # Initialize empty dict if there's an error
            if not hasattr(self, 'initial_point_widgets'):
                self.initial_point_widgets = {}
            self.update_status(f"Error setting up inputs: {str(e)}", self.colors['error'])

    def setup_features_selection(self):
        """Setup immutable features selection"""
        try:
            # Ensure feature_checkboxes exists
            if not hasattr(self, 'feature_checkboxes'):
                self.feature_checkboxes = {}

            # Clear existing widgets and references
            for widget in self.features_content_frame.winfo_children():
                widget.destroy()

            # Clear the checkboxes dictionary to prevent stale references
            self.feature_checkboxes.clear()

            # Ensure we have a valid dataframe
            if not hasattr(self, 'df') or self.df is None:
                print("No dataframe available for features selection")
                return

            # Create grid layout
            columns = 2
            row = 0
            col = 0

            for column in self.df.columns:
                if column == 'label':
                    continue

                var = tk.BooleanVar(value=False)

                checkbox = ctk.CTkCheckBox(
                    self.features_content_frame,
                    text=column,
                    font=scaled_font(22, "bold"),
                    variable=var,
                    fg_color=self.colors['accent'],
                    hover_color=self.colors['accent_hover'],
                    border_color=self.colors['border'],
                    checkmark_color=self.colors['bg_primary']
                )
                checkbox.grid(row=row, column=col, sticky='w', padx=20, pady=5)

                self.feature_checkboxes[column] = var

                col += 1
                if col >= columns:
                    col = 0
                    row += 1

        except Exception as e:
            print(f"Error in setup_features_selection: {e}")
            # Initialize empty dict if there's an error
            if not hasattr(self, 'feature_checkboxes'):
                self.feature_checkboxes = {}
            self.update_status(f"Error setting up features: {str(e)}", self.colors['error'])

    def get_initial_point(self):
        """Get initial point values"""
        if not hasattr(self, 'initial_point_widgets') or not self.initial_point_widgets:
            return None

        # Verify all widgets are still valid
        try:
            initial_point = {'label': 0}

            for column, widget in self.initial_point_widgets.items():
                if isinstance(widget, ctk.CTkComboBox):
                    value = widget.get()
                    if not value:  # Handle empty combobox
                        return None
                    initial_point[column] = value
                else:
                    value = widget.get()
                    if not value:  # Handle empty entry
                        return None
                    try:
                        initial_point[column] = float(value)
                    except ValueError:
                        initial_point[column] = value

            return initial_point

        except Exception as e:
            print(f"Error getting initial point: {e}")
            return None

    def preview_constraints(self):
        """Show constraints preview and editing dialog"""
        if not self.constraints_path.get():
            self.update_status("No constraints file selected", self.colors['error'])
            return

        try:
            # Load constraints from file
            with open(self.constraints_path.get(), 'r', encoding='utf-8') as f:
                constraints_text = f.read().strip()

            # Clean up garbled characters and normalize symbols
            constraints_text = self.clean_constraints_text(constraints_text)

            # Parse constraints
            constraints_list = [line.strip() for line in constraints_text.split('\n') if line.strip()]

            # Create constraints dialog
            self.show_constraints_dialog(constraints_list)

        except Exception as e:
            self.update_status(f"Error loading constraints: {str(e)}", self.colors['error'])

    def clean_constraints_text(self, text):
        """Clean up garbled characters in constraints text"""
        # Common character replacements for constraint symbols
        replacements = {
            'â': '¬',  # Fix negation symbol
            '∧': '∧',  # Keep logical AND
            '∨': '∨',  # Keep logical OR
            '≤': '<=',  # Convert to standard less-than-or-equal
            '≥': '>=',  # Convert to standard greater-than-or-equal
            '≠': '!=',  # Convert to standard not-equal
            '¬': '¬',  # Keep negation symbol
            # Handle various encodings of logical AND
            'âˆ§': '∧',
            'â^': '∧',
            # Handle various encodings of negation
            'Â¬': '¬',
            'â¬': '¬',
        }

        cleaned_text = text
        for old, new in replacements.items():
            cleaned_text = cleaned_text.replace(old, new)

        return cleaned_text

    def show_constraints_dialog(self, constraints_list):
        """Show comprehensive constraints management dialog - FIXED VERSION"""
        # Create dialog window
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Constraints Manager")
        dialog.geometry("1400x900")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog on screen
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (1400 // 2)
        y = (dialog.winfo_screenheight() // 2) - (900 // 2)
        dialog.geometry(f"1400x900+{x}+{y}")

        # Configure dialog appearance
        dialog.configure(fg_color=self.colors['bg_primary'])

        # Store dialog reference for button callbacks
        self.constraints_dialog = dialog

        # Main container - leave space for bottom buttons
        main_container = ctk.CTkFrame(dialog, fg_color="transparent")
        main_container.pack(fill='both', expand=True, padx=20, pady=(20, 80))  # Bottom padding for buttons

        # Title
        title_label = ctk.CTkLabel(
            main_container,
            text="🔧 Constraints Manager",
            font=scaled_font(28, "bold"),
            text_color=self.colors['text_primary']
        )
        title_label.pack(pady=(0, 20))

        # Create two-panel layout
        panels_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        panels_frame.pack(fill='both', expand=True)

        # Left panel - Existing constraints
        left_panel = ctk.CTkFrame(panels_frame, fg_color=self.colors['bg_secondary'], corner_radius=15)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Left panel header
        ctk.CTkLabel(
            left_panel,
            text="📋 Current Constraints",
            font=scaled_font(20, "bold"),
            text_color=self.colors['text_primary']
        ).pack(pady=15)

        # Constraints list frame
        constraints_frame = ctk.CTkScrollableFrame(
            left_panel,
            fg_color=self.colors['bg_primary'],
            corner_radius=10
        )
        constraints_frame.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Store constraints for editing
        self.dialog_constraints = constraints_list.copy()

        # Populate constraints list
        self.update_constraints_display(constraints_frame)

        # Right panel - Add new constraints
        right_panel = ctk.CTkFrame(panels_frame, fg_color=self.colors['bg_secondary'], corner_radius=15)
        right_panel.pack(side='right', fill='both', expand=True, padx=(10, 0))

        # Right panel header
        ctk.CTkLabel(
            right_panel,
            text="➕ Add New Constraint",
            font=scaled_font(20, "bold"),
            text_color=self.colors['text_primary']
        ).pack(pady=15)

        # Fixed control buttons frame at top
        controls_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        controls_frame.pack(fill='x', padx=15, pady=(0, 10))

        # Add component button (fixed position)
        add_comp_btn = AnimatedButton(
            controls_frame,
            text="➕ Add Component",
            width=150,
            height=35,
            font=scaled_font(12, "bold"),
            fg_color=self.colors['accent_secondary'],
            hover_color="#cc0056",
            command=None  # Will be set later
        )
        add_comp_btn.pack(side='left')

        # Add constraint button (fixed position)
        add_constraint_btn = AnimatedButton(
            controls_frame,
            text="➕ Add Constraint",
            width=150,
            height=35,
            font=scaled_font(12, "bold"),
            fg_color=self.colors['accent'],
            hover_color=self.colors['accent_hover'],
            command=None  # Will be set later
        )
        add_constraint_btn.pack(side='right')

        # Constraint builder frame (scrollable for components)
        builder_frame = ctk.CTkScrollableFrame(
            right_panel,
            fg_color=self.colors['bg_primary'],
            corner_radius=10
        )
        builder_frame.pack(fill='both', expand=True, padx=15, pady=(0, 15))

        # Initialize simplified constraint builder
        self.setup_simplified_constraint_builder(builder_frame, constraints_frame, add_comp_btn, add_constraint_btn)

        # Bottom buttons frame - FIXED: Create outside main_container and always visible
        bottom_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        bottom_frame.pack(fill='x', side='bottom', padx=20, pady=20)

        # Save and close buttons
        save_btn = AnimatedButton(
            bottom_frame,
            text="💾 Save Constraints",
            width=180,
            height=45,
            font=scaled_font(16, "bold"),
            fg_color=self.colors['success'],
            hover_color="#28a745",
            command=lambda: self.save_constraints_and_close(dialog)
        )
        save_btn.pack(side='left')

        cancel_btn = AnimatedButton(
            bottom_frame,
            text="❌ Cancel",
            width=120,
            height=45,
            font=scaled_font(16, "bold"),
            fg_color=self.colors['error'],
            hover_color="#dc3545",
            command=dialog.destroy
        )
        cancel_btn.pack(side='right')

    def save_constraints_and_close(self, dialog):
        """Save constraints back to file and close dialog"""
        try:
            # Clean constraints before saving to ensure proper formatting
            cleaned_constraints = []
            for constraint in self.dialog_constraints:
                # Ensure proper encoding and symbols
                cleaned = constraint.replace('∧', '∧').replace('¬', '¬')
                cleaned_constraints.append(cleaned)

            with open(self.constraints_path.get(), 'w', encoding='utf-8') as f:
                f.write('\n'.join(cleaned_constraints))

            self.update_status("Constraints saved successfully", self.colors['success'])
            dialog.destroy()

        except Exception as e:
            self.update_status(f"Error saving constraints: {str(e)}", self.colors['error'])

    def setup_simplified_constraint_builder(self, parent, constraints_display, add_comp_btn, add_constraint_btn):
        """Setup simplified constraint builder interface"""
        # Get dataset columns if available
        columns = []
        if hasattr(self, 'df') and self.df is not None:
            columns = [col for col in self.df.columns if col != 'label']

        # Binary constraint components list
        components = []

        def add_component():
            comp_frame = ctk.CTkFrame(parent, fg_color=self.colors['bg_tertiary'], corner_radius=8)
            comp_frame.pack(fill='x', pady=5)

            # Component header
            header_frame = ctk.CTkFrame(comp_frame, fg_color="transparent")
            header_frame.pack(fill='x', padx=10, pady=5)

            ctk.CTkLabel(
                header_frame,
                text=f"Component {len(components) + 1}:",
                font=scaled_font(14, "bold"),
                text_color=self.colors['text_primary']
            ).pack(side='left')

            # Remove button - using StaticButton to prevent hover animation
            StaticButton(
                header_frame,
                text="🗑️",
                width=30,
                height=25,
                font=scaled_font(12),
                fg_color=self.colors['error'],
                hover_color="#dc3545",
                command=lambda: remove_component(comp_frame)
            ).pack(side='right')

            # Main constraint line
            constraint_frame = ctk.CTkFrame(comp_frame, fg_color="transparent")
            constraint_frame.pack(fill='x', padx=10, pady=5)

            # Left side - first attribute
            left_combo = ctk.CTkComboBox(
                constraint_frame,
                values=["t0." + col for col in columns] + ["t1." + col for col in columns],
                font=scaled_font(12),
                width=140
            )
            if columns:
                left_combo.set("t0." + columns[0])
            left_combo.pack(side='left', padx=5)

            # Operator
            op_combo = ctk.CTkComboBox(
                constraint_frame,
                values=["==", "!=", "<=", ">=", "<", ">"],
                font=scaled_font(12),
                width=60
            )
            op_combo.set("==")
            op_combo.pack(side='left', padx=5)

            # Right side - attribute or value
            type_var = tk.StringVar(value="attribute")

            # Attribute combobox
            attr_combo = ctk.CTkComboBox(
                constraint_frame,
                values=["t0." + col for col in columns] + ["t1." + col for col in columns],
                font=scaled_font(12),
                width=140
            )
            if columns:
                attr_combo.set("t1." + columns[0])

            # Value inputs (entry and combobox for categorical)
            value_entry = ctk.CTkEntry(constraint_frame, font=scaled_font(12), width=140)
            value_combo = ctk.CTkComboBox(constraint_frame, values=[], font=scaled_font(12), width=140)

            # Show attribute initially
            attr_combo.pack(side='left', padx=5)

            def get_attribute_from_reference(attr_ref):
                """Extract attribute name from t0.attr or t1.attr reference"""
                if '.' in attr_ref:
                    return attr_ref.split('.', 1)[1]
                return attr_ref

            def update_value_options():
                """Update value input options based on selected left attribute"""
                if type_var.get() == "value":
                    left_attr_ref = left_combo.get()
                    left_attr = get_attribute_from_reference(left_attr_ref)

                    # Clear any existing value widgets first
                    value_entry.pack_forget()
                    value_combo.pack_forget()

                    if left_attr and hasattr(self, 'df') and self.df is not None and left_attr in self.df.columns:
                        if self.df[left_attr].dtype == 'object':  # Categorical
                            unique_values = sorted(self.df[left_attr].unique().astype(str))
                            value_combo.configure(values=unique_values)
                            if unique_values:
                                value_combo.set(unique_values[0])
                            value_combo.pack(side='left', padx=5)
                        else:  # Numerical
                            value_entry.pack(side='left', padx=5)

            # Update when left attribute changes
            left_combo.configure(command=lambda x: update_value_options())

            # Toggle between attribute and value
            def toggle_type():
                if type_var.get() == "attribute":
                    # Switch to value mode
                    attr_combo.pack_forget()
                    type_var.set("value")
                    update_value_options()  # Call after setting type
                    toggle_btn.configure(text="📊 Attr")
                else:
                    # Switch to attribute mode
                    value_entry.pack_forget()
                    value_combo.pack_forget()
                    attr_combo.pack(side='left', padx=5)
                    type_var.set("attribute")
                    toggle_btn.configure(text="🔢 Val")

            toggle_btn = AnimatedButton(
                constraint_frame,
                text="🔢 Val",
                width=70,
                height=25,
                font=scaled_font(10),
                fg_color=self.colors['accent_secondary'],
                command=toggle_type
            )
            toggle_btn.pack(side='left', padx=5)

            def get_constraint_value():
                """Get properly formatted constraint value"""
                if type_var.get() == "attribute":
                    return attr_combo.get()
                else:
                    left_attr_ref = left_combo.get()
                    left_attr = get_attribute_from_reference(left_attr_ref)

                    if left_attr and hasattr(self, 'df') and self.df is not None and left_attr in self.df.columns:
                        if self.df[left_attr].dtype == 'object':  # Categorical
                            return f'"{value_combo.get()}"'
                        else:  # Numerical
                            return value_entry.get()
                    return value_entry.get()

            components.append({
                'frame': comp_frame,
                'left': left_combo,
                'operator': op_combo,
                'get_right_value': get_constraint_value,
                'type': type_var
            })

        def remove_component(frame):
            components[:] = [comp for comp in components if comp['frame'] != frame]
            frame.destroy()

        # Set button commands now that functions are defined
        add_comp_btn.configure(command=add_component)
        add_constraint_btn.configure(command=lambda: self.add_constraint_from_components(
            components,
            constraints_display
        ))

        # Add initial component
        add_component()

    def add_constraint_from_components(self, components, constraints_display):
        """Add a constraint from components - CLEANED UP"""
        if not components:
            return

        # Build constraint components
        parts = []
        for comp in components:
            left_value = comp['left'].get()
            operator = comp['operator'].get()
            right_value = comp['get_right_value']()

            if not right_value:
                continue

            part = f"{left_value} {operator} {right_value}"
            parts.append(part)

        if not parts:
            return

        # Join with AND if multiple parts
        if len(parts) == 1:
            constraint_body = parts[0]
        else:
            constraint_body = " ∧ ".join(parts)

        # Always add negation (¬)
        constraint = f"¬{{ {constraint_body} }}"

        # Add to constraints list
        self.dialog_constraints.append(constraint)

        # Update display
        self.update_constraints_display(constraints_display)

    def update_constraints_display(self, constraints_frame):
        """Update the constraints display"""
        # Clear existing widgets
        for widget in constraints_frame.winfo_children():
            widget.destroy()

        if not self.dialog_constraints:
            ctk.CTkLabel(
                constraints_frame,
                text="No constraints defined",
                font=scaled_font(14),
                text_color=self.colors['text_dim']
            ).pack(pady=20)
            return

        # Display each constraint
        for i, constraint in enumerate(self.dialog_constraints):
            constraint_frame = ctk.CTkFrame(constraints_frame, fg_color=self.colors['bg_tertiary'], corner_radius=8)
            constraint_frame.pack(fill='x', pady=5, padx=10)

            # Create inner frame for better layout control
            inner_frame = ctk.CTkFrame(constraint_frame, fg_color="transparent")
            inner_frame.pack(fill='x', padx=10, pady=8)

            # Remove button - fixed position, using StaticButton
            remove_btn = StaticButton(
                inner_frame,
                text="🗑️",
                width=35,
                height=30,
                font=scaled_font(12),
                fg_color=self.colors['error'],
                hover_color="#dc3545",
                command=lambda idx=i: self.remove_constraint(idx, constraints_frame)
            )
            remove_btn.pack(side='right', padx=(10, 0))

            # Constraint text - takes remaining space
            ctk.CTkLabel(
                inner_frame,
                text=f"{i + 1}. {constraint}",
                font=scaled_font(13),
                text_color=self.colors['text_primary'],
                anchor='w',
                wraplength=400,  # Allow text wrapping for long constraints
                justify='left'
            ).pack(side='left', fill='both', expand=True)

    def remove_constraint(self, index, constraints_display):
        """Remove a constraint from the list"""
        if 0 <= index < len(self.dialog_constraints):
            self.dialog_constraints.pop(index)
            self.update_constraints_display(constraints_display)

    def get_fixed_features(self):
        """Get selected immutable features"""
        if not hasattr(self, 'feature_checkboxes') or not self.feature_checkboxes:
            return []

        fixed = []
        try:
            for column, var in self.feature_checkboxes.items():
                if var.get():
                    fixed.append(column)
        except Exception as e:
            print(f"Error getting fixed features: {e}")
            return []
        return fixed

    def generate_counterfactuals(self):
        """Generate counterfactuals with modern loading animation"""
        try:
            # Validate inputs
            initial_instance = self.get_initial_point()
            if not initial_instance:
                self.update_status("Invalid initial instance", self.colors['error'])
                return

            if not self.constraints_path.get() or not self.dataset_path.get():
                self.update_status("Missing required files", self.colors['error'])
                return

            # Update status
            self.update_status("Generating...", self.colors['warning'])

            # Show loading overlay
            self.show_loading_overlay()

            # Start computation in separate thread
            computation_thread = threading.Thread(
                target=self.computation_worker,
                args=(
                    initial_instance,
                    self.constraints_path.get(),
                    self.dataset_path.get(),
                    self.get_fixed_features(),
                    int(self.num_counterfactuals.get())
                )
            )
            computation_thread.daemon = True
            computation_thread.start()

        except Exception as e:
            print(f"Error in generate_counterfactuals: {e}")
            self.update_status(f"Generation error: {str(e)}", self.colors['error'])

    def show_loading_overlay(self):
        """Show modern loading overlay"""
        self.loading_overlay = ctk.CTkFrame(
            self.root,
            fg_color=(self.colors['bg_primary'], self.colors['bg_primary']),
            corner_radius=0
        )
        self.loading_overlay.place(relx=0.5, rely=0.5, anchor='center', relwidth=1, relheight=1)

        # Loading content
        loading_content = ctk.CTkFrame(
            self.loading_overlay,
            fg_color=self.colors['bg_secondary'],
            corner_radius=20,
            width=400,
            height=200
        )
        loading_content.place(relx=0.5, rely=0.5, anchor='center')
        loading_content.pack_propagate(False)

        # Loading animation
        self.loading_animation_frame = ctk.CTkFrame(
            loading_content,
            fg_color="transparent",
            height=80
        )
        self.loading_animation_frame.pack(pady=(30, 20))

        # Create spinning circles
        self.create_loading_animation()

        # Loading text
        self.loading_text = ctk.CTkLabel(
            loading_content,
            text="Initializing CODEC...",
            font=scaled_font(20, "bold"),
            text_color=self.colors['text_primary']
        )
        self.loading_text.pack(pady=10)

        # Store animation state
        self._animation_active = True

    def create_loading_animation(self):
        """Create modern loading animation with animated circles"""
        canvas = tk.Canvas(
            self.loading_animation_frame,
            width=80,
            height=80,
            bg=self.colors['bg_secondary'],
            highlightthickness=0
        )
        canvas.pack()

        # Create multiple circles in a circular pattern
        self.loading_circles = []
        num_circles = 8
        for i in range(num_circles):
            angle = i * (360 / num_circles)
            x = 40 + 25 * math.cos(math.radians(angle))
            y = 40 + 25 * math.sin(math.radians(angle))

            circle = canvas.create_oval(
                x - 4, y - 4, x + 4, y + 4,
                fill=self.colors['accent'],
                outline=""
            )
            self.loading_circles.append(circle)

        self.loading_canvas = canvas
        self.animate_loading()

    def animate_loading(self, index=0):
        """Animate loading circles with smooth rotation effect"""
        if hasattr(self, 'loading_canvas') and hasattr(self, '_animation_active') and self._animation_active:
            try:
                num_circles = len(self.loading_circles)
                for i, circle in enumerate(self.loading_circles):
                    # Calculate opacity based on position relative to current index
                    opacity_index = (index - i) % num_circles
                    opacity = 1.0 - (opacity_index / num_circles)

                    # Create color with varying opacity (simulate by changing size and color intensity)
                    size = 2 + opacity * 4

                    # Get original position
                    angle = i * (360 / num_circles)
                    center_x = 40 + 25 * math.cos(math.radians(angle))
                    center_y = 40 + 25 * math.sin(math.radians(angle))

                    # Update circle
                    self.loading_canvas.coords(
                        circle,
                        center_x - size, center_y - size,
                        center_x + size, center_y + size
                    )

                    # Update color intensity based on opacity
                    if opacity > 0.7:
                        color = self.colors['accent']
                    elif opacity > 0.4:
                        color = self.colors['accent_hover']
                    else:
                        color = self.colors['bg_tertiary']

                    self.loading_canvas.itemconfig(circle, fill=color)

                self.root.after(150, lambda: self.animate_loading((index + 1) % num_circles))
            except tk.TclError:
                # Canvas was destroyed, stop animation
                self._animation_active = False

    def stop_loading_animation(self):
        """Stop the loading animation"""
        self._animation_active = False

    def computation_worker(self, initial_instance, constraints_path, dataset_path, fixed_features, num_counterfactuals):
        """Worker function for computation thread - uses REAL algorithm"""
        try:
            # Notify UI of progress with simplified stages
            self.computation_queue.put({'type': 'progress', 'text': 'CoDeC generating counterfactuals...'})

            # Prepare query instance as your algorithm expects
            query_df = pd.DataFrame(initial_instance, [0])
            # Remove any extra columns that were added for display
            columns_to_drop = []
            if 'label' in query_df.columns:
                columns_to_drop.append('label')
            if 'prediction' in query_df.columns:
                columns_to_drop.append('prediction')
            query_df = query_df.drop(columns_to_drop, axis=1, errors='ignore')

            results = code_counterfactuals(
                query_instances=query_df,
                constraints_path=constraints_path,
                dataset_path=dataset_path,
                fixed_feat=fixed_features,
                k=num_counterfactuals,
                model_cache=self.cached_models,
                transformer_cache=self.cached_transformers,
                constraints_cache=self.cached_constraints,
                progress_queue=self.computation_queue,
                min_iter=MIN_ITER_ADULT if 'adult' in self.dataset_path.get() else MIN_ITER_NY,
                max_iter=MAX_ITER_ADULT if 'adult' in self.dataset_path.get() else MAX_ITER_NY,
                lr=LR_ADULT if 'adult' in self.dataset_path.get() else LR_NY
            )

            # Extract both DiCE and CoDeC results
            codec_results = results['codec_results']
            dice_results = results['dice_results']
            exp_random = results.get('exp_random')
            constraint_data = results.get('constraint_data')  # ADD: Get constraint data

            # Send both results back to UI thread
            self.computation_queue.put({
                'type': 'complete',
                'data': {
                    'codec': {
                        'positive_samples': codec_results[0],
                        'counterfactuals': [row.to_dict() for _, row in codec_results[0].iterrows()],
                        'diversity_score': codec_results[1],
                        'distances': codec_results[2],
                        'initial_instance': initial_instance
                    },
                    'dice': {
                        'positive_samples': dice_results[0],
                        'counterfactuals': [row.to_dict() for _, row in dice_results[0].iterrows()],
                        'diversity_score': dice_results[1],
                        'distances': dice_results[2],
                        'initial_instance': initial_instance
                    },
                    'exp_random': exp_random,
                    'constraint_data': constraint_data  # ADD: Pass constraint data
                }
            })

        except Exception as e:
            import traceback
            self.computation_queue.put({
                'type': 'error',
                'error': f"{str(e)}\n{traceback.format_exc()}"
            })

    def check_computation_queue(self):
        """Check for messages from computation thread"""
        try:
            while True:
                message = self.computation_queue.get_nowait()
                if message['type'] == 'progress':
                    # Filter out unwanted progress messages - only show our main message
                    text = message['text']
                    # Skip BFS iteration messages and other detailed progress
                    if 'BFS' in text or 'iteration' in text or 'Loading' in text or 'Preparing' in text or 'Running' in text or 'project' in text or 'took' in text or 'initial' in text:
                        continue
                    self.update_progress(text)
                elif message['type'] == 'complete':
                    self.handle_computation_complete(message['data'])
                elif message['type'] == 'error':
                    self.handle_computation_error(message['error'])
        except queue.Empty:
            pass

        self.root.after(50, self.check_computation_queue)

    def update_progress(self, text):
        """Update progress text during computation"""
        if hasattr(self, 'loading_text'):
            self.loading_text.configure(text=text)

    def handle_computation_complete(self, data):
        """Handle completion of computation in UI thread"""
        try:
            # Stop animation and close loading window
            self.stop_loading_animation()
            if hasattr(self, 'loading_overlay'):
                self.loading_overlay.destroy()

            # Update status to Ready
            self.update_status("Ready", self.colors['success'])

            # Store both results AND exp_random AND constraint_data
            self.stored_results = {
                'codec': data['codec'],
                'dice': data['dice'],
                'exp_random': data.get('exp_random'),
                'constraint_data': data.get('constraint_data')  # ADD: Store constraint data
            }

            # Start with CoDeC results by default
            self.current_results_mode = "codec"
            self.switch_results_mode("codec")

            # Switch to results view
            self.show_results_screen()

        except Exception as e:
            self.handle_computation_error(str(e))

    def handle_computation_error(self, error):
        """Handle computation error"""
        # Stop animation
        self.stop_loading_animation()

        # Remove loading overlay
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.destroy()

        # Update status
        self.update_status("Error", self.colors['error'])

        # Show error message
        messagebox.showerror("Computation Error", f"Failed to generate counterfactuals:\n{error}")

    def display_results(self, data):
        """Display results in scrollable table format showing all attributes"""
        # Clear existing results - destroy from bottom up to avoid reference issues
        children = self.results_table_frame.winfo_children()
        for widget in reversed(children):
            widget.destroy()

        # CRITICAL: Update the frame to process the destroy events
        self.results_table_frame.update()

        # Create title
        title_label = ctk.CTkLabel(
            self.results_table_frame,
            text="Counterfactual Results Comparison",
            font=scaled_font(34, "bold"),
            text_color=self.colors['text_primary']
        )
        title_label.pack(pady=20)

        # Create main scrollable area
        main_scroll_frame = ctk.CTkFrame(self.results_table_frame, fg_color="transparent")
        main_scroll_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        # Create canvas for horizontal and vertical scrolling
        canvas = tk.Canvas(
            main_scroll_frame,
            bg=self.colors['bg_secondary'],
            highlightthickness=0
        )

        # Create scrollbars
        v_scrollbar = ctk.CTkScrollbar(
            main_scroll_frame,
            command=canvas.yview,
            fg_color=self.colors['bg_tertiary'],
            button_color=self.colors['accent'],
            button_hover_color=self.colors['accent_hover']
        )
        v_scrollbar.pack(side='right', fill='y')

        h_scrollbar = ctk.CTkScrollbar(
            main_scroll_frame,
            orientation='horizontal',
            command=canvas.xview,
            fg_color=self.colors['bg_tertiary'],
            button_color=self.colors['accent'],
            button_hover_color=self.colors['accent_hover']
        )
        h_scrollbar.pack(side='bottom', fill='x')

        canvas.pack(side='left', fill='both', expand=True)
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Create scrollable frame inside canvas
        scrollable_frame = ctk.CTkFrame(canvas, fg_color="transparent")
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')

        # Get all features (excluding label)
        features = [k for k in data['initial_instance'].keys() if k not in ['label', 'prediction']]

        # Calculate dynamic column widths based on content
        def calculate_column_width(header_text, values_list):
            """Calculate optimal column width based on header and content"""
            max_len = len(header_text)
            for value in values_list:
                if isinstance(value, float):
                    val_str = f"{value:.2f}" if not value.is_integer() else str(int(value))
                else:
                    val_str = str(value)
                val_str = f"→ {val_str}"
                max_len = max(max_len, len(val_str))
            # ✅ Scale the multiplier and padding for font size
            base_width = max_len * int(10 * FONT_SCALE) + int(30 * FONT_SCALE)
            return max(int(120 * FONT_SCALE), min(int(300 * FONT_SCALE), base_width))

        # Calculate widths for each column
        col_widths = {
            'Instance': scaled_size(140),
            'Distance': scaled_size(150)
        }

        # Get all values for each feature to calculate optimal width
        feature_widths = {}
        for feature in features:
            all_values = [data['initial_instance'].get(feature, 'N/A')]
            all_values.extend([cf.get(feature, 'N/A') for cf in data['counterfactuals']])
            header_text = feature.replace('_', ' ').title()
            feature_widths[feature] = calculate_column_width(header_text, all_values)

        # Calculate total table width - INCREASED SPACING
        num_columns = len(features) + 2
        padding_per_column = 30  # CHANGED from 20 to 30
        extra_buffer = 500  # CHANGED from 400 to 500
        total_width = col_widths['Instance'] + col_widths['Distance'] + sum(feature_widths.values()) + (
                num_columns * padding_per_column) + extra_buffer

        # Calculate total height needed
        num_rows = 1 + len(data['counterfactuals'])
        row_height = scaled_size(54)  # Increased from 54
        header_height = scaled_size(65)
        total_height = header_height + (num_rows * row_height) + 20

        # Create table container
        table_container = ctk.CTkFrame(scrollable_frame, fg_color="transparent", width=total_width, height=total_height)
        table_container.pack()
        table_container.pack_propagate(False)

        # Header row
        header_frame = ctk.CTkFrame(table_container, fg_color=self.colors['bg_primary'], corner_radius=10, height=60,
                                    width=total_width)
        header_frame.pack(fill='x', pady=(0, 5))
        header_frame.pack_propagate(False)

        # Create header with fixed positioning
        header_inner = ctk.CTkFrame(header_frame, fg_color="transparent", width=total_width)
        header_inner.pack(fill='both', expand=True, padx=10, pady=10)
        header_inner.pack_propagate(False)

        # Create headers - INCREASED SPACING
        current_x = 0

        instance_header = ctk.CTkLabel(
            header_inner,
            text="Instance",
            font=scaled_font(24, "bold"),
            text_color=self.colors['accent'],
            width=col_widths['Instance'],
            anchor='w'
        )

        instance_header.place(x=current_x, y=0)
        current_x += col_widths['Instance'] + 20  # CHANGED from 10 to 20

        for feature in features:
            width = feature_widths[feature]
            header_label = ctk.CTkLabel(
                header_inner,
                text=feature.replace('_', ' ').title(),
                font=scaled_font(24, "bold"),
                text_color=self.colors['text_secondary'],
                width=width,
                anchor='center'  # ✅ Changed from 'w' to 'center'
            )
            header_label.place(x=current_x, y=0)
            current_x += width + 20  # CHANGED from 10 to 20

        distance_header = ctk.CTkLabel(
            header_inner,
            text="Distance",
            font=scaled_font(24, "bold"),
            text_color=self.colors['accent'],
            width=col_widths['Distance'],
            anchor='center'  # ✅ Changed from 'w' to 'center'
        )
        distance_header.place(x=current_x, y=0)

        # Data rows container
        rows_container = ctk.CTkFrame(table_container, fg_color="transparent", width=total_width)
        rows_container.pack(fill='both', expand=True)

        # Build all_instances list
        all_instances = [('Initial', data['initial_instance'], None)]
        for i in range(len(data['counterfactuals'])):
            cf = data['counterfactuals'][i]
            distance = data['distances'][i] if i < len(data['distances']) else 0.0
            all_instances.append((f'Option {i + 1}', cf, distance))

        # Create rows - INCREASED SPACING
        for row_idx, (label_text, instance, distance) in enumerate(all_instances):
            # Determine row styling
            if row_idx == 0:
                row_color = self.colors['bg_tertiary']
                label_color = self.colors['text_primary']
            else:
                row_color = self.colors['bg_primary']
                label_color = self.colors['success']

            # Create row frame
            row_frame = ctk.CTkFrame(rows_container, fg_color=row_color, corner_radius=8, height=50, width=total_width)
            row_frame.pack(fill='x', pady=2)
            row_frame.pack_propagate(False)

            # Row inner frame
            row_inner = ctk.CTkFrame(row_frame, fg_color="transparent", width=total_width)
            row_inner.pack(fill='both', expand=True, padx=10, pady=8)
            row_inner.pack_propagate(False)

            # Create row content - INCREASED SPACING
            current_x = 0

            instance_label = ctk.CTkLabel(
                row_inner,
                text=label_text,
                font=scaled_font(22, "bold"),
                text_color=label_color,
                width=col_widths['Instance'],
                anchor='w'
            )
            instance_label.place(x=current_x, y=0)
            current_x += col_widths['Instance'] + 20  # CHANGED from 10 to 20

            for feature in features:
                width = feature_widths[feature]
                value = instance.get(feature, 'N/A')

                if isinstance(value, float):
                    if value.is_integer():
                        value_text = str(int(value))
                    else:
                        value_text = f"{value:.2f}"
                else:
                    value_text = str(value)

                changed = False
                if row_idx > 0:
                    initial_value = data['initial_instance'].get(feature)
                    changed = self.has_changed(value, initial_value)
                    if changed:
                        value_text = f"→ {value_text}"

                value_label = ctk.CTkLabel(
                    row_inner,
                    text=value_text,
                    font=scaled_font(22, "bold" if changed else "normal"),
                    text_color=self.colors['accent'] if changed else self.colors['text_primary'],
                    width=width,
                    anchor='center'  # ✅ Center-aligned
                )
                value_label.place(x=current_x, y=0)
                current_x += width + 20  # CHANGED from 10 to 20

            dist_text = f"{distance:.3f}" if distance is not None else "--"
            dist_label = ctk.CTkLabel(
                row_inner,
                text=dist_text,
                font=scaled_font(22, "bold"),
                text_color=self.colors['warning'] if distance is not None else self.colors['text_dim'],
                width=col_widths['Distance'],
                anchor='center'  # ✅ Center-aligned
            )
            dist_label.place(x=current_x, y=0)

        # CRITICAL: Force the scrollable frame to update its size
        scrollable_frame.update_idletasks()

        # Configure scroll region AFTER all widgets are created and updated
        def configure_scroll():
            # Get the bounding box of all widgets in the scrollable frame
            canvas.update_idletasks()
            bbox = canvas.bbox("all")
            if bbox:
                # Set scroll region to the actual content size
                canvas.configure(scrollregion=bbox)
            else:
                # Fallback to calculated dimensions
                canvas.configure(scrollregion=(0, 0, total_width, total_height))

            # Ensure canvas window width is set
            canvas.itemconfig(canvas_window, width=max(total_width, canvas.winfo_width()))

        # Configure immediately and after a short delay
        configure_scroll()
        self.root.after(50, configure_scroll)
        self.root.after(100, configure_scroll)

        # Reset scroll position
        self.root.after(150, lambda: canvas.xview_moveto(0))
        self.root.after(150, lambda: canvas.yview_moveto(0))

        # Mouse wheel handling
        def on_mousewheel(event):
            shift = (event.state & 0x1) != 0
            ctrl = (event.state & 0x4) != 0
            if shift or ctrl:
                canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Bind mouse events
        def bind_mousewheel(widget):
            widget.bind("<MouseWheel>", on_mousewheel)
            widget.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
            widget.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
            widget.bind("<Shift-Button-4>", lambda e: canvas.xview_scroll(-1, "units"))
            widget.bind("<Shift-Button-5>", lambda e: canvas.xview_scroll(1, "units"))
            for child in widget.winfo_children():
                bind_mousewheel(child)

        bind_mousewheel(canvas)
        bind_mousewheel(scrollable_frame)

        # Keyboard navigation
        canvas.focus_set()
        canvas.bind("<Left>", lambda e: canvas.xview_scroll(-1, "units"))
        canvas.bind("<Right>", lambda e: canvas.xview_scroll(1, "units"))
        canvas.bind("<Up>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind("<Down>", lambda e: canvas.yview_scroll(1, "units"))
        canvas.bind("<Prior>", lambda e: canvas.yview_scroll(-1, "pages"))
        canvas.bind("<Next>", lambda e: canvas.yview_scroll(1, "pages"))
        canvas.bind("<Home>", lambda e: canvas.xview_moveto(0))
        canvas.bind("<End>", lambda e: canvas.xview_moveto(1))

    def has_changed(self, value1, value2):
        """Check if two values are different"""
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            return abs(value1 - value2) > 0.001
        return str(value1) != str(value2)

    def show_comparison_dialog(self):
        """Show unified comparison with DiCE and CoDeC CFs side-by-side"""
        if not self.stored_results or 'codec' not in self.stored_results or 'dice' not in self.stored_results:
            messagebox.showinfo("No Results", "Please generate counterfactuals first to compare results.")
            return

        # Create dialog
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("CoDeC vs DiCE Comparison")
        dialog.geometry("1600x1000")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (1600 // 2)
        y = (dialog.winfo_screenheight() // 2) - (1000 // 2)
        dialog.geometry(f"1600x1000+{x}+{y}")

        dialog.configure(fg_color=self.colors['bg_primary'])

        # Header
        header_frame = ctk.CTkFrame(dialog, fg_color=self.colors['bg_secondary'], height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)

        ctk.CTkLabel(
            header_frame,
            text="⚖️ Comparison",
            font=scaled_font(36, "bold"),
            text_color=self.colors['text_primary']
        ).pack(pady=20)

        # Metrics comparison at top
        metrics_container = ctk.CTkFrame(dialog, fg_color="transparent", height=scaled_size(160))
        metrics_container.pack(fill='x', padx=scaled_size(40), pady=scaled_size(25))
        metrics_container.pack_propagate(False)

        # DiCE metrics (left)
        dice_metrics = ctk.CTkFrame(metrics_container, fg_color=self.colors['bg_secondary'], corner_radius=15)
        dice_metrics.pack(side='left', fill='both', expand=True, padx=(0, scaled_size(15)))

        ctk.CTkLabel(
            dice_metrics,
            text="DiCE Results",
            font=scaled_font(28, "bold"),
            text_color=self.colors['warning']
        ).pack(pady=(scaled_size(20), scaled_size(10)))

        dice_stats = ctk.CTkFrame(dice_metrics, fg_color="transparent")
        dice_stats.pack(fill='x', padx=scaled_size(20), pady=(0, scaled_size(20)))

        dice_data = self.stored_results['dice']
        dice_avg_dist = sum(dice_data['distances']) / len(dice_data['distances']) if dice_data['distances'] else 0

        ctk.CTkLabel(
            dice_stats,
            text=f"Diversity: {dice_data['diversity_score']:.3f} | Avg. Distance: {dice_avg_dist:.3f}",
            font=scaled_font(20, "bold"),
            text_color=self.colors['text_primary']
        ).pack()

        # CoDeC metrics (right)
        codec_metrics = ctk.CTkFrame(metrics_container, fg_color=self.colors['bg_secondary'], corner_radius=15)
        codec_metrics.pack(side='right', fill='both', expand=True, padx=(scaled_size(15), 0))

        ctk.CTkLabel(
            codec_metrics,
            text="CoDeC Results",
            font=scaled_font(28, "bold"),
            text_color=self.colors['accent']
        ).pack(pady=(scaled_size(20), scaled_size(10)))

        codec_stats = ctk.CTkFrame(codec_metrics, fg_color="transparent")
        codec_stats.pack(fill='x', padx=scaled_size(20), pady=(0, scaled_size(20)))

        codec_data = self.stored_results['codec']
        codec_avg_dist = sum(codec_data['distances']) / len(codec_data['distances']) if codec_data['distances'] else 0

        ctk.CTkLabel(
            codec_stats,
            text=f"Diversity: {codec_data['diversity_score']:.3f} | Avg. Distance: {codec_avg_dist:.3f}",
            font=scaled_font(20, "bold"),
            text_color=self.colors['text_primary']
        ).pack()

        # Get constraint data
        constraint_data = self.stored_results.get('constraint_data')
        if constraint_data:
            cons_function = constraint_data['cons_function']
            constraints = constraint_data['constraints']
            cons_feat = constraint_data['cons_feat']
            dataset = constraint_data['dataset']

        # Reorder CoDeC CFs to match closest DiCE CFs
        exp_random = self.stored_results.get('exp_random')
        transformer = self.cached_transformers[self.dataset_path.get()]

        if exp_random is not None:
            codec_reordered, codec_distances_reordered = self.reorder_codec_by_closest_dice(
                self.stored_results['dice'],
                self.stored_results['codec'],
                exp_random,
                transformer
            )

            # Create reordered results for display
            reordered_results = {
                'dice': self.stored_results['dice'],
                'codec': {
                    **self.stored_results['codec'],
                    'counterfactuals': codec_reordered,
                    'distances': codec_distances_reordered
                }
            }
        else:
            reordered_results = self.stored_results

        # Main scrollable comparison table
        self.create_unified_comparison_table(dialog, reordered_results)

        # Close button
        close_btn = AnimatedButton(
            dialog,
            text="Close",
            width=150,
            height=45,
            font=scaled_font(16, "bold"),
            fg_color=self.colors['bg_tertiary'],
            hover_color=self.colors['accent_hover'],
            command=dialog.destroy
        )
        close_btn.pack(pady=20)

    def create_unified_comparison_table(self, parent, results):
        """Create a unified table showing DiCE and CoDeC CFs side-by-side"""
        dice_data = results['dice']
        codec_data = results['codec']

        # Table container
        table_frame = ctk.CTkFrame(parent, fg_color=self.colors['bg_secondary'], corner_radius=15)
        table_frame.pack(fill='both', expand=True, padx=40, pady=(0, 20))

        # Create canvas for scrolling
        canvas = tk.Canvas(
            table_frame,
            bg=self.colors['bg_secondary'],
            highlightthickness=0
        )

        # Scrollbars
        v_scrollbar = ctk.CTkScrollbar(
            table_frame,
            command=canvas.yview,
            fg_color=self.colors['bg_tertiary'],
            button_color=self.colors['accent'],
            button_hover_color=self.colors['accent_hover']
        )
        v_scrollbar.pack(side='right', fill='y')

        h_scrollbar = ctk.CTkScrollbar(
            table_frame,
            orientation='horizontal',
            command=canvas.xview,
            fg_color=self.colors['bg_tertiary'],
            button_color=self.colors['accent'],
            button_hover_color=self.colors['accent_hover']
        )
        h_scrollbar.pack(side='bottom', fill='x')

        canvas.pack(side='left', fill='both', expand=True)
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Scrollable frame
        scrollable_frame = ctk.CTkFrame(canvas, fg_color="transparent")
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')

        # Get features
        features = [k for k in dice_data['initial_instance'].keys() if k not in ['label', 'prediction']]

        # Calculate column widths
        def calculate_column_width(header_text, values_list):
            max_len = len(header_text)
            for value in values_list:
                if isinstance(value, float):
                    val_str = f"{value:.2f}" if not value.is_integer() else str(int(value))
                else:
                    val_str = str(value)
                val_str = f"→ {val_str}"
                max_len = max(max_len, len(val_str))
            return max(120, min(300, max_len * 10 + 30))

        col_widths = {
            'Method': scaled_size(180),
            'Distance': scaled_size(160)
        }

        # Calculate total width
        num_columns = len(features) + 2
        padding_per_column = scaled_size(50)  # Increased spacing
        extra_buffer = scaled_size(600)

        feature_widths = {}
        for feature in features:
            all_values = [dice_data['initial_instance'].get(feature, 'N/A')]
            all_values.extend([cf.get(feature, 'N/A') for cf in dice_data['counterfactuals']])
            all_values.extend([cf.get(feature, 'N/A') for cf in codec_data['counterfactuals']])
            header_text = feature.replace('_', ' ').title()
            feature_widths[feature] = calculate_column_width(header_text, all_values)

        # Calculate total width
        num_columns = len(features) + 2
        padding_per_column = scaled_size(45)  # Increased from 30
        extra_buffer = scaled_size(600)
        total_width = col_widths['Method'] + col_widths['Distance'] + sum(feature_widths.values()) + (
                num_columns * padding_per_column) + extra_buffer

        # Calculate total height - FIXED: Properly account for all rows
        num_cfs = max(len(dice_data['counterfactuals']), len(codec_data['counterfactuals']))
        # 1 initial row + (2 rows per CF pair) + separators
        num_rows = 1 + (num_cfs * 2)
        row_height = scaled_size(60)
        header_height = scaled_size(75)
        separator_height = 9  # 2px height + 5px padding top and bottom
        total_separators = num_cfs - 1 if num_cfs > 0 else 0
        total_height = header_height + (num_rows * row_height) + (total_separators * separator_height) + 50

        # Table container - REMOVED height constraint to let it expand
        table_container = ctk.CTkFrame(scrollable_frame, fg_color="transparent", width=total_width)
        table_container.pack()

        # Header row
        header_frame = ctk.CTkFrame(table_container, fg_color=self.colors['bg_primary'], corner_radius=10, height=60,
                                    width=total_width)
        header_frame.pack(fill='x', pady=(0, 5))
        header_frame.pack_propagate(False)

        header_inner = ctk.CTkFrame(header_frame, fg_color="transparent", width=total_width)
        header_inner.pack(fill='both', expand=True, padx=10, pady=10)
        header_inner.pack_propagate(False)

        # Create headers
        current_x = 0

        ctk.CTkLabel(
            header_inner,
            text="Method",
            font=scaled_font(24, "bold"),
            text_color=self.colors['text_primary'],
            width=col_widths['Method'],
            anchor='w'
        ).place(x=current_x, y=0)
        current_x += col_widths['Method'] + scaled_size(20)

        for feature in features:
            width = feature_widths[feature]
            ctk.CTkLabel(
                header_inner,
                text=feature.replace('_', ' ').title(),
                font=scaled_font(24, "bold"),
                text_color=self.colors['text_secondary'],
                width=width,
                anchor='center'  # ✅ Center-aligned
            ).place(x=current_x, y=0)
            current_x += width + scaled_size(40)

        ctk.CTkLabel(
            header_inner,
            text="Distance",
            font=scaled_font(24, "bold"),
            text_color=self.colors['text_primary'],
            width=col_widths['Distance'],
            anchor='center'  # ✅ Center-aligned
        ).place(x=current_x, y=0)

        # Rows container
        rows_container = ctk.CTkFrame(table_container, fg_color="transparent", width=total_width)
        rows_container.pack(fill='both', expand=True)

        # Initial instance row
        self.create_comparison_row(
            rows_container, "Initial", dice_data['initial_instance'], None,
            features, feature_widths, col_widths, total_width,
            self.colors['bg_tertiary'], self.colors['text_primary'], None
        )

        # CF pairs
        num_cfs = max(len(dice_data['counterfactuals']), len(codec_data['counterfactuals']))

        for i in range(num_cfs):
            # DiCE CF
            if i < len(dice_data['counterfactuals']):
                dice_cf = dice_data['counterfactuals'][i]
                dice_dist = dice_data['distances'][i] if i < len(dice_data['distances']) else 0.0
                self.create_comparison_row(
                    rows_container, f"CF {i + 1} (DiCE)", dice_cf, dice_dist,
                    features, feature_widths, col_widths, total_width,
                    "#1a1200", self.colors['warning'], dice_data['initial_instance']
                )

            # CoDeC CF
            if i < len(codec_data['counterfactuals']):
                codec_cf = codec_data['counterfactuals'][i]
                codec_dist = codec_data['distances'][i] if i < len(codec_data['distances']) else 0.0
                self.create_comparison_row(
                    rows_container, f"CF {i + 1} (CoDeC)", codec_cf, codec_dist,
                    features, feature_widths, col_widths, total_width,
                    "#001a1a", self.colors['accent'], codec_data['initial_instance']
                )

            # Separator between CF pairs
            if i < num_cfs - 1:
                separator = ctk.CTkFrame(rows_container, fg_color=self.colors['border'], height=2)
                separator.pack(fill='x', pady=5)

        # Configure scrolling - FIXED: Force update before calculating
        table_container.update_idletasks()
        scrollable_frame.update_idletasks()

        def configure_scroll():
            canvas.update_idletasks()
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=bbox)
            else:
                # Use calculated height as fallback
                canvas.configure(scrollregion=(0, 0, total_width, total_height))
            canvas.itemconfig(canvas_window, width=max(total_width, canvas.winfo_width()))

        configure_scroll()
        canvas.after(50, configure_scroll)
        canvas.after(100, configure_scroll)
        canvas.after(200, configure_scroll)  # Extra update for safety

        # Mouse wheel handling
        def on_mousewheel(event):
            shift = (event.state & 0x1) != 0
            ctrl = (event.state & 0x4) != 0
            if shift or ctrl:
                canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def bind_mousewheel(widget):
            widget.bind("<MouseWheel>", on_mousewheel)
            widget.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
            widget.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
            widget.bind("<Shift-Button-4>", lambda e: canvas.xview_scroll(-1, "units"))
            widget.bind("<Shift-Button-5>", lambda e: canvas.xview_scroll(1, "units"))
            for child in widget.winfo_children():
                bind_mousewheel(child)

        bind_mousewheel(canvas)
        bind_mousewheel(scrollable_frame)

        # Keyboard navigation
        canvas.focus_set()
        canvas.bind("<Left>", lambda e: canvas.xview_scroll(-1, "units"))
        canvas.bind("<Right>", lambda e: canvas.xview_scroll(1, "units"))
        canvas.bind("<Up>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind("<Down>", lambda e: canvas.yview_scroll(1, "units"))

    def create_comparison_row(self, parent, label, instance, distance, features, feature_widths,
                              col_widths, total_width, bg_color, label_color, initial_instance):
        """Create a single row for the comparison table"""
        row_frame = ctk.CTkFrame(parent, fg_color=bg_color, corner_radius=8, height=scaled_size(60), width=total_width)
        row_frame.pack(fill='x', pady=2)
        row_frame.pack_propagate(False)

        row_inner = ctk.CTkFrame(row_frame, fg_color="transparent", width=total_width)
        row_inner.pack(fill='both', expand=True, padx=10, pady=8)
        row_inner.pack_propagate(False)

        current_x = 0

        # Method label
        ctk.CTkLabel(
            row_inner,
            text=label,
            font=scaled_font(20, "bold"),
            text_color=label_color,
            width=col_widths['Method'],
            anchor='w'  # Keep left-aligned
        ).place(x=current_x, y=0)
        current_x += col_widths['Method'] + scaled_size(40)

        # Feature values
        for feature in features:
            width = feature_widths[feature]
            value = instance.get(feature, 'N/A')

            if isinstance(value, float):
                if value.is_integer():
                    value_text = str(int(value))
                else:
                    value_text = f"{value:.2f}"
            else:
                value_text = str(value)

            changed = False
            if initial_instance is not None:
                initial_value = initial_instance.get(feature)
                changed = self.has_changed(value, initial_value)
                if changed:
                    value_text = f"→ {value_text}"

            ctk.CTkLabel(
                row_inner,
                text=value_text,
                font=scaled_font(20, "bold" if changed else "normal"),
                text_color=label_color if changed else self.colors['text_primary'],
                width=width,
                anchor='center'  # ✅ Center-aligned
            ).place(x=current_x, y=0)
            current_x += width + scaled_size(40)

        # Distance
        dist_text = f"{distance:.3f}" if distance is not None else "--"
        ctk.CTkLabel(
            row_inner,
            text=dist_text,
            font=scaled_font(20, "bold"),
            text_color=self.colors['success'] if distance is not None else self.colors['text_dim'],
            width=col_widths['Distance'],
            anchor='center'  # ✅ Center-aligned
        ).place(x=current_x, y=0)


def main():
    root = ctk.CTk()
    app = ModernCounterfactualGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()