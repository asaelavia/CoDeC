import bisect
import itertools
import numbers
import pickle
import random
import re
import signal
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
from contextlib import contextmanager
import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os
import dice_ml
from class_models import Mlp, pretrain
from dice_ml.utils.helpers import DataTransfomer

import numpy as np
import pandas as pd
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)
global_rng = np.random.RandomState(seed)
def get_value_range_optimized(dataset, col, is_continuous):
    """Optimized value range generation"""
    if is_continuous:
        min_val, max_val = dataset[col].min(), dataset[col].max()
        if (max_val - min_val) > 100:
            step = max(1, (max_val - min_val) // 100)
            return list(range(int(min_val), int(max_val) + 1, step))
        else:
            return list(range(int(min_val), int(max_val) + 1))
    else:
        return list(dataset[col].unique())


def check_constraint_batch_optimized(dataset, row, constraint_indices, cons_function):
    """Optimized batch constraint checking maintaining original logic"""
    results = []
    row_dict = row.to_dict() if hasattr(row, 'to_dict') else row

    for cons in constraint_indices:
        try:
            non_follow_cons = cons_function(dataset, row_dict, cons)
            # Maintain original logic for violation tracking
            violating_rows = dataset.loc[non_follow_cons]
            results.append((cons, violating_rows))
        except Exception:
            results.append((cons, pd.DataFrame()))
    return results


def single_pred_cons_optimized(val_iter, col, constraints, dic_cols, unary_cons_lst_single):
    """Optimized single predicate constraints with exact original logic"""
    if col not in dic_cols:
        return val_iter

    common_elements = set(unary_cons_lst_single) & set(dic_cols[col])

    for cons in common_elements:
        try:
            _, op, constant_str = constraints[cons][0].split(' ')

            if op == '>':
                constant = float(constant_str)
                # Use bisect logic like original
                import bisect
                ind = bisect.bisect_right(val_iter, constant)
                val_iter = val_iter[:ind]
            elif op == '>=':
                constant = float(constant_str)
                import bisect
                ind = bisect.bisect_left(val_iter, constant)
                val_iter = val_iter[:ind]
            elif op == '<':
                constant = float(constant_str)
                import bisect
                ind = bisect.bisect_left(val_iter, constant)
                val_iter = val_iter[ind:]
            elif op == '<=':
                constant = float(constant_str)
                import bisect
                ind = bisect.bisect_right(val_iter, constant)
                val_iter = val_iter[ind:]
            elif op == '==':
                if constant_str in val_iter:
                    val_iter.remove(constant_str)
            elif op == '!=':
                if constant_str in val_iter:
                    val_iter = [constant_str]
        except (ValueError, IndexError):
            continue

    return val_iter

timeout_occurred = False
def timeout_handler():
    global timeout_occurred
    timeout_occurred = True

def project_constraints_exact_fast(row, cf_example, fixed_feat, cont_feat, dic_cols, constraints, cons_function,
                                   cons_feat, unary_cons_lst_single,bin_cons):
    global timeout_occurred
    # More efficient: use numeric hash instead of string operations
    original_instance = cf_example.test_instance_df.iloc[0]

    # Hash immutable parts (for consistency)
    instance_hash = 0
    for col in fixed_feat:
        if col in original_instance.index:
            instance_hash ^= hash((col, original_instance[col]))

    # Hash current row (for diversity)
    row_hash = 0
    for col, val in row.items():
        row_hash ^= hash((col, val))

    # Combine with XOR (very fast)
    projection_seed = (instance_hash ^ row_hash) % (2 ** 31)
    # print(f'$$$$$$$$$$$$$$$$$SEED IS {projection_seed}$$$$$$$$$$$$$$$$$$$$$')
    local_rng = np.random.RandomState(projection_seed)
    # Reset timeout flag and start timer
    timeout_occurred = False
    timer = threading.Timer(5.0, timeout_handler)
    timer.start()

    try:
        # dataset = d.data_df
        # dataset = cf_example.data_interface.data_df
        # dataset = pd.read_csv(args.dataset_path)
        dataset = cf_example.data_interface.data_df
        viable_cols = [col for col in dataset.columns if col not in fixed_feat]
        viable_cols = [col for col in viable_cols if col in dic_cols]
        must_cons = set()
        need_proj = False
        violation_set = []
        cons_set = [[]] * len(constraints)
        starting_viol = set()
        # starting_viol = set()
        # for cons in range(len(constraints)):
        #     indices = cons_function(dataset, row_val, cons, [val_col])
        #     val_cons_set = dataset[indices]
        #     if len(val_cons_set) != 0 and (cons not in dic_cols[val_col]):
        #         starting_viol.update(np.where(indices)[0])
        #         if len(starting_viol) > gamma:
        #             return None
        #     val_cons_dfs[cons] = val_cons_set

        for cons in range(len(constraints)):
            non_follow_cons = cons_function(dataset, row, cons)
            violation_set.append(dataset.loc[non_follow_cons])
            if len(non_follow_cons) == 0:
                continue
            count = non_follow_cons.sum()
            if count > 0:
                if cons in bin_cons:
                    indices = np.where(non_follow_cons)[0]
                    starting_viol.update(indices)
                    cons_set[cons] = indices
                need_proj = True
                must_cons.add(cons)
        if not need_proj:
            return row.copy()
        local_rng.shuffle(viable_cols)
        for i in range(1, len(viable_cols) + 1):
            combs = list(itertools.combinations(viable_cols, i))
            # print(combs)
            for comb in combs:
                if not all(any(feat in comb for feat in cons_feat[cons]) for cons in must_cons):
                    continue
                row_per = row.copy()
                comb_iters = []
                for col in comb[:-1]:
                    if col in cont_feat:
                        val_iter = get_value_range_optimized(dataset, col, True)
                        val_iter = single_pred_cons_optimized(val_iter, col, constraints, dic_cols, unary_cons_lst_single)
                        val_iter.sort(key=lambda x: abs(x - row_per[col]))
                        if row_per[col] in val_iter:
                            val_iter.remove(row_per[col])
                    else:
                        val_iter = get_value_range_optimized(dataset, col, False)
                        val_iter = single_pred_cons_optimized(val_iter, col, constraints, dic_cols, unary_cons_lst_single)
                        local_rng.shuffle(val_iter)
                        if row_per[col] in val_iter:
                            val_iter.remove(row_per[col])
                    comb_iters.append([(val, col) for val in val_iter])
                all_combinations = itertools.product(*comb_iters)
                for vals in all_combinations:
                    if timeout_occurred:
                        print("Projection function timed out")
                        return None
                    row_val = row_per.copy()
                    for val, val_col in vals:
                        row_val[val_col] = val
                    res = smart_project_optimized_safe(row_val, comb[-1], dataset, cont_feat, constraints, dic_cols, cons_function)
                    if res is not None:
                        row_val[comb[-1]] = res
                        print('Success')
                        print(row_val)
                        return row_val
        else:
            return None
    except TimeoutError:
        print("Projection function timed out")
        return None
    finally:
        timer.cancel()


def smart_project_optimized_safe(row_val, val_col, dataset, cont_feat, constraints, dic_cols, cons_function):
    """Safe smart projection that exactly matches original constraint logic"""
    val_cons_dfs = {}

    # EXACT original logic for constraint violation checking
    for cons in range(len(constraints)):
        val_cons_set = dataset[cons_function(dataset, row_val, cons, [val_col])]
        if len(val_cons_set) != 0 and (cons not in dic_cols.get(val_col, [])):
            return None
        val_cons_dfs[cons] = val_cons_set

    # Use optimized range generation
    if val_col in cont_feat:
        possible_values = list(range(dataset[val_col].min(), dataset[val_col].max() + 1))
    else:
        possible_values = list(dataset[val_col].unique())

    # EXACT original constraint processing logic
    for count, cons in enumerate(dic_cols.get(val_col, [])):
        prev_values = possible_values.copy()

        if len(val_cons_dfs[cons]) == 0:
            continue

        for pred in constraints[cons]:
            if pred.find('cf_row') == -1:
                continue

            first_clause, op, second_clause = pred.split(' ')
            cf_pred = first_clause if 'cf_row' in first_clause else second_clause
            cf_col = cf_pred.split('.')[1]

            if val_col != cf_col:
                continue

            df_pred = first_clause if cf_pred == second_clause else second_clause

            # Use vectorized operations but with EXACT original logic
            if 'df.' in df_pred:
                unique_vals = val_cons_dfs[cons][cf_col].unique()
                possible_values_np = np.array(possible_values)

                if op == '==':
                    mask = ~np.isin(possible_values_np, unique_vals)
                elif op == '!=':
                    mask = np.isin(possible_values_np, unique_vals)
                elif op == '>':
                    if cf_pred == second_clause:
                        max_val = unique_vals.max()
                        mask = possible_values_np >= max_val
                    else:
                        min_val = unique_vals.min()
                        mask = possible_values_np <= min_val
                elif op == '>=':
                    if cf_pred == second_clause:
                        max_val = unique_vals.max()
                        mask = possible_values_np > max_val
                    else:
                        min_val = unique_vals.min()
                        mask = possible_values_np < min_val
                elif op == '<':
                    if cf_pred == second_clause:
                        min_val = unique_vals.min()
                        mask = possible_values_np <= min_val
                    else:
                        max_val = unique_vals.max()
                        mask = possible_values_np >= max_val
                elif op == '<=':
                    if cf_pred == second_clause:
                        min_val = unique_vals.min()
                        mask = possible_values_np < min_val
                    else:
                        max_val = unique_vals.max()
                        mask = possible_values_np > max_val

                possible_values = possible_values_np[mask].tolist()
            else:
                # Handle constants with vectorized operations
                const_value = float(df_pred) if '"' not in df_pred else df_pred[1:-1]
                possible_values_np = np.array(possible_values)

                if op == '==':
                    mask = possible_values_np != const_value
                elif op == '!=':
                    mask = possible_values_np == const_value
                elif op == '>':
                    mask = possible_values_np <= const_value
                elif op == '>=':
                    mask = possible_values_np < const_value
                elif op == '<':
                    mask = possible_values_np >= const_value
                elif op == '<=':
                    mask = possible_values_np > const_value

                possible_values = possible_values_np[mask].tolist()

            # EXACT original failure handling
            if not possible_values:
                return None
            break

    # Original return logic
    if possible_values:
        if isinstance(possible_values[0], numbers.Number) and not isinstance(possible_values[0], bool):
            return min(possible_values, key=lambda x: abs(x - row_val[val_col]))
        return possible_values[0]

    return None

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


def check_constraint_batch(dataset, row, constraint_indices, cons_function):
    """Check a batch of constraints (for parallel processing)"""
    results = []
    for cons in constraint_indices:
        non_follow_cons = cons_function(dataset, row, cons)
        results.append((cons, dataset.loc[non_follow_cons]))
    return results


# def project_constraints_exact_fast(row, cf_example, fixed_feat, cont_feat, dic_cols, constraints, cons_function,
#                                    cons_feat, unary_cons_lst_single):
#     dataset = cf_example.data_interface.data_df
#     viable_cols = [col for col in dataset.columns if col not in fixed_feat]
#     viable_cols = [col for col in viable_cols if col in dic_cols]
#     must_cons = set()
#     need_proj = False
#     violation_set = []
#
#     # Parallel constraint checking for large constraint sets
#     if len(constraints) > 20:  # Only parallelize for many constraints
#         num_cores = min(mp.cpu_count(), 4)  # Limit cores to avoid overhead
#         chunk_size = len(constraints) // num_cores + 1
#         constraint_chunks = [list(range(i, min(i + chunk_size, len(constraints))))
#                              for i in range(0, len(constraints), chunk_size)]
#
#         with Pool(processes=num_cores) as pool:
#             results = pool.starmap(
#                 check_constraint_batch,
#                 [(dataset, row, chunk, cons_function) for chunk in constraint_chunks]
#             )
#
#         for batch_results in results:
#             for cons, violating_rows in batch_results:
#                 violation_set.append(violating_rows)
#                 if len(violating_rows) > 0:
#                     need_proj = True
#                     must_cons.add(cons)
#     else:
#         # Original sequential processing for small constraint sets
#         for cons in range(len(constraints)):
#             non_follow_cons = cons_function(dataset, row, cons)
#             violation_set.append(dataset.loc[non_follow_cons])
#             if len(non_follow_cons) == 0:
#                 continue
#             count = non_follow_cons.sum()
#             if count > 0:
#                 need_proj = True
#                 must_cons.add(cons)
#
#     if not need_proj:
#         return row.copy()
#
#     print(len(viable_cols))
#     for i in range(1, len(viable_cols) + 1):
#         random.shuffle(viable_cols)
#         combs = list(itertools.combinations(viable_cols, i))
#         for comb in combs:
#             if not all(any(feat in comb for feat in cons_feat[cons]) for cons in must_cons):
#                 continue
#             row_per = row.copy()
#             comb_iters = []
#             for col in comb[:-1]:
#                 if col in cont_feat:
#                     val_iter = get_value_range_optimized(dataset, col, True)
#                     val_iter = single_pred_cons_optimized(val_iter, col, constraints, dic_cols, unary_cons_lst_single)
#                     val_iter.sort(key=lambda x: abs(x - row_per[col]))
#                     if row_per[col] in val_iter:
#                         val_iter.remove(row_per[col])
#                 else:
#                     val_iter = get_value_range_optimized(dataset, col, False)
#                     val_iter = single_pred_cons_optimized(val_iter, col, constraints, dic_cols, unary_cons_lst_single)
#                     np.random.shuffle(val_iter)
#                     if row_per[col] in val_iter:
#                         val_iter.remove(row_per[col])
#                 comb_iters.append([(val, col) for val in val_iter])
#             all_combinations = itertools.product(*comb_iters)
#             for vals in all_combinations:
#                 row_val = row_per.copy()
#                 for val, val_col in vals:
#                     row_val[val_col] = val
#                 res = smart_project_optimized_safe(row_val, comb[-1], dataset, cont_feat,
#                                                    constraints, dic_cols, cons_function)
#                 if res is not None:
#                     row_val[comb[-1]] = res
#                     print('Success')
#                     print(row_val)
#                     return row_val
#
#     else:
#         return None


def project_instances(cf_example, df, cont_feat, d, fixed_feat, dic_cols, constraints, cons_function, cons_feat,
                      unary_cons_lst_single,bin_cons):
    project_cfs_df = cf_example.final_cfs_df_sparse.copy()

    for col in df.columns:
        if col in cont_feat + ['label']:
            continue
        project_cfs_df[col] = pd.Categorical(project_cfs_df[col],
                                             categories=d.data_df[col].cat.categories)
    projected_cfs_df = project_cfs_df.copy()
    projected_cfs_df = projected_cfs_df[1:0]
    for index, row in project_cfs_df.iterrows():
        print(f'project instance {index}')
        proj_row = project_constraints_exact_fast(row, cf_example, fixed_feat, cont_feat, dic_cols, constraints,
                                                  cons_function, cons_feat, unary_cons_lst_single,bin_cons)
        # proj_row = project_constraints_exact_fast_optimized(row, cf_example, fixed_feat, cont_feat, dic_cols, constraints,
        #                                           cons_function, cons_feat, unary_cons_lst_single)
        if proj_row is not None:
            projected_cfs_df.loc[len(projected_cfs_df)] = proj_row
    return projected_cfs_df


def project_counterfactuals(dice_exp, df, cont_feat, d, fixed_feat, dic_cols, constraints, cons_function, cons_feat,
                            unary_cons_lst_single,bin_cons):
    all_instances_cfs = []
    for cf_example in dice_exp.cf_examples_list:
        all_instances_cfs.append(
            project_instances(cf_example, df, cont_feat, d, fixed_feat, dic_cols, constraints, cons_function,
                              cons_feat, unary_cons_lst_single,bin_cons))
    return all_instances_cfs


def bfs_counterfactuals(exp, threshold, model, fixed_feat, exp_random, df, cont_feat, d, dic_cols, constraints,
                        cons_function, cons_feat, unary_cons_lst_single,bin_cons, transformer, progress_queue=None):
    with timed_section("Projecting counterfactuals", progress_queue):
        projected_cfs = project_counterfactuals(exp, df, cont_feat, d, fixed_feat, dic_cols, constraints, cons_function,
                                                cons_feat, unary_cons_lst_single,bin_cons)

    all_accepted_instances = []
    for i, cfs in enumerate(projected_cfs):
        accepted_final_cfs = []
        cfs = cfs.drop('label', axis=1)
        probs = model(torch.tensor(transformer.transform(cfs).values.astype('float32')).float()).detach().numpy()
        labels = np.round(probs)
        accepted = cfs[labels == 1]
        print('###########ACCEPTED#############')
        print(accepted)

        not_accepted = cfs[labels == 0]
        print('###########NOT ACCEPTED#############')
        print(not_accepted)
        if len(not_accepted) == 0:
            return accepted
        features_to_vary = list(df.columns)
        for feat in fixed_feat:
            features_to_vary.remove(feat)

        not_accepted_final_cfs = []
        iteration = 0
        while len(accepted) < threshold and iteration < 3:
            iteration += 1
            if progress_queue:
                progress_queue.put({'type': 'progress',
                                    'text': f'BFS iteration {iteration}, found {len(accepted)} valid counterfactuals'})

            try:
                dice_exp_random = exp_random.generate_counterfactuals(not_accepted, total_CFs=threshold,
                                                                      desired_class=1,
                                                                      verbose=True,
                                                                      learning_rate=0.1,  # Increased
                                                                      min_iter=5,  # Reduced
                                                                      max_iter=50,  # Reduced
                                                                      features_to_vary=features_to_vary)
            except:
                break

            projected_cfs_not_accepted = project_counterfactuals(dice_exp_random, df, cont_feat, d, fixed_feat,
                                                                 dic_cols, constraints, cons_function, cons_feat,
                                                                 unary_cons_lst_single,bin_cons)
            not_accepted = None
            for cfs in projected_cfs_not_accepted:
                cfs = cfs.drop('label', axis=1)
                probs = model(
                    torch.tensor(transformer.transform(cfs).values.astype('float32')).float()).detach().numpy()
                labels = np.round(probs)
                accepted_2 = cfs[labels == 1]
                not_accepted_final_cfs.append(accepted_2)
                if not_accepted is None:
                    not_accepted = cfs[labels == 0]
                else:
                    not_accepted = pd.concat([not_accepted, cfs[labels == 0]])
                if len(accepted_2) != 0:
                    accepted = pd.concat([accepted, accepted_2], ignore_index=True)
                    accepted_final_cfs.append(accepted_2)

        print('###########ACCEPTED NEW#############')
        print(accepted)
        return accepted
    return None


def single_pred_cons(val_iter, col, constraints, dic_cols, unary_cons_lst_single):
    if col not in dic_cols:
        return val_iter
    common_elements = set(unary_cons_lst_single) & set(dic_cols[col])
    for cons in common_elements:
        _, op, constant = constraints[cons][0].split(' ')
        if op == '>':
            constant = float(constant)
            ind = bisect.bisect_right(val_iter, constant)
            val_iter = val_iter[:ind]
        elif op == '>=':
            constant = float(constant)
            ind = bisect.bisect_left(val_iter, constant)
            val_iter = val_iter[:ind]
        elif op == '<':
            constant = float(constant)
            ind = bisect.bisect_left(val_iter, constant)
            val_iter = val_iter[ind:]
        elif op == '<=':
            constant = float(constant)
            ind = bisect.bisect_right(val_iter, constant)
            val_iter = val_iter[ind]
        elif op == '==':
            if constant in val_iter:
                val_iter.remove(constant)
        else:
            if constant in val_iter:
                val_iter = [constant]
    return val_iter


def smart_project(row_val, val_col, dataset, cont_feat, constraints, dic_cols, cons_function):
    val_cons_dfs = {}
    for cons in range(len(constraints)):
        val_cons_set = dataset[cons_function(dataset, row_val, cons, [val_col])]
        if len(val_cons_set) != 0 and (cons not in dic_cols[val_col]):
            return None
        val_cons_dfs[cons] = val_cons_set

    if val_col in cont_feat:
        possible_values = list(range(dataset[val_col].min(), dataset[val_col].max() + 1))
    else:
        possible_values = list(dataset[val_col].unique())

    for count, cons in enumerate(dic_cols[val_col]):
        prev_values = possible_values
        if len(val_cons_dfs[cons]) == 0:
            continue
        for pred in constraints[cons]:
            if pred.find('cf_row') == -1:
                continue
            first_clause, op, second_clause = pred.split(' ')
            cf_pred = first_clause if 'cf_row' in first_clause else second_clause
            cf_col = cf_pred.split('.')[1]

            if val_col != cf_col:
                continue
            df_pred = first_clause if cf_pred == second_clause else second_clause

            if 'df.' in df_pred:
                if op == '==':
                    possible_values = [elem for elem in possible_values if
                                       elem not in val_cons_dfs[cons][cf_col].unique()]
                elif op == '!=':
                    possible_values = [elem for elem in possible_values if
                                       elem in val_cons_dfs[cons][cf_col].unique()]
                elif op == '>':
                    if cf_pred == second_clause:
                        max_val = val_cons_dfs[cons][cf_col].unique().max()
                        possible_values_np = np.array(possible_values)
                        possible_values = list(possible_values_np[possible_values_np >= max_val])
                    else:
                        min_val = val_cons_dfs[cons][cf_col].unique().min()
                        possible_values_np = np.array(possible_values)
                        possible_values = list(possible_values_np[possible_values_np <= min_val])
                elif op == '>=':
                    if cf_pred == second_clause:
                        max_val = val_cons_dfs[cons][cf_col].unique().max()
                        possible_values_np = np.array(possible_values)
                        possible_values = list(possible_values_np[possible_values_np > max_val])
                    else:
                        min_val = val_cons_dfs[cons][cf_col].unique().min()
                        possible_values_np = np.array(possible_values)
                        possible_values = list(possible_values_np[possible_values_np < min_val])
                elif op == '<':
                    if cf_pred == second_clause:
                        min_val = val_cons_dfs[cons][cf_col].unique().min()
                        possible_values_np = np.array(possible_values)
                        possible_values = list(possible_values_np[possible_values_np <= min_val])
                    else:
                        max_val = val_cons_dfs[cons][cf_col].unique().max()
                        possible_values_np = np.array(possible_values)
                        possible_values = list(possible_values_np[possible_values_np >= max_val])
                elif op == '<=':
                    if cf_pred == second_clause:
                        min_val = val_cons_dfs[cons][cf_col].unique().min()
                        possible_values_np = np.array(possible_values)
                        possible_values = list(possible_values_np[possible_values_np < min_val])
                    else:
                        max_val = val_cons_dfs[cons][cf_col].unique().max()
                        possible_values_np = np.array(possible_values)
                        possible_values = list(possible_values_np[possible_values_np > max_val])

            else:
                const_value = float(df_pred) if '"' not in df_pred else df_pred[1:-1]

                if op == '==':
                    possible_values = [elem for elem in possible_values if
                                       elem not in [const_value]]
                elif op == '!=':
                    possible_values = [elem for elem in possible_values if
                                       elem in [const_value]]
                elif op == '>':
                    min_val = const_value
                    possible_values_np = np.array(possible_values)
                    possible_values = list(possible_values_np[possible_values_np <= min_val])
                elif op == '>=':
                    min_val = const_value
                    possible_values_np = np.array(possible_values)
                    possible_values = list(possible_values_np[possible_values_np < min_val])
                elif op == '<':
                    max_val = const_value
                    possible_values_np = np.array(possible_values)
                    possible_values = list(possible_values_np[possible_values_np >= max_val])
                elif op == '<=':
                    max_val = const_value
                    possible_values_np = np.array(possible_values)
                    possible_values = list(possible_values_np[possible_values_np > max_val])
            if not possible_values:
                return None
            break
    if isinstance(possible_values[0], numbers.Number) and not isinstance(possible_values[0], bool):
        return min(possible_values, key=lambda x: abs(x - row_val[val_col]))
    return possible_values[0]


class ProgressCallback:
    """Callback for DiCE to report progress"""

    def __init__(self, queue):
        self.queue = queue
        self.iteration = 0

    def __call__(self, *args, **kwargs):
        self.iteration += 1
        if self.iteration % 5 == 0:  # Update every 5 iterations
            self.queue.put({
                'type': 'progress',
                'text': f'Optimizing solutions... (iteration {self.iteration})'
            })


def code_counterfactuals(query_instances, constraints_path, dataset_path, fixed_feat, k,
                         model_cache, transformer_cache, constraints_cache, progress_queue=None):
    """Modified to use caches and progress queue"""

    # Load constraints with caching
    if constraints_path in constraints_cache:
        constraint_data = constraints_cache[constraints_path]
        constraints, dic_cols, cons_function, cons_feat, unary_cons_lst, unary_cons_lst_single, bin_cons = constraint_data
    else:
        with timed_section("Loading constraints", progress_queue):
            constraint_data = load_constraints(constraints_path)
            constraints_cache[constraints_path] = constraint_data
            constraints, dic_cols, cons_function, cons_feat, unary_cons_lst, unary_cons_lst_single, bin_cons = constraint_data

    # Load dataset
    with timed_section("Loading dataset", progress_queue):
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

    # FIXED: Use unified model loading that checks disk first, then cache, then trains
    cache_key = dataset_path
    model_state_dict_path = f'{dataset_path}_model_state_dict.dict'

    # Check if model is already in cache
    if cache_key in model_cache:
        model = model_cache[cache_key]
        transformer = transformer_cache[cache_key]
        d = dice_ml.Data(dataframe=train_dataset, continuous_features=cont_feat, outcome_name='label')
        print("Model loaded from cache")
    else:
        with timed_section("Preparing model and transformer", progress_queue):
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
                model = pretrain(model, 'cpu', train_loader, lr=1e-4, epochs=1000)
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

    with timed_section("Generating initial counterfactuals", progress_queue):
        dice_exp_random = exp_random.generate_counterfactuals(
            query_instances,
            total_CFs=k,
            desired_class=1,
            verbose=True,
            features_to_vary=features_to_vary,
            min_iter=5,  # Reduced
            max_iter=50,  # Reduced
            learning_rate=0.1  # Increased
        )
    dice_exp_random.cf_examples_list[0].final_cfs_df_sparse.to_csv('dice_gui.csv')
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', None,
                           'display.max_colwidth', None):
        print(dice_exp_random.cf_examples_list[0].final_cfs_df_sparse)

    algorithm_cfs = dice_exp_random.cf_examples_list[0].final_cfs_df_sparse.drop('label', axis=1)
    algorithm_cfs, dpp_score, distances = n_best_cfs_heuristic(algorithm_cfs, query_instances.iloc[0], k, transformer,
                                                               exp_random)
    # with open('dice_gui_metrics.pkl', 'wb') as f:
    #     pickle.dump((algorithm_cfs, dpp_score, distances), f)

    with timed_section("Running BFS for constraint satisfaction", progress_queue):
        algorithm_cfs = bfs_counterfactuals(dice_exp_random, k, model, fixed_feat, exp_random, df, cont_feat, d,
                                            dic_cols,
                                            constraints, cons_function, cons_feat, unary_cons_lst_single,bin_cons, transformer,
                                            progress_queue)

    algorithm_cfs, dpp_score, distances = n_best_cfs_heuristic(algorithm_cfs, query_instances.iloc[0], k, transformer,
                                                               exp_random)
    # with open('codec_gui_metrics.pkl', 'wb') as f:
    #     pickle.dump((algorithm_cfs, dpp_score, distances), f)

    return algorithm_cfs, dpp_score, distances

def code_counterfactuals_dice(query_instances, constraints_path, dataset_path, fixed_feat, k,
                         model_cache, transformer_cache, constraints_cache, progress_queue=None):
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

    # FIXED: Use unified model loading that checks disk first, then cache, then trains
    cache_key = dataset_path
    model_state_dict_path = f'{dataset_path}_model_state_dict.dict'

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
            model = pretrain(model, 'cpu', train_loader, lr=1e-4, epochs=1000)
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
        verbose=True,
        features_to_vary=features_to_vary,
        min_iter=5,  # Reduced
        max_iter=50,  # Reduced
        learning_rate=0.1  # Increased
    )
    dice_exp_random.cf_examples_list[0].final_cfs_df_sparse.to_csv('dice_gui.csv')
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
    codec_cfs = bfs_counterfactuals(dice_exp_random, k, model, fixed_feat, exp_random, df, cont_feat, d,
                                   dic_cols, constraints, cons_function, cons_feat, unary_cons_lst_single, bin_cons, transformer,
                                   progress_queue)

    codec_cfs_processed, codec_dpp_score, codec_distances = n_best_cfs_heuristic(codec_cfs, query_instances.iloc[0], k, transformer, exp_random)
    # with open('codec_gui_metrics.pkl', 'wb') as f:
    #     pickle.dump((codec_cfs_processed, codec_dpp_score, codec_distances), f)

    # Return both DiCE and CoDeC results
    return {
        'codec_results': (codec_cfs_processed, codec_dpp_score, codec_distances),
        'dice_results': (dice_cfs_processed, dice_dpp_score, dice_distances)
    }

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
            dic[i] = (dpp_score, proximity_distance, dpp_score - 0.5 * proximity_distance)
        best_index = max(dic, key=lambda x: dic[x][-1])
        best_cf = cfs_pool.loc[best_index]
        cfs_pool = cfs_pool.drop(best_index, axis=0)
        curr_best = pd.concat([curr_best, pd.DataFrame([best_cf])], axis=0)
        best_indices.append(best_index)
    return curr_best, dic[best_index][0], [dist_dic[i] for i in best_indices]


def load_dataset_chunked(filepath, chunksize=10000):
    """Load dataset in chunks for better memory usage"""
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


class CounterfactualGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CODEC: Constraints Guided Diverse Counterfactuals")
        self.root.geometry("1650x1080")

        # Initialize variables
        self.dataset_path = tk.StringVar()
        self.constraints_path = tk.StringVar()
        self.num_counterfactuals = tk.StringVar(value="3")
        self.model_type = tk.StringVar(value="MLP")
        self.fixed_features = set()

        # Initialize caches
        self.cached_models = {}
        self.cached_transformers = {}
        self.cached_constraints = {}

        # Initialize queue for thread communication
        self.computation_queue = queue.Queue()
        self.check_computation_queue()

        # Configure styles for larger fonts
        style = ttk.Style()
        style.configure('Title.TLabelframe.Label', font=('Segoe UI', 14, 'bold'))
        style.configure('Large.TLabel', font=('Segoe UI', 12))
        style.configure('Large.TEntry', font=('Segoe UI', 12))
        style.configure('Large.TButton', font=('Segoe UI', 12))
        style.configure('Treeview', font=('Segoe UI', 12))
        style.configure('Treeview.Heading', font=('Segoe UI', 12, 'bold'))

        # Configure tab font size
        style.configure('TNotebook.Tab', font=('Segoe UI', 12))

        # Create notebook
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)

        # Create frames
        self.input_frame = ttk.Frame(self.notebook)
        self.constraints_frame = ttk.Frame(self.notebook)
        self.results_frame = ttk.Frame(self.notebook)

        # Add frames to notebook with larger font
        self.notebook.add(self.input_frame, text='Input Parameters')
        self.notebook.add(self.constraints_frame, text='Constraints')
        self.notebook.add(self.results_frame, text='Results')

        # Setup tabs
        self.setup_input_tab()
        self.setup_constraints_tab()
        self.setup_results_tab()

    def browse_dataset(self):
        filename = filedialog.askopenfilename(
            title="Select Dataset CSV",
            initialdir="./",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filename:
            self.dataset_path.set(filename)
            self.update_features_list()
            # Preload model in background
            self.preload_model_async()

    def browse_constraints(self):
        filename = filedialog.askopenfilename(
            title="Select Constraints File",
            initialdir="./",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )
        if filename:
            self.constraints_path.set(filename)
            self.update_constraints_view()
            # Preload constraints in background
            threading.Thread(target=lambda: self.load_constraints_cached(filename), daemon=True).start()

    def get_initial_point(self):
        # Add safety check for widget existence
        if not hasattr(self, 'initial_point_widgets') or not self.initial_point_widgets:
            messagebox.showerror("Error", "Please load a dataset first")
            return None

        initial_point = {
            'label': 'Initial',
            'prediction': '0'  # Default prediction
        }

        for column, widget in self.initial_point_widgets.items():
            try:
                # Check if widget still exists
                if not widget.winfo_exists():
                    messagebox.showerror("Error", "Interface error: Please reload the dataset")
                    return None

                if isinstance(widget, ttk.Combobox):  # Categorical
                    initial_point[column] = widget.get()
                else:  # Continuous
                    try:
                        initial_point[column] = float(widget.get())
                    except ValueError:
                        messagebox.showerror("Error", f"Invalid value for {column}")
                        return None
            except tk.TclError:
                # Widget was destroyed
                messagebox.showerror("Error", "Interface error: Please reload the dataset")
                return None

        initial_point['label'] = 0
        return initial_point

    def get_fixed_features(self):
        return [self.features_list.get(idx) for idx in self.features_list.curselection()]

    def setup_results_tab(self):
        # Configure frame padding and background
        self.results_frame.configure(padding=(5, 5, 5, 5))

        # Create main table container
        table_container = ttk.Frame(self.results_frame)
        table_container.pack(fill='x', expand=False, pady=5)

        # Create Treeview with scrollbars
        self.results_tree = ttk.Treeview(table_container, show='headings', selectmode='none', height=4)

        # Add dummy frame to maintain compatibility with existing code
        self.scrollable_frame = ttk.Frame(table_container)

        # Scrollbars
        y_scroll = ttk.Scrollbar(table_container, orient='vertical', command=self.results_tree.yview)
        x_scroll = ttk.Scrollbar(table_container, orient='horizontal', command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        # Grid layout
        self.results_tree.grid(row=0, column=0, sticky='ew')
        y_scroll.grid(row=0, column=1, sticky='ns')
        x_scroll.grid(row=1, column=0, sticky='ew')
        table_container.grid_columnconfigure(0, weight=1)

        # Configure style for consistent fonts and increased row height
        style = ttk.Style()
        style.configure("Treeview",
                        font=('Segoe UI', 12),
                        rowheight=30)  # Increased row height
        style.configure("Treeview.Heading",
                        font=('Segoe UI', 12, 'bold'),
                        padding=(0, 5))  # Added padding to headers

        # Configure tag styles for different row types
        self.results_tree.tag_configure('initial', background='#f0f0f0')
        self.results_tree.tag_configure('ours', background='#f0fdf4')
        self.results_tree.tag_configure('changed', foreground='#1e40af', font=('Segue UI', 12, 'bold'))

        # Create metrics frame
        self.metrics_frame = ttk.Frame(self.results_frame)
        self.metrics_frame.pack(fill='x', expand=False, pady=(20, 0))

        # Configure metrics styles
        style.configure('Metric.TLabel', font=('Segoe UI', 13), padding=(10, 5))
        style.configure('MetricValue.TLabel', font=('Segoe UI', 13, 'bold'), padding=(10, 5))

        # Add metrics labels
        self.dpp_label = ttk.Label(self.metrics_frame, text="Diversity Score (DPP):", style='Metric.TLabel')
        self.dpp_label.pack(anchor='w')
        self.dpp_value = ttk.Label(self.metrics_frame, text="", style='MetricValue.TLabel')
        self.dpp_value.pack(anchor='w')

        ttk.Label(self.metrics_frame, text="Distance Scores:", style='Metric.TLabel').pack(anchor='w')
        self.distance_values = ttk.Label(self.metrics_frame, text="", style='MetricValue.TLabel')
        self.distance_values.pack(anchor='w')

    def add_result_row(self, data, row_type='Initial', distance=None, initial_instance=None):
        # Clear existing rows if this is the first row
        if not self.results_tree['columns']:
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)

            columns = ['Type'] + list(self.df.columns)
            self.results_tree['columns'] = columns

            # Configure columns
            for col in columns:
                self.results_tree.heading(col, text=col, anchor='w')
                width = {
                    'Type': 100,
                    'age': 70,
                    'workclass': 140,
                    'education': 140,
                    'education_num': 140,
                    'marital_status': 160,
                    'occupation': 140,
                    'relationship': 140,
                    'race': 160,
                    'sex': 100,
                    'hours_per_week': 140,
                    'native_country': 140,
                    'label': 100
                }.get(col, 120)
                self.results_tree.column(col, width=width, minwidth=50)

        # Prepare row values with highlighting for changed values
        values = []
        values.append(data.get('label', ''))  # Type column

        # Add other column values
        if row_type != 'Initial' and initial_instance is not None:
            for col in self.df.columns:
                # Get original and current values
                original_value = initial_instance.get(col, '')
                current_value = data.get(col, '')

                # Handle different data types for comparison
                if isinstance(original_value, (int, float)) and isinstance(current_value, (int, float)):
                    # For numeric values, compare the actual numbers
                    has_changed = original_value != current_value
                else:
                    # For other types, convert to string and compare
                    has_changed = str(original_value).strip() != str(current_value).strip()

                # Format the value
                formatted_value = self.format_value(col, current_value, row_type)

                # Only add arrow if the value actually changed
                if has_changed:
                    values.append('→ ' + formatted_value)
                else:
                    values.append(formatted_value)
        else:
            for col in self.df.columns:
                values.append(self.format_value(col, data.get(col, ''), row_type))

        # Insert row with tags for background color
        tags = ['initial'] if row_type == 'Initial' else ['ours']
        self.results_tree.insert('', 'end', values=values, tags=tags)

    def format_value(self, column, value, row_type):
        """Format value based on column type with improved consistency"""
        try:
            if column in ['age', 'education_num', 'hours_per_week']:
                # Consistent integer formatting
                if isinstance(value, (int, float)):
                    return str(int(float(value)))
                return str(value)
            elif column == 'label':
                # Special handling for label column
                if row_type == 'Initial':
                    return '0'
                else:
                    return '1'
            elif isinstance(value, (int, float)):
                # Handle numeric values
                if value == int(value):
                    return str(int(value))
                return str(round(value, 4))
            return str(value).strip()
        except (ValueError, TypeError):
            return str(value).strip()

    def setup_constraints_tab(self):
        # Create a frame for constraints visualization
        constraints_container = ttk.Frame(self.constraints_frame)
        constraints_container.pack(fill='both', expand=True, padx=10, pady=5)

        # Add title
        title_label = ttk.Label(constraints_container,
                                text="Active Denial Constraints",
                                font=('Helvetica', 14, 'bold'))
        title_label.pack(pady=10)

        # Create frame for text widget and scrollbar
        text_frame = ttk.Frame(constraints_container)
        text_frame.pack(fill='both', expand=True)

        # Create text widget for constraints display
        self.constraints_text = tk.Text(text_frame,
                                        wrap=tk.WORD,
                                        height=20,
                                        width=60,
                                        font=('Courier', 14))

        # Add scrollbar
        scrollbar = ttk.Scrollbar(text_frame,
                                  orient='vertical',
                                  command=self.constraints_text.yview)

        # Configure text widget and scrollbar
        self.constraints_text.configure(yscrollcommand=scrollbar.set)

        # Pack scrollbar and text widget
        scrollbar.pack(side='right', fill='y')
        self.constraints_text.pack(side='left', fill='both', expand=True)

        # Make text widget read-only
        self.constraints_text.configure(state='disabled')

    def update_constraints_view(self):
        """Update the constraints visualization with formatted constraints"""
        try:
            if not self.constraints_path.get():
                return

            # Explicitly specify UTF-8 encoding when reading the file
            with open(self.constraints_path.get(), 'r', encoding='utf-8') as file:
                constraints = file.readlines()

            # Enable text widget for updating
            self.constraints_text.configure(state='normal')
            self.constraints_text.delete(1.0, tk.END)

            # Add each constraint without replacing special characters
            for i, constraint in enumerate(constraints, 1):
                constraint = constraint.strip()
                if constraint:
                    self.constraints_text.insert(tk.END, f"Constraint {i}:\n")
                    self.constraints_text.insert(tk.END, constraint + "\n")
                    self.constraints_text.insert(tk.END, "-" * 60 + "\n\n")

            # Make text widget read-only again
            self.constraints_text.configure(state='disabled')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load constraints: {str(e)}")

    def show_loading_message(self, message):
        """Modified to return the loading window reference"""
        loading_window = tk.Toplevel(self.root)
        loading_window.title("Processing")

        # Set size and position
        window_width = 450
        window_height = 180
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2

        loading_window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        loading_window.transient(self.root)
        loading_window.grab_set()
        loading_window.resizable(False, False)

        # Create a frame with padding
        frame = ttk.Frame(loading_window, padding=20)
        frame.pack(fill='both', expand=True)

        # Add message label
        loading_window.message_label = ttk.Label(
            frame,
            text=message,
            font=('Segoe UI', 16, 'bold'),
            wraplength=400
        )
        loading_window.message_label.pack(pady=15)

        # Create animation frame
        animation_frame = ttk.Frame(frame)
        animation_frame.pack(pady=10)

        # Add spinning animation label
        loading_window.animation_label = ttk.Label(
            animation_frame,
            text="⠋",
            font=('Segoe UI', 24)
        )
        loading_window.animation_label.pack()

        return loading_window

    def update_loading_text(self, loading_window, new_message):
        """Update loading message text"""
        try:
            if hasattr(loading_window, 'message_label'):
                loading_window.message_label.configure(text=new_message)
                self.root.update_idletasks()
        except tk.TclError:
            # Window was destroyed
            pass

    def generate_counterfactuals(self):
        """Modified to run computation in separate thread with better error handling"""
        try:
            # Check if dataset is loaded
            if not hasattr(self, 'df') or self.df is None:
                messagebox.showerror("Error", "Please load a dataset first")
                return

            # Check if widgets exist
            if not hasattr(self, 'initial_point_widgets') or not self.initial_point_widgets:
                messagebox.showerror("Error", "Please load a dataset first")
                return

            # Get initial point with error checking
            initial_instance = self.get_initial_point()
            if not initial_instance:
                return

            # Validate required paths
            if not self.constraints_path.get():
                messagebox.showerror("Error", "Please select a constraints file")
                return

            if not self.dataset_path.get():
                messagebox.showerror("Error", "Please select a dataset file")
                return

            # Show loading window
            self.loading_window = self.show_loading_message("Initializing...")

            # Start animation
            self.start_loading_animation(self.loading_window)

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
            computation_thread.daemon = True  # Thread will die when main program exits
            computation_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start counterfactual generation: {str(e)}")

    def check_computation_queue(self):
        """Check for messages from the computation thread"""
        try:
            while True:
                message = self.computation_queue.get_nowait()
                if message['type'] == 'progress':
                    self.update_loading_text(self.loading_window, message['text'])
                elif message['type'] == 'complete':
                    self.handle_computation_complete(message['data'])
                elif message['type'] == 'error':
                    self.handle_computation_error(message['error'])
        except queue.Empty:
            pass

        # Schedule next check - reduced interval for more responsiveness
        self.root.after(50, self.check_computation_queue)

    def start_loading_animation(self, loading_window):
        """Start the loading animation in the UI thread"""
        if not hasattr(self, '_animation_active'):
            self._animation_active = False

        if self._animation_active:
            return

        self._animation_active = True
        animation_dots = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._animation_index = 0

        def animate():
            if self._animation_active and hasattr(loading_window, 'animation_label'):
                try:
                    loading_window.animation_label.config(text=animation_dots[self._animation_index])
                    self._animation_index = (self._animation_index + 1) % len(animation_dots)
                    self.root.after(100, animate)
                except tk.TclError:
                    # Window was destroyed
                    self._animation_active = False

        animate()

    def stop_loading_animation(self):
        """Stop the loading animation"""
        self._animation_active = False

    def load_or_get_model(self, dataset_path, train_dataset, cont_feat, y_train):
        """Load model from cache or disk, or train new one"""
        cache_key = dataset_path

        if cache_key in self.cached_models:
            return self.cached_models[cache_key], self.cached_transformers[cache_key]

        # Create transformer
        d = dice_ml.Data(dataframe=train_dataset, continuous_features=cont_feat, outcome_name='label')
        transformer = DataTransfomer('ohe-min-max')
        transformer.feed_data_params(d)
        transformer.initialize_transform_func()

        x_train = train_dataset.drop('label', axis=1)
        X_train = transformer.transform(x_train)

        # Create model
        model = Mlp(X_train.shape[1], [100, 1])
        model_state_dict_path = f'{dataset_path}_model_state_dict.dict'

        if os.path.exists(model_state_dict_path):
            model.load_state_dict(torch.load(model_state_dict_path))
            print("Model loaded from disk")
        else:
            # Train model
            model.train()
            X_train_fixed = X_train.values.astype(np.float32)
            train_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_train_fixed), torch.Tensor(y_train.values.astype('int'))),
                64, shuffle=True)
            model = pretrain(model, 'cpu', train_loader, lr=1e-4, epochs=1000)
            torch.save(model.state_dict(), model_state_dict_path)
            print('Model trained and saved')

        # Cache for future use
        self.cached_models[cache_key] = model
        self.cached_transformers[cache_key] = transformer

        return model, transformer

    def preload_model_async(self):
        """Preload model in background thread - FIXED to use same cache"""

        def worker():
            try:
                dataset_path = self.dataset_path.get()
                df = pd.read_csv(dataset_path)

                # Prepare data
                y = df['label']
                train_dataset, _, y_train, _ = train_test_split(df, y, test_size=0.2, random_state=0, stratify=y)

                cont_feat = []
                for col in df.columns:
                    if df[col].dtype != 'object' and col != 'label':
                        cont_feat.append(col)

                # FIXED: Use the same cache that will be passed to code_counterfactuals
                cache_key = dataset_path
                model_state_dict_path = f'{dataset_path}_model_state_dict.dict'

                # Check if already cached
                if cache_key in self.cached_models:
                    print("Model already in cache")
                    return

                # Check if model exists on disk first
                if os.path.exists(model_state_dict_path):
                    print("Model file exists on disk, will load when needed")
                    return

                # Only train if model doesn't exist anywhere
                print("No model found, training in background...")

                # Prepare data for training
                for col in df.columns:
                    if col in cont_feat + ['label']:
                        continue
                    df[col] = df[col].astype('object')

                d = dice_ml.Data(dataframe=train_dataset, continuous_features=cont_feat, outcome_name='label')
                transformer = DataTransfomer('ohe-min-max')
                transformer.feed_data_params(d)
                transformer.initialize_transform_func()

                x_train = train_dataset.drop('label', axis=1)
                X_train = transformer.transform(x_train)

                # Create and train model
                model = Mlp(X_train.shape[1], [100, 1])
                model.train()
                X_train_fixed = X_train.values.astype(np.float32)
                train_loader = DataLoader(
                    TensorDataset(torch.from_numpy(X_train_fixed), torch.Tensor(y_train.values.astype('int'))),
                    64, shuffle=True)
                model = pretrain(model, 'cpu', train_loader, lr=1e-4, epochs=1000)
                torch.save(model.state_dict(), model_state_dict_path)

                # Cache the trained model
                self.cached_models[cache_key] = model
                self.cached_transformers[cache_key] = transformer
                print('Model trained and cached in background')

            except Exception as e:
                print(f"Background model loading failed: {e}")

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def load_constraints_cached(self, path):
        """Load constraints with caching"""
        if path in self.cached_constraints:
            return self.cached_constraints[path]

        result = load_constraints(path)
        self.cached_constraints[path] = result
        return result

    def computation_worker(self, initial_instance, constraints_path, dataset_path, fixed_features, num_counterfactuals):
        """Worker function that runs in a separate thread - FIXED to pass GUI cache"""
        try:
            # Notify UI of progress
            self.computation_queue.put({'type': 'progress', 'text': 'Initializing DiCE...'})

            self.computation_queue.put({'type': 'progress', 'text': 'Generating initial counterfactuals...'})

            self.computation_queue.put({'type': 'progress', 'text': 'CoDeC processing constraints...'})

            self.computation_queue.put({'type': 'progress', 'text': 'Optimizing solutions...'})

            # Do the actual computation
            try:
                # Try loading pre-generated results first
                with open('dice_gui_metrics_aaaa.pkl', 'rb') as f:
                    dice_data = pickle.load(f)
                with open('codec_gui_metrics_aaaa.pkl', 'rb') as f:
                    codec_data = pickle.load(f)
                positive_samples, dpp, distances = codec_data
            except FileNotFoundError:
                # FIXED: Pass the GUI's cache to avoid duplicate training
                positive_samples, dpp, distances = code_counterfactuals(
                    pd.DataFrame(initial_instance, [0]).drop(['label', 'prediction'], axis=1),
                    constraints_path,
                    dataset_path,
                    fixed_features,
                    num_counterfactuals,
                    self.cached_models,  # Use GUI's cache
                    self.cached_transformers,  # Use GUI's cache
                    self.cached_constraints,
                    self.computation_queue
                )

            # Send results back to UI thread
            self.computation_queue.put({
                'type': 'complete',
                'data': {
                    'positive_samples': positive_samples,
                    'dpp': dpp,
                    'distances': distances,
                    'initial_instance': initial_instance,
                    'num_counterfactuals': num_counterfactuals
                }
            })

        except Exception as e:
            self.computation_queue.put({'type': 'error', 'error': str(e)})

    def handle_computation_complete(self, data):
        """Handle completion of computation in UI thread"""
        try:
            # Stop animation and close loading window
            self.stop_loading_animation()
            if hasattr(self, 'loading_window'):
                self.loading_window.destroy()

            positive_samples = data['positive_samples']
            dpp = data['dpp']
            distances = data['distances']
            initial_instance = data['initial_instance']
            num_counterfactuals = data['num_counterfactuals']

            # Clear existing results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            self.results_tree['columns'] = []

            # Clear existing results from scrollable frame
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()

            # Add initial instance
            initial_instance['label'] = 'Initial'
            self.add_result_row(initial_instance, 'Initial')

            # Sort counterfactuals by distance
            sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
            sorted_samples = positive_samples.iloc[sorted_indices].reset_index(drop=True)
            sorted_distances = [distances[i] for i in sorted_indices]

            # Update metrics display
            self.dpp_value.configure(text=f"{dpp:.4f}")
            distances_text = "\n".join([f"Option {i + 1}: {dist:.4f}" for i, dist in enumerate(sorted_distances)])
            self.distance_values.configure(text=distances_text)

            # Add counterfactual solutions in sorted order
            for i in range(num_counterfactuals):
                if i < len(sorted_samples):
                    solution = sorted_samples.iloc[i].to_dict()
                    solution['label'] = f'Option {i + 1}'
                    self.add_result_row(solution, 'Ours', sorted_distances[i], initial_instance)

            # Switch to results tab
            self.notebook.select(self.results_frame)

        except Exception as e:
            self.handle_computation_error(str(e))

    def handle_computation_error(self, error_message):
        """Handle computation error in UI thread"""
        self.stop_loading_animation()
        if hasattr(self, 'loading_window'):
            self.loading_window.destroy()
        messagebox.showerror("Error", f"Failed to generate counterfactuals: {error_message}")

    def setup_input_tab(self):
        # Create main frames
        left_frame = ttk.Frame(self.input_frame, width=800)
        left_frame.pack(side='left', fill='both', padx=5)
        left_frame.pack_propagate(False)

        right_frame = ttk.Frame(self.input_frame, width=450)
        right_frame.pack(side='left', fill='both', padx=5)
        right_frame.pack_propagate(False)

        # Input Parameters section
        input_frame = ttk.LabelFrame(left_frame, text="Input Parameters", style='Title.TLabelframe')
        input_frame.pack(fill='x', pady=5, padx=5)

        # Dataset CSV row
        ttk.Label(input_frame, text="Dataset:", style='Large.TLabel').grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(input_frame, textvariable=self.dataset_path, width=50, font=('Segoe UI', 14)).grid(row=0, column=1,
                                                                                                     padx=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_dataset, style='Large.TButton').grid(row=0, column=2)

        # Constraints file row
        ttk.Label(input_frame, text="Constraints:", style='Large.TLabel').grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(input_frame, textvariable=self.constraints_path, width=50, font=('Segoe UI', 14)).grid(row=1,
                                                                                                         column=1,
                                                                                                         padx=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_constraints, style='Large.TButton').grid(row=1,
                                                                                                            column=2)

        # Configure column weights
        input_frame.grid_columnconfigure(1, weight=1)  # Make column 1 expandable

        # Number of counterfactuals and model selector row
        ttk.Label(input_frame, text="Number of Counterfactuals:", style='Large.TLabel').grid(row=2, column=0,
                                                                                             sticky='w', pady=5)
        ttk.Entry(input_frame, textvariable=self.num_counterfactuals, width=10, font=('Segoe UI', 14)).grid(row=2,
                                                                                                            column=1,
                                                                                                            sticky='w',
                                                                                                            padx=5)

        # Dataset Preview section
        preview_frame = ttk.LabelFrame(left_frame, text="Dataset Preview (10 Random Samples)",
                                       style='Title.TLabelframe')
        preview_frame.pack(fill='both', expand=True, pady=5)

        # Create scrollbars for preview
        preview_scroll_y = ttk.Scrollbar(preview_frame)
        preview_scroll_y.pack(side='right', fill='y')

        preview_scroll_x = ttk.Scrollbar(preview_frame, orient='horizontal')
        preview_scroll_x.pack(side='bottom', fill='x')

        style = ttk.Style()
        style.configure("BigFont.Treeview", font=('Segoe UI', 15))
        style.configure("BigFont.Treeview.Heading", font=('Segoe UI', 16, 'bold'))
        style.configure("BigFont.Treeview", rowheight=40)

        # Create Treeview with increased height
        self.preview_tree = ttk.Treeview(preview_frame,
                                         height=7,
                                         yscrollcommand=preview_scroll_y.set,
                                         xscrollcommand=preview_scroll_x.set,
                                         style='BigFont.Treeview')
        self.preview_tree.pack(fill='both', expand=True)

        preview_scroll_y.config(command=self.preview_tree.yview)
        preview_scroll_x.config(command=self.preview_tree.xview)

        # Initial Point Selection section
        initial_point_frame = ttk.LabelFrame(right_frame,
                                             text="Input Instance",
                                             style='Title.TLabelframe')
        initial_point_frame.pack(fill='both', expand=True, pady=5)

        # Create canvas and scrollbar for initial point selection
        canvas = tk.Canvas(initial_point_frame)
        scrollbar = ttk.Scrollbar(initial_point_frame, orient="vertical", command=canvas.yview)

        self.initial_point_frame = ttk.Frame(canvas)
        self.initial_point_widgets = {}

        # Configure canvas
        canvas.create_window((0, 0), window=self.initial_point_frame, anchor="nw", width=400)

        # Update scrollregion when frame size changes
        self.initial_point_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Immutable Attributes section
        fixed_frame = ttk.LabelFrame(right_frame, text="Immutable Attributes", style='Title.TLabelframe')
        fixed_frame.pack(fill='x', pady=5)

        # Create frame for listbox and scrollbar
        list_frame = ttk.Frame(fixed_frame)
        list_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Create scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")

        # Create listbox with increased height
        self.features_list = tk.Listbox(list_frame,
                                        selectmode='multiple',
                                        height=6,
                                        font=('Segoe UI', 18),
                                        yscrollcommand=scrollbar.set)

        # Configure scrollbar
        scrollbar.config(command=self.features_list.yview)

        # Pack listbox and scrollbar
        self.features_list.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Generate button
        ttk.Button(right_frame, text="Generate Counterfactuals",
                   command=self.generate_counterfactuals,
                   style='Large.TButton').pack(pady=10)

    def update_preview(self):
        try:
            for col in self.preview_tree.get_children():
                self.preview_tree.delete(col)
            columns = list(self.df.columns)
            self.preview_tree['columns'] = columns
            self.preview_tree['show'] = 'headings'

            for col in columns:
                header_text = col

                # Calculate width based on header text length
                width = len(header_text) * 8  # Approximate pixels per character
                width = max(width, 40)  # Minimum width of 40 pixels

                self.preview_tree.heading(col, text=header_text, anchor='center')
                self.preview_tree.column(col, width=width, minwidth=30, anchor='center')

            random_samples = self.df.sample(n=min(10, len(self.df)))
            for _, row in random_samples.iterrows():
                values = [str(row[col]) for col in columns]
                item_id = self.preview_tree.insert('', 'end', values=values)

                if str(row['label']) == '1':
                    self.preview_tree.tag_configure('positive', background='#e6ffe6')
                    self.preview_tree.item(item_id, tags=('positive',))
                else:
                    self.preview_tree.tag_configure('negative', background='#ffe6e6')
                    self.preview_tree.item(item_id, tags=('negative',))

        except Exception as e:
            messagebox.showerror("Error", f"Failed to update preview: {str(e)}")

    def setup_initial_point_selection(self):
        try:
            # Clear existing widgets AND references
            for widget in self.initial_point_frame.winfo_children():
                widget.destroy()

            # IMPORTANT: Clear the widget references dictionary
            self.initial_point_widgets.clear()

            # Check if 'adult' is in the dataset filename (case insensitive)
            is_adult_dataset = 'adult' in self.dataset_path.get().lower()

            for i, column in enumerate(self.df.columns):
                if column == 'label':
                    continue

                frame = ttk.Frame(self.initial_point_frame)
                frame.pack(fill='x', padx=5, pady=2)

                # Attribute name with larger font
                label = ttk.Label(frame, text=f"{column}:",
                                  font=('Segoe UI', 14))
                label.pack(side='left', padx=(5, 10))

                if self.df[column].dtype == 'object':  # Categorical feature
                    unique_values = sorted(self.df[column].unique())
                    combo = ttk.Combobox(frame,
                                         values=unique_values,
                                         width=20,
                                         font=('Segoe UI', 14))

                    # Set default values based on column name and dataset
                    if is_adult_dataset:
                        if column == 'race':
                            default_value = 'Black'
                        elif column == 'native_country':
                            default_value = 'United_States'
                        elif column == 'relationship':
                            default_value = 'Not_in_family'
                        elif column == 'marital_status':
                            default_value = 'Never_married'
                        elif column == 'occupation':
                            default_value = 'Sales'
                        else:
                            default_value = unique_values[0] if unique_values else ""
                    else:
                        default_value = unique_values[0] if unique_values else ""

                    # Make sure the default value exists in the combobox options
                    if default_value in unique_values:
                        combo.set(default_value)
                    else:
                        # Fallback to first value if default isn't available
                        if unique_values:
                            combo.set(unique_values[0])

                    combo.pack(side='left', padx=5)
                    self.initial_point_widgets[column] = combo
                else:  # Continuous feature
                    min_val = int(self.df[column].min())
                    max_val = int(self.df[column].max())
                    var = tk.StringVar(value=str(min_val))

                    # Entry with larger font
                    entry = ttk.Entry(frame,
                                      textvariable=var,
                                      width=8,
                                      font=('Segoe UI', 14))
                    entry.pack(side='left', padx=5)

                    # Range label with larger font
                    range_label = ttk.Label(frame,
                                            text=f"Range: {min_val}-{max_val}",
                                            font=('Segoe UI', 14))
                    range_label.pack(side='left', padx=5)

                    self.initial_point_widgets[column] = entry

        except Exception as e:
            messagebox.showerror("Error", f"Failed to setup initial point selection: {str(e)}")

    def update_features_list(self):
        try:
            # Check if file path is set
            if not self.dataset_path.get():
                return

            # Clear any existing references first
            if hasattr(self, 'initial_point_widgets'):
                self.initial_point_widgets.clear()

            # Load dataset (use chunked loading for large files)
            file_size = os.path.getsize(self.dataset_path.get())
            if file_size > 100 * 1024 * 1024:  # 100MB
                self.df = load_dataset_chunked(self.dataset_path.get())
            else:
                self.df = pd.read_csv(self.dataset_path.get())

            # Clear and update features list
            self.features_list.delete(0, tk.END)
            for column in self.df.columns:
                if column == 'label':
                    continue
                self.features_list.insert(tk.END, column)

            # Update preview and initial point selection
            self.update_preview()
            self.setup_initial_point_selection()

        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found: {self.dataset_path.get()}")
        except pd.errors.EmptyDataError:
            messagebox.showerror("Error", "The CSV file is empty")
        except pd.errors.ParserError as e:
            messagebox.showerror("Error", f"Failed to parse CSV file: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")


import pandas as pd
import numpy as np
import random
import os
from datetime import datetime
import pickle

# Import the necessary functions from the GUI code
# Note: You'll need to have the paste.txt file's functions available
# Either copy them to this file or import from the original module

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)

# Define mappings and constraints
EDUCATION_NUM_MAP = {
    'HS_grad': 9,
    'Some_college': 10,
    'Bachelors': 13,
    'Masters': 14
}

# Attribute domains
WORKCLASS_VALUES = ['Federal_gov', 'Local_gov', 'Private', 'Self_emp_inc', 'State_gov']
EDUCATION_VALUES = ['Hs_grad', 'Some_college', 'Bachelors', 'Masters']
MARITAL_STATUS_VALUES = ['Never_married', 'Married_civ_spouse', 'Married_AF_spouse', 'Divorced']
OCCUPATION_VALUES = ['Sales', 'Tech_support', 'Craft_repair', 'Farming_fishing', 'Machine_op_inspct',
                     'Handlers_cleaners']
RELATIONSHIP_VALUES = ['Not_in_family', 'Husband', 'Wife', 'Other_relative', 'Own_child', 'Unmarried']
RACE_VALUES = ['Black', 'White', 'Asian_Pac_Islander']
SEX_VALUES = ['Male', 'Female']
NATIVE_COUNTRY_VALUES = ['United_States', 'Portugal', 'France', 'Taiwan', 'Vietnam', 'Canada', 'Jamaica']


def generate_realistic_profile():
    """Generate a realistic profile with logical consistency"""

    # Age distribution - mostly working age adults
    age = np.random.choice(
        range(22, 65),
        p=np.array([0.02 if x < 25 or x > 60 else 0.018 for x in range(22, 65)]) / sum(
            [0.02 if x < 25 or x > 60 else 0.018 for x in range(22, 65)])
    )

    # Sex - roughly equal distribution
    sex = random.choice(['Male', 'Female'])

    # Race - US demographics approximation
    race = np.random.choice(
        RACE_VALUES,
        p=[0.20, 0.70, 0.10]  # Black, White, Asian, Other
    )

    # Native country - mostly US
    native_country = np.random.choice(
        NATIVE_COUNTRY_VALUES
    )

    education = 'HS_grad'

    education_num = EDUCATION_NUM_MAP[education]

    # Marital status - age and sex appropriate
    if age < 25:
        marital_status = np.random.choice(['Never_married', 'Married_civ_spouse'], p=[0.8, 0.2])
    elif age < 35:
        marital_status = np.random.choice(MARITAL_STATUS_VALUES, p=[0.4, 0.5, 0.02, 0.08])
    else:
        marital_status = np.random.choice(MARITAL_STATUS_VALUES, p=[0.15, 0.65, 0.05, 0.15])

    # Relationship - must be consistent with marital status and sex
    if marital_status in ['Married_civ_spouse', 'Married_AF_spouse']:
        if sex == 'Male':
            relationship = 'Husband'
        else:
            relationship = 'Wife'
    elif marital_status == 'Never_married':
        if age < 25:
            relationship = np.random.choice(['Not_in_family', 'Own_child'], p=[0.6, 0.4])
        else:
            relationship = 'Not_in_family'
    else:  # Divorced
        relationship = np.random.choice(['Not_in_family', 'Unmarried'], p=[0.7, 0.3])

    # Workclass - education appropriate
    if education in ['Bachelors', 'Masters']:
        workclass = np.random.choice(WORKCLASS_VALUES, p=[0.15, 0.10, 0.60, 0.05, 0.10])
    else:
        workclass = np.random.choice(WORKCLASS_VALUES, p=[0.05, 0.15, 0.70, 0.05, 0.05])

    # Occupation - education and workclass appropriate
    if education in ['Bachelors', 'Masters']:
        # Higher education tends toward tech and sales
        occupation = np.random.choice(OCCUPATION_VALUES, p=[0.35, 0.35, 0.1, 0.05, 0.1, 0.05])
    else:
        # Lower education more evenly distributed
        occupation = np.random.choice(OCCUPATION_VALUES, p=[0.2, 0.1, 0.25, 0.1, 0.2, 0.15])

    # Hours per week - occupation appropriate
    if workclass in ['Self_emp_inc']:
        hours_per_week = np.random.randint(35, 60)
    elif occupation in ['Farming_fishing']:
        hours_per_week = np.random.randint(30, 55)
    else:
        hours_per_week = np.random.choice([40, 45, 50], p=[0.7, 0.2, 0.1])

    # Add some variation
    if random.random() < 0.1:  # 10% work part-time
        hours_per_week = np.random.randint(20, 35)

    return {
        'age': age,
        'workclass': workclass,
        'education': education,
        'education_num': education_num,
        'marital_status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'hours_per_week': hours_per_week,
        'native_country': native_country
    }


def select_immutable_features(profile):
    """Select immutable features for a profile"""
    # Always immutable
    immutable = ['age', 'race', 'sex', 'native_country','marital_status','relationship']

    # # Sometimes add education as immutable (30% of time)
    # if random.random() < 0.6:
    #     immutable.extend(['education', 'education_num'])
    #
    # # Sometimes add marital status as immutable for older married people
    # if random.random() < 0.5:
    #     immutable.append('marital_status')
    #     immutable.append('relationship')
    # else:
    #     immutable.append('workclass')
    #     immutable.append('occupation')
    # # If married status is fixed, relationship should be too

    return immutable


def generate_counterfactuals_for_profile(profile, immutable_features, dataset_path, constraints_path,cached_models,cached_transformers,cached_constraints,computation_queue):
    """Generate counterfactuals for a single profile"""

    # Create DataFrame for the profile
    query_instance = pd.DataFrame([profile])
    query_instance = query_instance.drop('label', axis=1, errors='ignore')

    try:

        # Generate counterfactuals
        counterfactuals, dpp_score, distances = code_counterfactuals(
            query_instances=query_instance,
            constraints_path=constraints_path,
            dataset_path=dataset_path,
            fixed_feat=immutable_features,
            k=2,  # Generate 3 counterfactuals
            model_cache=cached_models,transformer_cache=cached_transformers,constraints_cache=cached_constraints,progress_queue=computation_queue
        )

        return counterfactuals, dpp_score, distances

    except Exception as e:
        print(f"Error generating counterfactuals: {e}")
        return None, None, None


def save_results(profile, counterfactuals, immutable_features, output_dir, index):
    """Save the profile and counterfactuals to a CSV file"""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create filename
    filename = os.path.join(output_dir, f'profile_{index:04d}.csv')

    # Prepare data for saving
    all_data = []

    # Add original profile as first row
    profile_row = profile.copy()
    profile_row['type'] = 'original'
    profile_row['immutable_features'] = ','.join(immutable_features)
    all_data.append(profile_row)

    # Add counterfactuals
    if counterfactuals is not None and len(counterfactuals) > 0:
        for i, cf_row in counterfactuals.iterrows():
            cf_dict = cf_row.to_dict()
            cf_dict['type'] = f'counterfactual_{i + 1}'
            cf_dict['immutable_features'] = ''
            all_data.append(cf_dict)

    # Create DataFrame and save
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)

    return filename


# def main():
#     """Main function to generate profiles and counterfactuals"""
#
#     # Configuration
#     NUM_PROFILES = 50
#     DATASET_PATH = 'adult_clean.csv'  # Update with your dataset path
#     CONSTRAINTS_PATH = 'adult_good_adcs_test2.txt'  # Update with your constraints path
#     OUTPUT_DIR = 'counterfactual_results'
#
#     # Create output directory
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#
#     # Summary file
#     summary_data = []
#
#     print(f"Generating {NUM_PROFILES} profiles with counterfactuals...")
#
#     for i in range(NUM_PROFILES):
#         if i % 10 == 0:
#             print(f"Processing profile {i + 1}/{NUM_PROFILES}...")
#
#         # Generate profile
#         profile = generate_realistic_profile()
#
#         # Select immutable features
#         immutable_features = select_immutable_features(profile)
#         cached_models={}  # Use GUI's cache
#         cached_transformers={}  # Use GUI's cache
#         cached_constraints={}
#         computation_queue={}
#         # Generate counterfactuals
#         counterfactuals, dpp_score, distances = generate_counterfactuals_for_profile(
#             profile, immutable_features, DATASET_PATH, CONSTRAINTS_PATH,cached_models,cached_transformers,cached_constraints,computation_queue
#         )
#
#         # Save results
#         filename = save_results(profile, counterfactuals, immutable_features, OUTPUT_DIR, i)
#
#         # Add to summary
#         summary_entry = profile.copy()
#         summary_entry['profile_id'] = i
#         summary_entry['immutable_features'] = ','.join(immutable_features)
#         summary_entry['num_counterfactuals'] = len(counterfactuals) if counterfactuals is not None else 0
#         summary_entry['dpp_score'] = dpp_score if dpp_score is not None else None
#         summary_entry['output_file'] = filename
#         summary_data.append(summary_entry)
#
#     # Save summary
#     summary_df = pd.DataFrame(summary_data)
#     summary_df.to_csv(os.path.join(OUTPUT_DIR, 'summary.csv'), index=False)
#
#     print(f"\nCompleted! Results saved in '{OUTPUT_DIR}' directory.")
#     print(f"Summary saved as 'summary.csv'")
#
#     # Print statistics
#     print("\nGeneration Statistics:")
#     print(f"Total profiles generated: {len(summary_data)}")
#     print(f"Profiles with counterfactuals: {sum(1 for s in summary_data if s['num_counterfactuals'] > 0)}")
#
#     # Distribution statistics
#     print("\nAttribute Distributions:")
#     for attr in ['age', 'education', 'marital_status', 'race', 'sex']:
#         if attr == 'age':
#             print(f"{attr}: mean={summary_df[attr].mean():.1f}, std={summary_df[attr].std():.1f}")
#         else:
#             print(f"{attr}: {summary_df[attr].value_counts().to_dict()}")


# Alternative simplified version without the full counterfactual generation
def generate_profiles_only():
    """Generate only the profiles without counterfactuals (for testing)"""

    NUM_PROFILES = 100
    OUTPUT_DIR = 'profile_inputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_profiles = []

    print(f"Generating {NUM_PROFILES} realistic profiles...")

    for i in range(NUM_PROFILES):
        # Generate profile
        profile = generate_realistic_profile()

        # Select immutable features
        immutable_features = select_immutable_features(profile)

        # Add metadata
        profile['profile_id'] = i
        profile['immutable_features'] = ','.join(immutable_features)
        profile['label'] = 0  # Assuming all start with low income

        all_profiles.append(profile)

    # Save all profiles
    profiles_df = pd.DataFrame(all_profiles)
    profiles_df.to_csv(os.path.join(OUTPUT_DIR, 'all_profiles.csv'), index=False)

    # Save individual profile files
    for i, profile in enumerate(all_profiles):
        individual_df = pd.DataFrame([profile])
        individual_df.to_csv(os.path.join(OUTPUT_DIR, f'profile_{i:04d}.csv'), index=False)

    print(f"\nCompleted! {NUM_PROFILES} profiles saved in '{OUTPUT_DIR}' directory.")

    # Print statistics
    print("\nGeneration Statistics:")
    print("\nAge Distribution:")
    print(f"Mean: {profiles_df['age'].mean():.1f}, Std: {profiles_df['age'].std():.1f}")
    print(f"Min: {profiles_df['age'].min()}, Max: {profiles_df['age'].max()}")

    print("\nEducation Distribution:")
    print(profiles_df['education'].value_counts())

    print("\nMarital Status Distribution:")
    print(profiles_df['marital_status'].value_counts())

    print("\nSex Distribution:")
    print(profiles_df['sex'].value_counts())

    print("\nRace Distribution:")
    print(profiles_df['race'].value_counts())


# if __name__ == "__main__":
#     # Use this to generate profiles only (without counterfactuals)
#     generate_profiles_only()
#
#     # Uncomment this to run the full pipeline with counterfactual generation
#     main()

def main():
    root = tk.Tk()
    app = CounterfactualGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()