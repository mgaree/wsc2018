# -*- coding: utf-8 -*-
"""
Helper methods for preparing experimental design matrix.
"""

import csv
import pandas as pd

"""
class WSCModel:  # signature for reference
    def __init__(self, seed, collect_step_data, N, structure_instance_num, g,
                 randomize_update_seq, b_distro, b_0_is_zero, normalize_bs,
                 error_variance, normalize_yis, uninformed_rate):
"""

factors = dict(
    N=[100, 500, 1000],
    structure_instance_num=list(range(1, 14+1)),
    g=[1, 5, 'N/4', 'N'],
    randomize_update_seq=[True, False],
    b_distro=['U01', 'U-11', 'N01'],
    b_0_is_zero=[True, False],
    normalize_bs=['no', 'agent', 'network'],
    error_variance=[0.5, 1, 2],
    normalize_yis=[True, False],
    uninformed_rate=[0, 0.1, 0.25]
    )

def get_g_val(g, N):
    # Convert g=N/4 etc. into an actual number
    try:
        return int(g)
    except ValueError:
        if g == 'N':
            return N
        else:
            denom = int(g.split('/')[-1])
            return N / denom

def reverse_g_val(g_val, N):
    # Inverse operation of get_g_val
    for g in factors['g']:
        if get_g_val(g, N) == g_val:
            return g
    return False

# Used Sanchez (2011) spreadsheet to make design matrix, copied into new workbook,
# removed duplicate trials (I used the 'up to 29 factors' worksheet to get more
# design points), manually added column headers
# that match the keys in `factors` above (and match the kwargs for WSCModel),
# and saved to a csv.
#
design_matrix_integers_filename = 'experimental_design_matrix_integers.csv'
design_matrix_filename = 'design_matrix_WSC2018.csv'

def level_int_to_value(factor, level_int):
    level_int = int(level_int)
    try:
        return factors[factor][level_int-1]
    except Exception:
        print(factor, level_int, ' ERROR')
        raise

def create_level_value_design_matrix():
    """The matrix is coded to levels 1, 2, ..., so I need to map the level values
    onto that design, and marry up with a Trial column, then save to output
    file csv.

    """
    with open(design_matrix_integers_filename, 'rt', encoding='utf-8-sig', newline='') as f_in:
        with open(design_matrix_filename, 'w', newline='') as f_out:
            reader = csv.DictReader(f_in)
            fieldnames = reader.fieldnames

            writer = csv.DictWriter(f_out, fieldnames=['Trial'] + fieldnames)
            writer.writeheader()

            trial_num = 1
            for row in reader:
                row['Trial'] = trial_num
                trial_num += 1

                for factor in fieldnames:
                    row[factor] = level_int_to_value(factor, row[factor])
                    if factor == 'g':
                        row[factor] = get_g_val(row[factor], row['N'])
                writer.writerow(row)


if __name__ == '__main__':
    create_level_value_design_matrix()
    df = pd.read_csv(design_matrix_filename)

    # test design for class init
    from model_wsc import WSCModel
    from tqdm import tqdm

    with tqdm(total=len(df), desc='Total', unit='dp') as pbar_total:
        for index, row in df.iterrows():
            m = WSCModel(1, True, *list(row)[1:])
            m.step()
            pbar_total.update()