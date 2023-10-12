from cca_zoo.models import SCCA_PMD
import copy
from datetime import date
import dill
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection as fdrcorr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import time
import traceback
import sys
import os

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

today = str(date.today())
num = int(sys.argv[1])
print(num)
modality = str(sys.argv[2])
print(modality)

# Data paths
candpath = '/gpfs/milgram/pi/gee_dylan/candlab'
datapath = candpath + '/analyses/shapes/dwi/QSIPrep'
newri = '/gpfs/milgram/pi/gee_dylan/lms233/RI_Data/coded_output'
analysis = '/gpfs/milgram/pi/gee_dylan/candlab/analyses/shapes/dwi/QSIPrep/analysis'

# ********* EDIT HYPERPARAMETERS *********
cval_x = 0.5
cval_y = 0.5

full_df = pd.read_csv(
    analysis + '/DWI_RI_FullDataset_RegressedCovariates_n=107_2023-10-10_GFA_QA_RD_Adulthood_age_noEdu.csv')
# ****************************************

# Make output folder
if os.path.exists(analysis + '/model_output_{}_{}_{}_{}'.format(cval_x, cval_y, modality, today)):
    pass
else:
    os.mkdir(analysis + '/model_output_{}_{}_{}_{}'.format(cval_x,
                                                           cval_y, modality, today))

assert os.path.exists(
    analysis + '/model_output_{}_{}_{}_{}'.format(cval_x, cval_y, modality, today)) == True

outdir = analysis + \
    '/model_output_{}_{}_{}_{}'.format(cval_x, cval_y, modality, today)

# Run analysis
print("Running iteration {}, cval x={}, cval y = {}".format(num, cval_x, cval_y))

# ***FUNCTIONS***
# Resources: https://vitalflux.com/pca-explained-variance-concept-python-example/


def compute_covariance_explained(transformed_df1, transformed_df2):

    scaler = StandardScaler()
    if not np.allclose(transformed_df1.mean(), 0):
        transformed_df1 = pd.DataFrame(scaler.fit_transform(
            transformed_df1), columns=transformed_df1.columns)
    if not np.allclose(transformed_df2.mean(), 0):
        transformed_df2 = pd.DataFrame(scaler.fit_transform(
            transformed_df2), columns=transformed_df2.columns)

    # R code from Nat Comms paper: diag(covmat)^2 / sum(diag(covmat)^2)
    cov_arr = np.ones((transformed_df1.shape[1],))

    for i in range(0, len(transformed_df1.columns)):
        cov_val = np.cov(transformed_df1.iloc[:, i], transformed_df2.iloc[:, i], ddof=1)[
            0][1]  # Compute pairwise covariance between components in stress and dwi matrices
        cov_arr[i] = cov_val  # Append this value to empty array

    # covariance explained = per element in array, covariance^2 divided by all summed covariance^2
    exp_var = cov_arr**2 / np.sum(cov_arr**2)

    return exp_var


def split_sample(df, size, seed):
    from sklearn.model_selection import train_test_split

    # Randomly split and shuffle data
    p1, p2 = train_test_split(
        df, test_size=size, random_state=seed, shuffle=True)
    p1_resampled = p1.sample(replace=True, axis=0, frac=1, random_state=seed).iloc[0:len(
        p2), :]  # Replace p2 split with resampled p1 data from p1
    p_df = pd.DataFrame(pd.concat(
        [p1, p1_resampled], axis=0), columns=df.columns).reset_index(drop=True)

    return p_df


def fit_model(in_xmat, in_ymat, c_vals):
    n_components = in_xmat.shape[1]

    # Fit CCA model
    model = SCCA_PMD(latent_dims=n_components,
                     random_state=0,
                     scale=True,
                     centre=True,
                     max_iter=1000,
                     # maxvar=False,
                     c=c_vals)

    model.fit((in_xmat, in_ymat))
    model_score = model.score((in_xmat, in_ymat))
    model_results = model.pairwise_correlations((in_xmat, in_ymat))[0][1]

    gen_colnames_stress = []
    gen_colnames_dti = []

    for i in range(0, n_components):
        gen_colnames_stress.append('Component_{}_Stress'.format(i + 1))
        gen_colnames_dti.append('Component_{}_DTI'.format(i + 1))

    # Transform X and y matrices to see how CCA fitting changes variables
    tranf_df_bx = pd.DataFrame(model.transform((in_xmat, in_ymat))[
                               0], columns=gen_colnames_stress).dropna(axis=1)
    tranf_df_fa = pd.DataFrame(model.transform((in_xmat, in_ymat))[
                               1], columns=gen_colnames_dti).dropna(axis=1)

    return model.get_factor_loadings([in_xmat, in_ymat]), model.weights, model_score, model_results, tranf_df_bx, tranf_df_fa


# *************** Run Model ***************
analysis_columns = ['all_0.0_regr', 'all_1.0_regr', 'all_2.0_regr',
                    'all_3.0_regr', 'all_4.0_regr', 'all_5.0_regr',
                    'all_6.0_regr', 'all_7.0_regr', 'all_8.0_regr',
                    'all_9.0_regr', 'all_10.0_regr', 'all_11.0_regr',
                    'all_12.0_regr', 'all_13.0_regr', 'all_14.0_regr',
                    'all_15.0_regr', 'all_16.0_regr', 'all_17.0_regr',
                    'all_18.0_regr']

# Select resampled data
resampled_data = split_sample(
    full_df, size=.33, seed=num).reset_index(drop=True)

# Select x and y data
in_xmat_data = pd.DataFrame(resampled_data[analysis_columns].replace(
    np.nan, 0.0), columns=analysis_columns)  # Replace NaNs with 0 (representing no endorsement)
in_ymat_data = pd.DataFrame(resampled_data.loc[:, "{}_AF_left_regr".format(modality):"{}_ST_PREM_right_regr".format(modality)],
                            columns=resampled_data.loc[:, "{}_AF_left_regr".format(modality):"{}_ST_PREM_right_regr".format(modality)].columns)
print("ymat data: {}_AF_left_regr:{}_ST_PREM_right_regr".format(modality, modality))

assert np.nan not in in_xmat_data
assert np.nan not in in_ymat_data

# Set regularization parameters

loadings, weights, main_model_score, main_model_results, model_df_bx, model_df_fa = fit_model(
    in_xmat_data, in_ymat_data, [cval_x, cval_y])

# Compute covariance of components (Helpful: https://towardsdatascience.com/5-things-you-should-know-about-covariance-26b12a0516f1)
model_exp_var = compute_covariance_explained(model_df_bx, model_df_fa)

print("Variate 1 correlation strength: {}".format(main_model_score[0]))

# SHUFFLED
# Shuffle X-data on both axes
in_xmat_shuff = in_xmat_data.sample(
    frac=1, replace=True, random_state=num).reset_index(drop=True)

shuff_loadings, shuff_weights, model_score_shuff, model_results_shuff, tranf_bx_shuff, tranf_fa_shuff = fit_model(
    in_xmat_shuff, in_ymat_data, [cval_x, cval_y])

# Compute covariance of components (Helpful: https://towardsdatascience.com/5-things-you-should-know-about-covariance-26b12a0516f1)
exp_var = compute_covariance_explained(tranf_bx_shuff, tranf_fa_shuff)

pd.DataFrame(main_model_results).to_csv(
    outdir + '/Main_Bootstrapped_Results_Correlation_iteration{}.csv'.format(num))
pd.DataFrame(model_exp_var).to_csv(
    outdir + '/Main_Bootstrapped_Results_Covariation_iteration{}.csv'.format(num))
pd.DataFrame(loadings[0]).to_csv(
    outdir + '/Main_Loadings_AdvExp_iteration{}.csv'.format(num))
pd.DataFrame(loadings[1]).to_csv(
    outdir + '/Main_Loadings_DWI_iteration{}.csv'.format(num))
pd.DataFrame(weights[0]).to_csv(
    outdir + '/Main_Weights_AdvExp_iteration{}.csv'.format(num))
pd.DataFrame(weights[1]).to_csv(
    outdir + '/Main_Weights_DWI_iteration{}.csv'.format(num))
pd.DataFrame(model_df_bx).to_csv(
    outdir + '/Transformed_Data_AdvExp_iteration{}.csv'.format(num))
pd.DataFrame(model_df_fa).to_csv(
    outdir + '/Transformed_Data_DWI_iteration{}.csv'.format(num))

# Shuffled
pd.DataFrame(model_results_shuff).to_csv(
    outdir + '/Shuffled_Bootstrapped_Results_Correlation_iteration{}.csv'.format(num))
pd.DataFrame(exp_var).to_csv(
    outdir + '/Shuffled_Bootstrapped_Results_Covariation_iteration{}.csv'.format(num))
pd.DataFrame(shuff_loadings[0]).to_csv(
    outdir + '/Shuffled_Loadings_AdvExp_iteration{}.csv'.format(num))
pd.DataFrame(shuff_loadings[1]).to_csv(
    outdir + '/Shuffled_Loadings_DWI_iteration{}.csv'.format(num))
pd.DataFrame(shuff_weights[0]).to_csv(
    outdir + '/Shuffled_Weights_AdvExp_iteration{}.csv'.format(num))
pd.DataFrame(shuff_weights[1]).to_csv(
    outdir + '/Shuffled_Weights_DWI_iteration{}.csv'.format(num))
