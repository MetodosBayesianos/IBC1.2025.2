# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python [conda env:causal_book_py38]
#     language: python
#     name: conda-env-causal_book_py38-py
# ---

# %%
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.metrics import mean_absolute_percentage_error

import dowhy
from dowhy import CausalModel

from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from lightgbm import LGBMRegressor, LGBMClassifier

import networkx as nx

from tqdm import tqdm

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import graphviz

# %%
dowhy.__version__

# %%
COLORS = [
    '#00B0F0',
    '#FF0000',
    '#B0F000'
]

# %% [markdown]
# # Chapter 09
#
#

# %% [markdown]
# ## Matching

# %% [markdown]
# ### Read the data

# %%
earnings_data = pd.read_csv(r'./data/ml_earnings.csv')

# %%
earnings_data.shape

# %%
earnings_data.head()

# %%
earnings_groupby = earnings_data.groupby(['age', 'took_a_course']).mean()

# %%
# Compute naive estimate 
treatment_avg = earnings_data.query('took_a_course==1')['earnings'].mean()
cntrl_avg = earnings_data.query('took_a_course==0')['earnings'].mean()

treatment_avg - cntrl_avg

treatment_avg = earnings_data.query('took_a_course==1')['earnings'].mean()
cntrl_avg = earnings_data.query('took_a_course==0')['earnings'].mean()

mean_earnings = earnings_data.groupby(['age', 'took_a_course'])['earnings'].mean().unstack()
mean_earnings = mean_earnings.dropna(subset=[True, False])
mean_earnings['diferencia'] = mean_earnings[True] - mean_earnings[False]

print(mean_earnings[['diferencia']])


# %% [markdown]
# ### Define the graph

# %%
# Construct the graph (the graph is constant for all iterations)
nodes = ['took_a_course', 'earnings', 'age']
edges = [
    ('took_a_course', 'earnings'),
    ('age', 'took_a_course'),
    ('age', 'earnings')
]

# Generate the GML graph
gml_string = 'graph [directed 1\n'

for node in nodes:
    gml_string += f'\tnode [id "{node}" label "{node}"]\n'

for edge in edges:
    gml_string += f'\tedge [source "{edge[0]}" target "{edge[1]}"]\n'

gml_string += ']'

# %%
# Instantiate the CausalModel 
# Paso 1 (Estructura causal)
model = CausalModel(
    data=earnings_data,
    treatment='took_a_course',
    outcome='earnings',
    graph=gml_string
)


# %%
model.view_model()

# %% [markdown]
# ### Get the estimand

# %%
# Get the estimand
# Paso 2. Estimando (cómo vamos a identificar el efecto causal)
estimand = model.identify_effect()

print(estimand)

# %% [markdown]
# ### Estimate the effect

# %%
# Get estimate (Matching)
# Paso 3: Estimación
estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.distance_matching',
    target_units='ate',
    method_params={'distance_metric': 'minkowski', 'p': 2})


# %%
estimate.value

# %%
# Paso 4: validación,
refutation = model.refute_estimate(
    estimand=estimand, 
    estimate=estimate,
    method_name='random_common_cause')

# %%
print(refutation)

# %% [markdown]
# ## Inverse Probability Weighting (IPW)

# %%
# Get estimate (IPW weighting)
estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.propensity_score_weighting',
    target_units='ate'
)

# %%
estimate.value

# %%
(estimate.value - 10e3) / 10e3


# %% [markdown]
# ## S-Learner: The Lone Ranger

# %%
def plot_effect(effect_true, effect_pred, figsize=(10, 7), ylim=(5000, 22000)):
    plt.figure(figsize=figsize)
    plt.scatter(effect_true, effect_pred, color=COLORS[0])
    plt.plot(np.sort(effect_true), np.sort(effect_true), color=COLORS[1], alpha=.7, label='Perfect model')
    plt.xlabel('$True\ effect$', fontsize=14, alpha=.5)
    plt.ylabel('$Predicted\ effect$', fontsize=14, alpha=.5)
    plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.show()


# %% [markdown]
# ### Read the data

# %%
# Train set
earnings_interaction_train = pd.read_csv(r'./data/ml_earnings_interaction_train.csv')

# Test set
earnings_interaction_test = pd.read_csv(r'./data/ml_earnings_interaction_test.csv')

# %%
earnings_interaction_train.shape, earnings_interaction_test.shape

# %%
# Train 
earnings_interaction_train.head()

# %%
# Test
earnings_interaction_test.head()

# %% [markdown]
# ### Define the graph

# %%
# Construct the graph (the graph is constant for all iterations)
nodes = ['took_a_course', 'python_proficiency', 'earnings', 'age']
edges = [
    ('took_a_course', 'earnings'),
    ('age', 'took_a_course'),
    ('age', 'earnings'),
    ('python_proficiency', 'earnings')
]

# Generate the GML graph
gml_string = 'graph [directed 1\n'

for node in nodes:
    gml_string += f'\tnode [id "{node}" label "{node}"]\n'

for edge in edges:
    gml_string += f'\tedge [source "{edge[0]}" target "{edge[1]}"]\n'

gml_string += ']'

# %%
# Instantiate the CausalModel 
model = CausalModel(
    data=earnings_interaction_train,
    treatment='took_a_course',
    outcome='earnings',
    effect_modifiers='python_proficiency',
    graph=gml_string
)

# %%
model.view_model()

# %% [markdown]
# ### Get the estimand

# %%
# Get the estimand
# Paso 2.
estimand = model.identify_effect()

print(estimand)

# %% [markdown]
# ### Estimate the effect

# %%
# Get estimate (S-Learner)
# Paso 3. Estimación.
estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.metalearners.SLearner',
    target_units='ate',
    method_params={
        'init_params': {
            'overall_model': LGBMRegressor(n_estimators=500, max_depth=10)
        },
        'fit_params': {}
    })
)

# %%
estimate.value == np.array(estimate.cate_estimates).mean()


# %% [markdown]
# ### Refute

# %%
refutation = model.refute_estimate(
    estimand=estimand, 
    estimate=estimate,
    method_name='random_common_cause')

# %%
print(refutation)

# %%
refutation = model.refute_estimate(
    estimand=estimand, 
    estimate=estimate,
    method_name='placebo_treatment_refuter')

# %%
print(refutation)

# %% [markdown]
# ### Predict on test data

# %%
# Compute predictions

effect_pred = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.metalearners.SLearner',
    target_units= earnings_interaction_test.drop(
        ['true_effect', 'took_a_course'],
        axis=1
    ),
    method_params={
        'init_params': {
            'overall_model': LGBMRegressor(n_estimators=500, max_depth=10)
        },
        'fit_params': {}
    }
).cate_estimates


# Get the true effect
effect_true = earnings_interaction_test['true_effect'].values

# %%
# Compute the error 
mean_absolute_percentage_error(effect_true, effect_pred)

# %%
plot_effect(
    effect_true=effect_true,
    effect_pred=effect_pred,
)

# %% [markdown]
# ### What happens when your dataset is small?

# %%
# Instantiate the CausalModel 
model_small = CausalModel(
    data=earnings_interaction_train.sample(100),
    treatment='took_a_course',
    outcome='earnings',
    effect_modifiers='python_proficiency',
    graph=gml_string
)

# %%
# Get estimate (S-Learner)
estimate = model_small.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.metalearners.SLearner',
    target_units='ate',
    method_params={
        'init_params': {
            'overall_model': LGBMRegressor(n_estimators=500, max_depth=10)
        },
        'fit_params': {}
    })

# %%
# Compute predictions
effect_pred = model_small.causal_estimator.effect(earnings_interaction_test.drop(['true_effect', 'took_a_course'], axis=1))

# Get the true effect
effect_true = earnings_interaction_test['true_effect'].values

# %%
# Compute the error 
mean_absolute_percentage_error(effect_true, effect_pred)

# %%
plot_effect(
    effect_true=effect_true,
    effect_pred=effect_pred,
)

# %% [markdown]
# ## T-Learner: Together We Can Do More

# %% [markdown]
# ### Estimate the effect

# %%
# Get estimate (T-Learner)
estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.metalearners.TLearner',
    target_units='ate',
    method_params={
        'init_params': {
            'models': [
                LGBMRegressor(n_estimators=200, max_depth=10),
                LGBMRegressor(n_estimators=200, max_depth=10)
            ]
        },
        'fit_params': {}
    })

# %%
estimate.cate_estimates.mean()

# %% [markdown]
# ### Predict on test data

# %%
# Compute predictions
effect_pred = model.causal_estimator.effect(earnings_interaction_test.drop(['true_effect', 'took_a_course'], axis=1))

# Get the true effect
effect_true = earnings_interaction_test['true_effect'].values

# %%
# Compute the error 
mean_absolute_percentage_error(effect_true, effect_pred)

# %%
plot_effect(
    effect_true=effect_true,
    effect_pred=effect_pred,
)

# %% [markdown]
# ## X-Learner: A Step Further

# %% [markdown]
# ### Estimate the effect

# %%
# Get estimate (X-Learner)
estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.metalearners.XLearner',
    target_units='ate',
    method_params={
        'init_params': {
            'models': [
                LGBMRegressor(n_estimators=50, max_depth=10),
                LGBMRegressor(n_estimators=50, max_depth=10)
            ],
            'cate_models': [
                LGBMRegressor(n_estimators=50, max_depth=10),
                LGBMRegressor(n_estimators=50, max_depth=10)
            ]
        },
        'fit_params': {},
    })

# %%
# X-Learner with just one specified model - equivalent to the cell above
estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.metalearners.XLearner',
    target_units='ate',
    method_params={
        'init_params': {
            'models': LGBMRegressor(n_estimators=50, max_depth=10),
        },
        'fit_params': {},
    })

# %%
estimate.cate_estimates.mean()

# %% [markdown]
# ### Predict on test data

# %%
# Compute predictions
effect_pred = model.causal_estimator.effect(earnings_interaction_test.drop(['true_effect', 'took_a_course'], axis=1))

# Get the true effect
effect_true = earnings_interaction_test['true_effect'].values

# %%
# Compute the error 
mean_absolute_percentage_error(effect_true, effect_pred)

# %%
plot_effect(
    effect_true=effect_true,
    effect_pred=effect_pred,
)

# %% [markdown]
# ### X-Learner on small data

# %%
# Get estimate (X-Learner)
estimate = model_small.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.metalearners.XLearner',
    target_units='ate',
    method_params={
        'init_params': {
            'models': LGBMRegressor(n_estimators=50, max_depth=10)
        },
        'fit_params': {}
    })

# %%
# Compute predictions
effect_pred = model_small.causal_estimator.effect(earnings_interaction_test.drop(['true_effect', 'took_a_course'], axis=1))

# Get the true effect
effect_true = earnings_interaction_test['true_effect'].values

# %%
# Compute the error 
mean_absolute_percentage_error(effect_true, effect_pred)

# %%
plot_effect(
    effect_true=effect_true,
    effect_pred=effect_pred,
)

# %% [markdown]
# ## Meta-Learners on experiental data (Kevin Hillstrom)

# %%
# Read in the data
hillstrom_clean = pd.read_csv(r'./data/hillstrom_clean.csv')

# Read in labels mapping
with open(r'./data/hillstrom_clean_label_mapping.json', 'r') as f:
    hillstrom_labels_mapping = json.load(f)

# %%
hillstrom_clean.head()

# %%
hillstrom_clean.columns

# %% [markdown]
# ## Doubly Robust Methods: Let’s Get More!

# %% [markdown]
# ### Estimate the effect

# %%
# Get estimate (Doubly robust)
estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.dr.DRLearner',
    target_units='ate',
    method_params={
        'init_params': {
            'model_propensity': LogisticRegression(),
            'model_regression': LGBMRegressor(n_estimators=1000, max_depth=10)
        },
        'fit_params': {}
    })

# %%
estimate.cate_estimates.mean()

# %% [markdown]
# ### Predict on test data

# %%
# Compute predictions
effect_pred = model.causal_estimator.effect(earnings_interaction_test.drop(['true_effect', 'took_a_course'], axis=1))

# Get the true effect
effect_true = earnings_interaction_test['true_effect'].values

# %%
# Compute the error 
mean_absolute_percentage_error(effect_true, effect_pred)

# %%
plot_effect(
    effect_true=effect_true,
    effect_pred=effect_pred,
)

# %% [markdown]
# ## If Machine Learning is Cool, How About Double Machine Learning?

# %% [markdown]
# ### Estimate the effect

# %%
# Get estimate (DML)
estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name='backdoor.econml.dml.LinearDML',
    target_units='ate',
    method_params={
        'init_params': {
            'model_y': LGBMRegressor(n_estimators=500, max_depth=10),
            'model_t': LGBMRegressor(n_estimators=500, max_depth=10),
        },
        'fit_params': {}
    })

# %%
estimate.cate_estimates.mean()

# %% [markdown]
# ### Predict on test data

# %%
# Compute predictions
effect_pred = model.causal_estimator.effect(earnings_interaction_test.drop(['true_effect', 'took_a_course'], axis=1))

# Get the true effect
effect_true = earnings_interaction_test['true_effect'].values

# %%
# Compute the error 
mean_absolute_percentage_error(effect_true, effect_pred)

# %%
plot_effect(
    effect_true=effect_true,
    effect_pred=effect_pred,
)

# %% [markdown]
# ## Matching in the wild - simulation

# %% [markdown]
# ### Probability of finding at least one match per row

# %%
results_per_dimension = {}

for d in range(2, 19):
    results = []
    
    N_SAMPLES = 1000
    DIM_X = d

    for i in tqdm(range(100)):

        X = np.random.binomial(n=1, p=.5, size=(N_SAMPLES, DIM_X))

        local_results = []

        for row in range(X.shape[0]):
            # Did we find at least 1 match for `row`?
            success = (np.where(np.where(X == X[row], 1, 0).sum(axis=1) == DIM_X, 1, 0).sum() - 1) > 0
            local_results.append(success)

        results.append(np.array(local_results).mean())
        
    results_per_dimension[d] = results

# %%
# Compute statistics
mean_p = pd.DataFrame(results_per_dimension).mean(axis=0).values
sd_p = pd.DataFrame(results_per_dimension).std(axis=0).values

# %%
# Plot
plt.figure(figsize=(12, 8))
plt.fill_between(range(2, 19), mean_p - sd_p*2, mean_p + sd_p*2, color=COLORS[0], alpha=.2, label='$+/-2SD$')
plt.plot(range(2, 19), mean_p, color=COLORS[0], lw=2, label='$\mu$')
plt.legend()
plt.xlabel('$Dimensionalty \ of \ X$', alpha=.5)
plt.ylabel('$Probability \ of \ finding \ at \ least \ one \ match$', alpha=.5)
plt.show()
