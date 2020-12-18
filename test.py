import os
import numpy as np
import pandas as pd
from tqdm.notebook import trange, tqdm
# from IPython.display import display, HTML
from copy import deepcopy
import random
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from tti_explorer import config, utils
from tti_explorer.case import simulate_case, CaseFactors
from tti_explorer.contacts import EmpiricalContactsSimulator
from tti_explorer.strategies import TTIFlowModel, RETURN_KEYS

# import warnings
# warnings.filterwarnings('ignore')
# sns.set_style("darkgrid")

def print_doc(func):
    print(func.__doc__)


def load_csv(pth):
    return np.loadtxt(pth, dtype=int, skiprows=1, delimiter=",")


def do_simulation(policy_name, case_config, contacts_config, n_cases, 
                  random_state=None, dict_params=None):
    # Prepare simulation configuration
    case_config = case_config
    contacts_config = contacts_config
    
    if dict_params is None:
        policy_config = config.get_strategy_configs("delve", policy_name)[policy_name]
        factor_config = utils.get_sub_dictionary(policy_config, config.DELVE_CASE_FACTOR_KEYS)
        strategy_config = utils.get_sub_dictionary(policy_config, config.DELVE_STRATEGY_FACTOR_KEYS)
    else:
        for key, value in dict_params.items():
            policy_config = config.get_strategy_configs("delve", policy_name)[policy_name]
            policy_config[key] = value
            factor_config = utils.get_sub_dictionary(policy_config, config.DELVE_CASE_FACTOR_KEYS)
            strategy_config = utils.get_sub_dictionary(policy_config, config.DELVE_STRATEGY_FACTOR_KEYS)

    # Initialise configuration model
    if random_state is None:
        rng = np.random.RandomState(random.randint(0, 1000))
    else:
        rng = np.random.RandomState(random_state)

    simulate_contacts = EmpiricalContactsSimulator(over18, under18, rng)
    tti_model = TTIFlowModel(rng, **strategy_config)

    # Aggregates all cases outputs
    outputs = list()

    # Perform simulation
    for _ in range(n_cases):
        case = simulate_case(rng, **case_config)
        case_factors = CaseFactors.simulate_from(rng, case, **factor_config)
        contacts = simulate_contacts(case, **contacts_config)
        res = tti_model(case, contacts, case_factors)
        outputs.append(res)
    
    return outputs


def summarise_simulation_results(outputs, policy_name, case_config):
    to_show = [
        RETURN_KEYS.base_r,
        RETURN_KEYS.reduced_r,
        RETURN_KEYS.man_trace,
        RETURN_KEYS.app_trace,
        RETURN_KEYS.tests
    ]

    # Scale factor to turn simulation numbers into UK population numbers
    nppl = case_config['infection_proportions']['nppl']
    scales = [1, 1, nppl, nppl, nppl]

    results = pd.DataFrame(
        outputs
    ).mean(
        0
    ).loc[
        to_show
    ].mul(
        scales
    ).to_frame(
        name=f"Simulation results: {policy_name.replace('_', ' ')}"
    ).rename(
        index=lambda x: x + " (k per day)" if x.startswith("#") else x
    )

    display(results.round(1))


# Main
if __name__ == "__main__":

    # Load data
    over18 = load_csv("data/bbc-pandemic/contact_distributions_o18.csv")
    under18 = load_csv("data/bbc-pandemic/contact_distributions_u18.csv")

    policy_name_params = [f"{i}_{j}" for i in ["S1", "S2", "S3", "S4", "S5"] for j in ["no_TTI", "symptom_based_TTI", "test_based_TTI", "test_based_TTI_test_contacts"]]
    case_config = config.get_case_config("delve")
    contacts_config = config.get_contacts_config("delve")
    punder18_params = [0, 0.25, 0.5, 0.75, 1]
    n_experiments = 5 # number of simulations performed for a particular value

    # Used to store the experimentation result
    dict_age = {policy_name: {} for policy_name in policy_name_params}
    for punder18_param in punder18_params:
        for key in dict_age.keys():
            dict_age[key][punder18_param] = []
    
    # Try all possible policy and age parameters
    for policy_name in tqdm(policy_name_params):
        
        for punder18 in punder18_params:
            case_config["p_under18"] = punder18
            
            # For each case, experiment n times
            for _ in range(n_experiments):
                outputs = do_simulation(policy_name, case_config, 
                    contacts_config, n_cases=20000)
                dict_age[policy_name][punder18].append(outputs)
 
    
    key_params = [f"{i}_{j}" for i in ["S1", "S2", "S3", "S4", "S5"] for j in ["no_TTI", "symptom_based_TTI", "test_based_TTI", "test_based_TTI_test_contacts"]]
    df_result = {"strategy": [], 
                "gov_policy": [],
                "age_prob": [],
                "effective_r": [], 
                "base_r": [],
                "manual_traces": [], 
                "app_traces": [], 
                "test_needed": [],
                "persondays_quarantined": []}
    nppl = 120

    for key in key_params:
        strategy = key[:2]
        gov_policy = key[3:]
        
        for punder18 in punder18_params:

            for i in range(n_experiments):
                result_format = pd.DataFrame(dict_age[key][punder18][i]).mean(0)
                df_result["age_prob"].append(punder18)
                df_result["strategy"].append(strategy)
                df_result["gov_policy"].append(gov_policy)
                df_result["effective_r"].append(result_format.loc["Effective R"])
                df_result["base_r"].append(result_format.loc["Base R"])
                df_result["manual_traces"].append(result_format.loc["# Manual Traces"] * nppl)
                df_result["app_traces"].append(result_format.loc["# App Traces"] * nppl)
                df_result["test_needed"].append(result_format.loc["# Tests Needed"] * nppl)
                df_result["persondays_quarantined"].append(result_format.loc["# PersonDays Quarantined"])
    

    # Convert to dataframe
    df_result = pd.DataFrame(df_result)
    df_result.to_csv("experiments_result/experiment-age.csv", index=False)