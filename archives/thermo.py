from equilibrator_assets.local_compound_cache import LocalCompoundCache
from equilibrator_api import ComponentContribution, Q_
import cvxpy
import pymongo
import numpy as np

lc = LocalCompoundCache()
lc.load_cache('/Users/yashchainani96/PycharmProjects/pathway_discovery/chemaxon/'
              'equilibrator-assets-master/notebooks/compounds.sqlite')

cc = ComponentContribution(ccache=lc.ccache)

def parse_rxn(rxn_str: str):
    """
    Separates a reaction string into a list of reactants and of products
    reaction string is of the form substrate_smiles + cofactor_smiles = product_smiles + cofactor_smiles
    this will return [substrate, cofactor] and [product, cofactor]

    :param rxn_str (str): reaction string of the form: "smiles_A + smiles_B = smiles_C + smiles_D"
    :return reactants_list (list): list of reactant smiles
    :return products_list (list): list of product smiles
    """
    reactants, products = rxn_str.split(' = ')
    reactants_list = reactants.split(' + ')
    products_list = products.split(' + ')
    return reactants_list, products_list

def rxn_constructor_in_accession_IDs(reactants_list: list,
                                     products_list: list):
    """
    Construct a reaction string of the form substrate + cofactor = product + cofactor
    except instead of SMILES, eQuilibrator accession IDs are used instead for all species
    this output can directly be fed into eQuilibrator to calculate reaction dG

    :param reactants_list (list): list of reactants on the LHS (includes cofactors)
    :param products_list (list): list of products on the RHS (includes cofactors)
    :return rxn_str_in_db_accessions (str): reaction string consisting of eQuilibrator accession IDs
    """
    reactant_cpd_objects = lc.get_compounds(reactants_list)  # create eQuilibrator compound objects for all LHS species
    product_cpd_objects = lc.get_compounds(products_list)  # create eQuilibrator compound objects for all RHS species
    rxn_str_in_db_accessions = ''  # initialize empty string

    # try:
    for reactant in reactant_cpd_objects:

        # get eQuilibrator accession IDs for all LHS species (if decomposable)
        reactant_accession = reactant.get_accession()
        rxn_str_in_db_accessions += reactant_accession
        rxn_str_in_db_accessions += ' + '

    # remove extra ' + ' sign on the right
    rxn_str_in_db_accessions = rxn_str_in_db_accessions.rstrip(' + ')

    # add ' = ' sign before switching over to products side
    rxn_str_in_db_accessions += ' = '

    for product in product_cpd_objects:

        # get eQuilibrator accession IDs for all RHS species (if decomposable)
        product_accession = product.get_accession()
        rxn_str_in_db_accessions += product_accession
        rxn_str_in_db_accessions += ' + '

    # remove extra ' + ' sign on the right
    rxn_str_in_db_accessions = rxn_str_in_db_accessions.rstrip(' + ')

    # return reaction string in terms of eQuilibrator accession IDs
    return rxn_str_in_db_accessions

    # except Exception as e:
    #     print(e)
    #
    #     # if even one species on either LHS or RHS cannot be decomposed, return None
    #     return None

def calc_dG_frm_rxn_str(new_rxn_str: str):
    """
    Calculate dG value and estimation errors at physiological and standard conditions

    :param new_rxn_str: reaction string in eQuilibrator accession IDs

    :return rxn_object: eQuilibrator reaction object
    :return phys_dG_value (float): physiological reaction dG value (1 mM concentrations)
    :return phys_dG_error (float): error in estimating dG value at physiological conditions
    :return std_dG_value (float): standard reaction dG value (1 M concentrations)
    :return std_dG_error (float): error in estimating dG value at standard conditions
    """

    rxn_object = cc.parse_reaction_formula(new_rxn_str)

    cc.p_h = Q_(7.4)
    cc.p_mg = Q_(3.0)
    cc.ionic_strength = Q_("0.25M")
    cc.temperature = Q_("298.15K")

    phys_dG_value = float(str(cc.physiological_dg_prime(rxn_object).value).rstrip(' kilojoule/mole'))
    phys_dG_error = float(str(cc.physiological_dg_prime(rxn_object).error).rstrip(' kilojoule/mole'))
    std_dG_value = float(str(cc.standard_dg_prime(rxn_object).value).rstrip(' kilojoule/mole'))
    std_dG_error = float(str(cc.standard_dg_prime(rxn_object).error).rstrip(' kilojoule/mole'))

    return rxn_object, phys_dG_value, phys_dG_error, std_dG_value, std_dG_error

def calc_pathway_dG_vals(reformatted_rxn_strs_list: list):
    """
    Calculate dG of pathway

    :param reformatted_rxn_strs_list:

    :return pathway_rxn_objects_list (list): list of eQuilibrator reaction objects for all pathway reactions
    :return pathway_phys_dG_vals (list): list of reaction dG values at physiological conditions (1 mM concentrations)
    :return pathway_phys_dG_errors (list): list of dG errors in estimating dG at physiological conditions
    :return pathway_std_dG_vals (list): list of reaction dG at standard conditions (1M concentrations)
    :return pathway_std_dG_errors (list): list of dG errors in estimating dG at standard conditions
    """

    # initialize empty list to store eQuilibrator rxn objects for all reactions in pathway
    pathway_rxn_objects_list = []
    pathway_phys_dG_vals = []
    pathway_phys_dG_errors = []
    pathway_std_dG_vals = []
    pathway_std_dG_errors = []

    for rxn_str in reformatted_rxn_strs_list:

        # parse reaction string into a list of reactants and products (includes cofactors)
        reactants_list, products_list = parse_rxn(rxn_str)

        # generate new reaction string containing eQuilibrator accession IDs rather than SMILES
        new_rxn_str = rxn_constructor_in_accession_IDs(reactants_list, products_list)

        # generate new reaction string containing eQuilibrator accession IDs rather than SMILES
        new_rxn_str = rxn_constructor_in_accession_IDs(reactants_list, products_list)

        try:
            # get reaction object, physiological and standard dG vals as well as errors for this reaction
            rxn_object, phys_dG_value, phys_dG_error, std_dG_value, std_dG_error = calc_dG_frm_rxn_str(new_rxn_str)

            # store reaction object and dG values
            pathway_rxn_objects_list.append(rxn_object)
            pathway_phys_dG_vals.append(phys_dG_value)
            pathway_phys_dG_errors.append(phys_dG_error)
            pathway_std_dG_vals.append(std_dG_value)
            pathway_std_dG_errors.append(std_dG_error)

        except Exception as e:
            print(e)

    return pathway_rxn_objects_list, pathway_phys_dG_vals, pathway_phys_dG_errors, pathway_std_dG_vals, pathway_std_dG_errors

def get_mdf_frm_rxn_objects(pathway_rxn_objects_list: list, lb: float, ub: float):

    """
    Calculating pathway mdf from eQuilibrator reaction objects

    :param pathway_rxn_objects_list: list of eQuilibrator reaction objects
    :param lb: lower bound on metabolite concentration
    :param ub: upper bound on metabolite concentration

    :return max_df ( float or None ):
    :return best_case_rxn_energies ( list or None ):
    :return best_case_concentrations ( list or None ):
    :return eq_compound_ids ( list or None ):
    """

    standard_dgr_prime_mean, standard_dgr_Q = cc.standard_dg_prime_multi(pathway_rxn_objects_list,
                                                                         uncertainty_representation="fullrank")

    S = cc.create_stoichiometric_matrix_from_reaction_objects(pathway_rxn_objects_list)
    Nc, Nr = S.shape

    ln_conc = cvxpy.Variable(shape=Nc, name="metabolite log concentration")  # vector
    B = cvxpy.Variable()  # scalar
    dg_prime = -(standard_dgr_prime_mean.m_as("kJ/mol") + cc.RT.m_as("kJ/mol") * S.values.T @ ln_conc)

    constraints = [
        np.log(np.ones(Nc) * lb) <= ln_conc,  # lower bound on concentrations
        ln_conc <= np.log(np.ones(Nc) * ub),  # upper bound on concentrations
        np.ones(Nr) * B <= dg_prime]

    prob_max = cvxpy.Problem(cvxpy.Maximize(B), constraints)
    prob_max.solve()
    max_df = prob_max.value

    # best_case_rxn_energies = list(-dg_prime.value)
    # best_case_concentrations = list(np.exp(ln_conc.value))
    # eq_compound_ids = list(S.index)
    return max_df

