import os
import warnings
import pandas as pd
import pickle
from typing import Union, List, Tuple, Optional
from rdkit import RDLogger, Chem, DataStructs
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Chem.AtomPairs import Pairs
import numpy as np
from scipy.optimize import linear_sum_assignment
import doranet.modules.enzymatic as enzymatic
import doranet.modules.post_processing as post_processing
import json

# intra-package transfer (the try-except helps with remote HPC clusters)
try:
    import featurizations
except:
    from .featurizations import featurizations

# Suppress RDKit warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'

def weight(score: float) -> float:
    """
    A function that accepts a similarity score as the sole argument and returns a scaled value.
    This is useful for filtering molecules in Pickaxe expansions by their Tanimoto similarity to the target

    Parameters:
    ----------
    score: float
        Tanimoto similarity of input metabolite with respect to the target input into Pickaxe

    Returns:
    ----------
    score**4: float
        Weighted tanimoto similarity raised to the power of 4
        This makes it more likely for metabolites structurally similar to the target to be expanded upon
    """
    return score ** 4

def get_atom_atom_path_sim(m1: Chem.Mol, m2: Chem.Mol) -> float:
    """
    compute the Atom Atom Path Similarity for a pair of RDKit molecules.
    See Gobbi et al, J. ChemInf (2015) 7:11

    Parameters:
    ----------
    m1: rdkit.Chem.Mol
        Mol object created from the first metabolite using its SMILES
    m2: rdkit.Chem.Mol
        Mol object created from the second metabolite using its SMILES

    Returns:
    ----------
    AtomAtomPathSimilarity: float
        Atom-Atom path similarity between the two input metabolites
    """
    _BK_ = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 4
    }
    _BONDSYMBOL_ = {1: '-', 2: '=', 3: '#', 4: ':'}

    # _nAT_ = 217 # 108*2+1
    _nAT_ = 223  # Gobbi code actually uses the first prime higher than 217, not 217 itself
    _nBT_ = 5

    # def FindAllPathsOfLengthN_Gobbi(mol, length, rootedAtAtom=-1, uniquepaths=True):
    #	return FindAllPathsOfLengthMToN(mol, length, length, rootedAtAtom=rootedAtAtom, uniquepaths=uniquepaths)

    def FindAllPathsOfLengthMToN_Gobbi(mol, minlength, maxlength, rootedAtAtom=-1, uniquepaths=True):
        '''this function returns the same set of bond paths as the Gobbi paper.  These differ a little from the rdkit FindAllPathsOfLengthMToN function'''
        paths = []
        for atom in mol.GetAtoms():
            if rootedAtAtom == -1 or atom.GetIdx() == rootedAtAtom:
                path = []
                visited = {atom.GetIdx()}
                #			visited = set()
                _FindAllPathsOfLengthMToN_Gobbi(atom, path, minlength, maxlength, visited, paths)

        if uniquepaths:
            uniquepathlist = []
            seen = set()
            for path in paths:
                if path not in seen:
                    reversepath = tuple([i for i in path[::-1]])
                    if reversepath not in seen:
                        uniquepathlist.append(path)
                        seen.add(path)
            return uniquepathlist
        else:
            return paths

    def _FindAllPathsOfLengthMToN_Gobbi(atom, path, minlength, maxlength, visited, paths):
        for bond in atom.GetBonds():
            if bond.GetIdx() not in path:
                bidx = bond.GetIdx()
                path.append(bidx)
                if len(path) >= minlength and len(path) <= maxlength:
                    paths.append(tuple(path))
                if len(path) < maxlength:
                    a1 = bond.GetBeginAtom()
                    a2 = bond.GetEndAtom()
                    if a1.GetIdx() == atom.GetIdx():
                        nextatom = a2
                    else:
                        nextatom = a1
                    nextatomidx = nextatom.GetIdx()
                    if nextatomidx not in visited:
                        visited.add(nextatomidx)
                        _FindAllPathsOfLengthMToN_Gobbi(nextatom, path, minlength, maxlength, visited, paths)
                        visited.remove(nextatomidx)
                path.pop()

    def getpathintegers(m1, uptolength=7):
        '''returns a list of integers describing the paths for molecule m1.  This uses numpy 16 bit unsigned integers to reproduce the data in the Gobbi paper.  The returned list is sorted'''
        bondtypelookup = {}
        for b in m1.GetBonds():
            bondtypelookup[b.GetIdx()] = _BK_[b.GetBondType()], b.GetBeginAtom(), b.GetEndAtom()
        pathintegers = {}
        for a in m1.GetAtoms():
            idx = a.GetIdx()
            pathintegers[idx] = []
            #		for pathlength in range(1, uptolength+1):
            #			for path in rdmolops.FindAllPathsOfLengthN(m1, pathlength, rootedAtAtom=idx):
            for ipath, path in enumerate(
                    FindAllPathsOfLengthMToN_Gobbi(m1, 1, uptolength, rootedAtAtom=idx, uniquepaths=False)):
                strpath = []
                currentidx = idx
                res = []
                for ip, p in enumerate(path):
                    bk, a1, a2 = bondtypelookup[p]
                    strpath.append(_BONDSYMBOL_[bk])
                    if a1.GetIdx() == currentidx:
                        a = a2
                    else:
                        a = a1
                    ak = a.GetAtomicNum()
                    if a.GetIsAromatic():
                        ak += 108
                    # trying to get the same behaviour as the Gobbi test code - it looks like a circular path includes the bond, but not the closure atom - this fix works
                    if a.GetIdx() == idx:
                        ak = None
                    if ak is not None:
                        astr = a.GetSymbol()
                        if a.GetIsAromatic():
                            strpath.append(astr.lower())
                        else:
                            strpath.append(astr)
                    res.append((bk, ak))
                    currentidx = a.GetIdx()
                pathuniqueint = np.ushort(0)  # work with 16 bit unsigned integers and ignore overflow...
                for ires, (bi, ai) in enumerate(res):
                    # use 16 bit unsigned integer arithmetic to reproduce the Gobbi ints
                    #					pathuniqueint = ((pathuniqueint+bi)*_nAT_+ai)*_nBT_
                    val1 = pathuniqueint + np.ushort(bi)
                    val2 = val1 * np.ushort(_nAT_)
                    # trying to get the same behaviour as the Gobbi test code - it looks like a circular path includes the bond, but not the closure atom - this fix works
                    if ai is not None:
                        val3 = val2 + np.ushort(ai)
                        val4 = val3 * np.ushort(_nBT_)
                    else:
                        val4 = val2
                    pathuniqueint = val4
                pathintegers[idx].append(pathuniqueint)
        # sorted lists allow for a quicker comparison algorithm
        for p in pathintegers.values():
            p.sort()
        return pathintegers

    def getcommon(l1, ll1, l2, ll2):
        '''returns the number of items sorted lists l1 and l2 have in common.  ll1 and ll2 are the list lengths'''
        ncommon = 0
        ix1 = 0
        ix2 = 0
        while (ix1 < ll1) and (ix2 < ll2):
            a1 = l1[ix1]
            a2 = l2[ix2]
            # a1 is < or > more often that ==
            if a1 < a2:
                ix1 += 1
            elif a1 > a2:
                ix2 += 1
            else:  # a1 == a2:
                ncommon += 1
                ix1 += 1
                ix2 += 1
        return ncommon

    def getsimaibj(aipaths, bjpaths, naipaths, nbjpaths):
        '''returns the similarity of two sorted path lists.  Equation 2'''
        nc = getcommon(aipaths, naipaths, bjpaths, nbjpaths)
        sim = float(nc + 1) / (max(naipaths, nbjpaths) * 2 - nc + 1)
        return sim

    def getmappings(simmatrixarray):
        '''return a mapping of the atoms in the similarity matix using the heuristic algorithm described in the paper'''

        costarray = np.ones(simmatrixarray.shape) - simmatrixarray

        it = np.nditer(costarray, flags=['multi_index'], op_flags=['writeonly'])
        dsu = []
        for a in it:
            dsu.append((a, it.multi_index[0], it.multi_index[1]))
        dsu.sort()

        seena = set()
        seenb = set()
        mappings = []
        for sim, a, b in dsu:
            if a not in seena and b not in seenb:
                seena.add(a)
                seenb.add(b)
                mappings.append((a, b))

        return mappings[:min(simmatrixarray.shape)]

    def gethungarianmappings(simmatrixarray):
        '''return a mapping of the atoms in the similarity matrix - the Hungarian algorithm is used because it is invariant to atom ordering.  Requires scipy'''
        costarray = np.ones(simmatrixarray.shape) - simmatrixarray
        row_ind, col_ind = linear_sum_assignment(costarray)
        res = zip(row_ind, col_ind)
        return res

    def getsimab(mappings, simmatrixdict):
        '''return the similarity for a set of mapping.  See Eqn 3'''
        naa, nab = simmatrixdict.shape

        score = 0.0
        for a, b in mappings:
            score += simmatrixdict[a][b]
        simab = score / (max(naa, nab) * 2 - score)
        return simab

    def getsimmatrix(m1, m1pathintegers, m2, m2pathintegers):
        '''generate a matrix of atom atom similarities.  See Figure 4'''

        aidata = [((ai.GetAtomicNum(), ai.GetIsAromatic()), ai.GetIdx()) for ai in m1.GetAtoms()]
        bjdata = [((bj.GetAtomicNum(), bj.GetIsAromatic()), bj.GetIdx()) for bj in m2.GetAtoms()]

        simmatrixarray = np.zeros((len(aidata), len(bjdata)))

        for ai, (aitype, aiidx) in enumerate(aidata):
            aipaths = m1pathintegers[aiidx]
            naipaths = len(aipaths)
            for bj, (bjtype, bjidx) in enumerate(bjdata):
                if aitype == bjtype:
                    bjpaths = m2pathintegers[bjidx]
                    nbjpaths = len(bjpaths)
                    simmatrixarray[ai][bj] = getsimaibj(aipaths, bjpaths, naipaths, nbjpaths)
        return simmatrixarray

    def AtomAtomPathSimilarity(m1, m2, m1pathintegers=None, m2pathintegers=None):
        '''compute the Atom Atom Path Similarity for a pair of RDKit molecules.  See Gobbi et al, J. ChemInf (2015) 7:11
            the most expensive part of the calculation is computing the path integers - we can precompute these and pass them in as an argument'''
        if m1pathintegers is None:
            m1pathintegers = getpathintegers(m1)
        if m2pathintegers is None:
            m2pathintegers = getpathintegers(m2)

        simmatrix = getsimmatrix(m1, m1pathintegers, m2, m2pathintegers)

        #	mappings = getmappings(simmatrix)
        mappings = gethungarianmappings(simmatrix)

        simab = getsimab(mappings, simmatrix)

        return simab

    return AtomAtomPathSimilarity(m1, m2)

def has_carboxylic_acid(mol: Chem.Mol) -> bool:
    """
    Check if the molecule has a carboxylic acid group.

    Parameters:
    ----------
    mol: rdkit.Chem.Mol
        Mol object created from input metabolite using its SMILES

    Returns:
    ----------
    mol.HasSubstructMatch(pattern): bool
        True if the molecule has a carboxylic acid group, False otherwise
    """
    # Define a SMARTS pattern for carboxylic acid
    carboxylic_acid_smarts = "[#6](=[O])[OH]"
    pattern = Chem.MolFromSmarts(carboxylic_acid_smarts)
    return mol.HasSubstructMatch(pattern)

def double_reduce(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    Doubly reduce an acid group

    Parameters:
    ----------
    mol: rdkit.Chem.Mol
        Mol object created from input metabolite using its SMILES

    Returns:
    ----------
    RDKit molecule object after decarboxylation or None if double reduction is not possible.

    """
    # Define a reaction for decarboxylation
    rxn = AllChem.ReactionFromSmarts("[#6:1](-[#6](=[O])-[OH])>>[#6:1]-[#6]")
    products = rxn.RunReactants((mol,))
    if products:
        return products[0][0]
    return None

class feasibility_classifier:
    """
    A wrapper class for enzymatic reaction feasibility prediction using an XGBoost classification model.
    This classifier uses a pre-trained model to predict the feasibility of a given enzymatic reaction based
    on its reaction string. Input reaction strings can be featurized into reaction fingerprints via a combination
    of ways to arrange the molecular fingerprints of substrates, products, and cofactors in each reaction.

    Attributes:
        cofactor_positioning (str): Describes the ordering of cofactors in the reaction fingerprint
        feasibility_model (XGBoost model): The loaded XGBoost model for predicting reaction feasibility.
        threshold (float): The threshold for deciding between feasible (1) and infeasible (0) reactions.
        cofactors_set_wo_stereo (set): Set of cofactors without stereochemistry information.
    """

    def __init__(self, ML_model_type: str, cofactors_path: str):
        """
        Initialize the feasibility classifier by loading the appropriate XGBoost model based on the provided
        ML_model_type and setting up the required configurations for reaction fingerprinting.

        Parameters:
        - ML_model_type (str): Type of the model, which determines how the cofactors are positioned in the
                               reaction fingerprint. Options include 'by_ascending_MW', 'by_descending_MW',
                               'add_concat', and 'add_subtract'.
        - cofactors_path (str): Filepath to the cofactors' data, which is used for generating the reaction
                                fingerprint.
        """
        model_type = 'xgboost'
        fp_type = 'ecfp4'
        max_species = 4
        nBits = 2048

        if ML_model_type == 'by_ascending_MW':
            cofactor_positioning = 'by_ascending_MW'
            model_path = dir_path + f'../models/{model_type}_{fp_type}_{nBits}_{max_species}_{cofactor_positioning}.pkl'
            self.cofactor_positioning = cofactor_positioning
            self.feasibility_model = pickle.load(open(model_path, 'rb'))
            self.threshold = 0.668

        if ML_model_type == 'by_descending_MW':
            cofactor_positioning = 'by_descending_MW'
            model_path = dir_path + f'../models/{model_type}_{fp_type}_{nBits}_{max_species}_{cofactor_positioning}.pkl'
            self.cofactor_positioning = cofactor_positioning
            self.feasibility_model = pickle.load(open(model_path, 'rb'))
            self.threshold = 0.362

        if ML_model_type == 'add_concat':
            cofactor_positioning = 'add_concat'
            model_path = dir_path + f'../models/{model_type}_{fp_type}_{nBits}_{max_species}_{cofactor_positioning}.pkl'
            self.cofactor_positioning = cofactor_positioning
            self.feasibility_model = pickle.load(open(model_path, 'rb'))
            self.threshold = 0.372

        if ML_model_type == 'add_subtract':
            cofactor_positioning = 'add_subtract'
            model_path = dir_path + f'../models/{model_type}_{fp_type}_{nBits}_{max_species}_{cofactor_positioning}.pkl'
            self.cofactor_positioning = cofactor_positioning
            self.feasibility_model = pickle.load(open(model_path, 'rb'))
            self.threshold = 0.407

        dummy_rxn_str = "" # create a dummy reaction object to extract out the list of cofactors
        dummy_rxn_object = featurizations.reaction(dummy_rxn_str)
        self.cofactors_set_wo_stereo = dummy_rxn_object.print_full_cofactors_set_wo_stereo(dir_path + cofactors_path)

    def predict_proba(self, rxn_str:str) -> float:
        """
        Predict the probability of a reaction being feasible.

        Parameters:
        - rxn_str (str): The reaction string representing a balanced enzymatic reaction.

        Returns:
        - float: The predicted probability of the reaction being feasible.
        """

        # initialize a reaction object with custom-built featurizations package
        rxn_object = featurizations.reaction(rxn_str)

        # convert reaction string to a reaction fingerprint
        # substrates go first followed by cofactors on the LHS in order of descending MW for positions 1-4
        # same with products and cofactors on the RHS for positions 5-8
        rxn_fp = rxn_object.rxn_2_fp_w_positioning(fp_type = 'ecfp4',
                                                   nBits = 2048,
                                                   is_folded = True,
                                                   dim = 2048,
                                                   max_species = 4,
                                                   cofactor_positioning = self.cofactor_positioning,
                                                   reaction_rule = None,
                                                   all_cofactors_wo_stereo = self.cofactors_set_wo_stereo)

        rxn_fp = rxn_fp.reshape(1, -1)  # reshape since only single sample
        rxn_feasib_score = self.feasibility_model.predict_proba(rxn_fp)[:,1][0]

        return rxn_feasib_score

    def predict_label(self, rxn_str: str) -> int:
        """
        Predict whether a reaction is feasible or not based on the threshold set for the classifier.

        Parameters:
        - rxn_str (str): The reaction string representing the enzymatic reaction.

        Returns:
        - int: The predicted label for the reaction feasibility, where 1 indicates feasible and 0 indicates infeasible.
        """

        # initialize a reaction object with custom-built featurizations package
        rxn_object = featurizations.reaction(rxn_str)

        # convert reaction string to a reaction fingerprint
        # substrates go first followed by cofactors on the LHS in order of descending MW for positions 1-4
        # same with products and cofactors on the RHS for positions 5-8
        rxn_fp = rxn_object.rxn_2_fp_w_positioning(fp_type='ecfp4',
                                                   nBits=2048,
                                                   is_folded=True,
                                                   dim=2048,
                                                   max_species=4,
                                                   cofactor_positioning=self.cofactor_positioning,
                                                   reaction_rule=None,
                                                   all_cofactors_wo_stereo=self.cofactors_set_wo_stereo)

        rxn_fp = rxn_fp.reshape(1, -1)  # reshape since only single sample
        rxn_feasib_score = self.feasibility_model.predict_proba(rxn_fp)[:, 1][0]

        if rxn_feasib_score >= self.threshold:
            return 1
        else:
            return 0

class biosynth_pipeline:
    def __init__(self,
                pathway_sequence: list,
                target_smiles: str,
                target_name: str,
                feasibility_classifier: any,
                pks_release_mechanism: str,
                config_filepath: str = '../scripts/config.json'):
        """
        Combined pipeline for pathway design using multifunctional and monofunctional enzymes

        Parameters:
        ----------
        pathway_sequence: list
            order of transformations using multifunctional and monofunctional enzymes
            currently, we only support the order ['pks','non_pks']
            this ensures that multifunctional enzymes are used to first make the carbon backbone of a target molecule
            any further modifications would then be performed by monofunctional enzymes, which are fairly precise
            such an approach also resembles the natural arrangement of enzymes within biosynthetic gene clusters.
            if users would like to perform pathway design using only PKSs, however, set this to: ['pks']
            similarly, if users would like to use only monofunctional enzymes, set this to ['non_pks']

        target_smiles: str
            SMILES string of target molecule, e.g. 'CCC' for propane

        target_name: str
            Name of target molecule, e.g. 'propane'

        feasibility_classifier: any
            Our previously published enzymatic reaction feasibility classification ML model

        pks_release_mechanism: str
            preferred termination reaction to release bound substrate from PKS chain
            currently, we support 'cyclization' and 'thiolysis'

        pks_starters_filepath (default = '../biosynth_pipeline/retrotide/data/starters.smi'): str
            filepath of .smi file of PKS starter units

        pks_extenders_filepath (default = '../biosynth_pipeline/retrotide/data/extenders.smi'): str
            filepath of .smi file of PKS extender units

        pks_starters (default = 'all'): list of str or str
            list of PKS starter units that users would like to use
            e.g. ['mal', 'mmal'] would use only malonyl-CoA and methylmalonyl-CoA as starters
            or if 'all', then all PKS starter units originally in Retrotide will be used

        pks_extenders (default = 'all'): list of str or str
            list of PKS extender units that users would like to use
            e.g. ['mal', 'mmal'] would use only malonyl-CoA and methylmalonyl-CoA as extenders
            or if 'all', then all PKS extender units originally in Retrotide will be used

        pks_similarity_metric (default = 'mcs_without_stereo'): str
            metric to use when comparing the structural similarity of PKS product to target

        non_pks_similarity_metric (default = 'mcs_without_stereo'): str
            metric to use when comparing the structural similarity of non-PKS product to target

        consider_target_stereo (default = False): bool
            remove stereochemistry from the input target's SMILES string if False

        known_metabolites_filepath (default = '../data/all_known_metabolites.txt'): str
            filepath of known unique, metabolites found in BRENDA, KEGG, and METACYC

        non_pks_cofactors_filepath (default = '../data/coreactants_and_rules/all_cofactors.tsv'): str
            filepath of cofactors to use for pickaxe expansion using monofunctional enzymes

        input_cpd_dir (default = '../data/input_compounds/'): str
            filepath to store the SMILES string of the compound for pickaxe expansion using monofunctional enzymes

        non_pks_rules (default = 'intermediate_non_dimerization'): str
            reaction rules for pickaxe, choose from 'biological_generalized', 'biological_intermediate'
            or 'intermediate_non_dimerization', which is optimal because these rules are most specific
            and also suggest fewer false positives

        non_pks_steps (default = 1): int
            number of reaction steps to run pickaxe using monofunctional enzymes

        non_pks_cores (default = 1): int
            number of cores to run pickaxe using monofunctional enzymes

        non_pks_sim_score_filter (default = False): bool
            if True, then tanimoto similarity filters are used in pickaxe expansion

        non_pks_sim_score_cutoffs (default = None): None or list of floats
            tanimoto similarity cutoffs to be used for each generation

        non_pks_sim_sample (default = False): bool
            if True, then similarity sampling will be used during pickaxe expansions

        non_pks_sim_sample_size (default = None): None or int
            number of intermediates to sample at each pickaxe generation

        stopping_criteria (default = 'first_product_formation'): str
            by default, everything stops when the target product is formed

        Attributes
        ----------
        pathway_sequence: list
            order of transformations, e.g. ['pks','non_pks','chem']

        target_smiles: str
            SMILES string of target compound

        feasibility_classifier: any
            Loaded feasibility classification model to predict the feasibilities of monofunctional enzymatic reactions

        pks_release_mechanism: str
            termination reaction that will release the bound substrate from the PKS chain

        pks_extenders_filepath: str
            filepath for a .smi file of extender units that Retrotide will use to design chimeric PKSs

        pks_starters_filepath: str
            filepath for a .smi file of starter units that Retrotide will use to design chimeric PKSs

        pks_starters: list of str or str
            user-selected starter units (can be 'all' or a list like ['mal','mmal']

        pks_extenders: list of str or str
            user-selected extender units (can be 'all' or a list like ['mal','mmal']

        pks_similarity_metric: str
            user-selected metric to compare structural similarity between pks product and target product

        non_pks_similarity_metric: str
            user-selected metric to compare structural similarity between any non-pks product and target product

        consider_target_stereo: bool
            if false, stereochemical information will be removed from the input target SMILES in self.target_smiles

        known_metabolites: set
            set of unique metabolites across Brenda, KEGG, and MetaCyc loaded from the input known_metabolites_filepath

        non_pks_cofactors: str
            filepath of cofactors to use for pickaxe expansion with monofunctional enzymes

        input_cpd_dir: str
            filepath to the directory that will house the .tsv file of a SMILES string for pickaxe expansion

        non_pks_rules: str
            user-input rule choice for reaction rules to use for pickaxe expansion
            this can be either 'biological_generalized', 'biological_intermediate' or 'intermediate_non_dimerization'

        rule_filepath: str
            filepath to the rule choice for pickaxe expansion

        non_pks_steps: int
            number of steps to use for a pickaxe biological expansion only

        non_pks_cores: int
            number of computing cores to use for a pickaxe biological expansion only

        non_pks_sim_score_filer: bool
            if True, a tanimoto similarity score cutoff filter will be used for pickaxe biological expansion only

        non_pks_sim_score_cutoffs: None or list of floats
            tanimoto similarity cutoffs to filter intermediates in each step of a pickaxe biological expansion only

        non_pks_sim_sample: bool
            if True, a similarity sampling filter will be used for pickaxe biologicale expansions only

        non_pks_sim_sample_size: None or int
            similarity sampling size to use to filter intermediates in each step of a pickaxe biological expansion only

        stopping_criteria: str
            criteria for when to stop searching for new products

        pks_designs: list
            List of potential PKS module designs from Retrotide

        pks_top_final_product: str
            Sanitized SMILES string of final PKS product using the best PKS design

        non_pks_compounds_df: Pandas DataFrame
            DataFrame of compounds generated from a pickaxe biological expansion only

        non_pks_pathways: Dict
            pathways generated from a pickaxe biological expansion ony

        current_pickaxe_graph: Networkx graph
            Networkx graph representing a pickaxe biological expansion only

        current_pickaxe_precursor: str
            SMILES string of PKS product for pickaxe biological expansion

        results_logs: list
            result logs

        Methods
        ----------
        _canon_smi:
            internal function to canonicalize SMILES string

        _remove_stereo:
            internal function to remove stereochemical information from a SMILES string

        _reformat_pickaxe_rxn_str:
            internal function to reformat a reaction equation from a biological pickaxe expansion for feasibility

        _calculate_similarity:
            internal function to calculate the chemical similarity between two input SMILES strings

        _select_starters:
            internal function to load PKS starter units based on user inputs

        _select_extenders:
            internal function to load PKS extender units based on user inputs

        _initialize_retrotide:
            internal function to import Retrotide with selected starters and extenders

        run_pks_synthesis: run retrotide with predetermined pks release reaction
        run_non_pks_synthesis
        """

        self.pathway_sequence = pathway_sequence
        self.target_smiles = self._canon_smi(target_smiles) # canonicalize input target SMILES but retain stereo first
        self.target_name = target_name
        self.feasibility_classifier = feasibility_classifier
        self.pks_release_mechanism = pks_release_mechanism
        self.config_filepath = config_filepath

        with open(self.config_filepath, 'r') as file:
            self.config_dict = json.load(file)

        self.pks_extenders_filepath = dir_path + self.config_dict['pks_extenders_filepath']
        self.pks_starters_filepath = dir_path + self.config_dict['pks_starters_filepath']
        self.pks_starters = self.config_dict['pks_starters']
        self.pks_extenders = self.config_dict['pks_extenders']

        self._select_extenders(self.pks_extenders)  # load PKS extenders based on user input
        self._select_starters(self.pks_starters)  # load PKS starters based on user input
        self._initialize_retrotide()  # then re-import retrotide (slightly inefficient)

        self.pks_similarity_metric = self.config_dict['pks_similarity_metric']
        self.non_pks_similarity_metric = self.config_dict['non_pks_similarity_metric']

        # remove stereochemistry if user does not want to consider stereo
        # stereochemistry is not currently supported anyway but is on the docket for future versions
        self.consider_target_stereo = bool(self.config_dict['consider_target_stereo'])
        if not self.consider_target_stereo:
            self.target_smiles = self._remove_stereo(target_smiles)

        self.known_metabolites_filepath = self.config_dict['known_metabolites_filepath']
        self.known_metabolites = set(line.strip() for line in open(dir_path + self.known_metabolites_filepath))
        self.non_pks_cofactors_filepath = self.config_dict['non_pks_cofactors_filepath']
        self.non_pks_cofactors = dir_path + self.non_pks_cofactors_filepath
        self.input_cpd_dir = dir_path + self.config_dict['input_cpd_dir']

        # load in reaction rules for DORAnet - choose from generalized, intermediate or 'intermediate_non_dimerization
        self.non_pks_rules = self.config_dict['non_pks_rules']
        if self.non_pks_rules == 'biological_intermediate':
            self.rule_filepath =  dir_path + '../data/coreactants_and_rules/JN3604IMT_rules.tsv'

        elif self.non_pks_rules == 'biological_generalized':
            self.rule_filepath =  dir_path + '../data/coreactants_and_rules/JN1224MIN_rules.tsv'

        elif self.non_pks_rules == 'intermediate_non_dimerization':
            self.rule_filepath =  dir_path + '../data/coreactants_and_rules/non_dimerization_imt_rules.tsv'

        self.non_pks_steps = int(self.config_dict['non_pks_steps'])
        self.non_pks_cores = int(self.config_dict['non_pks_cores'])
        self.non_pks_sim_score_filter = bool(self.config_dict['non_pks_sim_score_filter'])
        self.non_pks_sim_score_cutoffs = [float(val) for val in self.config_dict['non_pks_sim_score_cutoffs']]
        self.non_pks_sim_sample = bool(self.config_dict['non_pks_sim_sample'])
        self.non_pks_sim_sample_size = int(self.config_dict['non_pks_sim_sample_size'])
        self.stopping_criteria = self.config_dict['stopping_criteria']

        ### initialize attributes that will have values further updated afterwards
        self.pks_designs = None
        self.pks_top_final_product = None
        self.non_pks_compounds_df = None
        self.pathways_found = None
        self.current_pickaxe_graph = None
        self.current_pickaxe_precursor = None
        self.results_logs = []
        self.pks_top_final_product = None
        self.non_pks_pathways = None
        self.ranked_metbolites = None

    ###########################
    # Cheminformatics methods #
    ###########################

    @staticmethod
    def _canon_smi(smiles: str) -> str:
        """
        Canonicalize a given SMILES (Simplified Molecular Input Line Entry System) string

        This method converts an input SMILES string into its canonical form using RDKit.
        Canonical SMILES should be unique for each molecule, regardless of the order of atoms in the input string.
        Here, we take an input SMILES string then covert it into an RDKit mol object first.
        This RDKit mol object is then converted back into a SMILES string to achieve the canonical form.

        Parameters:
        ----------
        smiles: str
            A SMILES string representing the structure of a molecule. This input may or may not be canonicalized

        Returns:
        ----------
        canonicalized_smiles: str
            The canonical form of the input SMILES string.

        Raises:
        ----------
        ValueError:
            If the input SMILES string is invalid and cannot be parsed into a molecule.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: '{smiles}'")
        canonicalized_smiles = Chem.MolToSmiles(mol)
        return canonicalized_smiles

    @staticmethod
    def _remove_stereo(smiles: str) -> str:
        """
        Processes the input SMILES string and return a version without stereochemical information.

        The function uses RDKit to parse the SMILES string into a molecule object, removes its stereochemistry,
        and then converts it back to a SMILES string. This is useful for Pickaxe operations since biological rules
        do not take stereochemistry into account

        Parameters:
        ----------
        smiles: str
            A SMILES string that may include stereochemical information.

        Returns:
        ----------
        smi_wo_stereo: str
            The SMILES string with stereochemical information removed.

        Raises:
        ----------
        ValueError:
            If the input is not a valid SMILES string or conversion to a molecule object fails.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: '{smiles}'")
        Chem.RemoveStereochemistry(mol)
        smi_wo_stereo = Chem.MolToSmiles(mol)
        return smi_wo_stereo

    @staticmethod
    def simplify_reaction(reaction_smiles: str,
                          reaction_stoich: str) -> str:
        """
        Converts a reaction into a readable reaction with stoichiometry applied.

        The function converts a reaction formatted by DORAnet into a readable, balanced equation.

        Parameters:
        ----------
        reaction_smiles: str
            the SMILES string of a reaction generated by DORAnet.

        reaction_stoich: str
            the stoichiometry of the reaction input.

        Returns:
        ----------
        rxn_string: str
            a readable reaction string with stoichiometry applied.
        """
        reactants, products = reaction_smiles.split('>>')
        reactant_list = reactants.split('.')
        product_list = products.split('.')

        rxn_string = ''
        reactant_stoich, product_stoich = reaction_stoich.split('$')

        reactant_stoich = reactant_stoich.strip('()')
        reactant_stoich = reactant_stoich.split(',')
        reactant_stoich = list(filter(None, reactant_stoich))

        for i in range(0, len(reactant_list)):
            for j in range(0, int(reactant_stoich[i])):
                rxn_string += reactant_list[i]

                if i < len(reactant_list) - 1:
                    rxn_string += ' + '

        rxn_string += ' = '

        product_stoich = product_stoich.strip('()')
        product_stoich = product_stoich.split(',')
        product_stoich = list(filter(None, product_stoich))

        for i in range(0, len(product_list)):
            for j in range(0, int(product_stoich[i])):
                rxn_string += product_list[i]
                if i < len(product_list) - 1:
                    rxn_string += ' + '

        return rxn_string

    @staticmethod
    def _calculate_similarity(smiles1: str, smiles2: str, metric: str) -> float:
        """
        Calculate the similarity between two given metabolites with various metrics.

        Parameters:
        ----------
        smiles1: str
            SMILES string of the first metabolite in its canonical form.

        smiles2: str
            SMILES string of the second metabolite in its canonical form.

        metric: str
            Cheminformatics similarity metric to use.
            Users can choose between "Tanimoto", "mcs_with_stereo", "mcs_without_stereo", "atompairs", "atompaths"

        Returns:
        ----------
        score: str
            Chemical similarity score computed based on the chosen similarity metric.

        """
        # Convert SMILES strings to molecule objects
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            raise ValueError("Invalid SMILES string provided.")

        if metric == 'Tanimoto':
            # Generate morgan/ ecfp4 fingerprints for the molecules first before computing similarity
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
            score = DataStructs.FingerprintSimilarity(fp1, fp2)
            return score

        if metric == 'mcs_with_stero':
            # Calculate the maximum common substructures between both metabolites while factoring in stereochemistry
            result = rdFMCS.FindMCS([mol1, mol2],
                                    timeout=1,
                                    matchValences = True,
                                    matchChiralTag = True, # pay attention to stereochemistry
                                    bondCompare=Chem.rdFMCS.BondCompare.CompareOrderExact)

            if result.canceled:
                print('MCS timeout')

            # With the maximum common substructure, calculate the Tanimoto similarity
            score = result.numAtoms / (len(mol1.GetAtoms()) + len(mol2.GetAtoms()) - result.numAtoms)
            return score

        if metric == 'mcs_without_stereo':
            # Calculate the maximum common substructures between both metabolites without factoring in stereochemistry
            result = rdFMCS.FindMCS([mol1, mol2],
                                    timeout = 1,
                                    matchValences = True,
                                    matchChiralTag = False, # don't pay attention to stereochemistry
                                    bondCompare = Chem.rdFMCS.BondCompare.CompareOrderExact)

            if result.canceled:
                print('MCS timeout')

            score = result.numAtoms / (len(mol1.GetAtoms()) + len(mol2.GetAtoms()) - result.numAtoms)
            return score

        if metric == 'atompairs':
            ms = [mol1, mol2]
            pairFps = [Pairs.GetAtomPairFingerprint(x) for x in ms]
            score = DataStructs.TanimotoSimilarity(pairFps[0], pairFps[1])
            return score

        if metric == 'atompaths':
            score = get_atom_atom_path_sim(mol1, mol2)
            return score

    ##########################
    # Retrotide/ PKS methods #
    ##########################

    @staticmethod
    def _initialize_retrotide() -> None:
        """
        Initializes the Retrotide module by importing necessary components.

        This static method is responsible for setting up the Retrotide environment by importing the main
        Retrotide module, as well as specific submodules 'structureDB' and 'designPKS' from the Retrotide package.
        It ensures that these components are available in the current namespace for further operations related
        to Retrotide within the class.

        Parameters:
        ----------
        None

        Returns:
        ----------
        None

        Notes:
        ----------
        - This method does not return any value.
        - It's assumed that the RETROTIDE package is correctly installed and accessible in the Python environment.
        - The method should be called within the class when RETROTIDE initialization is required before performing
          RETROTIDE-related operations.
        - The method is static, meaning it can be called on the class itself without needing an instance of the class.
        """
        from .retrotide import retrotide
        from .retrotide import structureDB, designPKS

    def _select_extenders(self, extenders: Union[List[str], str]) -> None:
        """
        Generates a file containing SMILES strings and metadata for a user-specified list of PKS extenders.
        This function takes an input list of PKS extender 'shortNames' and writes their corresponding SMILES strings,
        identifiers, types, and short names to an RDKit .smi file.
        If 'all' is passed as an argument for extenders, then a .smi file for all available extenders is created.

        Parameters:
        ----------
        extenders: list[str] or str:
            A list of short names representing the desired extenders to be included in the output file.
            If the string 'all' is passed, then all extenders will be included when designing a PKS.
            But if say ['mal'] is passed, then only malonyl-CoA is used as an extender when designing PKSs.

        Returns:
        ----------
        None

        Raises:
        ----------
        ValueError:
            If any of the provided extender shortNames are not found in the list of available molecules,
            this indicates an incorrect or unknown shortName was provided.

        Notes:
        ----------
        A file named 'extenders.smi' will be created in the '../biosynth_pipeline/retrotide/data/' directory containing
        the SMILES strings and associated data for the selected extenders. This function assumes that the provided
        'extenders' are valid shortNames present in the molecules_data list. SMILES strings are validated using the
        RDKit Chem.MolFromSmiles function; invalid strings result in a warning. This function will not terminate upon
        encountering an invalid SMILES string but will skip writing that molecule.

        Examples:
        ----------
        To generate an SMI file for specific extenders:
            # >>> _select_extenders(['mal', 'mmal'])

        To generate an SMI file for all available extenders:
            # >>> _select_extenders('all')
        """
        molecules_data = [
        {"smiles": "O=C(O)CC(=O)[S]", "id": "Malonyl-CoA", "type": "CoA", "shortName": "mal"},
        {"smiles": "C[C@@H](C(=O)O)C(=O)[S]", "id": "Methylmalonyl-CoA", "type": "CoA", "shortName": "mmal"},
        {"smiles": "C=CC[C@@H](C(=O)O)C(=O)[S]", "id": "Allylmalonyl-CoA", "type": "CoA", "shortName": "allylmal"},
        {"smiles": "CO[C@@H](C(=O)O)C(=O)[S]", "id": "Methoxymalonyl-CoA", "type": "CoA", "shortName": "mxmal"},
        {"smiles": "CC[C@@H](C(=O)O)C(=O)[S]", "id": "Ethylmalonyl-CoA", "type": "CoA", "shortName": "emal"},
        {"smiles": "CCCC[C@@H](C(=O)O)C(=O)[S]", "id": "Butyrylmalonyl-CoA", "type": "CoA", "shortName": "butmal"},
        {"smiles": "OC(C(C([S])=O)O)=O", "id": "Hydroxymalonyl-CoA", "type": "CoA", "shortName": "hmal"},
        {"smiles": "[S]C([C@@H](C(O)=O)CC(C)C)=O", "id": "Isobutyrylmalonyl-CoA", "type": "CoA", "shortName": "isobutmal"},
        {"smiles": "[S]C([C@H](C(O)=O)CC(C)C)=O", "id": "D-isobutyrylmalonyl-CoA", "type": "CoA", "shortName": "d-isobutmal"},
        {"smiles": "ClC1=C(Cl)NC=C1CCCCC(C(O)=O)C([S])=O", "id": "DCP", "type": "CoA", "shortName": "DCP"},
        {"smiles": "CCCCCC[C@@H](C(=O)O)C(=O)[S]", "id": "Hexylmalonyl-CoA", "type": "CoA", "shortName": "hexmal"}]

        # Create a set of all valid shortNames
        valid_short_names = {molecule['shortName'] for molecule in molecules_data}

        with open(self.pks_extenders_filepath, "w") as file:
            file.write("smiles\tid\ttype\tshortName\n")

            if extenders == "all":
                for molecule in molecules_data:
                    smiles = molecule["smiles"]

                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        print(f"Warning: Invalid SMILES string for {molecule['id']}")
                        continue

                    file.write(f"{smiles}\t{molecule['id']}\t{molecule['type']}\t{molecule['shortName']}\n")

            else:
                for extender in extenders:
                    if extender not in valid_short_names:  # Check if the shortName is valid
                        raise ValueError(f"Invalid shortName provided: '{extender}'")

                for molecule in molecules_data:
                    if molecule["shortName"] in extenders:
                        smiles = molecule["smiles"]

                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None:
                            print(f"Warning: Invalid SMILES string for {molecule['id']}")
                            continue

                        file.write(f"{smiles}\t{molecule['id']}\t{molecule['type']}\t{molecule['shortName']}\n")

        print("\nExtender units successfully chosen for polyketide synthases")

    def _select_starters(self, starters: Union[List[str], str]) -> None:
        """
        Generates a file containing SMILES strings and metadata for a specified list of PKS starters.
        This function takes a list of PKS starter 'shortNames' and writes their corresponding SMILES strings,
        identifiers, types, and short names to a SMI file. If 'all' is passed as an argument,
        then this function creates an RDKit .smi file for all available starters.

        Parameters:
        ----------
        starters (list[str] or str):
            A list of short names representing the desired starters to be included in the output file.
            If the string 'all' is passed, all starters will be included.

        Returns:
        ----------
        None

        Raises:
        ----------
        ValueError:
            If any of the provided starter shortNames are not found in the list of available molecules.
            This would indicate an incorrect or unknown shortName.

        Notes:
        ----------
        - The function assumes that the provided 'starters' are valid shortNames present in the molecules_data list.
        - SMILES strings are validated using the RDKit Chem.MolFromSmiles function; invalid strings result in a warning.
        - The function will not terminate upon encountering an invalid SMILES string but will skip writing that molecule.

        Examples:
        ----------
        To generate an SMI file for specific starters:
            # >>> select_starters(['mal', 'mmal'])

        To generate an SMI file for all available starters:
            # >>> select_starters('all')
        """
        molecules_data = [
            {"smiles": "CC(=O)[S]", "id": "Acetyl-CoA", "type": "CoA", "shortName": "Acetyl-CoA"},
            {"smiles": "CCC(=O)[S]", "id": "Propionyl-CoA", "type": "CoA", "shortName": "prop"},
            {"smiles": "CC(=O)[S]", "id": "Malonyl-CoA", "type": "CoA", "shortName": "mal"},
            {"smiles": "CCC(=O)[S]", "id": "Methylmalonyl-CoA", "type": "CoA", "shortName": "mmal"},
            {"smiles": "C=CCCC([S])=O", "id": "Allylmalonyl-CoA", "type": "CoA", "shortName": "allylmal"},
            {"smiles": "COCC(=O)[S]", "id": "Methoxymalonyl-CoA", "type": "CoA", "shortName": "mxmal"},
            {"smiles": "[S]C(CO)=O", "id": "Hydroxymalonyl-CoA", "type": "CoA", "shortName": "hmal"},
            {"smiles": "[S]C(CCC(O)=O)=O", "id": "Succinyl-CoA_non_decarboxylated", "type": "CoA", "shortName": "succ-coa_non_decarboxylated"}, # this is left as a dicarboxylic acid (Amin's AiCHE paper)
            {"smiles": "[S]C(CC)=O", "id": "Succinyl-CoA_decarboxylated", "type": "CoA", "shortName": "succ-coa_decarboxylated"},
            {"smiles": "CCCCCC([S])=O", "id": "Butyrylmalonyl-CoA", "type": "CoA", "shortName": "butmal"},
            {"smiles": "CC(C)C(=O)[S]", "id": "Isobutyrylmalonyl-CoA", "type": "CoA", "shortName": "isobutmal"},
            {"smiles": "CCC(C)C(=O)[S]", "id": "2-methylbutyrylmalonyl-CoA", "type": "CoA", "shortName": "2metbutmal"},
            {"smiles": "[S]C(C1=CNC(Cl)=C1Cl)=O", "id": "DCP", "type": "CoA", "shortName": "DCP"},
            {"smiles": "CC(=O)[S]", "id": "cemal", "type": "CoA", "shortName": "cemal"},
            {"smiles": "C1CCCCC1C(=O)[S]", "id": "CHC-CoA", "type": "CoA", "shortName": "CHC-CoA"},
            {"smiles": "C1CC[C@@H](C(=O)O)[C@@H]1C(=O)[S]", "id": "trans-1,2-CPDA", "type": "CoA", "shortName": "trans-1,2-CPDA"},
            {"smiles": "C1(=O)C(=CCC1)C(=O)[S]", "id": "cyclopentene", "type": "CoA", "shortName": "cyclopentene"},
            {"smiles": "P[S]C(C1=CC=CN1)=O", "id": "pyr", "type": "CoA", "shortName": "pyr"},
            {"smiles": "O=C([S])/C=C/C1=CC=CC=C1", "id": "cin", "type": "CoA", "shortName": "cin"},
            {"smiles": "[S]C(C1=CC(O)=CC(N)=C1)=O", "id": "ABHA", "type": "CoA", "shortName": "ABHA"},
            {"smiles": "CC(CC([S])=O)C", "id": "isoval", "type": "CoA", "shortName": "isoval"},
            {"smiles": "NC1=CC=C(C([S])=O)C=C1", "id": "PABA", "type": "CoA", "shortName": "PABA"},
            {"smiles": "NC(NCC([S])=O)=[NH2+]", "id": "guan", "type": "CoA", "shortName": "guan"},
            {"smiles": "CC1=NC(C([S])=O)=CS1", "id": "mthz", "type": "CoA", "shortName": "mthz"},
            {"smiles": "O[C@H]1[C@H](O)CCC(C([S])=O)C1", "id": "DHCH", "type": "CoA", "shortName": "DHCH"},
            {"smiles": "O[C@H]1[C@H](O)CC=C(C([S])=O)C1", "id": "DHCHene", "type": "CoA", "shortName": "DHCHene"},
            {"smiles": "O=C([S])CC1=CC=CC=C1", "id": "plac", "type": "CoA", "shortName": "plac"},
            {"smiles": "[S]C(C1=CC=CC=C1)=O", "id": "benz", "type": "CoA", "shortName": "benz"},
            {"smiles": "[S]C(C1=CC=C([N+]([O-])=O)C=C1)=O", "id": "PNBA", "type": "CoA", "shortName": "PNBA"},
            {"smiles": "[S]C([C@@H](CC)C(N)=O)=O", "id": "ema", "type": "CoA", "shortName": "ema"},
            {"smiles": "[S]C([C@@H](C)CNC([C@@H](N)C)=O)=O", "id": "3measp", "type": "CoA", "shortName": "3measp"}]

        # Create a set of all valid shortNames
        valid_short_names = {molecule['shortName'] for molecule in molecules_data}

        with open(self.pks_starters_filepath, "w") as file:
            file.write("smiles\tid\ttype\tshortName\n")

            if starters == "all":
                for molecule in molecules_data:
                    smiles = molecule["smiles"]

                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        print(f"Warning: Invalid SMILES string for {molecule['id']}")
                        continue
                    file.write(f"{smiles}\t{molecule['id']}\t{molecule['type']}\t{molecule['shortName']}\n")

            else:
                for starter in starters:
                    if starter not in valid_short_names:  # Check if the shortName is valid
                        raise ValueError(f"Invalid shortName provided: '{starter}'")

                for molecule in molecules_data:
                    if molecule["shortName"] in starters:
                        smiles = molecule["smiles"]

                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None:
                            print(f"Warning: Invalid SMILES string for {molecule['id']}")
                            continue

                        file.write(f"{smiles}\t{molecule['id']}\t{molecule['type']}\t{molecule['shortName']}\n")

        print("\nStarter units successfully chosen for polyketide synthases")

    def run_pks_termination(self, pks_design_num: int, pks_release_mechanism: str) -> Union[str, None]:
        """
        Executes the final termination reaction in the polyketide synthase (PKS) design process.
        This function retrieves the product bound to the final PKS module and performs the specified release reaction.
        Currently, we only support thiolysis and cyclization reactions for offloading.

        Parameters:
        ----------
        pks_design_num: int
            Index of the PKS design from which to release the final product.

        pks_release_mechanism: str
            The offloading reaction by which the bound product is released to give a final PKS product.
            Currently, only 'thiolysis' and 'cyclization' offloadings are supported

        Returns:
        ----------
        product_smiles: str
            SMILES string of the PKS product

        Raises:
        ----------
        ValueError:
            Unsupported pks_release_mechanism is provided or cyclization release reaction cannot be performed
        """
        from .retrotide import retrotide

        # first, we compute the product bound to the PKS modules in the final PKS step
        bound_product_mol_object = self.pks_designs[-1][pks_design_num][0].computeProduct(retrotide.structureDB)

        # then, we run offloading reactions to detach this bound product

        if pks_release_mechanism == 'thiolysis':
            Chem.SanitizeMol(bound_product_mol_object)  # run detachment reaction to produce terminal acid group
            rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])[S:3]>>[C:1](=[O:2])[O].[S:3]')
            product = rxn.RunReactants((bound_product_mol_object,))[0][0]
            Chem.SanitizeMol(product)
            product_smiles = Chem.MolToSmiles(product)
            return product_smiles

        if pks_release_mechanism == 'cyclization':
            Chem.SanitizeMol(bound_product_mol_object)  # run detachment reaction to cyclize bound substrate
            rxn = AllChem.ReactionFromSmarts(
                    '([C:1](=[O:2])[S:3].[O,N:4][C:5][C:6])>>[C:1](=[O:2])[*:4][C:5][C:6].[S:3]')
            try:
                product = rxn.RunReactants((bound_product_mol_object,))[0][0]
                Chem.SanitizeMol(product)
                product_smiles = Chem.MolToSmiles(product)
                return product_smiles

            # if the bound substrate cannot be cyclized, then return None
            except:
                raise ValueError("\nUnable to perform cyclization reaction")

        else:
            raise ValueError(f"\nUnsupported pks_release_mechanism: {pks_release_mechanism}")

    def run_pks_synthesis(self) -> None:
        """
        Executes the synthesis process for polyketide synthases (PKSs) using Retrotide.
        This function generates and then stores potential PKS designs based on the input target molecule.
        It initiates the PKS design process then selects the best PKS design based on chemical similarity.
        This similarity is computed between the PKS product and the final target.
        Finally, a PKS termination/ offloading reaction is run to obtain the final PKS product.

        Following is a detailed breakdown of the key steps this function performs:
        1. re-imports retrotide to ensure only user-defined starter and extender units are used
        2. runs retrotide to design PKSs and then stores the designs (by assigning them to self.pks_designs)
        3. prints the best PKS design based on the chemical similarity between the PKS product and final target
        4. runs the specified PKS termination reaction on the top PKS design to get the final PKS product

        Parameters:
        ----------
        None
        """
        print("\nStarting PKS synthesis with Retrotide")
        print("---------------------------------------------")

        # re-import Retrotide to ensure only user-defined starter and extender units are used (slightly hacky)
        from .retrotide import retrotide

        # run Retrotide then generate and store Retrotide's PKS designs
        self.pks_designs = retrotide.designPKS(targetMol = Chem.MolFromSmiles(self.target_smiles),
                                               similarity = self.pks_similarity_metric)

        print('\nBest PKS design: ' + repr(self.pks_designs[-1][0][0].modules))

        # perform offloading/ termination reaction with the top PKS design
        try:
            self.pks_top_final_product = self.run_pks_termination(pks_design_num = 0,
                                                                  pks_release_mechanism = self.pks_release_mechanism)
        except ValueError:
            self.pks_top_final_product = None

        if self.pks_top_final_product:
            print(f"\nClosest final product is: {self.pks_top_final_product}")

            # if the final product is reached, then the user can stop here
            if self._remove_stereo(self.pks_top_final_product) == self.target_smiles:
                print(f"\nFinished PKS retrobiosynthesis: target reached by PKSs alone.")

            # if the final product is not reached, then the user can progress onto non-PKS modifications
            if self.pks_top_final_product != self.target_smiles:
                print(f"\nFinished PKS synthesis: closest product to the target using the top PKS design of "
                      f"{repr(self.pks_designs[-1][0][0].modules)} is {self.pks_top_final_product}.")

        else:
            print(f"\nNo product can be released by terminating the top PKS design")

    ###########################
    # DORAnet / biology methods #
    ###########################

    def run_biology(self, precursor_smiles: str,
                    target_name: str,
                    target_smiles: str,
                    direction: str,
                    num_generations: int,
                    PKS_design_number: int) -> dict:
        """
        A wrapper to run DORAnet biology and return biological pathways from the top PKS product generated by Retrotide
        to the target metabolite.

        Parameters:
        ----------
        precursor_smiles: str
            SMILES string of top PKS product.

        target_name: str
            name of the target molecule to be used for the job name.

        target_smiles: str
            SMILES string of target product after bio modifications.

        direction: str
            direction of the reaction; either forward or reverse.

        num_generations: int
            number of generations.

        PKS_design_number: int
            the number of the PKS product biologically expanded upon by DORAnet.

        Returns:
        ----------
        pathways_found: boolean
            indicates whether biological pathways were successfully found from the top PKS product to the target.

        pathways_dict: dict
            a dictionary of pathways found, with the key being pathway number, and the value being a dictionary of
            reactions, with the reaction number as the key and values as reaction strings.

        ranked_metabolites: dict
            a dictionary of most  similar molecules to the target metabolite as determined by the MCS algorithm
            in the case that the target cannot be biologically produced from the top PKS product with the given
            number of generations.

        Raises:
        ----------
        Value Error:
            Ensures that the top PKS product is a valid molecule.
        """

        # 1. Ensure precursor SMILES are in their canonical form and without stereochemistry
        precursor_mol = Chem.MolFromSmiles(precursor_smiles)
        if precursor_mol is None:
            raise ValueError(f'Invalid SMILES string: {precursor_smiles}')
        Chem.RemoveStereochemistry(precursor_mol)
        canon_precursor_smiles = Chem.MolToSmiles(precursor_mol)

        # 2. Run DORAnet
        job_name = f'{target_name}_PKS{PKS_design_number}_BIO{num_generations}'

        forward_network = enzymatic.generate_network(
            job_name = job_name,  # name of the job, can be anything
            starters = {f'{canon_precursor_smiles}'},  # starting molecule(s)
            gen = num_generations,  # number of generations
            direction = f'{direction}',  # direction of operators, here forward direction
        )

        # 3. Extract all products from running DORAnet and check if target is present
        # to check if target is present, we must also canonicalize and remove stereo from target_smiles
        # then if we can get a list of metabolites from forward_network, let's check if target is in there

        target_mol = Chem.MolFromSmiles(target_smiles)
        if target_mol is None:
            raise ValueError(f'Invalid SMILES string: {precursor_smiles}')
        Chem.RemoveStereochemistry(target_mol)
        canon_target_smiles = Chem.MolToSmiles(target_mol)

        # 4. When biological pathways that reach the target molecule are found, they are processed into a pathways.txt
        # file, which is parsed through to create a dictionary of pathways compatible as an input for the feasibility
        # classifier.

        try:
            post_processing.one_step(
                networks = {
                    forward_network,
                },
                total_generations = num_generations,
                starters = {f'{canon_precursor_smiles}'},
                target = f'{canon_target_smiles}',
                job_name = job_name,
                search_depth = num_generations,
                max_num_rxns = num_generations,
                min_rxn_atom_economy = 0,
                num_process = num_generations,
            )
            self.pathways_found = True
            pathways_dict = {}

            with open(f'{job_name}_pathways.txt', 'r') as file:
                pathway_counter = 0
                lines = file.readlines()
                for i, line in enumerate(lines):
                    if 'reaction SMILES stoichiometry' in line:
                        pathway_counter += 1
                        reactions_list = []
                        rules_list = []
                        stoich_list = line.strip('reaction SMILES stoichiometry ')
                        stoich_list = stoich_list.lstrip("'[").rstrip("]'\n")
                        stoich_list = stoich_list.split("', '")

                        for j in range(0, len(stoich_list)):
                            reaction = lines[i + j + 2].rstrip('\n')
                            rxn_string = self.simplify_reaction(reaction, stoich_list[j])
                            reactions_list.append(rxn_string)
                            rule = lines[i + j + 3].rstrip('\n')
                            rules_list.append(rule)

                        pathways_dict.update({f'pathway {pathway_counter}': {'reactions (SMILES)': reactions_list,
                                                                             'reaction rules': rules_list}})

            self.non_pks_pathways = pathways_dict

        # In the case that pathways are not found, the top 10 most MCS similar molecules to the target molecule are
        # found to provide the user with potential starting points.

        except FileNotFoundError:
            self.pathways_found = False
            mol_dict = {}
            for mol in forward_network.mols:
                mol_dict[mol.uid] = self._calculate_similarity(target_smiles, mol.uid, 'mcs_without_stereo')
            ranked_metabolites = dict(sorted(mol_dict.items(), key=lambda item: item[1], reverse=True))

            self.ranked_metabolites = ranked_metabolites

    def run_pickaxe(self, precursor_smiles: str) -> None:
        """
        Conducts a biological Pickaxe expansion on a specified precursor molecule. This could be a PKS product.
        This function performs multiple steps, such as, the canonicalization of the input SMILES string, tsv file
        preparation for starting and target compounds, initialization of the biological pickaxe object with user-defined
        or default rules and coreactants, loading of the compounds into Pickaxe, application of filters,
        execution of the transformation process, and extraction as well as storage of the resulting compounds
        and pathways. The detailed order of operations for performing  a Pickaxe expansion and extracting pathways
        is given below.

        Parameters:
        ----------
        precursor_smiles: str
            The SMILES string of the precursor molecule for the Pickaxe expansion

        Returns:
        ----------
        None

        Notes:
        ----------
        - this function modifies the `non_pks_compounds_df`, `non_pks_pathways` attributes of the class instance
        - this function assumes that `target_smiles`, `non_pks_cofactors`, `rule_filepath`, `input_cpd_dir`,
          `non_pks_sim_score_filter`, `non_pks_sim_score_cutoffs`, `non_pks_sim_sample`, `non_pks_sim_sample_size`,
          `non_pks_cores`, `non_pks_steps`, and `known_metabolites` are properly set attributes of the instance.
        - this function requires the 'pickaxe_utils' module and associated functions and classes
          (e.g., `SimilarityFilter`, `SimilaritySamplingFilter`, `Pickaxe`) to be properly imported and configured
        - the function may raise FileNotFoundError if specified rule files are not found


        Processes:
        ----------
        1. canonicalizes the input precursor_smiles for consistency.
        2. writes the starting and target compounds to respective .tsv files
        3. initializes a pickaxe object with user-defined or default rules and coreactants
        4. loads the starting and target compounds into the biological pickaxe object
        5. applies similarity-based filters if specified by the user
        6. executes the transformation process in pickaxe across the specified number of generations
        7. assigns ID to the generated compounds
        8. extracts non-PKS reactions and utilizes them to create a graph
        9. extracts sequences and pathways from the graph that lead from the PKS product to the target molecule
        """
        print(f"\nStarting pickaxe expansion on {precursor_smiles}")
        print("---------------------------------------------")
        self.current_pickaxe_precursor = self._canon_smi(self._remove_stereo(precursor_smiles))

        # write starting compound (PKS product) to a tsv file
        precursor_filepath = pickaxe_utils.write_cpds_to_tsv(cpd_name = self.current_pickaxe_precursor,
                                                             cpd_smi = self.current_pickaxe_precursor,
                                                             input_cpd_dir = self.input_cpd_dir)

        # write target compound (expected Pickaxe product) to a tsv file
        target_filepath = pickaxe_utils.write_cpds_to_tsv(cpd_name = self.target_smiles,
                                                          cpd_smi = self.target_smiles,
                                                          input_cpd_dir = self.input_cpd_dir)

        # initialize a Pickaxe object
        try:
            pk = Pickaxe(coreactant_list = self.non_pks_cofactors,
                         rule_list = self.rule_filepath)
        except FileNotFoundError:
            print('\nInvalid reaction rules filepath. Switching to intermediate rules.')
            pk = Pickaxe(coreactant_list = self.non_pks_cofactors,
                         rule_list = '/.root/data/coreactants_and_rules/JN3604IMT_rules.tsv')

        # load starting compound (PKS product) and target compound (expected Pickaxe product) into Pickaxe
        pk.load_compound_set(compound_file = precursor_filepath)
        pk.load_targets(target_compound_file = target_filepath)

        ## incorporate filters into pickaxe
        if self.non_pks_sim_score_filter is True:
            sample_fingerprint_method = "Morgan"
            cutoff_fingerprint_method = "Morgan"
            cutoff_fingerprint_args = {"radius": 2}
            cutoff_similarity_method = "Tanimoto"  # Similarity filter

            crit_similarity = taniFilter = SimilarityFilter(
                crit_similarity = self.non_pks_sim_score_cutoffs,
                increasing_similarity = False, # i.e metabolites don't have to strictly increase in similarity
                fingerprint_method = sample_fingerprint_method,
                fingerprint_args = cutoff_fingerprint_args,
                similarity_method = cutoff_similarity_method)

            pk.filters.append(crit_similarity)

        if self.non_pks_sim_sample is True:
            # Similarity sampling filter
            sample_size = self.non_pks_sim_sample_size
            sample_fingerprint_method = "Morgan"
            sample_fingerprint_args = {"radius": 2}
            sample_similarity_method = "Tanimoto"

            taniSampleFilter = SimilaritySamplingFilter(
                sample_size = sample_size,
                weight=weight,
                fingerprint_method=sample_fingerprint_method,
                fingerprint_args = sample_fingerprint_args,
                similarity_method=sample_similarity_method)
            pk.filters.append(taniSampleFilter)

        ### most important - run Pickaxe
        pk.transform_all(processes = self.non_pks_cores,
                         generations = self.non_pks_steps)
        pk.assign_ids()

        # create and store dataframe of compounds generated by Pickaxe
        compounds_df = pickaxe_utils.create_compounds_df(pk)

        # store the dataframe of compounds generated by Pickaxe
        self.non_pks_compounds_df = compounds_df

        # extract non-PKS reactions from Pickaxe object
        pk_rxn_keys = [key for key in pk.reactions.keys()]

        all_pk_rxn_ids = [pk.reactions[key]['ID'] for key in pk_rxn_keys]
        all_rxn_strs_in_cpd_ids = [pk.reactions[key]['ID_rxn'] for key in pk_rxn_keys]
        all_rxn_strs_in_SMILES = [pk.reactions[key]['SMILES_rxn'] for key in pk_rxn_keys]
        all_rxn_rules = [list(pk.reactions[key]['Operators']) for key in pk_rxn_keys]

        # use extracted reactions and Pickaxe object to create a graph
        self.current_pickaxe_graph = pickaxe_utils.create_graph(all_rxn_strs_in_cpd_ids,
                                                                self.current_pickaxe_precursor)

        # get and store sequences from Graph
        sequences = pickaxe_utils.get_sequences_from_graph(G = self.current_pickaxe_graph,
                                                           compounds_df = self.non_pks_compounds_df,
                                                           precursor_smiles = self.current_pickaxe_precursor,
                                                           target_smiles = self.target_smiles,
                                                           num_generations = self.non_pks_steps)

        ### If Pickaxe was able to generate the target molecule, extract the sequences
        if sequences:
            # initialize a dictionary to store all sequences between the PKS product and eventual target product
            all_sequences_dict = {}

            for i, seq in enumerate(sequences):
                seq_SMILES = [list(compounds_df[compounds_df["ID"] == id]["SMILES"])[0] for id in seq]

                all_sequences_dict.update({f"seq {i}":
                                               {"seq_num": str(i),
                                                "seq (IDs)": seq,
                                                "seq (SMILES)": seq_SMILES}})

            # extract pathways between the PKS product and eventual target product
            self.non_pks_pathways = pickaxe_utils.get_pathways_from_graph_proto(sequences,
                                                                                self.known_metabolites,
                                                                                self.non_pks_compounds_df,
                                                                                pk)
        else:
            self.non_pks_pathways = None

    def pick_closest_non_pks_product(self) -> Tuple[str,float]:
        """
        Identifies the most chemically similar non-PKS product to the final, target product. This function filters
        compounds from a pickaxe expansion and which are stored in 'self.non_pks_compounds_df'. It then calculates the
        chemical similarity between each predicted non-PKS compound and the target product and sorts predicted compounds based on their similarity scores in descending order.

        The function returns the SMILES string of the most similar non-PKS compound and its corresponding similarity score.

        Parameters:
        ----------

        Returns:
        ----------
        - tuple: A tuple containing two elements:
            1. str: The SMILES string of the non-PKS compound that is closest to the target product.
            2. float: The similarity score of this compound to the target product.

        Notes:
        ----------
        - The function assumes `non_pks_compounds_df` and `target_smiles` are properly set attributes of the instance.
        - The `_calculate_similarity` method of the instance is used for computing similarity scores.
        """
        # extract out predicted compounds and their SMILES strings from pickaxe expansion
        predicted_cpds_df = self.non_pks_compounds_df[self.non_pks_compounds_df["Type"] == "Predicted"]
        predicted_cpds_smiles = list(predicted_cpds_df['SMILES'])

        print('\nCalculating chemical similarities between all non-PKS products and final target product')

        # initialize an empty list to calculate the structural similarity score of each biological pickaxe product
        # with respect to the final, downstream target molecule
        sim_scores = []

        for cpd in predicted_cpds_smiles:
            sim_scores.append(self._calculate_similarity(smiles1 = cpd,
                                                         smiles2 = self.target_smiles,
                                                         metric = self.non_pks_similarity_metric))

        sim_df = pd.DataFrame({'SMILES': predicted_cpds_smiles,
                              'Similarity': sim_scores}).sort_values(by = 'Similarity', ascending = False)

        best_non_pks_product = sim_df.iloc[0, :]['SMILES']
        best_sim_score = sim_df.iloc[0, :]['Similarity']

        return best_non_pks_product, best_sim_score

    def calculate_non_pks_pathway_feasibilities(self) -> None:

        # for each pathway between pks product and final downstream target
        for pathway in self.non_pks_pathways.keys():

            # initialize an empty list of reaction feasibility scores
            pathway_feasibilities = []
            rxns = self.non_pks_pathways[pathway]['reactions (SMILES)']

            # for each reaction in a pathway
            for rxn_str in rxns:
                # calculate feasibility score
                feasibility_score = self.feasibility_classifier.predict_proba(rxn_str)
                # store this score (convert to str for ease of saving to json)
                pathway_feasibilities.append(str(feasibility_score))

            # calculate net feasibility for this pathway
            net_feasibility = 1
            for score in pathway_feasibilities:
                net_feasibility *= float(score)

            # update the existing pathways dictionary
            self.non_pks_pathways[pathway].update({'feasibilities': pathway_feasibilities})
            self.non_pks_pathways[pathway].update({'net feasibility': str(net_feasibility)})

    def run_combined_synthesis(self, max_designs: int) -> Union[dict, None]:

        if self.pathway_sequence == ['pks']:
            self.run_pks_synthesis()

            # considering only the best PKS design, compute chemical similarity between the top pks product and the target
            top_pks_product_sim_score = self._calculate_similarity(smiles1=self.pks_top_final_product,
                                                                   smiles2=self.target_smiles,
                                                                   metric=self.pks_similarity_metric)

            # If the target product has been reached with PKSs,
            if self._canon_smi(self._remove_stereo(self.pks_top_final_product)) == self.target_smiles:
                results_entry = {'pks_design_num': 0,
                                 'pks_design': self.pks_designs[-1][0][0].modules, # top pks design so middle index is 0
                                 'pks_product': self.pks_top_final_product,
                                 'pks_product_similarity': top_pks_product_sim_score}

                self.results_logs.append(results_entry)  # store the successful PKS design

            # if the target product has not been reached with PKSs, we all PKS designs from Retrotide
            else:
                for i in range(0, len(self.pks_designs[-1])):

                    # calculate the PKS product from this i-th PKS design
                    pks_product = self.run_pks_termination(pks_design_num = i,
                                                           pks_release_mechanism = self.pks_release_mechanism)

                    # calculate the chemical similarity between UNBOUND pks product and the target SMILES
                    # if you want to store chemical similarity between BOUND pks product and target SMILES instead,
                    # store self.pks_designs[-1][i][1] in the results_logs under 'pks_product_similarity'
                    pks_product_vs_target_sim_score = self._calculate_similarity(smiles1 = pks_product,
                                                                                 smiles2 = self.target_smiles,
                                                                                 metric = self.pks_similarity_metric)

                    # store each PKS design in the remaining set
                    self.results_logs.append({'pks_design_num': i,
                                              'pks_design': self.pks_designs[-1][i][0].modules,
                                              'pks_product': pks_product,
                                              'pks_product_similarity': pks_product_vs_target_sim_score})

        if self.pathway_sequence == ['pks','bio']:

            ### Run PKS synthesis first by calling on Retrotide
            self.run_pks_synthesis()

            # considering only the best PKS design, compute chemical similarity between the top pks product and the target
            top_pks_product_sim_score = self._calculate_similarity(smiles1 = self.pks_top_final_product,
                                                                   smiles2 = self.target_smiles,
                                                                   metric = self.non_pks_similarity_metric)

            # If the target product has been reached with PKSs,
            if self._canon_smi(self._remove_stereo(self.pks_top_final_product)) == self.target_smiles:
                results_entry = {'pks_design_num': 0,
                                 'pks_design': self.pks_designs[-1][0][0].modules, # top pks design hence middle index is 0
                                 'pks_product': self.pks_top_final_product, # top pks product would be the user's target
                                 'pks_product_similarity': top_pks_product_sim_score}

                self.results_logs.append(results_entry) # store the successful PKS design

                # If user wants to stop the moment their target product is synthesized, terminate here
                if self.stopping_criteria == 'first_product_formation':
                    return

            # If target is not reached with PKS, run pickaxe on this top PKS product for user-specified number of steps
            print(f'\nMoving onto non-PKS modifications...')
            self.run_biology(precursor_smiles = self.pks_top_final_product, target_name = self.target_name,
                             target_smiles = self.target_smiles, direction = 'forward',
                             num_generations = self.non_pks_steps, PKS_design_number = 0)

            ## If pathways between the top PKS product and final target are found, then the user's job is complete
            if self.pathways_found:
                print(f"Pathways found in {self.non_pks_steps} step/s between the top PKS product "
                     f"{self.pks_top_final_product} and the eventual target product {self.target_smiles} !!!")

                self.calculate_non_pks_pathway_feasibilities()

                # record and track the relevant similarity scores as well as PKS and non-PKS products
                results_entry = {'pks_design_num': 0,
                                 'pks_design': self.pks_designs[-1][0][0].modules,
                                 'pks_product': self.pks_top_final_product,
                                 'pks_product_similarity': top_pks_product_sim_score,
                                 'non_pks_product': self.target_smiles,
                                 'non_pks_product_similarity': 1.0,
                                 'non_pks_pathways': self.non_pks_pathways}

                self.results_logs.append(results_entry)

            ## If pathways between the top PKS product and final target are not found, consider alternative PKS designs
            else:

                # still record and track the relevant similarity scores as well as PKS and non-PKS products
                results_entry = {'pks_design_num': 0,
                                 'pks_design': self.pks_designs[-1][0][0].modules,
                                 'pks_product': self.pks_top_final_product,
                                 'pks_product_similarity': top_pks_product_sim_score,
                                 'non_pks_product': list(self.ranked_metabolites.keys())[0],
                                 'non_pks_product_similarity': self.ranked_metabolites[list(self.ranked_metabolites.keys())[0]]}

                self.results_logs.append(results_entry)

                print(f"\nNo pathways to target are found using non-PKS enzymes for {self.non_pks_steps} step/s and the "
                      f"top PKS module design")

                print(f"\nAttempting non-PKS pathways for {self.non_pks_steps} step/s on PKS product from the next {max_designs} "
                      f"best PKS designs that produce unique PKS products.")

                # initialize a list to track all PKS products generated and all designs enumerated
                all_PKS_products_list = [self.pks_top_final_product]
                all_PKS_design_count = 0

                # initialize another list to track only PKS designs that give unique products
                unique_PKS_products_list = [self.pks_top_final_product]
                unique_PKS_design_count = 0

                while unique_PKS_design_count <= max_designs:
                    all_PKS_design_count += 1 # increase this counter by 1 first so that the top PKS design isn't repeated

                    try:
                        # extract the PKS product from this next PKS design
                        current_pks_product = self.run_pks_termination(pks_design_num = all_PKS_design_count,
                                                                       pks_release_mechanism = self.pks_release_mechanism)

                    except IndexError: # we have exhausted all PKS designs
                        print(f"\nAll PKS designs have been exhausted and {unique_PKS_design_count} unique PKS designs were considered")
                        print(f"\nIn these {unique_PKS_design_count} unique PKS designs, no enzymatic pathways were found "
                              f"between the PKS product and the final, downstream target")
                        break

                    # track the current PKS product
                    all_PKS_products_list.append(current_pks_product)

                    # check if this current PKS product is unique and do nothing if PKS product has been generated before
                    if current_pks_product in unique_PKS_products_list:
                        pass

                    # if PKS product is unique and has not been generated before
                    if current_pks_product not in unique_PKS_products_list:
                        unique_PKS_design_count += 1 # first, update the number of unique PKS designs enumerated
                        unique_PKS_products_list.append(current_pks_product)  # then update all unique PKS products

                        # calculate chemical similarity between this current PKS product and the final target product
                        current_pks_product_sim_score = self._calculate_similarity(smiles1 = current_pks_product,
                                                                                  smiles2 = self.target_smiles,
                                                                                  metric = self.non_pks_similarity_metric)

                        # run pickaxe with this current PKS product
                        # self.run_pickaxe(precursor_smiles = current_pks_product)
                        self.run_biology(precursor_smiles = current_pks_product, target_name = self.target_name,
                        target_smiles = self.target_smiles, direction = 'forward',
                        num_generations = self.non_pks_steps, PKS_design_number = unique_PKS_design_count)

                        # extract closest non-PKS product score
                        current_non_pks_product = list(self.ranked_metabolites.keys())[0]
                        current_non_PKS_sim_score = self.ranked_metabolites[current_non_pks_product]

                        # if enzymatic pathways are found between this PKS product and the final target
                        # then the synthesis process can be terminated here
                        if self.non_pks_pathways:
                            print(f"\nPathways found with non-PKS enzymes in {self.non_pks_steps} step/s "
                                  f"using unique PKS design #{unique_PKS_design_count} out of all Retrotide designs !!!")
                            current_non_PKS_sim_score = 1.0
                            self.calculate_non_pks_pathway_feasibilities()

                            # record and track the relevant similarity scores as well as PKS and non-PKS products
                            results_entry = {'pks_design_num': all_PKS_design_count,
                                             'pks_design': self.pks_designs[-1][all_PKS_design_count][0].modules,
                                             'pks_product': current_pks_product,
                                             'pks_product_similarity': current_pks_product_sim_score,
                                             'non_pks_product': self.target_smiles,
                                             'non_pks_product_similarity': current_non_PKS_sim_score,
                                             'non_pks_pathways': self.non_pks_pathways}

                            self.results_logs.append(results_entry)

                            if self.stopping_criteria == "first_product_formation":
                                break
                            else:
                                print("Continuing to see if even more pathways can be found")

                        # if no enzymatic pathways are found between this PKS product and the final target
                        else:

                            print(f"\nNo pathways found with non-PKS enzymes in {self.non_pks_steps} step/s "
                                  f"using unique PKS design #{unique_PKS_design_count} out of all Retrotide designs. "
                                  f"Moving onto the next unique PKS design")

                            results_entry = {'pks_design_num': all_PKS_design_count,
                                             'pks_design': self.pks_designs[-1][all_PKS_design_count][0].modules,
                                             'pks_product': current_pks_product,
                                             'pks_product_similarity': current_pks_product_sim_score,
                                             'non_pks_product': current_non_pks_product,
                                             'non_pks_product_similarity': current_non_PKS_sim_score}

                            self.results_logs.append(results_entry)

    def run_non_pks_synthesis_retro(self,pks_release_mechanism):
        """
        Use Pickaxe to perform a retro-expansion on the final, downstream target molecule to generate upstream precursors
        See if any PKS designs can synthesize these upstream precursors
        :return:
        """
        print(f"\nStarting a reverse pickaxe expansion on the final, downstream target {self.target_smiles}")
        print('')

        downstream_target_smiles = pickaxe_utils.canonicalize_smiles(self.target_smiles)

        # write starting compound (PKS product) to a tsv file
        precursor_filepath = pickaxe_utils.write_cpds_to_tsv(cpd_name = downstream_target_smiles,
                                                             cpd_smi = downstream_target_smiles,
                                                             input_cpd_dir = self.input_cpd_dir)

        # initialize a Pickaxe object
        pk = Pickaxe(coreactant_list=self.non_pks_cofactors, rule_list=self.rule_filepath)

        # load starting compound (PKS product) and target compound (expected Pickaxe product) into Pickaxe
        pk.load_compound_set(compound_file=precursor_filepath)

        ## incorporate filters into pickaxe

        if self.non_pks_sim_score_filter is True:
            # Similarity filter
            sample_fingerprint_method = "Morgan"
            cutoff_fingerprint_method = "Morgan"
            cutoff_fingerprint_args = {"radius": 2}
            cutoff_similarity_method = "Tanimoto"

            crit_similarity = taniFilter = SimilarityFilter(
                crit_similarity=self.non_pks_sim_score_cutoffs,
                increasing_similarity=False,  # i.e metabolites don't have to strictly increase in similarity
                fingerprint_method=sample_fingerprint_method,
                fingerprint_args=cutoff_fingerprint_args,
                similarity_method=cutoff_similarity_method)

            pk.filters.append(crit_similarity)

        if self.non_pks_sim_sample is True:
            # Similarity sampling filter
            sample_size = self.non_pks_sim_sample_size
            sample_fingerprint_method = "Morgan"
            sample_fingerprint_args = {"radius": 2}
            sample_similarity_method = "Tanimoto"

            taniSampleFilter = SimilaritySamplingFilter(
                sample_size=sample_size,
                weight=weight,
                fingerprint_method=sample_fingerprint_method,
                fingerprint_args=sample_fingerprint_args,
                similarity_method=sample_similarity_method)
            pk.filters.append(taniSampleFilter)

        # run Pickaxe
        pk.transform_all(processes=self.non_pks_cores, generations=self.non_pks_steps)
        pk.assign_ids()

        # create a dataframe of compounds generated by Pickaxe
        compounds_df = pickaxe_utils.create_compounds_df(pk)

        # extract non-PKS reactions from Pickaxe object
        pk_rxn_keys = [key for key in pk.reactions.keys()]

        all_pk_rxn_ids = [pk.reactions[key]['ID'] for key in pk_rxn_keys]
        all_rxn_strs_in_cpd_ids = [pk.reactions[key]['ID_rxn'] for key in pk_rxn_keys]
        all_rxn_strs_in_SMILES = [pk.reactions[key]['SMILES_rxn'] for key in pk_rxn_keys]
        all_rxn_rules = [list(pk.reactions[key]['Operators']) for key in pk_rxn_keys]

        upstream_precursors = list(compounds_df[compounds_df['Type'] == 'Predicted']["SMILES"])

        print(f'\nThere are {len(upstream_precursors)} precursors (located by enzymatic reactions upstream) of the final target molecule')
        print('\n----------------------------------------------------------')

        desired_upstream_precursors = []

        for upstream_precursor_smiles in upstream_precursors:

            # if the upstream precursor has a carboxylic acid group
            if has_carboxylic_acid(Chem.MolFromSmiles(upstream_precursor_smiles)):
                print(f'\n Detected a carboxylic acid group in this upstream precusor {upstream_precursor_smiles}')

                # doubly reduce the acid group on this precursor first
                # the double_reduce function returns a mol object
                reduced_intermediate_mol = double_reduce(Chem.MolFromSmiles(upstream_precursor_smiles))
                reduced_intermediate_smi = Chem.MolToSmiles(reduced_intermediate_mol)
                print(f'\n Doubly reducing this acid group gives: {reduced_intermediate_smi}')
                print(f'Checking if feeding {reduced_intermediate_smi} as a precursor to RetroTide can give the upstream precursor {upstream_precursor_smiles}:')

                designs = retrotide.designPKS((Chem.MolFromSmiles(reduced_intermediate_smi)))
                top_design = designs[-1][0][0]
                bound_product_mol_object = top_design.computeProduct(retrotide.structureDB)

                if pks_release_mechanism == 'thiolysis':
                    ## Run detachment reaction via a decarboxylation
                    Chem.SanitizeMol(bound_product_mol_object)
                    rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])[S:3]>>[C:1](=[O:2])[O].[S:3]')
                    product = rxn.RunReactants((bound_product_mol_object,))[0][0]
                    Chem.SanitizeMol(product)
                    pks_product = Chem.MolToSmiles(product)
                    print(f'\n This PKS product is {pks_product}')

                    mol = Chem.MolFromSmiles(pks_product)
                    Chem.RemoveStereochemistry(mol)
                    pks_product = Chem.MolToSmiles(mol)
                    print(f'\n After removing stereochemistry, the PKS product is {pks_product}')

                    if pks_product == upstream_precursor_smiles:
                        print('\nThus, this upstream precursor can be synthesized with PKSs !!!')

            else:
                print(f'\nRunning PKS retrosynthesis on {upstream_precursor_smiles}')


                designs = retrotide.designPKS(Chem.MolFromSmiles(upstream_precursor_smiles))
                top_design = designs[-1][0][0]
                bound_product_mol_object = top_design.computeProduct(retrotide.structureDB)

                if pks_release_mechanism == 'thiolysis':
                    ## Run detachment reaction via a decarboxylation
                    Chem.SanitizeMol(bound_product_mol_object)
                    rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])[S:3]>>[C:1](=[O:2])[O].[S:3]')
                    product = rxn.RunReactants((bound_product_mol_object,))[0][0]
                    Chem.SanitizeMol(product)
                    pks_top_final_product = Chem.MolToSmiles(product)

                    print(f"The PKS product is {pks_top_final_product}")

                else:
                    pks_top_final_product = None

                mol = Chem.MolFromSmiles(pks_top_final_product)
                Chem.RemoveStereochemistry(mol)
                pks_top_final_product_no_stereo = Chem.MolToSmiles(mol)

                if pks_top_final_product_no_stereo == upstream_precursor_smiles:
                    print('\nPKSs can be used to make this upstream precursor')

            print('\n----------------------------------------------------------')

    def save_results_logs(self):
        os.makedirs(dir_path + f'../data/results_logs/', exist_ok=True)

        # if only PKSs are used, then the top PKS product indicates if the final target has been reached or not
        if self.pathway_sequence == ['pks']:
            output_filepath = dir_path + self.config_dict['results_dir'] + f'/{self.target_name}_PKS_only.txt'
            output_config_filepath = dir_path + self.config_dict['results_dir'] + f'/{self.target_name}_PKS_only_config.json'
            with open(output_config_filepath, 'w') as file:
                json.dump(self.config_dict, file)
            with open(output_filepath, 'w') as file:
                for result in self.results_logs:
                    file.write(f"\nPKS design number: {result['pks_design_num']}")
                    file.write(f"\nPKS design:")
                    for module in result['pks_design']:
                        file.write(f"\n    {module}")
                    file.write(f"\nPKS product: {result['pks_product']}")
                    file.write(f"\nPKS product similarity: {result['pks_product_similarity']}")
                    file.write(f"\n----------------")

        # if PKSs and biology is used,
        if self.pathway_sequence == ['pks', 'bio']:
            output_filepath = dir_path + self.config_dict['results_dir'] + f'/{self.target_name}_PKS_BIO{self.non_pks_steps}.txt'
            output_config_filepath = dir_path + self.config_dict['results_dir'] + f'/{self.target_name}_PKS_BIO{self.non_pks_steps}_config.json'
            with open(output_config_filepath, 'w') as file:
                json.dump(self.config_dict, file)
            with open(output_filepath,'w') as file:
                for result in self.results_logs:
                    file.write(f"\nPKS design number: {result['pks_design_num']}")
                    file.write(f"\nPKS design:")

                    for module in result['pks_design']:
                        file.write(f"\n    {module}")

                    file.write(f"\nPKS product: {result['pks_product']}")
                    file.write(f"\nPKS product similarity: {result['pks_product_similarity']}")
                    file.write(f"\nBio product: {result['non_pks_product']}")
                    file.write(f"\nBio product similarity: {result['non_pks_product_similarity']}")

                    if 'non_pks_pathways' in list(result.keys()):
                        file.write("\nBio pathways:")
                        d = result['non_pks_pathways']
                        pathways_list = [d[key] for key in d.keys()]

                        sorted_pathways_list = sorted(pathways_list,
                                                      key=lambda x: float(x['net feasibility']),
                                                      reverse=True)

                        # sort the pathways in descending order of their net feasibilitie
                        for i, pathway in enumerate(sorted_pathways_list):
                            file.write(f"\n  Pathway #{f'{i}'}:")
                            pickaxe_rxn_strs = pathway['reactions (SMILES)']

                            for pickaxe_rxn_str in pickaxe_rxn_strs:
                                file.write(f"\n    {pickaxe_rxn_str}")

                            reaction_feasibilities = pathway['feasibilities']
                            reaction_feasibilities = [f"{float(x):.3f}" for x in reaction_feasibilities]
                            net_feasibility = float(pathway['net feasibility'])

                            file.write(f"\n    reaction feasibilities: {reaction_feasibilities}")
                            file.write(f"\n    net feasibility: {net_feasibility:.3f}")
                            file.write(f"\n    reaction rules: {pathway['reaction rules']}")

                    file.write("\n")