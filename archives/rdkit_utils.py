#! /usr/bin/env python

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import re
import collections
import itertools
import copy
import json


def postsanitize_smiles(smiles_list):
    """
    Sanitize and tautomerize SMILES with aromatic nitrogens

    Parameters
    ----------
    smiles_list : list
        list of SMILES with aromatic nitrogens to be cleaned up

    Returns
    -------
    list of all lists of tautomerized SMILES
    """

    sanitized_list = []
    tautomer_smarts = '[#7H1X3&a:1]:[#6&a:2]:[#7H0X2&a:3]>>[#7H0X2:1]:[#6:2]:[#7H1X3:3]'

    for s in smiles_list:

        temp_mol = Chem.MolFromSmiles(s, sanitize=False)

        # # pickaxe
        # temp_mol = Chem.rdmolops.RemoveHs(temp_mol)

        aromatic_bonds = [i.GetIdx() for i in temp_mol.GetBonds() if i.GetBondType() == Chem.rdchem.BondType.AROMATIC]

        for i in temp_mol.GetBonds():
            if i.GetBondType() == Chem.rdchem.BondType.UNSPECIFIED:
                i.SetBondType(Chem.rdchem.BondType.SINGLE)

        try:
            Chem.SanitizeMol(temp_mol)
            Chem.rdmolops.RemoveStereochemistry(temp_mol)
            temp_smiles = Chem.MolToSmiles(temp_mol)

        except Exception as msg:
            if 'Can\'t kekulize mol' in str(msg):
                # unkekulized_indices = [int(i) for i in str(msg).split('Unkekulized atoms: ')[1].split('.')[0].rstrip(' \n').split(' ')]
                pyrrole_indices = [i[0] for i in temp_mol.GetSubstructMatches(Chem.MolFromSmarts('n'))]

                # indices to sanitize
                # for s_i in set(unkekulized_indices).intersection(set(pyrrole_indices)):
                for s_i in pyrrole_indices:
                    temp_mol = Chem.MolFromSmiles(s, sanitize=False)
                    if temp_mol.GetAtomWithIdx(s_i).GetNumExplicitHs() == 0:
                        temp_mol.GetAtomWithIdx(s_i).SetNumExplicitHs(1)
                    elif temp_mol.GetAtomWithIdx(s_i).GetNumExplicitHs() == 1:
                        temp_mol.GetAtomWithIdx(s_i).SetNumExplicitHs(0)
                    try:
                        Chem.SanitizeMol(temp_mol)

                        processed_pyrrole_indices = [i[0] for i in
                                                     temp_mol.GetSubstructMatches(Chem.MolFromSmarts('n'))]
                        processed_aromatic_bonds = [i.GetIdx() for i in
                                                    temp_mol.GetBonds() if i.GetBondType() == Chem.rdchem.BondType.AROMATIC]
                        if processed_pyrrole_indices != pyrrole_indices or aromatic_bonds != processed_aromatic_bonds:
                            continue

                        Chem.rdmolops.RemoveStereochemistry(temp_mol)
                        temp_smiles = Chem.MolToSmiles(temp_mol)
                        break
                    except:
                        continue
                if 'temp_smiles' not in vars():
                    Chem.rdmolops.RemoveStereochemistry(temp_mol)
                    temp_smiles = Chem.MolToSmiles(temp_mol)
                    sanitized_list.append([temp_smiles])
                    continue
            else:
                Chem.rdmolops.RemoveStereochemistry(temp_mol)
                temp_smiles = Chem.MolToSmiles(temp_mol)
                sanitized_list.append([temp_smiles])
                continue
        rxn = AllChem.ReactionFromSmarts(tautomer_smarts)

        try:
            tautomer_mols = rxn.RunReactants((Chem.MolFromSmiles(temp_smiles), ))
        except:
            try:
                tautomer_mols = rxn.RunReactants((Chem.MolFromSmiles(temp_smiles, sanitize=False),))
            except:
                continue

        tautomer_smiles = [Chem.MolToSmiles(m[0]) for m in tautomer_mols]
        sanitized_list.append(sorted(set(tautomer_smiles + [temp_smiles])))

    return list(itertools.product(*sanitized_list))


def get_cofactors(cofactor_list_path, cofactor_pair_path):
    """
    Get cofactor designation information for JN1224min

    Parameters
    ----------
    cofactor_list_path : str
        path to cofactor_list_alldb.tsv
    cofactor_pair_path : str
        path to cofactor_pair_alldb.json

    Returns
    -------
    cofactor_name_dict : dict
        {cofactor designation in JN1224min: SEED compound ID of the cofactor}
    cofactor_list_dict : dict
        {compound ID (capitalized) for database compounds recognized as cofactors: cofactor designation in JN1224min}
    cofactor_pair_dict : dict
        {tuple(lhs compound ID, rhs compound ID) (capitalized) for database compound pairs recognized as cofactor pairs:
        cofactor pair designation in JN1224min}
    """

    # cofactor to cpd id dict
    cofactor_name_dict = {}

    # cofactor list name designation
    cofactor_list_dict = {}
    for k, v in pd.read_csv(cofactor_list_path, sep='\t', index_col=0).iterrows():
        cofactor_list_dict[k.upper()] = v['replacement']
        if v['replacement'] not in cofactor_name_dict and k.startswith('cpd'):
            cofactor_name_dict[v['replacement']] = k

    # cofactor pair name designation
    cofactor_pair_dict = {}
    with open(cofactor_pair_path) as f:
        cofactor_pair_read_json = json.loads(f.read())
    for k, v in cofactor_pair_read_json.items():
        for pair in v:
            cofactor_pair_dict[(pair[1].upper(), pair[2].upper())] = k
            cofactor_pair_dict[(pair[2].upper(), pair[1].upper())] = '%s,%s' % (k.split(',')[1], k.split(',')[0])
            if k.split(',')[0] not in cofactor_name_dict and pair[1].startswith('cpd') and pair[2].startswith('cpd'):
                cofactor_name_dict[k.split(',')[0]] = pair[1]
                cofactor_name_dict[k.split(',')[1]] = pair[2]

    return cofactor_name_dict, cofactor_list_dict, cofactor_pair_dict


def label_cofactor(input_reactant_molfile, input_product_molfile, cofactor_list_dict, cofactor_pair_dict):
    """
    Label cofactor designation based on reactant & product compound IDs

    Parameters
    ----------
    input_reactant_molfile : list
        list of reactant MetaCyc or KEGG or BRENDA IDs
    input_product_molfile : list
        list of product MetaCyc or KEGG or BRENDA IDs
    cofactor_list_dict : dict
        {compound ID (capitalized) for database compounds recognized as cofactors: cofactor designation in JN1224min}
    cofactor_pair_dict : dict
        {tuple(lhs compound ID, rhs compound ID) (capitalized) for database compound pairs recognized as cofactor pairs:
        cofactor pair designation in JN1224min}

    Returns
    -------
    tuple
        (lhs cofactor designations, rhs cofactor designations)
    """

    # reactant & product names
    reactant_molfile = [':'.join(m.upper().split(':')[0:max(1, len(m.split(':')) - 1)]) for m in input_reactant_molfile]
    product_molfile = [':'.join(m.upper().split(':')[0:max(1, len(m.split(':')) - 1)]) for m in input_product_molfile]

    # new substrate labels
    reactant_names = ['Any'] * len(reactant_molfile)
    product_names = ['Any'] * len(product_molfile)

    # get cofactor pairs
    for i_lhs, lhs in enumerate(reactant_molfile):
        for i_rhs, rhs in enumerate(product_molfile):

            # skip if already assigned
            if product_names[i_rhs] != 'Any':
                continue

            # assign cofactor pair designation
            try:
                temp_pair = cofactor_pair_dict[(lhs, rhs)]
                reactant_names[i_lhs] = temp_pair.split(',')[0]
                product_names[i_rhs] = temp_pair.split(',')[1]
                break
            except KeyError:
                continue

        # assign cofactor list if no cofactor pair assigned
        if reactant_names[i_lhs] == 'Any':
            try:
                reactant_names[i_lhs] = cofactor_list_dict[lhs]
            except KeyError:
                continue

    # assign cofactor list for rhs
    for i_rhs, rhs in enumerate(product_molfile):

        # assign cofactor list if no cofactor pair assigned
        if product_names[i_rhs] == 'Any':
            try:
                product_names[i_rhs] = cofactor_list_dict[rhs]
            except KeyError:
                continue

    return ';'.join(reactant_names), ';'.join(product_names)


def get_atom_count(mol):
    """
    Molecule atom counter, adopted from Pickaxe by James Jeffryes

    Parameters
    ----------
    mol : str or Mol
        SMILES or Mol object of input molecule

    Returns
    -------
    atoms : dict
        atom counter dict
    """

    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    atoms = collections.Counter()

    # Find all strings of the form A# in the molecular formula where A
    # is the element (e.g. C) and # is the number of atoms of that
    # element in the molecule. Pair is of form [A, #]
    for pair in re.findall('([A-Z][a-z]*)(\d*)', AllChem.CalcMolFormula(mol)):
        # Add # to atom count, unless there is no # (in which case
        # there is just one of that element, as ones are implicit in
        # chemical formulas)
        if pair[1]:
            atoms[pair[0]] += int(pair[1])
        else:
            atoms[pair[0]] += 1

        # radical = any([atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()])
        # if radical:
        #     atoms["*"] += 1

    return atoms


def neutralize_mol(mol_dict):
    """
    Neutralize molecules to the best extent

    Parameters
    ----------
    mol_dict : dict
        {compound ID: Mol}

    Returns
    -------
    mol_dict : dict
        {compound ID: neutralized Mol}
    neutralize_flag : bool
        if one of the molecules is not completely neutralized
    """

    def _InitialiseNeutralisationReactions():
        """Substructure SMARTS to neutralize, adopted from Hans de Winter"""
        patts = (
            # Imidazoles
            ('[n+;H]', 'n'),
            # Amines
            ('[N+;!H0]', 'N'),
            # Carboxylic acids and oxygen anions
            ('[$([O-])]', 'O'),
            # Thiols
            ('[S-;X1]', 'S'),
            # Sulfonamides
            ('[$([N-;X2]S(=O)=O)]', 'N'),
            # Enamines
            ('[$([N-;X2][C,N]=C)]', 'N'),
            # Tetrazoles
            ('[n-]', '[nH]'),
            # Sulfoxides
            ('[$([S-]=O)]', 'S'),
            # Amides
            ('[$([N-]C=O)]', 'N'),
            # Phosphates
            ('[$([O-][P]=O)]', 'O'),
            # Carbanions
            ('[C;-]', 'C'),
            # Non hydrogen elements
            ('[!#1;D0]', '*')
        )
        return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]

    # check for hydrogens
    hydrogens = [h[1].GetAtomWithIdx(0).GetNumExplicitHs() for h in _InitialiseNeutralisationReactions()]
    neutralize_flag = True

    # for each mol in mol dict
    for mol_k, mol_v in mol_dict.items():

        # check for smarts pattern
        matches = []
        for (reactant, product) in _InitialiseNeutralisationReactions():  # for each pattern
            matches.append([m[0] for m in mol_v.GetSubstructMatches(reactant)])

        # try to neutralize each atom
        for m, h in zip(matches, hydrogens):
            if not m:
                continue

            for index in m:
                atom = mol_v.GetAtomWithIdx(index)
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(h)

        # update mol_dict
        mol_dict[mol_k] = Chem.MolFromSmiles(Chem.MolToSmiles(mol_v))

        # if mol not completely neutralized
        if Chem.rdmolops.GetFormalCharge(mol_v) != 0:
            neutralize_flag = False

    return mol_dict, neutralize_flag


def get_smarts(smarts_str):
    """
    Get SMARTS and seed SMARTS patterns

    Parameters
    ----------
    smarts_str : str
        SMARTS from rules in JN1224min

    Returns
    -------
    smarts_list : list
        SMARTS split by different components
    seedsmarts_list : list
        seed SMARTS split by different components for reactant MCS search in JN3604imt
    """

    # lhs smarts pattern
    lhs_smarts = smarts_str.split('>>')[0]
    lhs_smarts = re.sub(r':[0-9]+]', ']', lhs_smarts)

    # identify each fragment
    smarts_list = []
    temp_fragment = []

    # append complete fragments only
    for fragment in lhs_smarts.split('.'):
        temp_fragment += [fragment]
        if '.'.join(temp_fragment).count('(') == '.'.join(temp_fragment).count(')'):
            smarts_list.append('.'.join(temp_fragment))
            temp_fragment = []

            # remove component grouping for substructure matching
            if '.' in smarts_list[-1]:
                smarts_list[-1] = smarts_list[-1].replace('(', '', 1)[::-1].replace(')', '', 1)[::-1]

    # lhs seedsmarts pattern
    lhs_seedsmarts = copy.deepcopy(lhs_smarts)
    for atom in set(re.findall(r'\[#[0-9]+\]', lhs_smarts)):
        temp_atom = '[' + str(atom).rstrip(']').split('#')[1] + str(atom).replace('#', '99#').lstrip('[')
        lhs_seedsmarts = lhs_seedsmarts.replace(atom, temp_atom)

    # identify each fragment
    seedsmarts_list = []
    temp_fragment = []

    # append complete fragments only
    for fragment in lhs_seedsmarts.split('.'):
        temp_fragment += [fragment]
        if '.'.join(temp_fragment).count('(') == '.'.join(temp_fragment).count(')'):
            seedsmarts_list.append('.'.join(temp_fragment))
            temp_fragment = []

            # add * for only one atom in reaction center
            if seedsmarts_list[-1].count('[') == 1:
                seedsmarts_list[-1] = '*' + seedsmarts_list[-1]

            # remove component grouping for substructure matching
            if '.' in seedsmarts_list[-1]:
                seedsmarts_list[-1] = seedsmarts_list[-1].replace('(', '', 1)[::-1].replace(')', '', 1)[::-1]

    return smarts_list, seedsmarts_list


class MapRules:
    """Map reactions with JN1224min or JN3604imt reaction rules"""

    def __init__(self, rules_path=None, metacyc_coreactants_path=None, cofactor_list_path=None, cofactor_pair_path=None,
                 cofactors=True):
        """
        Parameters
        ----------
        rules_path : str
            path to JN1224min or JN3604imt
        metacyc_coreactants_path : str
            path to metacyc_coreactant.tsv for pickaxe
        cofactor_list_path : str
            path to cofactor_list_alldb.tsv
        cofactor_pair_path : str
            path to cofactor_pair_alldb.json
        cofactors : bool
            default True, whether to include cofactor designations
        """

        self.rules = pd.read_csv(rules_path, sep='\t', index_col=0)
        if cofactors:
            self.cofactor_name_dict, self.cofactor_list_dict, self.cofactor_pair_dict = get_cofactors(cofactor_list_path, cofactor_pair_path)
            self.metacyc_coreactants = pd.read_csv(metacyc_coreactants_path, sep='\t', index_col=0)
        else:
            self.cofactor_name_dict = {}
            self.cofactor_list_dict = {}
            self.cofactor_pair_dict = {}

    def map_pickaxe_rules(self, lhs_dict, rhs_dict, rule_current, return_reaction_center=False):
        """
        Map a balanced reaction with reaction rules

        Parameters
        ----------
        lhs_dict, rhs_dict : dict
            {'cpdA:0': 'cpdA_SMILES', 'cpdB:0': 'cpdB_SMILES', 'cpdB:1': 'cpdB_SMILES', 'cofactor:0': 'cofactor_SMILES'}
        rule_current : str
            rule id in self.rules (for example 'rule0002')
        return_reaction_center : bool
            default False, if True returns the atom indices of reaction center if reaction is mapped

        Returns
        -------
        match_index
            if reaction mapped, return ([], []) (index of sorted reactant/product name at each rule component)
            if reaction unmapped, return (None, None)
            if cofactor incorrect, raise ValueError
            if return_reaction_center = True, returns the atom indices of reaction center if reaction is mapped
        """

        rxn_df = self.rules.loc[rule_current.split(';')[0]]
        rule = rxn_df['SMARTS']
        reactants = rxn_df['Reactants']
        products = rxn_df['Products']

        # remove cofactor, sanitize mols
        lhs_list, rhs_list = self._process_substrates(lhs_dict, rhs_dict, rule_current)

        # match index with pickaxe
        match_index = self._map_rules(rule, lhs_list, rhs_list, reactants, products, return_reaction_center)

        return match_index

    def _process_substrates(self, lhs_dict, rhs_dict, rule_current):
        """
        Process compounds and return noncofactor compounds only
        """

        # check cofactor designation
        rule_reactant_names = self.rules.loc[rule_current, 'Reactants']
        rule_product_names = self.rules.loc[rule_current, 'Products']

        reactant_names, product_names = \
            label_cofactor(sorted(lhs_dict), sorted(rhs_dict), self.cofactor_list_dict, self.cofactor_pair_dict)

        if sorted(rule_reactant_names.split(';')) != sorted(reactant_names.split(';')) \
                or sorted(rule_product_names.split(';')) != sorted(product_names.split(';')):
            raise ValueError('Cofactor designation error.')

        # create list from dict
        lhs_list = [lhs_dict[k] for i, k in enumerate(sorted([k for k in lhs_dict]))
                    if reactant_names.split(';')[i] == 'Any']
        rhs_list = [rhs_dict[k] for i, k in enumerate(sorted([k for k in rhs_dict]))
                    if product_names.split(';')[i] == 'Any']

        # sanitize
        for i, m in enumerate(lhs_list):
            try:
                temp_mol = Chem.MolFromSmiles(m)
                Chem.rdmolops.RemoveStereochemistry(temp_mol)
                lhs_list[i] = Chem.MolToSmiles(temp_mol)
            except:
                temp_mol = Chem.MolFromSmiles(m, sanitize=False)
                Chem.rdmolops.RemoveStereochemistry(temp_mol)
                lhs_list[i] = Chem.MolToSmiles(temp_mol)

        for i, m in enumerate(rhs_list):
            try:
                temp_mol = Chem.MolFromSmiles(m)
                Chem.rdmolops.RemoveStereochemistry(temp_mol)
                rhs_list[i] = Chem.MolToSmiles(temp_mol)
            except:
                temp_mol = Chem.MolFromSmiles(m, sanitize=False)
                Chem.rdmolops.RemoveStereochemistry(temp_mol)
                rhs_list[i] = Chem.MolToSmiles(temp_mol)

        return lhs_list, rhs_list

    def _map_rules(self, rule, lhs, rhs, reactants, products, return_reaction_center):
        """
        Map rules and return reaction center if specified
        """

        rxn = Chem.rdChemReactions.ReactionFromSmarts(rule)
        reactants = reactants.split(';')
        cofactor_index_reactants = [i for i, r in enumerate(reactants) if r != 'Any']

        products = products.split(';')
        cofactor_index_products = [i for i, p in enumerate(products) if p != 'Any']

        # if number of reactants does not match reactant template
        if len(lhs) > reactants.count('Any'):
            repetitive_mols = set(lhs).intersection(set(rhs))

            while repetitive_mols:
                lhs.remove(sorted(repetitive_mols)[0])
                rhs.remove(sorted(repetitive_mols)[0])
                repetitive_mols = set(lhs).intersection(set(rhs))

        lhs_set = set()
        for lhs_perm in itertools.permutations(lhs):
            lhs_set.add(lhs_perm)

        for lhs_perm in lhs_set:
            lhs_temp = list(lhs_perm)

            for c in cofactor_index_reactants:
                lhs_temp[c:c] = [self.metacyc_coreactants.loc[reactants[c], 'SMILES']]

            # pruned MetaCyc
            try:
                lhs_tuple = tuple([Chem.MolFromSmiles(i) for i in lhs_temp])
                outputs = rxn.RunReactants(lhs_tuple)
            except:
                try:
                    lhs_tuple = tuple([Chem.MolFromSmiles(i, sanitize=False) for i in lhs_temp])
                    outputs = rxn.RunReactants(lhs_tuple)
                except:
                    continue

            # # pickaxe
            # lhs_tuple_list = []
            # for i in lhs_temp:
            #     try:
            #         temp_mol = Chem.MolFromSmiles(i)
            #         temp_mol = AllChem.AddHs(temp_mol)
            #         AllChem.Kekulize(temp_mol, clearAromaticFlags=True)
            #     except:
            #         temp_mol = Chem.MolFromSmiles(i, sanitize=False)
            #     lhs_tuple_list.append(temp_mol)
            # lhs_tuple = tuple(lhs_tuple_list)
            # outputs = rxn.RunReactants(lhs_tuple)

            for rxn_output in outputs:

                rhs_run = [Chem.MolToSmiles(rhs_mols) for rhs_mols in rxn_output]
                rhs_list = copy.deepcopy(rhs_run)

                for c in cofactor_index_products:
                    rhs_list.remove(rhs_run[c])

                # for all tautomer possibilities of clean rhs
                for rhs in postsanitize_smiles(rhs):
                    rhs = list(rhs)

                    for rhs_list in postsanitize_smiles(rhs_list):

                        # pruned MetaCyc
                        if sorted(list(rhs_list)) == sorted(rhs):
                            # lhs_index = [int(np.where(np.argsort(lhs) == i)[0]) for i in np.argsort(lhs_perm)]
                            # rhs_index = [int(np.where(np.argsort(rhs) == i)[0]) for i in np.argsort(rhs_list)]
                            lhs_index = [lhs.index(i) for i in lhs_perm]
                            rhs_index = [rhs.index(i) for i in rhs_list]

                            # return atom index of reaction center
                            if return_reaction_center:

                                # try to append lhs reactants
                                lhs_mols = []
                                for l in lhs_perm:
                                    lhs_mols.append(Chem.MolFromSmiles(l))
                                    if not lhs_mols[-1]:
                                        lhs_mols[-1] = Chem.MolFromSmiles(l, sanitize=False)

                                smarts_list, _ = get_smarts(rule)
                                smarts_list = [s for i, s in enumerate(smarts_list)
                                               if i not in cofactor_index_reactants]

                                # possible reaction center
                                temp_lhs_match = [Chem.MolFromSmiles(l, sanitize=False).GetSubstructMatches(
                                    Chem.MolFromSmarts(smarts_list[i])) for i, l in enumerate(lhs_perm)]
                                reaction_center_set = [set(itertools.chain(*l)) for l in temp_lhs_match]
                                lhs_all_matches = itertools.product(*temp_lhs_match)

                                # for all possible reaction centers
                                for lhs_match in lhs_all_matches:

                                    # iterate over all reactants
                                    for l_idx, match in enumerate(lhs_match):
                                        for protect in reaction_center_set[l_idx] - set(match):
                                            lhs_mols[l_idx].GetAtomWithIdx(protect).SetProp('_protected', '1')

                                    # add cofactors
                                    lhs_temp_mol = list(lhs_mols)
                                    for c in cofactor_index_reactants:
                                        lhs_temp_mol[c:c] = [Chem.MolFromSmiles(
                                            self.metacyc_coreactants.loc[reactants[c], 'SMILES'])]

                                    # for all possible reaction outcomes
                                    for rhs_rxn in rxn.RunReactants(tuple(lhs_temp_mol)):

                                        for rhs_smiles in postsanitize_smiles([Chem.MolToSmiles(r) for r in rhs_rxn]):

                                            # found match
                                            if tuple(r for i, r in enumerate(rhs_smiles)
                                                     if i not in cofactor_index_products) == rhs_list:
                                                return lhs_index, rhs_index, list(lhs_match)

                                    # else remove protection
                                    for l_idx, match in enumerate(lhs_match):
                                        for deprotect in reaction_center_set[l_idx] - set(match):
                                            lhs_mols[l_idx].GetAtomWithIdx(deprotect).ClearProp('_protected')

                            else:
                                return lhs_index, rhs_index

                        # # pickaxe
                        # if sorted(list(rhs_list)) == sorted(rhs):
                        #     lhs_index = [int(np.where(np.argsort(lhs) == i)[0]) for i in np.argsort(lhs_perm)]
                        #     rhs_index = [int(np.where(np.argsort(rhs) == i)[0]) for i in np.argsort(rhs_list)]
                        #     return lhs_index, rhs_index

        return None, None


if __name__ == '__main__':

    # postsanitize_smiles
    print('Sanitize & tautomerize SMILES for a list of compounds containing imidazole')
    print(postsanitize_smiles(['CCO', 'NC(Cc1c[nH]cn1)C(=O)O']))
    print()

    # get_cofactors
    input_cofactor_list_path = '../JN1224MIN/cofactor/cofactor_list_alldb.tsv'
    input_cofactor_pair_path = '../JN1224MIN/cofactor/cofactor_pair_alldb.json'
    output_cofactor_name_dict, output_cofactor_list_dict, output_cofactor_pair_dict\
        = get_cofactors(input_cofactor_list_path, input_cofactor_pair_path)
    print('Total number of cofactor designations in JN1224min: %s' % str(len(output_cofactor_name_dict)))
    print('Total number of database compounds recognized as cofactors: %s'
          % str(len(output_cofactor_list_dict)))
    print('Total number of database compound pairs recognized as cofactor pairs: %s'
          % str(len(output_cofactor_pair_dict)))
    print()

    # label_cofactor
    print('Cofactor designation of reactions using compound IDs')
    # 1.1.1.308 using MetaCyc compound IDs
    print(label_cofactor(['CPD-12692:0', 'NAD:0', 'WATER:0', 'NAD:1'], ['CPD-367:0', 'NADH:0', 'NADH:1'],
                         output_cofactor_list_dict, output_cofactor_pair_dict))
    # 4.1.1.3 using KEGG compound IDs
    print(label_cofactor(['cpd00032'], ['cpd00020', 'cpd00011'],
                         output_cofactor_list_dict, output_cofactor_pair_dict))
    # 1.1.1.1 using BRENDA compound IDs
    print(label_cofactor(['NAD+', 'ethanol'], ['acetaldehyde', 'NADH'],
                         output_cofactor_list_dict, output_cofactor_pair_dict))
    print()

    # get_atom_count
    print('Atom count for ethanol')
    print(get_atom_count(Chem.MolFromSmiles('CCO')))
    print(get_atom_count('CCO'))
    print()

    # neutralize_mol
    print('Neutralize molecules to the best extent')
    input_neutralize_mol_dict = {'cpd00029': Chem.MolFromSmiles('CC(=O)[O-]'),
                                 'cpd00202': Chem.MolFromSmiles('CC(C)=CCOP(=O)([O-])OP(=O)([O-])O'),
                                 'cpd00266': Chem.MolFromSmiles('C[N+](C)(C)CC(O)CC(=O)[O-]')}
    output_neutralize_mol_dict = neutralize_mol(input_neutralize_mol_dict)[0]
    print({k: Chem.MolToSmiles(v) for k, v in output_neutralize_mol_dict.items()})
    print()

    # get_smarts
    print('Split SMARTS strings for reaction rules by component')
    print(get_smarts('[#6:1]-[#8:2].([#6:3].[#6:4]-[#6:5]).[#6:6]-[#6:7]-[#8:8]'))
    print()

    # MapRules
    input_MR = MapRules(rules_path='../JN1224MIN/JN1224MIN_rules.tsv',
                        metacyc_coreactants_path='../JN1224MIN/cofactor/metacyc_coreactants.tsv',
                        cofactor_list_path=input_cofactor_list_path, cofactor_pair_path=input_cofactor_pair_path)
    rxn_dict = {
        '1.1.1.6':
            [{'NAD+:0': 'NC(=O)c1ccc[n+](C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(O)C3O)C(O)C2O)c1',
              'glycerol:0': 'OCC(O)CO'},
             {'NADH:0': 'NC(=O)C1=CN(C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(O)C3O)C(O)C2O)C=CC1',
              'dihydroxyacetone:0': 'O=C(CO)CO'}],
        '3.2.2.3':
            [{'cpd00249': 'O=c1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1', 'cpd00001': 'O'},
             {'cpd00092': 'O=c1cc[nH]c(=O)[nH]1', 'cpd00105': 'OC[C@H]1OC(O)[C@H](O)[C@@H]1O'}],
        'MetaCyc_RXN-12395':
            [{"CPD1F-114:0": "CC(C)=CCC/C(C)=C/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C=C(\\C)CCC=C(C)C",
              "OXYGEN-MOLECULE:0": "O=O", "OXYGEN-MOLECULE:1": "O=O"},
             {"CPD-13371:0": "CC(=O)/C=C/C=C(\\C)CCC=C(C)C",
              "CPD-7207:0": "C/C(C=O)=C\\C=C\\C(C)=C\\C=C\\C=C(C)\\C=C\\C=O", "GERANIAL:0": "CC(C)=CCC/C(C)=C/C=O"}]
    }

    # 1.1.1.6
    # mapped
    print('Map 1.1.1.6 with the correct reaction rule')
    print('%s %s' %
          ('rule0002', input_MR.map_pickaxe_rules(rxn_dict['1.1.1.6'][0], rxn_dict['1.1.1.6'][1], 'rule0002')))
    # unmapped
    print('%s %s' %
          ('rule0018', input_MR.map_pickaxe_rules(rxn_dict['1.1.1.6'][0], rxn_dict['1.1.1.6'][1], 'rule0018')))
    # cofactor designation error
    try:
        input_MR.map_pickaxe_rules(rxn_dict['1.1.1.6'][0], rxn_dict['1.1.1.6'][1], 'rule0024')
    except Exception as e:
        print('%s %s' % ('rule0024', str(e)))
    print()

    # 3.2.2.3
    # mapped
    print('Map 3.2.2.3 with the correct reaction rule')
    print('%s %s' %
          ('rule0013', input_MR.map_pickaxe_rules(rxn_dict['3.2.2.3'][0], rxn_dict['3.2.2.3'][1], 'rule0013')))
    # unmapped
    print('%s %s' %
          ('rule0085', input_MR.map_pickaxe_rules(rxn_dict['3.2.2.3'][0], rxn_dict['3.2.2.3'][1], 'rule0085')))
    # cofactor designation error
    try:
        input_MR.map_pickaxe_rules(rxn_dict['3.2.2.3'][0], rxn_dict['3.2.2.3'][1], 'rule0002')
    except Exception as e:
        print('%s %s' % ('rule0002', str(e)))
    print()

    # MetaCyc RXN-12395
    print('MetaCyc RXN-12395')
    print('Non-cofactor reactants: %s, products: %s' %
          input_MR._process_substrates(rxn_dict['MetaCyc_RXN-12395'][0], rxn_dict['MetaCyc_RXN-12395'][1], 'rule0265'))
    output_RXN12395_lhs_cofactor, output_RXN12395_rhs_cofactor =\
        label_cofactor(sorted(rxn_dict['MetaCyc_RXN-12395'][0]), sorted(rxn_dict['MetaCyc_RXN-12395'][1]),
                       output_cofactor_list_dict, output_cofactor_pair_dict)
    print('Designate cofactors reactants: %s, products: %s'
          % (output_RXN12395_lhs_cofactor, output_RXN12395_rhs_cofactor))
    input_RXN12395_lhs_cpd = [r for i_r, r in enumerate(sorted(rxn_dict['MetaCyc_RXN-12395'][0])) if
                              output_RXN12395_lhs_cofactor.split(';')[i_r] == 'Any']
    input_RXN12395_rhs_cpd = [r for i_r, r in enumerate(sorted(rxn_dict['MetaCyc_RXN-12395'][1])) if
                              output_RXN12395_rhs_cofactor.split(';')[i_r] == 'Any']
    print('Sorted non-cofactor reactants: %s, products: %s' % (input_RXN12395_lhs_cpd, input_RXN12395_rhs_cpd))
    print('Mapped MetaCyc RXN-12395 with %s, index of sorted reactant & product names at each rule component %s' %
          ('rule0265', input_MR.map_pickaxe_rules(
              rxn_dict['MetaCyc_RXN-12395'][0], rxn_dict['MetaCyc_RXN-12395'][1], 'rule0265')))
