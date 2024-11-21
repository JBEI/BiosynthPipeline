### pickaxe-generic stuff
import pandas as pd
import pickaxe_generic as pg
from rdkit import Chem
from pickaxe_generic.filters import (
    ChainFilter,
    CoreactantUIDPreFilter,
    LessThanNElementTypeFilter,
    TanimotoSimilarityFilter
)

from pickaxe_generic.interfaces import MolDatRDKit, ReactionFilter
import typing
import dataclasses
from pickaxe_generic import metadata, interfaces
import collections.abc
import time
from datetime import datetime
import os
from Reaction_Smarts_List2 import op_smarts
import numpy as np

from biosynth_pipeline import biosynth_pipeline
import pickle

### User-defined parameters
pathway_sequence = ['pks', 'non_pks']  # run retrotide first then pickaxe
target_smiles = 'C=CC(=O)O'
target_name = 'Acrylic_acid'
pks_release_mechanism = 'thiolysis' # choose from 'macrolactonization' or 'thiolysis'
pks_starters_filepath = '../biosynth_pipeline/retrotide/data/starters.smi'
pks_extenders_filepath = '../biosynth_pipeline/retrotide/data/extenders.smi'

# For each of these, set as 'all', or specify list like ['mal', 'mmal']
pks_starters = ['mal','mmal','hmal','mxmal','allylmal']
pks_extenders = ['mal','mmal','hmal','mxmal','allylmal']

# For each of these, choose from 'atompairs', 'mcs_with_stereo', 'mcs_without_stereo', 'atomatompath'
pks_similarity_metric = 'mcs_without_stereo'
non_pks_similarity_metric = 'mcs_without_stereo'

known_metabolites_filepath = '../data/all_known_metabolites.txt'
non_pks_cofactors_filepath = '../data/coreactants_and_rules/all_cofactors.tsv'
input_cpd_dir = '../data/coreactants_and_rules/'
feasibility_cofactors = '../data/coreactants_and_rules/all_cofactors_updated.csv'
consider_target_stereo = True
non_pks_rules = 'biological_intermediate'
non_pks_steps = 1
non_pks_cores = 1
non_pks_sim_score_filter = False
non_pks_sim_score_cutoffs = []
non_pks_sim_sample = False
non_pks_sim_sample_size = None

stopping_criteria = 'first_product_formation'

### Create an object that is an instance of the feasibility classification model
PX = biosynth_pipeline.feasibility_classifier(ML_model_type='add_concat',
                                              cofactors_path=feasibility_cofactors)

### Create an object that is an instance of Biosynth Pipeline
biosynth_pipeline_object = biosynth_pipeline.biosynth_pipeline(
                                             pathway_sequence = pathway_sequence,
                                             target_smiles = target_smiles,
                                             feasibility_classifier=PX,
                                             pks_release_mechanism = pks_release_mechanism,
                                             pks_starters_filepath = pks_starters_filepath,
                                             pks_extenders_filepath = pks_extenders_filepath,
                                             pks_starters = pks_starters,
                                             pks_extenders = pks_extenders,
                                             pks_similarity_metric = pks_similarity_metric,
                                             non_pks_similarity_metric = non_pks_similarity_metric,
                                             consider_target_stereo = consider_target_stereo,
                                             known_metabolites = known_metabolites_filepath,
                                             non_pks_cofactors = non_pks_cofactors_filepath,
                                             input_cpd_dir = input_cpd_dir,
                                             non_pks_rules = non_pks_rules,
                                             non_pks_steps = non_pks_steps,
                                             non_pks_cores = non_pks_cores,
                                             non_pks_sim_score_filter = non_pks_sim_score_filter,
                                             non_pks_sim_score_cutoffs = non_pks_sim_score_cutoffs,
                                             non_pks_sim_sample = non_pks_sim_sample,
                                             non_pks_sim_sample_size = non_pks_sim_sample_size,
                                             stopping_criteria = stopping_criteria)

### ------ Chem ------

incorrect_reactions_soi = []
pgthermo_none_list = set()
@typing.final
@dataclasses.dataclass(frozen=True)  # , slots=True)
class EnthalpyCalculator(metadata.MolPropertyCalc[float]):  # Calculate Hf for molecules
    Enthalpy_key: collections.abc.Hashable

    @property
    def key(self) -> collections.abc.Hashable:
        return self.Enthalpy_key

    @property
    def meta_required(self) -> interfaces.MetaKeyPacket:
        return interfaces.MetaKeyPacket(molecule_keys={self.Enthalpy_key})

    @property
    def resolver(self) -> metadata.MetaDataResolverFunc[float]:
        return metadata.TrivialMetaDataResolverFunc

    def __call__(
            self,
            data: interfaces.DataPacketE[interfaces.MolDatBase],
            prev_value: typing.Optional[float] = None,
    ) -> typing.Optional[float]:
        if prev_value is not None:
            return prev_value
        item = data.item
        if data.meta is not None and self.Enthalpy_key in data.meta:
            return None
        if not isinstance(item, interfaces.MolDatRDKit):
            raise NotImplementedError(
                f"EnthalpyCalculator has not been implemented for molecule type {type(item)}"
            )

        _enthalpy_f = Hf(item.uid)
        if _enthalpy_f is None:
            pgthermo_none_list.add(item.uid)

            # print("None Enthalpy returned by molecule:", item.uid)  # tell user which molecule is causing problems
            return float('nan')
        return _enthalpy_f

@typing.final
@dataclasses.dataclass(frozen=True)
class MaxEnthalpyFilter(metadata.ReactionFilterBase):
    __slots__ = ("max_H", "H_key")
    max_H: float
    H_key: collections.abc.Hashable

    def __call__(self, recipe: interfaces.ReactionExplicit) -> bool:
        dH = 0.0
        for idx, mol in enumerate(recipe.products):
            if mol.meta[self.H_key] == float('nan'):
                return False
            dH = dH + mol.meta[self.H_key] * recipe.operator.meta['products_stoi'][idx]
        for idx, mol in enumerate(recipe.reactants):
            if mol.meta[self.H_key] == float('nan'):
                return False
            dH = dH - mol.meta[self.H_key] * recipe.operator.meta['reactants_stoi'][idx]
        if recipe.operator.meta['enthalpy_correction'] is not None:
            dH = dH + recipe.operator.meta['enthalpy_correction']
        if dH / recipe.operator.meta['number_of_steps'] < self.max_H:
            return True
        return False

    @property
    def meta_required(self) -> interfaces.MetaKeyPacket:
        return interfaces.MetaKeyPacket(molecule_keys={self.H_key},
                                        operator_keys={'reactants_stoi', "products_stoi", "enthalpy_correction",
                                                       "number_of_steps"})

@typing.final
@dataclasses.dataclass(frozen=True)
class Carbon_Chain_Reduce_Filter(metadata.ReactionFilterBase):
    #    __slots__ = ("max_H", "H_key")
    #    max_H: float
    #    H_key: float

    def __call__(self, recipe: interfaces.ReactionExplicit) -> bool:
        rea_max_carbon_num = 0
        pro_max_carbon_num = 0

        for mol in recipe.products:
            pro_max_carbon_num = max(pro_max_carbon_num,
                                     len(mol.item.rdkitmol.GetSubstructMatches(Chem.MolFromSmarts("[#6]"))))
        for mol in recipe.reactants:
            rea_max_carbon_num = max(rea_max_carbon_num,
                                     len(mol.item.rdkitmol.GetSubstructMatches(Chem.MolFromSmarts("[#6]"))))
        if rea_max_carbon_num >= pro_max_carbon_num:
            return True
        return False

@typing.final
@dataclasses.dataclass(frozen=True)
class exclude_structure_filter(metadata.ReactionFilterBase):
    __slots__ = ("exclude_smarts")
    exclude_smarts: str

    def __call__(self, recipe: interfaces.ReactionExplicit) -> bool:
        structure = Chem.MolFromSmarts(self.exclude_smarts)
        for mol in recipe.products:
            if mol.item.rdkitmol.HasSubstructMatch(structure):
                return False
        return True

@typing.final
@dataclasses.dataclass(frozen=True)
class Ring_Issues_Filter(metadata.ReactionFilterBase):
    def __call__(self, recipe: interfaces.ReactionExplicit) -> bool:
        rea_carbon_num = 0
        pro_carbon_num = 0
        rea_H_num = 0
        pro_H_num = 0

        if recipe.operator.meta['ring_issue'] is True and recipe.operator.meta[
            'enthalpy_correction'] is None:  # check balance
            for idx, mol in enumerate(recipe.reactants):
                rea_carbon_num += len(mol.item.rdkitmol.GetSubstructMatches(Chem.MolFromSmarts("[#6]"))) * \
                                  recipe.operator.meta['reactants_stoi'][idx]
                mol_H = 0
                for atom in mol.item.rdkitmol.GetAtoms():
                    mol_H += atom.GetTotalNumHs()
                rea_H_num += mol_H * recipe.operator.meta['reactants_stoi'][idx]
                if mol.item.uid == "[H][H]":
                    rea_H_num += 2 * recipe.operator.meta['reactants_stoi'][idx]

            for idx, mol in enumerate(recipe.products):

                if len(Chem.GetMolFrags(
                        mol.item.rdkitmol)) != 1:  # if there're fragments in a mol, indicates invalid rxn
                    return False

                pro_carbon_num += len(mol.item.rdkitmol.GetSubstructMatches(Chem.MolFromSmarts("[#6]"))) * \
                                  recipe.operator.meta['products_stoi'][idx]
                mol_H = 0
                for atom in mol.item.rdkitmol.GetAtoms():
                    mol_H += atom.GetTotalNumHs()
                pro_H_num += mol_H * recipe.operator.meta['products_stoi'][idx]
                if mol.item.uid == "[H][H]":
                    pro_H_num += 2 * recipe.operator.meta['products_stoi'][idx]

            if rea_carbon_num != pro_carbon_num or rea_H_num != pro_H_num:
                return False
        return True

    @property
    def meta_required(self) -> interfaces.MetaKeyPacket:
        return interfaces.MetaKeyPacket(
            operator_keys={'reactants_stoi', "products_stoi", "ring_issue", "enthalpy_correction"})

@typing.final
@dataclasses.dataclass(frozen=True)
class Retro_Not_Aromatic_Filter(metadata.ReactionFilterBase):
    def __call__(self, recipe: interfaces.ReactionExplicit) -> bool:
        rea_aro_ring_num = 0
        pro_aro_ring_num = 0

        if recipe.operator.meta['Retro_Not_Aromatic'] is True:
            for idx, mol in enumerate(recipe.reactants):
                rea_aro_ring_num += Chem.rdMolDescriptors.CalcNumAromaticRings(mol.item.rdkitmol) * \
                                    recipe.operator.meta['reactants_stoi'][idx]

            for idx, mol in enumerate(recipe.products):
                pro_aro_ring_num += Chem.rdMolDescriptors.CalcNumAromaticRings(mol.item.rdkitmol) * \
                                    recipe.operator.meta['products_stoi'][idx]

            if rea_aro_ring_num < pro_aro_ring_num:
                return False
        return True

    @property
    def meta_required(self) -> interfaces.MetaKeyPacket:
        return interfaces.MetaKeyPacket(operator_keys={'reactants_stoi', "products_stoi", "Retro_Not_Aromatic"})

@typing.final
@dataclasses.dataclass(frozen=True)
class Check_balance_filter(metadata.ReactionFilterBase):
    def __call__(self, recipe: interfaces.ReactionExplicit) -> bool:
        rea_carbon_num = 0
        pro_carbon_num = 0
        rea_H_num = 0
        pro_H_num = 0

        if recipe.operator.meta['enthalpy_correction'] is None and recipe.operator.meta['ring_issue'] is False:
            for idx, mol in enumerate(recipe.reactants):
                rea_carbon_num += len(mol.item.rdkitmol.GetSubstructMatches(Chem.MolFromSmarts("[#6]"))) * \
                                  recipe.operator.meta['reactants_stoi'][idx]
                mol_H = 0
                for atom in mol.item.rdkitmol.GetAtoms():
                    mol_H += atom.GetTotalNumHs()
                rea_H_num += mol_H * recipe.operator.meta['reactants_stoi'][idx]
                if mol.item.uid == "[H][H]":
                    rea_H_num += 2 * recipe.operator.meta['reactants_stoi'][idx]

            for idx, mol in enumerate(recipe.products):
                pro_carbon_num += len(mol.item.rdkitmol.GetSubstructMatches(Chem.MolFromSmarts("[#6]"))) * \
                                  recipe.operator.meta['products_stoi'][idx]
                mol_H = 0
                for atom in mol.item.rdkitmol.GetAtoms():
                    mol_H += atom.GetTotalNumHs()
                pro_H_num += mol_H * recipe.operator.meta['products_stoi'][idx]
                if mol.item.uid == "[H][H]":
                    pro_H_num += 2 * recipe.operator.meta['products_stoi'][idx]

            if rea_carbon_num != pro_carbon_num or rea_H_num != pro_H_num:

                if recipe.operator.meta['name'] not in incorrect_reactions_soi:
                    incorrect_reactions_soi.append(recipe.operator.meta['name'])
                    incorrect_reactions_soi.append(recipe.reactants)
                    incorrect_reactions_soi.append(recipe.products)
                return False
        return True

    @property
    def meta_required(self) -> interfaces.MetaKeyPacket:
        return interfaces.MetaKeyPacket(
            operator_keys={'reactants_stoi', "products_stoi", "ring_issue", "enthalpy_correction", "name"})

    ###############################################################################

def run_chemistry(my_start):
    mol_smiles = (  # FDCA C1=C(OC(=C1)C(=O)O)C(=O)O
        "O",
        "O=O",
        "[H][H]",
        "O=C=O",
        "C=O",
        "[C-]#[O+]",
        "Br",
        "[Br][Br]",
        "CO",
        "[C-]#[O+]",
        "C=C",
        "O=S(O)O",
        "N",
        "O=S(=O)(O)O",
        "O=NO",
        "N#N",
        "O=[N+]([O-])O",
        "NO",
        "C#N",
        "S",
        "O=S=O")

    engine = pg.create_engine()
    network = engine.new_network()

    # get library objects from engine; must be in this order to have correct initializers
    mol_lib, op_lib, rxn_lib = engine.Libs()

    for smiles in mol_smiles:
        network.add_mol(engine.mol.rdkit(smiles), )
    my_start_i = network.add_mol(engine.mol.rdkit(my_start))
    for smarts in op_smarts:
        if smarts.kekulize_flag is False:
            network.add_op(engine.op.rdkit(smarts.smarts),
                           meta={"name": smarts.name,
                                 "reactants_stoi": smarts.reactants_stoi,
                                 "products_stoi": smarts.products_stoi,
                                 "enthalpy_correction": smarts.enthalpy_correction,
                                 "ring_issue": smarts.ring_issue,
                                 "kekulize_flag": smarts.kekulize_flag,
                                 "Retro_Not_Aromatic": smarts.Retro_Not_Aromatic,
                                 "number_of_steps": smarts.number_of_steps,
                                 }
                           )
        if smarts.kekulize_flag is True:
            network.add_op(engine.op.rdkit(smarts.smarts, kekulize=True, ),
                           meta={"name": smarts.name,
                                 "reactants_stoi": smarts.reactants_stoi,
                                 "products_stoi": smarts.products_stoi,
                                 "enthalpy_correction": smarts.enthalpy_correction,
                                 "ring_issue": smarts.ring_issue,
                                 "kekulize_flag": smarts.kekulize_flag,
                                 "Retro_Not_Aromatic": smarts.Retro_Not_Aromatic,
                                 "number_of_steps": smarts.number_of_steps,
                                 }
                           )

    strat = engine.strat.cartesian(network)
    max_atoms_filter = engine.filter.reaction.max_atoms(max_atoms=6, proton_number=6)
    coreactants_filter = engine.filter.bundle.coreactants(tuple(range(my_start_i)))
    reaction_plan = max_atoms_filter >> Ring_Issues_Filter() >> Check_balance_filter()
    ini_number = len(network.mols)

    strat.expand(num_iter=my_num_gens,
                 reaction_plan=reaction_plan,
                 bundle_filter=coreactants_filter,
                 save_unreactive=False)

    end_number = 0
    tt_number = 0
    target = engine.mol.rdkit(my_goal)
    close_match_list = []
    similarity_scores_list = []
    smiles_list = []

    for mol in network.mols:
        tt_number += 1
        print(mol.uid)
        sim_score = biosynth_pipeline_object.calculate_similarity(smiles1 = mol.uid,
                                                                  smiles2 = target_smiles,
                                                                  metric = non_pks_similarity_metric)
        similarity_scores_list.append(sim_score)
        smiles_list.append(mol.uid)
        # print("reactive:", network.reactivity[network.mols.i(mol.uid)])
        if network.reactivity[network.mols.i(mol.uid)] is True:
            end_number += 1
            if mol.uid == target.uid:
                print("target found")

    print("number of generations:", my_num_gens)
    print("number of operators:", len(op_smarts))
    print("number of molecules before expansion:", ini_number)
    print("number of molecules after expansion:", end_number)
    print("total number of molecules after expansion:", tt_number)

    return similarity_scores_list, smiles_list

if __name__ == '__main__':
    biosynth_pipeline_object.run_pks_synthesis()

    results = []
    for i in range(0,7):
        current_pks_product = biosynth_pipeline_object.run_pks_termination(pks_design_num = i,
                                                                           pks_release_mechanism = 'thiolysis')
        my_start = (current_pks_product)
        my_start = biosynth_pipeline_object._remove_stereo(my_start)
        my_goal = target_smiles
        print(f'My start: {my_start}')
        my_num_gens = 2
        # run pickaxe generic
        similarity_scores_list, smiles_list = run_chemistry(my_start)
        print(f"Max similarity of product: {max(similarity_scores_list)}")

        # print(pd.DataFrame({'Similarity':similarity_scores_list,'smiles':smiles_list})
        #       .sort_values(by='Similarity',ascending=False).head(10))



        entry = {'pks_design_num': i,
                 'pks_product_similarity': biosynth_pipeline_object.calculate_similarity(smiles1 = current_pks_product,
                                                                                         smiles2 = target_smiles,
                                                                                         metric = non_pks_similarity_metric),
                 'non_pks_product_similarity': max(similarity_scores_list)}


        results.append(entry)

        if max(similarity_scores_list) > 0.999:
            break

    with open(f'../data/hybrid_pathways_analysis/{target_name}_pks_plus_chem.pkl', 'wb') as f:
        pickle.dump(results, f)

