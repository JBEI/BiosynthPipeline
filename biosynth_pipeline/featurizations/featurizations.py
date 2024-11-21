"""
Author: Yash Chainani
Date last updated: April 24, 2023

This is a custom-designed package built atop of several rdkit functionalities
The goal of this package is to provide various fingerprinting methods to featurize both compounds and reactions
Compounds can be featurized by directly converting their SMILES into ecfp4, atom-pair, modred, MAP4, and Min-Hashed fingerprints
Reactions can be featurized by concatenating the fingerprints of constituent compounds together into a single vector
"""

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from mordred import Calculator, descriptors
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import Tuple

# Silence non-critical RDKit warnings to minimize unnecessary outputs
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

class compound:
    """
    Compound class for various fingerprinting methods for a single compound
    """

    def __init__(self, smiles: str):
        self.smiles = smiles

    def _canonicalize_smiles(self) -> str:
        """
        Canonicalize smiles string
        :return: Canonicalized smiles or original smiles if canonicalization fails
        """
        uncanon_smi = self.smiles  # assume original smiles not canonical

        try:
            canon_smi = Chem.MolToSmiles(
                Chem.MolFromSmiles(uncanon_smi)
            )  # try to canonicalize
        except:
            canon_smi = uncanon_smi  # return original if canonicalization failed

        return canon_smi

    def remove_stereo(self) -> str:
        """
        Removes stereochemistry if any is present after first canonicalizing smiles string
        :return: Smiles string without stereochemistry information
        """

        # canonicalize input smiles first then assume original smiles have stereochemistry
        smiles_w_stereo = self._canonicalize_smiles()

        try:
            mol = Chem.MolFromSmiles(smiles_w_stereo)
            Chem.RemoveStereochemistry(mol)
            smiles_wo_stereo = Chem.MolToSmiles(mol)
        except:
            smiles_wo_stereo = smiles_w_stereo

        return smiles_wo_stereo

    def _smiles_2_morganfp(self, radius: int, nBits: int) -> np.ndarray:
        """
        Generate Morgan fingerprints for a compound after canonicalizing smiles and removing stereochemistry
        :param radius: Radius for fragmentation
        :param nBits: Output dimensions of fingerprint
        :return: Morgan/ ecfp4 fingerprint of specified radius and dimension
        """
        try:
            canon_smi_wo_stereo = self.remove_stereo()
            mol = Chem.MolFromSmiles(canon_smi_wo_stereo)
        except Exception as E:
            return None

        if mol:
            fp = Chem.AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=radius, nBits=nBits
            )
            return np.array(fp) # srun

        return None

    def _smiles_2_MACCS(self) -> np.ndarray:
        """
        Generate MACCS keys of a compound after canonicalizing smiles and removing stereochemistry
        Some code is inspired from the following link:
        https://chem.libretexts.org/Courses/Intercollegiate_Courses/Cheminformatics/06%3A_Molecular_Similarity/6.04%3A_Python_Assignment
        :return: MACCS fingerprint
        """
        try:
            canon_smi_wo_stereo = self.remove_stereo()
            mol = Chem.MolFromSmiles(canon_smi_wo_stereo)
        except Exception as E:
            return None

        if mol:
            fp = MACCSkeys.GenMACCSKeys(mol)
            return np.array(fp)

        return None

    def _smiles_2_atompair(self) -> np.ndarray:
        """
        Generate atom pair fingerprints of a compound after canonicalizing smiles and removing stereochemistry
        :return: atom pair fingerprint
        """
        try:
            canon_smi_wo_stereo = self.remove_stereo()
            mol = Chem.MolFromSmiles(canon_smi_wo_stereo)
        except Exception as E:
            return None

        if mol:
            fp = Pairs.GetAtomPairFingerprintAsBitVect(mol)
            return np.array(fp)

        return None

    def _smiles_2_modred(self) -> np.ndarray:
        """
        Generate modred fingerprints and replace None/ Nan values with zero
        :return:
        """
        try:
            canon_smi_wo_stereo = self.remove_stereo()
            mol = Chem.MolFromSmiles(canon_smi_wo_stereo)
        except Exception as E:
            return None

        if mol:
            calc = Calculator(descriptors, ignore_3D=True)
            fp = calc(mol)
            fp = np.array(fp).reshape(-1, 1)
            fp_no_nan = np.nan_to_num(
                fp, copy=True
            )  # converts any nan values in modred descriptor to zero

            # Scale the vector since descriptors can vary significantly in magnitudes
            scaler = StandardScaler()
            scaler.fit(fp_no_nan)
            fp_scaled = scaler.transform(fp_no_nan)

            return fp_scaled.reshape(len(fp_scaled))

        return None

    # def _smiles_2_MAP4(self, is_folded: bool, dim: int) -> np.ndarray:
    #     """
    #     Generate MAP4 fingerprints for a compound after canonicalizing smiles and removing stereochemistry
    #     :param is_folded: False as default (resolve TMAP attribute errors) but set to True for fixed dimensions
    #     :param dim: Output dimensions of fingerprint
    #     :return: MAP4 fingerprint
    #     """
    #     try:
    #         canon_smi_wo_stereo = self.remove_stereo()
    #         mol = Chem.MolFromSmiles(canon_smi_wo_stereo)
    #     except Exception as E:
    #         return None
    #
    #     if mol:
    #         MAP4_folded = MAP4Calculator(dimensions=dim, is_folded=is_folded)
    #         fp = MAP4_folded.calculate(mol)
    #         return np.array(fp)
    #
    #     return None

    # def _smiles_2_MHFP(self) -> np.ndarray:
    #     mhfp_encoder = MHFPEncoder()
    #
    #     try:
    #         canon_smi_wo_stereo = self.remove_stereo()
    #         mol = Chem.MolFromSmiles(canon_smi_wo_stereo)
    #     except Exception as E:
    #         return None
    #
    #     if mol:
    #         fp = mhfp_encoder.encode(canon_smi_wo_stereo)
    #         return np.array(fp)
    #
    #     return None

    #TODO - test this function
    def smiles_2_fp(self, fp_type, radius: int = None, nBits:int = None, is_folded: bool = None, dim: int = None) -> np.ndarray:

        if fp_type == "morgan" or fp_type == "ecfp4":
            reactant_fp = self._smiles_2_morganfp(radius=2, nBits=2048)
            return reactant_fp

        if fp_type == "MACCS":
            reactant_fp = self._smiles_2_MACCS()
            return reactant_fp

        if fp_type == "atom_pair":
            reactant_fp = self._smiles_2_atompair()
            return reactant_fp

        if fp_type == "modred":
            reactant_fp = self._smiles_2_modred()
            return reactant_fp

        # if fp_type == "MAP4":
        #     reactant_fp = self._smiles_2_MAP4(is_folded=True, dim=2048)
        #     return reactant_fp
        #
        # if fp_type == "MHFP6":
        #     reactant_fp = self._smiles_2_MHFP()
        #     return reactant_fp

        else:
            raise ValueError("Please enter a valid fingerprinting method, such as: morgan/ecfp4, MACCS, atom_pair, modred, MAP4, or MHFP6")

class reaction:
    """
    Reaction class for various fingerprinting methods for a full reaction
    """
    def __init__(self, rxn_str: str):
        self.rxn_str = rxn_str

    def _rxn_2_cpds(self) -> Tuple[str, str]:
        """
        Parse a reaction string to return two lists: a reactants list and a products list
        :return reactants_str: string of reactants on the LHS of the reaction (index 0)
        :return products_str: string of products on the RHS of the reaction (index 1)
        """
        rxn_str = self.rxn_str
        reactants_str = rxn_str.split(" = ")[0]
        products_str = rxn_str.split(" = ")[1]
        return reactants_str, products_str

    def _load_cofactors_set_w_stereo(self, cofactors_filepath: str) -> set:
        """
        Get the full list of cofactors available on memory
        Stereochemistry of cofactors is NOT removed here
        :param cofactors_filepath: relative path to local cofactors list
        :return: set of cofactors WITH stereochemistry
        """
        cofactors_df = pd.read_csv(cofactors_filepath)
        cofactors_set_w_stereo = set(cofactors_df["SMILES"])
        return cofactors_set_w_stereo

    def print_full_cofactors_set_wo_stereo(self, cofactors_filepath: str) -> set:
        """
        Get the full list of cofactors available on memory
        Stereochemistry of cofactors IS removed here (even if the initial file did not contain stereochemistry)
        :param cofactors_filepath: relative path to local cofactors list
        :return: set of cofactors WITHOUT any stereochemistry
        """
        cofactors_set_w_stereo = self._load_cofactors_set_w_stereo(cofactors_filepath)
        cofactors_set_wo_stereo = []
        for cofactor_smiles in cofactors_set_w_stereo:
            cpd_w_stereo = compound(cofactor_smiles)
            cpd_wo_stereo = cpd_w_stereo.remove_stereo()
            cofactors_set_wo_stereo.append(cpd_wo_stereo)
        return set(cofactors_set_wo_stereo)

    def get_substrates(self, all_cofactors_wo_stereo: set) -> list:
        """
        Extract the substrate/s from a reaction string of the form "A + B = C + D"
        :param all_cofactors_wo_stereo: set of cofactors
        :return:
        """
        (
            reactants_str,
            products_str,
        ) = (
            self._rxn_2_cpds()
        )  # separate rxn str into two lists of strings - reactants and products
        substrates = []

        for reactant_smiles in reactants_str.split(" + "):
            reactant = compound(
                reactant_smiles
            )  # create compound object for this reactant's SMILES string
            canon_smi_wo_stereo = (
                reactant.remove_stereo()
            )  # canonicalize SMILES and remove stereochemistry

            if (
                canon_smi_wo_stereo not in all_cofactors_wo_stereo
            ):  # if canonicalized SMILES are not in the set of cofactors
                substrates.append(canon_smi_wo_stereo)  # then this is a substrate

        return substrates

    def get_products(self, all_cofactors_wo_stereo: set) -> list:
        """
        Extract the product/s from a reaction string of the form "A + B = C + D"
        :param all_cofactors_wo_stereo:
        :return:
        """
        reactants_str, products_str = self._rxn_2_cpds()  # separate reaction
        products = []

        for product_smiles in products_str.split(" + "):
            product = compound(
                product_smiles
            )  # create compound object for this product's SMILES string
            canon_smi_wo_stereo = (
                product.remove_stereo()
            )  # canonicalize SMILES and remove stereo

            if (
                canon_smi_wo_stereo not in all_cofactors_wo_stereo
            ):  # if canonicalized SMILES are not in the set of cofactors
                products.append(canon_smi_wo_stereo)  # then this is a product

        return products

    def get_lhs_cofactors(self, all_cofactors_wo_stereo: set) -> list:
        nadplus_incorrect = "*OC1C(O)C(COP(=O)(O)OP(=O)(O)OCC2OC([n+]3cccc(C(N)=O)c3)C(O)C2O)OC1n1cnc2c(N)ncnc21"
        nadh_incorrect = "*OC1C(O)C(COP(=O)(O)OP(=O)(O)OCC2OC(N3C=CCC(C(N)=O)=C3)C(O)C2O)OC1n1cnc2c(N)ncnc21"
        nadplus_correct = "NC(=O)c1ccc[n+](C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(O)C3O)C(O)C2O)c1"
        nadh_correct = "NC(=O)C1=CN(C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(O)C3O)C(O)C2O)C=CC1"

        reactants_str, products_str = self._rxn_2_cpds()  # separate reaction
        lhs_cofactors = []

        for reactant_smiles in reactants_str.split(" + "):
            reactant = compound(
                reactant_smiles
            )  # create compound object for this reactant's SMILES string
            canon_smi_wo_stereo = (
                reactant.remove_stereo()
            )  # canonicalize SMILES and remove stereo

            if (
                canon_smi_wo_stereo in all_cofactors_wo_stereo
            ):  # if canonicalized SMILES are in cofactors list
                if canon_smi_wo_stereo == nadplus_incorrect:
                    canon_smi_wo_stereo = nadplus_correct

                if canon_smi_wo_stereo == nadh_incorrect:
                    canon_smi_wo_stereo = nadh_correct

                lhs_cofactors.append(
                    canon_smi_wo_stereo
                )  # then this is a cofactor on the lhs

        return lhs_cofactors

    def get_rhs_cofactors(self, all_cofactors_wo_stereo: set) -> list:
        nadplus_incorrect = "*OC1C(O)C(COP(=O)(O)OP(=O)(O)OCC2OC([n+]3cccc(C(N)=O)c3)C(O)C2O)OC1n1cnc2c(N)ncnc21"
        nadh_incorrect = "*OC1C(O)C(COP(=O)(O)OP(=O)(O)OCC2OC(N3C=CCC(C(N)=O)=C3)C(O)C2O)OC1n1cnc2c(N)ncnc21"
        nadplus_correct = "NC(=O)c1ccc[n+](C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(O)C3O)C(O)C2O)c1"
        nadh_correct = "NC(=O)C1=CN(C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(O)C3O)C(O)C2O)C=CC1"

        reactants_str, products_str = self._rxn_2_cpds()  # separate reaction
        rhs_cofactors = []

        for product_smiles in products_str.split(" + "):
            product = compound(
                product_smiles
            )  # create compound object for this reactant's SMILES string
            canon_smi_wo_stereo = (
                product.remove_stereo()
            )  # canonicalize SMILES and remove stereo

            if (
                canon_smi_wo_stereo in all_cofactors_wo_stereo
            ):  # if canonicalized SMILES are in cofactors list
                if canon_smi_wo_stereo == nadplus_incorrect:
                    canon_smi_wo_stereo = nadplus_correct

                if canon_smi_wo_stereo == nadh_incorrect:
                    canon_smi_wo_stereo = nadh_correct
                rhs_cofactors.append(
                    canon_smi_wo_stereo
                )  # then this is a cofactor on the rhs

        return rhs_cofactors

    def _reorder_cofactors_by_MW(self, cofactors_list: list, ascending: bool) -> list:
        """
        Rearrange cofactors in ascending or descending molecular weights
        :param cofactors_list: list of cofactor SMILES
        :param ascending: cofactors will be arranged from lowest to highest molecular weight if true
        :return: list of rearranged cofactors
        """
        MW_list = []
        for cofactor_smiles in cofactors_list:
            mol = Chem.MolFromSmiles(cofactor_smiles)
            mw = Descriptors.ExactMolWt(mol)
            MW_list.append(mw)

        sorted_cofactors_list = [val for (_, val) in sorted(zip(MW_list, cofactors_list), key=lambda x: x[0])]

        if ascending:
            return sorted_cofactors_list

        else:
            return sorted_cofactors_list[::-1]

    def _reorder_substrates_by_MW(self, substrates_list: list, ascending: bool) -> list:
        """
        Rearrange substrates in ascending or descending molecular weights
        :param substrates_list: list of substrate SMILES
        :param ascending: substrates will be arranged from lowest to highest molecular weight if true
        :return: list of rearranged substrates
        """
        MW_list = []
        for substrate_SMILES in substrates_list:
            mol = Chem.MolFromSmiles(substrate_SMILES)
            mw = Descriptors.ExactMolWt(mol)
            MW_list.append(mw)

        sorted_substrates_list = [val for (_, val) in sorted(zip(MW_list, substrates_list), key=lambda x: x[0])]

        if ascending:
            return sorted_substrates_list

        else:
            return sorted_substrates_list[::-1]

    def _reorder_products_by_MW(self, products_list: list, ascending: bool) -> list:
        """
        Rearrange products in ascending or descending molecular weights
        :param products_list: list of product SMILES
        :param ascending: products will be arranged from lowest to highest molecular weight if true
        :return: list of rearranged products
        """
        MW_list = []
        for product_SMILES in products_list:
            mol = Chem.MolFromSmiles(product_SMILES)
            mw = Descriptors.ExactMolWt(mol)
            MW_list.append(mw)

        sorted_products_list = [val for (_, val) in sorted(zip(MW_list, products_list), key=lambda x: x[0])]

        if ascending:
            return sorted_products_list

        else:
            return sorted_products_list[::-1]

    def rxn_2_fp(self, type: str, max_species: int) -> np.ndarray:
        """
        Fingerprint a reaction string of form "substrate_smiles + cofactor_smiles = product_smiles + cofactor_smiles"


        :param type: Type of fingerprint to generate (eg: morgan, ecfp4, MACCS)
        :param max_species: Number of species on either side of reaction
                            If the number of species specified if less than the actual number, pad with extra zeroes
        :return:
        """
        reactants_str, products_str = self._rxn_2_cpds()
        all_reactants_fp = np.array([])
        all_products_fp = np.array([])

        # initialize counter to track reactants that are featurized
        reactant_counter = 0

        # initialize counter to track products that are featurized
        product_counter = 0

        ### Featurize all reactants on the LHS of the reaction string (includes cofactors)
        for reactant_smiles in reactants_str.split(" + "):
            reactant_counter += 1
            reactant_object = compound(reactant_smiles)

            if type == "morgan" or type == "ecfp4":
                reactant_fp = reactant_object._smiles_2_morganfp(radius=2, nBits=2048)

            if type == "MACCS":
                reactant_fp = reactant_object._smiles_2_MACCS()

            if type == "atom_pair":
                reactant_fp = reactant_object._smiles_2_atompair()

            if type == "modred":
                reactant_fp = reactant_object._smiles_2_modred()

            if type == "MAP4":
                reactant_fp = reactant_object._smiles_2_MAP4(is_folded=True, dim=2048)

            # else:
            #     raise Exception("Please enter a valid fingerprinting method")

            num_features = len(reactant_fp)  # length of each reactant's fingerprint
            all_reactants_fp = np.concatenate(
                (all_reactants_fp, reactant_fp), axis=None
            )

        # if number of reactants featurized is less than the maximum number of species
        # then pad the LHS fingerprint with extra zeroes (dummy reactants)
        if reactant_counter < max_species:
            lhs_diff = max_species - reactant_counter
            dummy_fp_for_lhs = np.zeros(
                num_features
            )  # create a dummy fingerprint of the same length

            # add as many dummy fingerprints as the difference in number of reactants and max species
            for i in range(0, lhs_diff):
                all_reactants_fp = np.concatenate(
                    (all_reactants_fp, dummy_fp_for_lhs), axis=None
                )

        ### Featurize all products on the RHS of the products string (includes cofactors)
        for product_smiles in products_str.split(" + "):
            product_counter += 1
            product_object = compound(product_smiles)

            if type == "morgan" or type == "ecfp4":
                product_fp = product_object._smiles_2_morganfp(radius=2, nBits=2048)

            if type == "MACCS":
                product_fp = product_object._smiles_2_MACCS()

            if type == "atom_pair":
                product_fp = product_object._smiles_2_atompair()

            if type == "modred":
                product_fp = product_object._smiles_2_modred()

            if type == "MAP4":
                product_fp = product_object._smiles_2_MAP4(is_folded=True, dim=2048)

            all_products_fp = np.concatenate((all_products_fp, product_fp), axis=None)

            num_features = len(product_fp)  # length of each product's fingerprint

        # if number of products featurized is less than the maximum number of species
        # then pad the RHS fingerprint with extra zeroes (dummy products)
        if product_counter < max_species:
            rhs_diff = max_species - product_counter
            dummy_fp_for_rhs = np.zeros(
                num_features
            )  # create a dummy fingerprint of the same length

            # add as many dummy fingerprints as the difference in number of reactants and max species
            for i in range(0, rhs_diff):
                all_products_fp = np.concatenate(
                    (all_products_fp, dummy_fp_for_rhs), axis=None
                )

        ### Finally, concatenate fingerprints on both sides of the reaction for a full reaction fingerprint
        reaction_fp = np.concatenate((all_reactants_fp, all_products_fp), axis=None)

        return reaction_fp

    def rxn_2_fp_w_positioning(self,
                               fp_type: str,
                               radius: int = 2,
                               nBits: int = 2048,
                               is_folded: bool = True,
                               dim: int = 2048,
                               max_species: int = 4,
                               cofactor_positioning: str = None,
                               reaction_rule: str = None,
                               all_cofactors_wo_stereo: set = None) -> np.ndarray:
        """
        :param fp_type: Type of reaction fingerprint to generate ( 'morgan/ecfp4', 'MACCS', 'atom_pair', 'MAP4')
        :param radius: Radius of fragmentation if using morgan or ecfp4
        :param nBits: Number of bits if using morgan or ecfp4
        :param is_folded: If fingerprint should be folded or not when using MAP4
        :param dim: Number of bits if using MAP4
        :param max_species: Maximum number of species on each side (pad zeroes if less species than max are present)
        :param cofactor_positioning: Arrangement of cofactors: in increasing MW, decreasing MW, or as per reaction rule
        :param reaction_rule: reaction rule
        :param all_cofactors_wo_stereo: set of all cofactors without stereochemistry
        :return: reaction fingerprint
        """

        # extract the substrates and cofactors on the LHS of the reaction string
        substrates_list = self.get_substrates(all_cofactors_wo_stereo)
        lhs_cofactors_list = self.get_lhs_cofactors(all_cofactors_wo_stereo)

        # extract the products and cofactors on the RHS of the reaction string
        products_list = self.get_products(all_cofactors_wo_stereo)
        rhs_cofactors_list = self.get_rhs_cofactors(all_cofactors_wo_stereo)

        # initialize empty arrays to store fingerprints for both the LHS and RHS of reaction
        all_lhs_fp = np.array([])
        all_rhs_fp = np.array([])

        # initialize counter to track reactants that are featurized
        reactant_counter = 0

        # initialize counter to track products that are featurized
        product_counter = 0

        if cofactor_positioning == "by_ascending_MW":

            # If cofactors present, rearrange from lightest to heaviest (ascending molecular weights)
            if lhs_cofactors_list:
                lhs_cofactors = self._reorder_cofactors_by_MW(lhs_cofactors_list, ascending=True)

            if rhs_cofactors_list:
                rhs_cofactors = self._reorder_cofactors_by_MW(rhs_cofactors_list, ascending=True)

            if substrates_list:
                reordered_substrates = self._reorder_substrates_by_MW(substrates_list, ascending=True)

            if products_list:
                reordered_products = self._reorder_products_by_MW(products_list, ascending=True)

            # Featurize all reactants on the LHS of the reaction string - begin with substrates then do cofactors
            if substrates_list:
                for substrate_smiles in reordered_substrates:
                    reactant_counter += 1
                    reactant_object = compound(substrate_smiles)
                    reactant_fp = reactant_object.smiles_2_fp(fp_type = fp_type,
                                                              radius = radius,
                                                              nBits = nBits,
                                                              is_folded = is_folded,
                                                              dim = dim)

                    num_features = len(reactant_fp)  # length of each reactant's fingerprint
                    all_lhs_fp = np.concatenate((all_lhs_fp, reactant_fp), axis=None)

            # then repeat for cofactors on the LHS
            if lhs_cofactors_list:
                for lhs_cofactor_smiles in lhs_cofactors:
                    reactant_counter += 1
                    lhs_cofactor_object = compound(lhs_cofactor_smiles)
                    lhs_cofactor_fp = lhs_cofactor_object.smiles_2_fp(fp_type = fp_type,
                                                                      radius = radius,
                                                                      nBits = nBits,
                                                                      is_folded = is_folded,
                                                                      dim = dim)

                    num_features = len(lhs_cofactor_fp)  # length of each reactant's fingerprint
                    all_lhs_fp = np.concatenate((all_lhs_fp, lhs_cofactor_fp), axis=None)

            # if number of reactants featurized is less than the maximum number of species
            # then pad the LHS fingerprint with extra zeroes (dummy reactants)
            if reactant_counter < max_species:
                lhs_diff = max_species - reactant_counter
                dummy_fp_for_lhs = np.zeros(num_features)  # create a dummy fingerprint of the same length

                # add as many dummy fingerprints as the difference in number of reactants and max species
                for i in range(0, lhs_diff):
                    all_lhs_fp = np.concatenate((all_lhs_fp, dummy_fp_for_lhs), axis=None)

            # Featurize all products on the RHS of the reaction string - begin with products then do cofactors
            if products_list:
                for product_smiles in reordered_products:
                    product_counter += 1
                    product_object = compound(product_smiles)
                    product_fp = product_object.smiles_2_fp(fp_type = fp_type,
                                                            radius = radius,
                                                            nBits = nBits,
                                                            is_folded = is_folded,
                                                            dim = dim)

                    num_features = len(product_fp)  # length of each reactant's fingerprint
                    all_rhs_fp = np.concatenate((all_rhs_fp, product_fp), axis=None)

            # then repeat for cofactors on the RHS
            if rhs_cofactors_list:
                for rhs_cofactor_smiles in rhs_cofactors:
                    product_counter += 1
                    rhs_cofactor_object = compound(rhs_cofactor_smiles)
                    rhs_cofactor_object = rhs_cofactor_object.smiles_2_fp(fp_type = fp_type,
                                                            radius = radius,
                                                            nBits = nBits,
                                                            is_folded = is_folded,
                                                            dim = dim)

                    num_features = len(rhs_cofactor_object)  # length of each reactant's fingerprint
                    all_rhs_fp = np.concatenate((all_rhs_fp, rhs_cofactor_object), axis=None)

            # if number of products featurized is less than the maximum number of species
            # then pad the RHS fingerprint with extra zeroes (dummy products)
            if product_counter < max_species:
                rhs_diff = max_species - product_counter
                dummy_fp_for_rhs = np.zeros(num_features)

                # add as many dummy fingerprints as the difference in number of products and max species
                for i in range(0, rhs_diff):
                    all_rhs_fp = np.concatenate((all_rhs_fp, dummy_fp_for_rhs), axis=None)

            ### Finally, concatenate fingerprints on both sides of the reaction for a full reaction fingerprint
            reaction_fp = np.concatenate((all_lhs_fp, all_rhs_fp), axis=None)

            return reaction_fp

        if cofactor_positioning == "by_descending_MW":
            
            # If cofactors present, rearrange from lightest to heaviest (ascending molecular weights)
            if lhs_cofactors_list:
                lhs_cofactors = self._reorder_cofactors_by_MW(lhs_cofactors_list, ascending=False)

            if rhs_cofactors_list:
                rhs_cofactors = self._reorder_cofactors_by_MW(rhs_cofactors_list, ascending=False)

            if substrates_list:
                reordered_substrates = self._reorder_substrates_by_MW(substrates_list, ascending=False)

            if products_list:
                reordered_products = self._reorder_products_by_MW(products_list, ascending=False)
                
            # Featurize all reactants on the LHS of the reaction string - begin with substrates then do cofactors
            if substrates_list:
                
                for substrate_smiles in reordered_substrates:
                    reactant_counter += 1
                    reactant_object = compound(substrate_smiles)
                    
                    reactant_fp = reactant_object.smiles_2_fp(fp_type = fp_type,
                                                              radius = radius,
                                                              nBits = nBits,
                                                              is_folded = is_folded,
                                                              dim = dim)

                    num_features = len(reactant_fp)  # length of each reactant's fingerprint
                    all_lhs_fp = np.concatenate((all_lhs_fp, reactant_fp), axis=None)

            # then repeat for cofactors on the LHS
            if lhs_cofactors_list:
                
                for lhs_cofactor_smiles in lhs_cofactors:
                    reactant_counter += 1
                    lhs_cofactor_object = compound(lhs_cofactor_smiles)
                    lhs_cofactor_fp = lhs_cofactor_object.smiles_2_fp(fp_type = fp_type,
                                                                      radius = radius,
                                                                      nBits = nBits,
                                                                      is_folded = is_folded,
                                                                      dim = dim)

                    num_features = len(lhs_cofactor_fp)  # length of each reactant's fingerprint
                    all_lhs_fp = np.concatenate((all_lhs_fp, lhs_cofactor_fp), axis=None)

            # if number of reactants featurized is less than the maximum number of species
            # then pad the LHS fingerprint with extra zeroes (dummy reactants)
            if reactant_counter < max_species:
                lhs_diff = max_species - reactant_counter
                dummy_fp_for_lhs = np.zeros(num_features)  # create a dummy fingerprint of the same length

                # add as many dummy fingerprints as the difference in number of reactants and max species
                for i in range(0, lhs_diff):
                    all_lhs_fp = np.concatenate((all_lhs_fp, dummy_fp_for_lhs), axis=None)

            # Featurize all products on the RHS of the reaction string - begin with products then do cofactors
            if products_list:
                
                for product_smiles in reordered_products:
                    product_counter += 1
                    product_object = compound(product_smiles)
                    product_fp = product_object.smiles_2_fp(fp_type = fp_type,
                                                            radius = radius,
                                                            nBits = nBits,
                                                            is_folded = is_folded,
                                                            dim = dim)

                    num_features = len(product_fp)  # length of each reactant's fingerprint
                    all_rhs_fp = np.concatenate((all_rhs_fp, product_fp), axis=None)

            # then repeat for cofactors on the RHS
            if rhs_cofactors_list:
                
                for rhs_cofactor_smiles in rhs_cofactors:
                    product_counter += 1
                    rhs_cofactor_object = compound(rhs_cofactor_smiles)
                    rhs_cofactor_object = rhs_cofactor_object.smiles_2_fp(fp_type = fp_type,
                                                            radius = radius,
                                                            nBits = nBits,
                                                            is_folded = is_folded,
                                                            dim = dim)

                    num_features = len(rhs_cofactor_object)  # length of each reactant's fingerprint
                    all_rhs_fp = np.concatenate((all_rhs_fp, rhs_cofactor_object), axis=None)

            # if number of products featurized is less than the maximum number of species
            # then pad the RHS fingerprint with extra zeroes (dummy products)
            if product_counter < max_species:
                rhs_diff = max_species - product_counter
                dummy_fp_for_rhs = np.zeros(num_features)

                # add as many dummy fingerprints as the difference in number of products and max species
                for i in range(0, rhs_diff):
                    all_rhs_fp = np.concatenate((all_rhs_fp, dummy_fp_for_rhs), axis=None)
            
            ### Finally, concatenate fingerprints on both sides of the reaction for a full reaction fingerprint
            reaction_fp = np.concatenate((all_lhs_fp, all_rhs_fp), axis=None)
            
            return reaction_fp

        #TODO - not complete
        if cofactor_positioning == "by_rule":

            cofactors_dict = {'PYROPHOSPHATE_DONOR_CoF': 'Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C1O', # ATP
                              'PYROPHOSPHATE_ACCEPTOR_CoF': 'Nc1ncnc2c1ncn2C1OC(COP(=O)(O)O)C(O)C1O', # AMP
                              'FAD_CoF': 'Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(CC(O)C(O)C(O)COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(O)C3O)c2cc1C', # FAD
                              'FADH2_CoF': 'Cc1cc2c(cc1C)N(CC(O)C(O)C(O)COP(=O)(O)OP(=O)(O)OCC1OC(n3cnc4c(N)ncnc43)C(O)C1O)c1[nH]c(=O)[nH]c(=O)c1N2', # FADH2
                              'PHOSPHATE_DONOR_CoF': 'Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C1O', # ATP
                              'PHOSPHATE_ACCEPTOR_CoF': 'Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)O)C(O)C1O', # ADP
                              'NAD_CoF': 'NC(=O)c1ccc[n+](C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(OP(=O)(O)O)C3O)C(O)C2O)c1', # NADP
                              'NADH_CoF': 'NC(=O)C1=CN(C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(OP(=O)(O)O)C3O)C(O)C2O)C=CC1', # NADPH
                              'SULFATE_DONOR_CoF': 'Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OS(=O)(=O)O)C(OP(=O)(O)O)C1O', # PAPS
                              'SULFATE_ACCEPTOR_CoF': 'Nc1ncnc2c1ncn2C1OC(COP(=O)(O)O)C(OP(=O)(O)O)C1O', # 3,5-ADP
                              'METHYL_DONOR_CoF': 'C[S+](CCC(N)C(=O)O)CC1OC(n2cnc3c(N)ncnc32)C(O)C1O', # S-adenosylmethionine
                              'METHYL_ACCEPTOR_CoF': 'Nc1ncnc2c1ncn2C1OC(CSCCC(N)C(=O)O)C(O)C1O', # Adenosyl-homo-cys
                              'GLUCOSYL_DONOR_CoF': 'O=c1ccn(C2OC(COP(=O)(O)OP(=O)(O)OC3OC(CO)C(O)C(O)C3O)C(O)C2O)c(=O)[nH]1', # CPD-12575
                              'GLUCOSYL_ACCEPTOR_CoF': 'O=c1ccn(C2OC(COP(=O)(O)OP(=O)(O)O)C(O)C2O)c(=O)[nH]1', # UDP
                              'Ubiquinols_CoF': '*c1c(*)c(O)c(*)c(*)c1O', # ETR-Quinols
                              'Ubiquinones_CoF': '*C1=C(*)C(=O)C(*)=C(*)C1=O', # ETR-Quinones
                              'PRENYL_DONOR_CoF': 'CC(C)=CCOP(=O)(O)OP(=O)(O)O', # CPD-4211
                              'PRENYL_ACCEPTOR_CoF': 'O=P(O)(O)OP(=O)(O)O', # PPI
                              'CARBONYL_CoF': 'O=C(O)CCC(=O)C(=O)O', # 2-Ketoglutarate
                              'AMINO_CoF': 'NC(CCC(=O)O)C(=O)O', # GLT
                              'FORMYL_DONOR_CoF': '*C(=O)CCC(NC(=O)c1ccc(N(C=O)CC2CNc3nc(N)[nH]c(=O)c3N2)cc1)C(=O)O', # Formyl-thf-glu-N
                              'FORMYL_ACCEPTOR_CoF': '*C(=O)CCC(NC(=O)c1ccc(NCC2CNc3nc(N)[nH]c(=O)c3N2)cc1)C(=O)O',
                              'ASCORBATE_RADICAL_CoF': '',
                              'ASCORBATE_CoF': 'O=C1OC(C(O)CO)C(O)=C1O', # Ascorbate
                              'Oxidized-Factor-F420_CoF': '*C(=O)C(C)OP(=O)(O)OCC(O)C(O)C(O)Cn1c2nc(=O)[nH]c(=O)c-2cc2ccc(O)cc21', # Oxidized factor F420
                              'Reduced-Factor-F420_CoF': '*C(=O)C(C)OP(=O)(O)OCC(O)C(O)C(O)CN1c2cc(O)ccc2Cc2c1[nH]c(=O)[nH]c2=O', # Reduced factor F420
                              'ACETYL-COA': 'CC(=O)SCCNC(=O)CCNC(=O)C(O)C(C)(C)COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O', # Acetyl CoA
                              'CO': '',
                              'CO2': 'O=C=O', # CO2
                              'CO3': 'O=C(O)O', # CO3
                              'CoA': 'CC(C)(COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O)C(O)C(=O)NCCC(=O)NCCS', # Carbon dioxide
                              'H+': '[H+]', # Proton
                              'H2O2': 'OO', # Hydrogen peroxide
                              'HBr': 'Br', # Bromide
                              'HCN': '',
                              'Hcl': 'Cl', # Chloride
                              'HF': 'F', # Fluoride
                              'HI': 'I', # Iodide
                              'NH3': 'N', # Ammonia,
                              'O2': 'O=O', # Oxygen,
                              'PPI': 'O=P(O)(O)OP(=O)(O)O', # PPI
                              'Pi': 'O=P(O)(O)O', # PI
                              'SULFATE': 'O=S(=O)(O)O', # Sulfate
                              'SULFITE': 'O=S(O)O',
                              'WATER': 'O',
                              'OXYGEN': 'O=O'}

            rules_dict = {'rule0001': [ ['Any'], ['Any'] ],
                          'rule0002': [ ['Any', 'NAD_CoF'], ['Any', 'NADH_CoF'] ],
                          'rule0003': [ ['Any', 'NADH_CoF'], ['Any', 'NAD_CoF'] ],
                          'rule0004': [ ['Any', 'NADH_CoF', 'O2'], ['Any', 'NAD_CoF', 'WATER'] ],
                          'rule0005': [ ['Any', 'NAD_CoF', 'WATER'], ['Any', 'NADH_CoF', 'O2'] ],
                          'rule0023': [ ['Any','CO2'], ['Any'] ],
                          'rule0024': [ ['Any'], ['Any', 'CO2'] ],
                          'rule0025': [ ['Any', 'NADH_CoF'], ['Any', 'NAD_CoF', 'WATER'] ],
                          'rule0026': [ ['Any', 'NAD_CoF', 'WATER'], ['Any', 'NADH_CoF'] ]}

            # Get positioning of species on the LHS and RHS as per the rule
            lhs_config = rules_dict[reaction_rule][0]
            rhs_config = rules_dict[reaction_rule][1]

            for i, lhs_species_type in enumerate(lhs_config):
                reactant_counter += 1

                if lhs_species_type == 'Any':
                    substrate_smiles = substrates_list[i]
                    reactant_object = compound(substrate_smiles)
                    reactant_fp = reactant_object.smiles_2_fp(fp_type=fp_type,
                                                              radius=radius,
                                                              nBits=nBits,
                                                              is_folded=is_folded,
                                                              dim=dim)

                    num_features = len(reactant_fp)  # length of each reactant's fingerprint
                    all_lhs_fp = np.concatenate((all_lhs_fp, reactant_fp), axis=None)

                else:
                    lhs_cofactor_identity = lhs_config[i]
                    lhs_cofactor_smiles = cofactors_dict[lhs_cofactor_identity]
                    lhs_cofactor_object = compound(lhs_cofactor_smiles)
                    lhs_cofactor_fp = lhs_cofactor_object.smiles_2_fp(fp_type=fp_type,
                                                              radius=radius,
                                                              nBits=nBits,
                                                              is_folded=is_folded,
                                                              dim=dim)

                    num_features = len(lhs_cofactor_fp)  # length of each reactant's fingerprint
                    all_lhs_fp = np.concatenate((all_lhs_fp, lhs_cofactor_fp), axis=None)

            # if number of reactants featurized is less than maximum number of species then pad LHS with zeroes
            if reactant_counter < max_species:
                lhs_diff = max_species - reactant_counter
                dummy_fp_for_lhs = np.zeros(num_features)  # create a dummy fingerprint of the same length

                # add as many dummy fingerprints as the difference in number of reactants and max species
                for i in range(0, lhs_diff):
                    all_lhs_fp = np.concatenate((all_lhs_fp, dummy_fp_for_lhs), axis=None)

            for i, rhs_species_type in enumerate(rhs_config):
                product_counter += 1

                if rhs_species_type == 'Any':
                    product_smiles = products_list[i]
                    product_object = compound(product_smiles)
                    product_fp = product_object.smiles_2_fp(fp_type=fp_type,
                                                              radius=radius,
                                                              nBits=nBits,
                                                              is_folded=is_folded,
                                                              dim=dim)

                    num_features = len(product_fp)  # length of each reactant's fingerprint
                    all_rhs_fp = np.concatenate((all_rhs_fp, product_fp), axis=None)

                else:
                    rhs_cofactor_identity = rhs_config[i]
                    rhs_cofactor_smiles = cofactors_dict[rhs_cofactor_identity]
                    rhs_cofactor_object = compound(rhs_cofactor_smiles)
                    rhs_cofactor_fp = rhs_cofactor_object.smiles_2_fp(fp_type=fp_type,
                                                              radius=radius,
                                                              nBits=nBits,
                                                              is_folded=is_folded,
                                                              dim=dim)

                    num_features = len(rhs_cofactor_fp)  # length of each reactant's fingerprint
                    all_rhs_fp = np.concatenate((all_rhs_fp, rhs_cofactor_fp), axis=None)

            # if number of reactants featurized is less than maximum number of species then pad RHS with zeroes
            if product_counter < max_species:
                rhs_diff = max_species - product_counter
                dummy_fp_for_rhs = np.zeros(num_features)

                # add as many dummy fingerprints as the difference in number of products and max species
                for i in range(0, rhs_diff):
                    all_rhs_fp = np.concatenate((all_rhs_fp, dummy_fp_for_rhs), axis=None)

            ### Finally, concatenate fingerprints on both sides of the reaction for a full reaction fingerprint
            reaction_fp = np.concatenate((all_lhs_fp, all_rhs_fp), axis=None)

            return reaction_fp

        #TODO - think again
        if cofactor_positioning == "by_one_hot":

            # for positioning cofactors by one-hot encoding, the feature vector will be longer than other positionings
            # this is because 2 substrates (with padding) will be featurized first
            # only 2 substrates will be featurized instead of 4 because cofactors
            # then 2 products (with padding) will be featurized next
            # finally, the generalized reaction rule will be one-hot encoded as a vector of length 1224
            # this

            # redefine max_species for one-hot encoding
            max_species_ohe = 2

            rule_number = int(reaction_rule[4:]) # get rule number as an integer
            ohe_rule = np.zeros(1224)
            ohe_rule[rule_number-1] = 1 # turn on the relevant reaction rule, leave the rest as zeros

            # Featurize all reactants on the LHS of the reaction string - begin with substrates then do cofactors
            for substrate_smiles in substrates_list:
                reactant_counter += 1
                reactant_object = compound(substrate_smiles)
                reactant_fp = reactant_object.smiles_2_fp(fp_type = fp_type,
                                                          radius = radius,
                                                          nBits = nBits,
                                                          is_folded = is_folded,
                                                          dim = dim)

                num_features = len(reactant_fp)  # length of each reactant's fingerprint
                all_lhs_fp = np.concatenate((all_lhs_fp, reactant_fp), axis=None)

            # if number of reactants featurized is less than the maximum number of species
            # then pad the LHS fingerprint with extra zeroes (dummy reactants)
            if reactant_counter < max_species_ohe:
                lhs_diff = max_species_ohe - reactant_counter
                dummy_fp_for_lhs = np.zeros(num_features)  # create a dummy fingerprint of the same length

                # add as many dummy fingerprints as the difference in number of reactants and max species
                for i in range(0, lhs_diff):
                    all_lhs_fp = np.concatenate((all_lhs_fp, dummy_fp_for_lhs), axis=None)

            # Featurize all products on the RHS of the reaction string - begin with products then do cofactors
            for product_smiles in products_list:
                product_counter += 1
                product_object = compound(product_smiles)
                product_fp = product_object.smiles_2_fp(fp_type = fp_type,
                                                        radius = radius,
                                                        nBits = nBits,
                                                        is_folded = is_folded,
                                                        dim = dim)

                num_features = len(product_fp)  # length of each reactant's fingerprint
                all_rhs_fp = np.concatenate((all_rhs_fp, product_fp), axis=None)

            # if number of products featurized is less than the maximum number of species
            # then pad the RHS fingerprint with extra zeroes (dummy products)
            if product_counter < max_species_ohe:
                rhs_diff = max_species_ohe - product_counter
                dummy_fp_for_rhs = np.zeros(num_features)

                # add as many dummy fingerprints as the difference in number of products and max species
                for i in range(0, rhs_diff):
                    all_rhs_fp = np.concatenate((all_rhs_fp, dummy_fp_for_rhs), axis=None)

            ### Concatenate fingerprints on both sides of the reaction
            reaction_fp = np.concatenate((all_lhs_fp, all_rhs_fp), axis=None)

            #### Finally, concatenate fingerprints
            reaction_fp = np.concatenate((ohe_rule, reaction_fp), axis=None)

            return reaction_fp

        if cofactor_positioning == "add_concat":
            # generate ecfp4 fingerprints for all species
            # add up all reactants' fingerprints elementwise into a single vector
            # add up all products' fingerprints elementwise into a single vector
            # concatenate the resultant reactants' and products' fingerprints side by side
            # this approach does not require any padding

            # initialize a fingerprint for all reactants on the LHS (incl. cofactors)
            lhs_fp = np.zeros(2048)

            # initialize a fingerprint for all products on the RHS (incl. cofactors)
            rhs_fp = np.zeros(2048)

            for substrate_smiles in substrates_list:
                compound_object = compound(substrate_smiles)
                fp = compound_object._smiles_2_morganfp(radius = 2, nBits = 2048)
                lhs_fp = lhs_fp + fp

            for lhs_cofactor_smiles in lhs_cofactors_list:
                compound_object = compound(lhs_cofactor_smiles)
                fp = compound_object._smiles_2_morganfp(radius = 2, nBits = 2048)
                lhs_fp = lhs_fp + fp

            for product_smiles in products_list:
                compound_object = compound(product_smiles)
                fp = compound_object._smiles_2_morganfp(radius = 2, nBits = 2048)
                rhs_fp = rhs_fp + fp

            for rhs_cofactor_smiles in rhs_cofactors_list:
                compound_object = compound(rhs_cofactor_smiles)
                fp = compound_object._smiles_2_morganfp(radius = 2, nBits = 2048)
                rhs_fp = rhs_fp + fp

            ### Finally, concatenate fingerprints on both sides of the reaction for a full reaction fingerprint
            reaction_fp = np.concatenate((lhs_fp, rhs_fp), axis=None)

            return reaction_fp

        if cofactor_positioning == "half_random":
            # randomly rearrange reactants anywhere along positions 1 to 4 of the final reaction fingerprint
            # randomly rearrange products anywhere along positions 5 to 8 for the final reaction fingerprint
            # concatenate the two
            # this approach will still require padding

            # since our initial vector already has 16384 zeros, we actually don't need to perform extra padding
            rxn_fp = np.zeros(16384)

            random_positions_for_lhs_species = np.random.permutation(4)
            random_positions_for_rhs_species = 4 + np.random.permutation(4)

            for substrate_smiles in substrates_list:
                reactant_counter += 1 # keep track of substrate
                compound_object = compound(substrate_smiles)
                substrate_fp = compound_object._smiles_2_morganfp(radius = 2, nBits = 2048)

                # grab a position for the substrate
                substrate_position = random_positions_for_lhs_species[0]

                # remove this position from the list of positions since it is no longer available
                random_positions_for_lhs_species = np.delete(random_positions_for_lhs_species,0)

                # slot the substrate fingerprint into its allocated position
                if substrate_position == 0:
                    rxn_fp[:2048] = substrate_fp
                    assert np.array_equal(rxn_fp[:2048], substrate_fp)

                else:
                    rxn_fp[ (substrate_position-1) * 2048 : substrate_position * 2048 ] = substrate_fp
                    assert np.array_equal(rxn_fp[ (substrate_position-1) * 2048 : substrate_position * 2048 ],
                                          substrate_fp)

            for lhs_cofactor_smiles in lhs_cofactors_list:
                reactant_counter += 1 # keep track of lhs_cofactor
                compound_object = compound(lhs_cofactor_smiles)
                lhs_cofactor_fp = compound_object._smiles_2_morganfp(radius = 2, nBits = 2048)

                # grab a position for the lhs_cofactor
                lhs_cofactor_position = random_positions_for_lhs_species[0]

                # remove this position from the list of positions since it is no longer available
                random_positions_for_lhs_species = np.delete(random_positions_for_lhs_species,0)

                # slot the lhs_cofactor fingerprint into its allocated position
                if lhs_cofactor_position == 0:
                    rxn_fp[:2048] = lhs_cofactor_fp
                    assert np.array_equal(rxn_fp[:2048], lhs_cofactor_fp)

                else:
                    rxn_fp[ (lhs_cofactor_position-1) * 2048 : lhs_cofactor_position * 2048 ] = lhs_cofactor_fp
                    assert np.array_equal( rxn_fp[ (lhs_cofactor_position-1) * 2048 : lhs_cofactor_position * 2048 ],
                                           lhs_cofactor_fp)

            for product_smiles in products_list:
                product_counter += 1 # keep track of product
                compound_object = compound(product_smiles)
                product_fp = compound_object._smiles_2_morganfp(radius = 2, nBits = 2048)

                # grab a position for the product
                product_position = random_positions_for_rhs_species[0]

                # remove this position from the list of positions since it is no longer available
                random_positions_for_rhs_species = np.delete(random_positions_for_rhs_species,0)

                # slot the product fingerprint into its allocated position
                rxn_fp[(product_position - 1) * 2048: product_position * 2048] = product_fp
                assert np.array_equal(rxn_fp[(product_position - 1) * 2048: product_position * 2048],
                                      product_fp)

            for rhs_cofactor_smiles in rhs_cofactors_list:
                product_counter += 1 # keep track of rhs_cofactor
                compound_object = compound(rhs_cofactor_smiles)
                rhs_cofactor_fp = compound_object._smiles_2_morganfp(radius = 2, nBits = 2048)

                # grab a position for the rhs_cofactor
                rhs_cofactor_position = random_positions_for_rhs_species[0]

                # remove this position from the list of positions since it is no longer available
                random_positions_for_rhs_species = np.delete(random_positions_for_rhs_species,0)

                # slot the rhs fingerprint into its allocated position
                rxn_fp[(rhs_cofactor_position - 1) * 2048: rhs_cofactor_position * 2048] = rhs_cofactor_fp
                assert np.array_equal(rxn_fp[(rhs_cofactor_position - 1) * 2048: rhs_cofactor_position * 2048],
                                      rhs_cofactor_fp)


            return rxn_fp

        if cofactor_positioning == "full_random":
            # randomly rearrange reactants anywhere along positions 1 to 4 of the final reaction fingerprint
            # randomly rearrange products anywhere along positions 5 to 8 for the final reaction fingerprint
            # concatenate the two
            # this approach will still require padding

            rxn_fp = np.zeros(16384)

            random_positions_for_all_species = np.random.permutation(8)

            for substrate_smiles in substrates_list:
                reactant_counter += 1 # keep track of substrate
                compound_object = compound(substrate_smiles)
                substrate_fp = compound_object._smiles_2_morganfp(radius = 2, nBits = 2048)

                # grab a position for the substrate
                substrate_position = random_positions_for_all_species[0]

                # remove this position from the list of positions since it is no longer available
                random_positions_for_all_species = np.delete(random_positions_for_all_species,0)

                # slot the substrate fingerprint into its allocated position
                if substrate_position == 0:
                    rxn_fp[:2048] = substrate_fp
                    assert np.array_equal(rxn_fp[:2048], substrate_fp)

                else:
                    rxn_fp[ (substrate_position-1) * 2048 : substrate_position * 2048 ] = substrate_fp
                    assert np.array_equal(rxn_fp[ (substrate_position-1) * 2048 : substrate_position * 2048 ],
                                          substrate_fp)

            for lhs_cofactor_smiles in lhs_cofactors_list:
                reactant_counter += 1 # keep track of lhs_cofactor
                compound_object = compound(lhs_cofactor_smiles)
                lhs_cofactor_fp = compound_object._smiles_2_morganfp(radius = 2, nBits = 2048)

                # grab a position for the lhs_cofactor
                lhs_cofactor_position = random_positions_for_all_species[0]

                # remove this position from the list of positions since it is no longer available
                random_positions_for_all_species = np.delete(random_positions_for_all_species,0)

                # slot the lhs_cofactor fingerprint into its allocated position
                if lhs_cofactor_position == 0:
                    rxn_fp[:2048] = lhs_cofactor_fp
                    assert np.array_equal(rxn_fp[:2048], lhs_cofactor_fp)

                else:
                    rxn_fp[ (lhs_cofactor_position-1) * 2048 : lhs_cofactor_position * 2048 ] = lhs_cofactor_fp
                    assert np.array_equal( rxn_fp[ (lhs_cofactor_position-1) * 2048 : lhs_cofactor_position * 2048 ],
                                           lhs_cofactor_fp)

            for product_smiles in products_list:
                product_counter += 1 # keep track of rhs_cofactor
                compound_object = compound(product_smiles)
                product_fp = compound_object._smiles_2_morganfp(radius = 2, nBits = 2048)

                # grab a position for the lhs_cofactor
                product_position = random_positions_for_all_species[0]

                # remove this position from the list of positions since it is no longer available
                random_positions_for_all_species = np.delete(random_positions_for_all_species,0)

                # slot the lhs_cofactor fingerprint into its allocated position
                if product_position == 0:
                    rxn_fp[:2048] = product_fp
                    assert np.array_equal(rxn_fp[:2048], product_fp)

                else:
                    rxn_fp[ (product_position-1) * 2048 : product_position * 2048 ] = product_fp
                    assert np.array_equal( rxn_fp[ (product_position-1) * 2048 : product_position * 2048 ], product_fp)

            for rhs_cofactor_smiles in rhs_cofactors_list:
                product_counter += 1 # keep track of rhs_cofactor
                compound_object = compound(rhs_cofactor_smiles)
                rhs_cofactor_fp = compound_object._smiles_2_morganfp(radius = 2, nBits = 2048)

                # grab a position for the lhs_cofactor
                rhs_cofactor_position = random_positions_for_all_species[0]

                # remove this position from the list of positions since it is no longer available
                random_positions_for_all_species = np.delete(random_positions_for_all_species,0)

                # slot the lhs_cofactor fingerprint into its allocated position
                if rhs_cofactor_position == 0:
                    rxn_fp[:2048] = rhs_cofactor_fp
                    assert np.array_equal(rxn_fp[:2048], rhs_cofactor_fp)

                else:
                    rxn_fp[ (rhs_cofactor_position-1) * 2048 : rhs_cofactor_position * 2048 ] = rhs_cofactor_fp
                    assert np.array_equal( rxn_fp[ (rhs_cofactor_position-1) * 2048 : rhs_cofactor_position * 2048 ],rhs_cofactor_fp)

            return rxn_fp

        if cofactor_positioning == "add_subtract":
            # generate ecfp4 fingerprints for all species
            # add up all reactants' fingerprints elementwise into a single vector
            # add up all products' fingerprints elementwise into a single vector
            # concatenate the resultant reactants' and products' fingerprints side by side
            # this approach does not require any padding

            # initialize a fingerprint for all reactants on the LHS (incl. cofactors)
            lhs_fp = np.zeros(2048)

            # initialize a fingerprint for all products on the RHS (incl. cofactors)
            rhs_fp = np.zeros(2048)

            for substrate_smiles in substrates_list:
                compound_object = compound(substrate_smiles)
                fp = compound_object._smiles_2_morganfp(radius = 2, nBits = 2048)
                lhs_fp = lhs_fp + fp

            for lhs_cofactor_smiles in lhs_cofactors_list:
                compound_object = compound(lhs_cofactor_smiles)
                fp = compound_object._smiles_2_morganfp(radius = 2, nBits = 2048)
                lhs_fp = lhs_fp + fp

            for product_smiles in products_list:
                compound_object = compound(product_smiles)
                fp = compound_object._smiles_2_morganfp(radius = 2, nBits = 2048)
                rhs_fp = rhs_fp + fp

            for rhs_cofactor_smiles in rhs_cofactors_list:
                compound_object = compound(rhs_cofactor_smiles)
                fp = compound_object._smiles_2_morganfp(radius = 2, nBits = 2048)
                rhs_fp = rhs_fp + fp

            ### Finally, concatenate fingerprints on both sides of the reaction for a full reaction fingerprint
            reaction_fp = rhs_fp - lhs_fp

            return reaction_fp
