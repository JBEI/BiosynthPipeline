"""
A combined RetroTide and Pickaxe pipeline
Authors: Tyler Backman and Yash Chainani
"""

import warnings
from rdkit import RDLogger

# Suppress RDKit warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

import retrotide

from rdkit import Chem
from rdkit.Chem import AllChem
from minedatabase.pickaxe import Pickaxe

from src.biosynth_pipeline.pickaxe_utils import pickaxe_utils

#TODO - get list of chemical operators from Quan
#TODO - use a reaction operator to detach the final bound molecule from the thioesterase module in the pks framework
# (the most common pks termination is a lactonization)
# (but there are a lot of different terminations to try in general so we may want to take these into account)
# TODO - even though reductive termination is most useful, carboxylic acid formation is most common
# Dan was able to do it

class biosynth_pipeline:
    def __init__(self,
                pathway_sequence: list,
                target_smiles: str,
                remove_stereo: bool,
                known_metabolites: str,
                non_pks_cofactors: str,
                non_pks_rules: str,
                non_pks_steps: int,
                non_pks_cores: int,
                non_pks_sim_score_filter: bool = False,
                non_pks_sim_score_cutoffs: list = [],
                non_pks_sim_sample: bool = False,
                non_pks_sim_sample_size: int = None):

        """
        Combined pipeline for biosynthetic pathway design
        :param pathway_sequence: order of pks and non_pks transformations, e.g. ['pks','non_pks'] or ['non_pks','pks']
        :param target_smiles: SMILES string of target molecule, e.g. 'CCC' for propane
        :param remove_stereo: remove stereochemistry from SMILES string if True
        :param known_metabolites: list of known metabolites from BRENDA, KEGG, and METACYC
        :param non_pks_cofactors: cofactors to use for pickaxe expansion
        :param non_pks_rules: reaction rules for pickaxe, choose from 'generalized', 'intermediate', or 'chemical'
        :param non_pks_steps: number of reaction steps to run pickaxe
        :param non_pks_cores: number of cores to run Pickaxe
        :param non_pks_sim_score_filter: tanimoto similarity cutoffs during pickaxe expansion will be used if True
        :param non_pks_sim_score_cutoffs: list of tanimoto similarity cutoffs for each generation. e.g. [0.2, 0.4]
        :param non_pks_sim_sample: similarity sampling during pickaxe expansion will be used if True
        :param non_pks_sim_sample_size: number of intermediates to sample at each pickaxe generation
        :return:
        """

        # always canonicalize input SMILES string
        self.target_smiles = self._canon_smi(target_smiles)

        # remove stereochemistry if selected
        if remove_stereo:
            self.target_smiles = self._remove_stereo(target_smiles)

        # load in reaction rules for pickaxe - choose from generalized, intermediate or chemical rules
        if non_pks_rules == 'generalized':
            self.rule_filepath = '../data/coreactants_and_rules/JN1224MIN_rules.tsv'

        if non_pks_rules == 'intermediate':
            self.rule_filepath = '../data/coreactants_and_rules/JN3604IMT_rules.tsv'

        if non_pks_rules == 'chemical':
            pass

        self.known_metabolites = set(line.strip() for line in open(known_metabolites))
        self.non_pks_cofactors = non_pks_cofactors
        self.non_pks_steps = non_pks_steps
        self.non_pks_cores = non_pks_cores
        self.non_pks_sim_score_filter = non_pks_sim_score_filter
        self.non_pks_sim_score_cutoffs = non_pks_sim_score_cutoffs
        self.non_pks_sim_sample = non_pks_sim_sample
        self.non_pks_sim_sample_size = non_pks_sim_sample_size

    def _canon_smi(self,smi: str) -> str:
        """
        Canonicalize an input SMILES string
        :param smi: input SMILES string
        :return: output SMILES string in canonical form
        """
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    def _remove_stereo(self, smi: str) -> str:
        """
        Remove stereochemistry from SMILES if selected
        :param smi: input SMILES string
        :return: output SMILES string without stereochemistry
        """
        mol = Chem.MolFromSmiles(smi)
        Chem.RemoveStereochemistry(mol)
        smi_wo_stereo = Chem.MolToSmiles(mol)
        return smi_wo_stereo

    def get_pks_product(self,termination_method):
        """
        Run retrotide and get the final product from the pks modules
        :return:
        """
        print("\nStarting pks retrobiosynthesis with retrotide")
        print("---------------------------------------------")

        designs = retrotide.designPKS(Chem.MolFromSmiles(self.target_smiles))

        print('\nBest PKS design: ' + repr(designs[-1][0][0].modules))

        pks_product_sim_score = str(designs[-1][0][1])

        bound_product_mol_object = designs[-1][0][0].computeProduct(retrotide.structureDB)

        if termination_method == 'decarboxylation':
            ## Run detachment reaction via a decarboxylation
            Chem.SanitizeMol(bound_product_mol_object)
            rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])[S:3]>>[C:1](=[O:2])[O].[S:3]')
            product = rxn.RunReactants((bound_product_mol_object,))[0][0]
            Chem.SanitizeMol(product)
            self.pks_final_product = Chem.MolToSmiles(product)

        # self.pks_final_product = bound_product_smiles
        print(f"\nClosest final product is: {Chem.MolToSmiles(product)}")

        if pks_product_sim_score != "1.0":
            print(f"\nFinished pks retrobiosynthesis - closest product to the target has a similarity score of: {pks_product_sim_score}")

        if pks_product_sim_score == "1.0":
            print(f"\nFinished pks retrobiosynthesis - target reached")
            exit()

    def run_non_pks(self):
        print(f"\nStarting pickaxe expansion on {self.pks_final_product}")
        print('')

        self.pks_final_product = pickaxe_utils.canonicalize_smiles(self.pks_final_product)

        # write starting compound (PKS product) to a tsv file
        precursor_filepath = pickaxe_utils.write_cpds_to_tsv(cpd_name=self.pks_final_product, cpd_smi= self.pks_final_product)

        # write target compound (expected Pickaxe product) to a tsv file
        target_filepath = pickaxe_utils.write_cpds_to_tsv(cpd_name=self.target_smiles, cpd_smi=self.target_smiles)

        # initialize a Pickaxe object
        pk = Pickaxe(coreactant_list=self.non_pks_cofactors, rule_list=self.rule_filepath)

        # load starting compound (PKS product) and target compound (expected Pickaxe product) into Pickaxe
        pk.load_compound_set(compound_file=precursor_filepath)
        pk.load_targets(target_compound_file=target_filepath)

        # run Pickaxe for enzymatic transformations on PKS product
        pk.transform_all(generations=self.non_pks_steps, processes=self.non_pks_cores)
        pk.assign_ids()

        # create a dataframe of compounds generated by Pickaxe
        compounds_df = pickaxe_utils.create_compounds_df(pk)


        # extract non-PKS reactions from Pickaxe object
        pk_rxn_keys = [key for key in pk.reactions.keys()]

        all_pk_rxn_ids = [pk.reactions[key]['ID'] for key in pk_rxn_keys]
        all_rxn_strs_in_cpd_ids = [pk.reactions[key]['ID_rxn'] for key in pk_rxn_keys]
        all_rxn_strs_in_SMILES = [pk.reactions[key]['SMILES_rxn'] for key in pk_rxn_keys]
        all_rxn_rules = [list(pk.reactions[key]['Operators']) for key in pk_rxn_keys]

        # use extracted reactions and Pickaxe object to create a graph
        G = pickaxe_utils.create_graph(all_rxn_strs_in_cpd_ids, self.pks_final_product)

        # get and store sequences from Graph
        sequences = pickaxe_utils.get_sequences_from_graph(G,
                                                           compounds_df,
                                                           self.pks_final_product,
                                                           self.target_smiles,
                                                           self.non_pks_steps)

        # initialize a dictionary to store all sequences
        all_sequences_dict = {}

        for i, seq in enumerate(sequences):
            seq_SMILES = [list(compounds_df[compounds_df["ID"] == id]["SMILES"])[0] for id in seq]

            all_sequences_dict.update({f"seq {i}":
                                           {"seq_num": str(i),
                                            "seq (IDs)": seq,
                                            "seq (SMILES)": seq_SMILES}})

        non_pks_pathways = pickaxe_utils.get_pathways_from_graph_proto(sequences,
                                                                       self.known_metabolites,
                                                                       compounds_df,
                                                                       pk)

        print(non_pks_pathways)

if __name__ == "__main__":

    # specify parameters
    pathway_sequence = ['pks', 'non_pks'] # do retrotide first then pickaxe
    target_smiles = 'CCC(=O)O'
    remove_stereo = False # leave stereochemistry on
    cofactors_filepath = '../data/coreactants_and_rules/all_cofactors.tsv'
    known_metabolites_filepath = '../../cell_free_biosensing/data/raw/all_known_metabolites.txt'

    reported_biological_compounds = set(line.strip() for line in open(
        '../../cell_free_biosensing/data/raw/all_known_metabolites.txt'))

    non_pks_rules = 'intermediate' # intermediate enzymatic reaction rules for pickaxe
    non_pks_steps = 2
    non_pks_cores = 4
    non_pks_sim_score_filter = False
    non_pks_sim_score_cutoffs = []
    non_pks_sim_sample = False
    non_pks_sim_sample_size = None

    # create an instance of the biosynth_pipeline class
    biosynth_pipeline_object =  biosynth_pipeline(pathway_sequence = pathway_sequence,
                                                  target_smiles = target_smiles,
                                                  remove_stereo = remove_stereo,
                                                  known_metabolites = known_metabolites_filepath,
                                                  non_pks_cofactors = cofactors_filepath,
                                                  non_pks_rules = non_pks_rules,
                                                  non_pks_steps = non_pks_steps,
                                                  non_pks_cores = non_pks_cores,
                                                  non_pks_sim_score_filter = non_pks_sim_score_filter,
                                                  non_pks_sim_score_cutoffs = non_pks_sim_score_cutoffs,
                                                  non_pks_sim_sample = non_pks_sim_sample,
                                                  non_pks_sim_sample_size = non_pks_sim_sample_size)

    biosynth_pipeline_object.get_pks_product(termination_method='decarboxylation')

    biosynth_pipeline_object.run_non_pks()
