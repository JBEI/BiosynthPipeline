from biosynth_pipeline import biosynth_pipeline
from rdkit import Chem

def test_gluconate():
    ### User-defined parameters
    pathway_sequence = ['pks']  # choose between ['pks'] or ['pks','bio']
    target_smiles = 'O=C(O)C(O)C(O)C(O)C(O)CO'
    target_name = 'D-gluconate'
    pks_release_mechanism = 'thiolysis' # choose from 'cyclization' or 'thiolysis'
    feasibility_cofactors = '../data/coreactants_and_rules/all_cofactors_updated.csv'

    ### Create an object that is an instance of the feasibility classification model
    PX = biosynth_pipeline.feasibility_classifier(ML_model_type = 'add_concat',
                                                  cofactors_path = feasibility_cofactors)

    ### Create an object that is an instance of Biosynth Pipeline
    biosynth_pipeline_object = biosynth_pipeline.biosynth_pipeline(
                                                 pathway_sequence = pathway_sequence,
                                                 target_smiles = target_smiles,
                                                 target_name = target_name,
                                                 feasibility_classifier = PX,
                                                 pks_release_mechanism = pks_release_mechanism,
                                                 config_filepath = f'{target_name}_input_config.json')

    biosynth_pipeline_object.run_combined_synthesis(max_designs = 4)
    biosynth_pipeline_object.save_results_logs()

    with open(f'./test_molecules/{target_name}_results/{target_name}_PKS_only.txt', 'r') as file:
        mol = Chem.MolFromSmiles(target_smiles)
        canonical_smiles = Chem.MolToSmiles(mol)

        counter = 0
        for line in file:
            if 'product similarity: 1.0' in line or f'product: {canonical_smiles}' in line:
                counter +=1

        assert counter >= 1