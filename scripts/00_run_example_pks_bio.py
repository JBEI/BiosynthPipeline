from biosynth_pipeline import biosynth_pipeline
import os
import pickle
import warnings

warnings.simplefilter('ignore')

dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'

### User-defined parameters
pathway_sequence = ['pks','bio']  # choose between ['pks'] or ['pks','bio']
target_smiles = 'CCCCCCC'
target_name = 'heptane'
pks_release_mechanism = 'thiolysis' # choose from 'cyclization' or 'thiolysis'
feasibility_cofactors = '../data/coreactants_and_rules/all_cofactors_updated.csv'

config_filepath = 'input_config_file.json'

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
                                             config_filepath = config_filepath)

### ----- Start synthesis -----
if __name__ == "__main__":
    biosynth_pipeline_object.run_combined_synthesis(max_designs = 4)
    biosynth_pipeline_object.save_results_logs()

    with open(dir_path + f'../data/hybrid_pathways_analysis/{target_name}_pks_plus_bio.pkl', 'wb') as f:
        pickle.dump(biosynth_pipeline_object.results_logs, f)
