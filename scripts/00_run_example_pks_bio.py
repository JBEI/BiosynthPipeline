from biosynth_pipeline import biosynth_pipeline
from DORA_XGB import DORA_XGB
import os
import warnings
warnings.simplefilter('ignore')

dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'

### User-defined parameters
pathway_sequence = ['pks','bio']  # choose between ['pks'] or ['pks','bio']
target_smiles = 'CCC=O'
target_name = 'propionaldehyde'
pks_release_mechanism = 'thiolysis' # choose from 'cyclization' or 'thiolysis'

config_filepath = 'input_config_file.json'

post_pks_rxn_model = DORA_XGB.feasibility_classifier(cofactor_positioning = 'add_concat',
                                                     model_type = "spare")

### Create an object that is an instance of Biosynth Pipeline
biosynth_pipeline_object = biosynth_pipeline.biosynth_pipeline(
                                             pathway_sequence = pathway_sequence,
                                             target_smiles = target_smiles,
                                             target_name = target_name,
                                             feasibility_classifier = post_pks_rxn_model,
                                             pks_release_mechanism = pks_release_mechanism,
                                             config_filepath = config_filepath)

### ----- Start synthesis -----
if __name__ == "__main__":
    biosynth_pipeline_object.run_combined_synthesis(max_designs = 4)
    biosynth_pipeline_object.save_results_logs()
