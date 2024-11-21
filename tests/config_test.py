import json
from biosynth_pipeline import biosynth_pipeline
import os
import pickle
import warnings
from rdkit import Chem

warnings.simplefilter('ignore')

dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'
file_path = ('./D-gluconate_input_config.json')

with open(file_path, 'r') as file:
    data = json.load(file)

def test_typedata():
    assert isinstance(data, dict)

def test_empty():
    assert data is not None

def test_lendata():
    assert len(data) == 20

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
                                             config_filepath ='D-gluconate_input_config.json')
def test_config_type():
    assert isinstance(biosynth_pipeline_object.config_filepath, str)

def test_valid_smiles():
    mol = Chem.MolFromSmiles(biosynth_pipeline_object.target_smiles)
    assert mol is not None

def test_pks_starters_filepath():
    assert biosynth_pipeline_object.pks_starters_filepath.endswith('.smi')

def test_pks_extenders_filepath():
    assert biosynth_pipeline_object.pks_extenders_filepath.endswith('.smi')

def test_pks_starters():
    if isinstance(biosynth_pipeline_object.pks_starters, list):
        assert all(isinstance(item, str) for item in biosynth_pipeline_object.pks_starters)
    else:
        assert biosynth_pipeline_object.pks_starters == 'all'

def test_pks_extenders():
    if isinstance(biosynth_pipeline_object.pks_extenders, list):
        assert all(isinstance(elem, str) for elem in biosynth_pipeline_object.pks_extenders)
    else:
        assert biosynth_pipeline_object.pks_extenders == 'all'

def test_metrics():
    assert isinstance(biosynth_pipeline_object.pks_similarity_metric, str)
    assert isinstance(biosynth_pipeline_object.non_pks_similarity_metric, str)

def test_known_metabolites():
    assert biosynth_pipeline_object.known_metabolites_filepath.endswith('.txt')

def test_non_pks_cofactors():
    assert biosynth_pipeline_object.non_pks_cofactors_filepath.endswith('.tsv')

def test_input_cpd_dir():
    assert biosynth_pipeline_object.input_cpd_dir.endswith('/')

def test_feasibility_cofactors():
    assert feasibility_cofactors.endswith('.csv')

def test_stereo():
    assert isinstance(biosynth_pipeline_object.consider_target_stereo, bool)

def test_non_pks_rules():
    assert isinstance(biosynth_pipeline_object.non_pks_rules, str)
    assert (biosynth_pipeline_object.non_pks_rules == 'biological_intermediate' or
            biosynth_pipeline_object.non_pks_rules == 'biological_generalized' or
            biosynth_pipeline_object.non_pks_rules == 'intermediate_non_dimerization')

def test_score_filter_and_cutoffs():
    assert isinstance(biosynth_pipeline_object.non_pks_sim_score_filter, bool)
    if biosynth_pipeline_object.non_pks_sim_score_filter:
        assert isinstance(biosynth_pipeline_object.non_pks_sim_score_cutoffs, list)
        assert all(isinstance(elem, float) for elem in biosynth_pipeline_object.non_pks_sim_score_cutoffs)
    else:
        assert biosynth_pipeline_object.non_pks_sim_score_cutoffs == []

def test_non_pks_sim_sample():
    assert isinstance(biosynth_pipeline_object.non_pks_sim_sample, bool)
def test_steps_cores_sample_size():
    assert isinstance(biosynth_pipeline_object.non_pks_steps, int)
    assert isinstance(biosynth_pipeline_object.non_pks_cores, int)
    assert isinstance(biosynth_pipeline_object.non_pks_sim_sample_size, int)

def test_stopping_criteria():
    assert isinstance(biosynth_pipeline_object.stopping_criteria, str)
    assert biosynth_pipeline_object.stopping_criteria == 'first_product_formation'