from biosynth_pipeline import biosynth_pipeline
import retrotide

pathway_sequence = ['pks', 'non_pks']  # do retrotide first then pickaxe
target_smiles = 'O=C(C)CCC' # 2-pentanone
remove_stereo = False

cofactors_filepath = '../data/coreactants_and_rules/all_cofactors.tsv'
known_metabolites_filepath = '../data/all_known_metabolites.txt'
input_cpd_dir = '../data/coreactants_and_rules/'
feasibility_model = '../models/updated_model_Apr28'
feasibility_calibration_model = '../models/updated_model_Apr28_calibration'
feasibility_cofactors = '../data/coreactants_and_rules/all_cofactors_updated.csv'
fp_type = 'ecfp4'
nBits = 2048
max_species = 4
cofactor_positioning = 'by_descending_MW'

PX = biosynth_pipeline.feasibility_classifier(feasibility_model_path = feasibility_model,
                            calibration_model_path = feasibility_calibration_model,
                        cofactors_path = feasibility_cofactors,
                        fp_type = fp_type,
                        nBits = nBits,
                        max_species = max_species,
                        cofactor_positioning = cofactor_positioning)

non_pks_rules = 'biological_intermediate'  # intermediate enzymatic reaction rules for pickaxe (can choose chemical too)
non_pks_steps = 1
non_pks_cores = 4
non_pks_sim_score_filter = False
non_pks_sim_score_cutoffs = []
non_pks_sim_sample = False
non_pks_sim_sample_size = None

if __name__ == "__main__":

    # create an instance of the biosynth_pipeline class
    biosynth_pipeline_object = biosynth_pipeline.biosynth_pipeline(pathway_sequence = pathway_sequence,
                                             target_smiles = target_smiles,
                                             feasibility_classifier = PX,
                                             remove_stereo = remove_stereo,
                                             known_metabolites = known_metabolites_filepath,
                                             input_cpd_dir = input_cpd_dir,
                                             non_pks_cofactors = cofactors_filepath,
                                             non_pks_rules = non_pks_rules,
                                             non_pks_steps = non_pks_steps,
                                             non_pks_cores = non_pks_cores,
                                             non_pks_sim_score_filter = non_pks_sim_score_filter,
                                             non_pks_sim_score_cutoffs = non_pks_sim_score_cutoffs,
                                             non_pks_sim_sample = non_pks_sim_sample,
                                             non_pks_sim_sample_size = non_pks_sim_sample_size)

    # ----- Start synthesis -----
    biosynth_pipeline_object.run_pks_synthesis(pks_release_mechanism = 'thiolysis')
    non_pks_pathways = biosynth_pipeline_object.run_non_pks_synthesis_post_pks(max_designs=5)
    non_pks_pathways = biosynth_pipeline_object.rank_non_pks_pathways(non_pks_pathways)