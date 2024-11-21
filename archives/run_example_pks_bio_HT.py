from biosynth_pipeline import biosynth_pipeline
import pickle

### User-defined parameters
pathway_sequence = ['pks', 'non_pks']  # run retrotide first then pickaxe
target_smiles = 'CC=O'
target_name = 'test'
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

### ----- Start synthesis -----
biosynth_pipeline_object.run_combined_synthesis(max_designs = 4)
print(biosynth_pipeline_object.results_logs)

with open(f'../data/hybrid_pathways_analysis/{target_name}_pks_plus_bio.pkl', 'wb') as f:
    pickle.dump(biosynth_pipeline_object.results_logs, f)

# Define the filename for the output .txt file
output_filename = f'../data/hybrid_pathways_analysis/{target_name}_pks_plus_bio_designs.txt'

# Write the list of dictionaries to a .txt file in a readable format
with open(output_filename, 'w') as file:
    for design in biosynth_pipeline_object.results_logs:
        file.write(f"PKS Design Number: {design['pks_design_num']}\n")
        file.write(f"PKS Design: {design['pks_design']}\n")
        file.write(f"PKS Product: {design['pks_product']}\n")
        try:
            file.write(f"PKS Product Similarity: {design['pks_product_similarity']:.2f}\n")
            file.write(f"Non-PKS Product: {design['non_pks_product']}\n")
            file.write(f"Non-PKS Product Similarity: {design['non_pks_product_similarity']:.2f}\n")
            file.write("\n")  # Add an empty line between designs
        except KeyError:
            pass

        try:
            non_pks_pathways = design['non_pks_pathways']
            non_pks_pathways = dict(sorted(non_pks_pathways.items(),
                                            key=lambda item: item[1]['net feasibility'],
                                            reverse=True))

            file.write(f"Non-PKS pathways:\n")
            for pathway_num in non_pks_pathways:
                reactions = non_pks_pathways[pathway_num]['reactions (SMILES)']
                reaction_rules = non_pks_pathways[pathway_num]['reaction rules']
                feasibilities = non_pks_pathways[pathway_num]['feasibilities']
                net_feasibility = non_pks_pathways[pathway_num]['net feasibility']
                file.write(f'     Reactions: {reactions}\n')
                file.write(f'     Reaction rules: {reaction_rules}\n')
                file.write(f'     Feasibilities: {feasibilities}\n')
                file.write(f'     Net feasibility: {net_feasibility}\n')
                file.write('\n')

        except KeyError:
            pass

        except AttributeError:
            pass
