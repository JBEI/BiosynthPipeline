"""
This script processes the LASER database of reported metabolic pathways
This database was downloaded from https://bitbucket.org/jdwinkler/laser_release/downloads/
And the publication on the LASER database can be found at https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5779719/
"""

import glob
import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

### Helper functions to process the LASER metabolic database
def get_smiles_from_pubchem(compound_name):
    # Base URL for the PubChem Compound API
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    # Construct the URL for querying by compound name
    query_url = f"{base_url}/compound/name/{compound_name}/property/CanonicalSMILES/JSON"

    try:
        # Send the GET request
        response = requests.get(query_url)
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            # Extract the SMILES string
            smiles = data.get('PropertyTable', {}).get('Properties', [])[0].get('CanonicalSMILES', None)
            return smiles
        else:
            # If response is not successful, return None
            return None
    except Exception as e:
        # If there is any error (e.g., network error), print the error and return None
        print(f"An error occurred: {e}")
        return None

def neutralize_atoms(mol):
    """Neutralize charged atoms in a molecule and return the neutralized molecule."""
    # Patterns for positively and negatively charged atoms
    patt_pos = Chem.MolFromSmarts('[NH4+,NH3+,NH2+,NH+,N+,OH2+,OH+,O+]')
    patt_neg = Chem.MolFromSmarts('[O-,OH-]')

    # Create a copy of the molecule
    new_mol = Chem.Mol(mol)

    # Neutralize positive charges
    while new_mol.HasSubstructMatch(patt_pos):
        for atom in new_mol.GetAtoms():
            if atom.GetFormalCharge() > 0:
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(max(atom.GetTotalNumHs() - 1, 0))  # Reduce hydrogen count if positive

    # Neutralize negative charges
    while new_mol.HasSubstructMatch(patt_neg):
        for atom in new_mol.GetAtoms():
            if atom.GetFormalCharge() < 0:
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)  # Add hydrogen if negative

    AllChem.SanitizeMol(new_mol)
    return new_mol

def neutralize_smiles(smiles):
    """Convert a SMILES string to a neutralized SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        neutral_mol = neutralize_atoms(mol)
        return Chem.MolToSmiles(neutral_mol, isomericSmiles=True)
    else:
        return None

def process_smiles(smiles):
    neutralized_smiles = neutralize_smiles(smiles)
    canonicalized_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(neutralized_smiles))

    return neutralized_smiles

pathway_records_filepaths = glob.glob('../data/LASER_db_2015_release/*.txt')

target_names_list = [] # initialize a list to store names of target compounds
DOI_list = [] # initialize a list to store DOIs of papers reporting biosynthesis of target compounds

for record_filepath in pathway_records_filepaths:
    with open(record_filepath, 'r') as record:
        for line in record:
            if 'Mutant1.TargetMolecule' in line.strip():
                target_name = line.split(' = ')[1]
                target_name = target_name[1:-2] # remove front and back quotation marks
                target_names_list.append(target_name)

            if 'DOI ' in line.strip():
                DOI = line.split(' = ')[1]
                DOI = DOI[1:-2] # remove front and back quotation marks
                DOI_list.append(DOI)

target_SMILES_list = [] # initialize a list to store SMILES strings of target compounds

for target_name in target_names_list:
    smiles_from_pubchem = get_smiles_from_pubchem(target_name)
    if smiles_from_pubchem:
        processed_smiles = process_smiles(smiles_from_pubchem)
        target_SMILES_list.append(processed_smiles)
    else:
        target_SMILES_list.append(None)

targets_df = pd.DataFrame({'DOI': DOI_list,
                           'target name': target_names_list,
                           'target SMILES': target_SMILES_list})

# remove entries for which the SMILES strings are duplicates (i.e. identical target compounds)
targets_df = targets_df.drop_duplicates(subset='target SMILES', keep='first')

targets_df.to_csv('../data/LASER_db_targets.csv')

