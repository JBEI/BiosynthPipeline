import requests


def get_kegg_smiles(compound_ids):
    """
    Query KEGG API for given compound IDs and return their SMILES strings.

    :param compound_ids: List of KEGG compound IDs (e.g., ['C00002', 'C00003']).
    :return: Dictionary of compound IDs and their SMILES strings.
    """
    smiles_dict = {}
    base_url = "http://rest.kegg.jp/get/cpd:"

    for cid in compound_ids:
        url = f"{base_url}{cid}"
        response = requests.get(url)
        if response.status_code == 200:
            # Split the response into lines and iterate through them
            for line in response.text.split('\n'):
                print(line)
                # Check if the line contains the SMILES string
                if line.startswith("            "):  # SMILES lines start with spaces
                    # Extract the SMILES string, which follows ' '
                    smiles = line.strip()
                    smiles_dict[cid] = smiles
                    break  # Stop searching once the SMILES is found
        else:
            print(f"Failed to fetch data for compound ID {cid}")

    return smiles_dict


# Example usage
compound_ids = ['C00002', 'C00003']
smiles_dict = get_kegg_smiles(compound_ids)
for cid, smiles in smiles_dict.items():
    print(f"Compound ID: {cid}, SMILES: {smiles}")
