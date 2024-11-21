from archives.utils import utils
import pandas as pd


def test_canonicalizing_smiles_01(smiles="OCC"):
    assert utils.canonicalize_smiles(smiles) == "CCO"


def test_canonicalizing_smiles_02(smiles="OCCC"):
    assert utils.canonicalize_smiles(smiles) == "CCCO"


def test_canonicalizing_smiles_03(smiles="C(CO)C(CO)O"):
    assert utils.canonicalize_smiles(smiles) == "OCCC(O)CO"


def test_searching_biological_dbs_01(smiles="O=C=O"):
    biological_compounds = set(
        line.strip() for line in open("../../cell_free_biosensing/data/raw/all_known_metabolites.txt")
    )
    assert (
            utils.search_SMILES_in_biological_databases(smiles, biological_compounds)
            == True
    )


def test_searching_biological_dbs_02(smiles="Cc1ncc(COP(=O)(O)O)c(C)c1O"):
    biological_compounds = set(
        line.strip() for line in open("../../cell_free_biosensing/data/raw/all_known_metabolites.txt")
    )
    assert (
            utils.search_SMILES_in_biological_databases(smiles, biological_compounds)
            == True
    )


def test_searching_biological_dbs_03(smiles="CCCCCCCCCCCCCC(=O)NC(CCO)C(=O)O"):
    biological_compounds = set(
        line.strip() for line in open("../../cell_free_biosensing/data/raw/all_known_metabolites.txt")
    )
    assert (
            utils.search_SMILES_in_biological_databases(smiles, biological_compounds)
            == True
    )


def test_picking_rules_01(
    rules_type="generalized", rules_range=None, specific_rule=None
):
    filepath = utils.pick_rules(
        rules_type=rules_type, rules_range=rules_range, specific_rule=specific_rule
    )
    assert filepath == "../data/coreactants_and_rules/JN1224MIN_rules.tsv"


def test_picking_rules_02(
    rules_type="intermediate", rules_range=None, specific_rule=None
):
    filepath = utils.pick_rules(
        rules_type=rules_type, rules_range=rules_range, specific_rule=specific_rule
    )
    assert filepath == "../data/coreactants_and_rules/JN3604IMT_rules.tsv"


def test_picking_rules_03(
    rules_type="generalized", rules_range=100, specific_rule=None
):
    filepath = utils.pick_rules(
        rules_type=rules_type, rules_range=rules_range, specific_rule=specific_rule
    )
    assert filepath == "../data/coreactants_and_rules/input_rules.tsv"

    df = pd.read_csv(filepath, delimiter="\t")

    assert df.iloc[0, :]["Name"] == "rule0001"

    assert df.iloc[-1, :]["Name"] == "rule0100"


def test_picking_rules_04(
    rules_type="intermediate", rules_range=90, specific_rule=None
):
    filepath = utils.pick_rules(
        rules_type=rules_type, rules_range=rules_range, specific_rule=specific_rule
    )
    assert filepath == "../data/coreactants_and_rules/input_rules.tsv"

    df = pd.read_csv(filepath, delimiter="\t")

    assert df.iloc[0, :]["Name"] == "rule0001_01"

    assert df.iloc[-1, :]["Name"] == "rule0001_90"


def test_picking_rules_05(
    rules_type="generalized", rules_range=None, specific_rule="rule0002"
):
    filepath = utils.pick_rules(
        rules_type=rules_type, rules_range=rules_range, specific_rule=specific_rule
    )
    assert filepath == "../data/coreactants_and_rules/input_rules.tsv"

    df = pd.read_csv(filepath, delimiter="\t")

    assert df.iloc[0, :]["Name"] == "rule0002"

    assert df.shape[0] == 1


def test_picking_rules_06(
    rules_type="intermediate", rules_range=None, specific_rule="rule0001_80"
):
    filepath = utils.pick_rules(
        rules_type=rules_type, rules_range=rules_range, specific_rule=specific_rule
    )
    assert filepath == "../data/coreactants_and_rules/input_rules.tsv"

    df = pd.read_csv(filepath, delimiter="\t")

    assert df.iloc[0, :]["Name"] == "rule0001_80"

    assert df.shape[0] == 1
