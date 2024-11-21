import sys
import joblib
import pickle
sys.path.append('../src/biosynth_pipeline/featurizations')
from src.biosynth_pipeline.featurizations import featurizations


class feasibility_classifier:
    """
    Wrapper for enzymatic reaction feasibility prediction via an XGBoost classification model
    """

    def __init__(
        self,
        feasibility_model_path: str,
        calibration_model_path: str,
        cofactors_path: str,
        fp_type: str,
        nBits: int,
        max_species: int,
        cofactor_positioning: str
    ):
        """
        Initialize feasibility classifier
        :param feasibility_model_path: filepath to feasibility XGBoost model
        :param calibration_model_path: filepath to an isotonic regression model that calibrates probabilities
        :param cofactors_path: filepath to cofactors
        """
        self.feasibility_model = joblib.load(open(feasibility_model_path,'rb'))
        self.calibration_model = pickle.load(open(calibration_model_path,'rb'))
        dummy_rxn_str = ""
        dummy_rxn_object = featurizations.reaction(dummy_rxn_str)
        self.cofactors_set_wo_stereo = dummy_rxn_object.print_full_cofactors_set_wo_stereo(cofactors_path)

        self.fp_type = fp_type
        self.nBits = nBits
        self.max_species = max_species
        self.cofactor_positioning = cofactor_positioning

    def predict_proba(self, rxn_str):

        # initialize a reaction object with custom-built featurizations package
        rxn_object = featurizations.reaction(rxn_str)

        # convert reaction string to a reaction fingerprint
        # substrates go first followed by cofactors on the LHS in order of descending MW for positions 1-4
        # same with products and cofactors on the RHS for positions 5-8
        rxn_fp = rxn_object.rxn_2_fp_w_positioning(fp_type = self.fp_type,
                                                   nBits = self.nBits,
                                                   is_folded = True,
                                                   dim = None,
                                                   max_species = self.max_species,
                                                   cofactor_positioning = self.cofactor_positioning,
                                                   reaction_rule = None,
                                                   all_cofactors_wo_stereo = self.cofactors_set_wo_stereo)

        rxn_fp = rxn_fp.reshape(1, -1)  # reshape since only single sample
        rxn_feasib_score = self.calibration_model.predict(self.feasibility_model.predict_proba(rxn_fp).T)

        return rxn_feasib_score[1]
