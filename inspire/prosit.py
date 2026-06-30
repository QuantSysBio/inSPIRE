""" Code for running Prosit models on CPU.
    Large sections based on https://github.com/kusterlab/prosit.
"""

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers, constraints, initializers
from tensorflow.keras.layers import Layer # pylint: disable=no-name-in-module
import yaml

from inspire.constants import (
    MAX_CHARGE,
    MAX_FRAG_CHARGE,
    MAX_SEQ_LEN,
    N_ION_TYPES,
    N_LOSSES,
    PROSIT_ALPHABET,
    PROSIT_MASK_VALUE,
    PROSIT_PRED_BATCH_SIZE,
    PROSIT_UNMOD_ALPHA_S,
)

PROSIT_IONS = np.array(
    [
        'y1', 'y1^2)', 'y1^3)', 'b1', 'b1^2)', 'b1^3)',
        'y2', 'y2^2)', 'y2^3)', 'b2', 'b2^2)', 'b2^3)',
        'y3', 'y3^2)', 'y3^3)', 'b3', 'b3^2)', 'b3^3)',
        'y4', 'y4^2)', 'y4^3)', 'b4', 'b4^2)', 'b4^3)',
        'y5', 'y5^2)', 'y5^3)', 'b5', 'b5^2)', 'b5^3)',
        'y6', 'y6^2)', 'y6^3)', 'b6', 'b6^2)', 'b6^3)',
        'y7', 'y7^2)', 'y7^3)', 'b7', 'b7^2)', 'b7^3)',
        'y8', 'y8^2)', 'y8^3)', 'b8', 'b8^2)', 'b8^3)',
        'y9', 'y9^2)', 'y9^3)', 'b9', 'b9^2)', 'b9^3)',
        'y10', 'y10^2', 'y10^3', 'b10', 'b10^2', 'b10^3',
        'y11', 'y11^2', 'y11^3', 'b11', 'b11^2', 'b11^3',
        'y12', 'y12^2', 'y12^3', 'b12', 'b12^2', 'b12^3',
        'y13', 'y13^2', 'y13^3', 'b13', 'b13^2', 'b13^3',
        'y14', 'y14^2', 'y14^3', 'b14', 'b14^2', 'b14^3',
        'y15', 'y15^2', 'y15^3', 'b15', 'b15^2', 'b15^3',
        'y16', 'y16^2', 'y16^3', 'b16', 'b16^2', 'b16^3',
        'y17', 'y17^2', 'y17^3', 'b17', 'b17^2', 'b17^3',
        'y18', 'y18^2', 'y18^3', 'b18', 'b18^2', 'b18^3',
        'y19', 'y19^2', 'y19^3', 'b19', 'b19^2', 'b19^3',
        'y20', 'y20^2', 'y20^3', 'b20', 'b20^2', 'b20^3',
        'y21', 'y21^2', 'y21^3', 'b21', 'b21^2', 'b21^3',
        'y22', 'y22^2', 'y22^3', 'b22', 'b22^2', 'b22^3',
        'y23', 'y23^2', 'y23^3', 'b23', 'b23^2', 'b23^3',
        'y24', 'y24^2', 'y24^3', 'b24', 'b24^2', 'b24^3',
        'y25', 'y25^2', 'y25^3', 'b25', 'b25^2', 'b25^3',
        'y26', 'y26^2', 'y26^3', 'b26', 'b26^2', 'b26^3',
        'y27', 'y27^2', 'y27^3', 'b27', 'b27^2', 'b27^3',
        'y28', 'y28^2', 'y28^3', 'b28', 'b28^2', 'b28^3',
        'y29', 'y29^2', 'y29^3', 'b29', 'b29^2', 'b29^3',
    ],
    dtype='str',
)

class PrositAttention(Layer):
    """ The modified attention layer used by Prosit.
    """
    def __init__(
        self,
        context=False,
        w_regularizer=None,
        b_regularizer=None,
        u_regularizer=None,
        w_constraint=None,
        b_constraint=None,
        u_constraint=None,
        bias=True,
        **kwargs
    ):
        self.supports_masking = True
        self.init = initializers.get("glorot_uniform")
        self.w_regularizer = regularizers.get(w_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.w_constraint = constraints.get(w_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.bias = bias
        self.context = context
        self.b = None
        self.w = None
        self.u = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.w = self.add_weight(
            shape=(input_shape[-1],),
            initializer=self.init,
            name=f'{self.name}_W',
            regularizer=self.w_regularizer,
            constraint=self.w_constraint,
        )
        if self.bias:
            self.b = self.add_weight(
                shape=(input_shape[1],),
                initializer="zero",
                name=f'{self.name}_b',
                regularizer=self.b_regularizer,
                constraint=self.b_constraint,
            )
        else:
            self.b = None
        if self.context:
            self.u = self.add_weight(
                shape=(input_shape[-1],),
                initializer=self.init,
                name=f'{self.name}_u',
                regularizer=self.u_regularizer,
                constraint=self.u_constraint,
            )

        self.built = True

    def call(self, inputs, *args, **kwargs):
        """ Function to execute the PrositAttention layer.
        """
        processed_tensor = K.squeeze(K.dot(inputs, K.expand_dims(self.w)), axis=-1)
        if self.bias:
            processed_tensor += self.b
        processed_tensor = K.tanh(processed_tensor)
        if self.context:
            processed_tensor = K.squeeze(K.dot(inputs, K.expand_dims(self.u)), axis=-1)
        processed_tensor = K.exp(processed_tensor)

        processed_tensor /= K.cast(
            K.sum(processed_tensor, axis=1, keepdims=True) + K.epsilon(), K.floatx()
        )
        processed_tensor = K.expand_dims(processed_tensor)
        weighted_input = inputs * processed_tensor

        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = {
            "bias": self.bias,
            "context": self.context,
            "w_regularizer": regularizers.serialize(self.w_regularizer),
            "b_regularizer": regularizers.serialize(self.b_regularizer),
            "u_regularizer": regularizers.serialize(self.u_regularizer),
            "w_constraint": constraints.serialize(self.w_constraint),
            "b_constraint": constraints.serialize(self.b_constraint),
            "u_constraint": constraints.serialize(self.u_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

def load_model(config_loc, model_loc, weights_loc):
    """ Function to load a Prosit model from disk.

    Parameters
    ----------
    config_loc : str
        The location of the model config file.
    model_loc : str
        The location of the model file.
    weights_loc : str
        The location of the model weights file.

    Returns
    -------
    model_dict : dict
        The loaded model dictionary.
    """
    with open(config_loc, 'r', encoding='UTF-8') as stream:
        config_dict = yaml.safe_load(stream)

    with open(model_loc, 'r', encoding='UTF-8') as model_file:
        model = tf.keras.models.model_from_yaml(
            model_file.read(), custom_objects={"Attention": PrositAttention}
        )
    model.load_weights(weights_loc)

    model_dict = {}
    model_dict["graph"] = tf.Graph()

    model_dict["session"] = tf.compat.v1.Session()
    with model_dict["session"].as_default():
        model_dict["model"], model_dict["config"] = model, config_dict
        model_dict["model"].compile(optimizer="adam", loss="mse")

    return model_dict

def get_precursor_charge_onehot(charges):
    """ Function to get one hot array from precursor charge.

    Parameters
    ----------
    charges : np.array of int
        The charge states observed.

    Return
    ------
    charge_one_hot : np.array of np.array
        Array of the one hot encoded charge states.
    """
    charge_one_hot = np.zeros([len(charges), MAX_CHARGE], dtype=int)
    for i, precursor_charge in enumerate(charges):
        charge_one_hot[i, precursor_charge - 1] = 1
    return charge_one_hot


def peptide_parser(peptide):
    """ Function to yield residues possible with modification.

    Parameters
    ----------
    peptide : str
        The input peptide.

    Yields
    ------
    residue : str
        Either a single character residue or M(ox).
    """
    pep_len = len(peptide)
    i = 0
    while i < pep_len:
        if i < pep_len - 3 and peptide[i + 1] == '(':
            j = peptide[i + 2 :].index(')')
            offset = i + j + 3
            yield PROSIT_ALPHABET[peptide[i:offset]]
            i = offset
        else:
            yield PROSIT_ALPHABET[peptide[i]]
            i += 1

def get_sequence_integer(sequences):
    """ Function to create np arrays of integers from peptides using the Prosit alphabet.

    Parameters
    ----------
    sequences : list of str
        The peptide sequences.

    Returns
    -------
    sequence_integers : np.array
        The encoded arrays.
    """
    sequence_integers = np.zeros([len(sequences), MAX_SEQ_LEN], dtype=int)
    for seq_idx, sequence in enumerate(sequences):
        for res_idx, residue in enumerate(peptide_parser(sequence)):
            sequence_integers[seq_idx, res_idx] = residue
    return sequence_integers


def process_csv_file(input_loc):
    """ Function to process csv file for Prosit input.

    Parameters
    ----------
    input_loc : str
        The location of the Prosit input csv file.

    Returns
    -------
    input_df : pd.DataFrame
        The DataFrame read in from the csv file.
    data : dict
        Dictionary of the Prosit input arrays.
    """
    input_df = pd.read_csv(input_loc)
    data = {
        'collision_energy_aligned_normed': (
            np.expand_dims(np.array(input_df['collision_energy']).astype(float), axis=1) / 100.0
        ),
        'sequence_integer': get_sequence_integer(input_df['modified_sequence']),
        'precursor_charge_onehot': get_precursor_charge_onehot(input_df['precursor_charge']),
    }

    return input_df, data

def sanitize(data):
    """ Function to sanitize Prosit predicted MS2 spectra.

    Parameters
    ----------
    data : dict
        The output of the Prosit model.

    Returns
    -------
    data : dict
        The Prosit output after cleaning and normalising the spectrum.
    """
    sequence_lengths = np.count_nonzero(data["sequence_integer"], axis=1)
    intensities = data["intensities_pred"]
    charges = list(data["precursor_charge_onehot"].argmax(axis=1) + 1)

    intensities[intensities < 0] = 0
    maxima = intensities.max(axis=1)
    intensities /= maxima[:, np.newaxis]

    intensities = intensities.reshape(
        [intensities.shape[0], MAX_SEQ_LEN - 1, N_ION_TYPES, N_LOSSES, MAX_FRAG_CHARGE]
    )

    for i in range(intensities.shape[0]):
        intensities[i, sequence_lengths[i] - 1 :, :, :, :] = PROSIT_MASK_VALUE
    for i in range(intensities.shape[0]):
        if charges[i] < 3:
            intensities[i, :, :, :, charges[i] :] = PROSIT_MASK_VALUE

    flat_dim = [
        intensities.shape[0],
        (MAX_SEQ_LEN - 1)*N_ION_TYPES*N_LOSSES*MAX_FRAG_CHARGE,
    ]
    intensities = intensities.reshape(flat_dim)
    data["intensities_pred"] = intensities

    return data


def prosit_predict(data, d_model):
    """ Function to predict MS2 spectra or iRT using a Prosit model.

    Parameters
    ----------
    data : dict
        A dictionary containing modified_sequence and precursor_charge keys as well as
        collision_energy if MS2 spectrum is being predicted.
    d_model : dict
        A dictionary containing the model details.

    Return
    ------
    data : dict
        The input dictionary updated to contain the prediction data.
    """
    # check for mandatory keys
    input_data = [data[key] for key in d_model["config"]["x"]]

    keras.backend.set_session(d_model["session"])

    prediction = d_model["model"].predict(
        input_data, verbose=True, batch_size=PROSIT_PRED_BATCH_SIZE
    )

    if d_model["config"]["prediction_type"] == "intensity":
        data["intensities_pred"] = prediction
        data = sanitize(data)
    elif d_model["config"]["prediction_type"] == "iRT":
        scal = float(d_model["config"]["iRT_rescaling_var"])
        mean = float(d_model["config"]["iRT_rescaling_mean"])
        data["iRT"] = prediction * np.sqrt(scal) + mean
    else:
        raise ValueError("model_config misses parameter")

    return data


def generate_mods_string_tuples(sequence_integer):
    """ Function to generate tuples of the modifications in a sequence.

    Parameters
    ----------
    sequence_integer : np.array of int
        The peptide sequence encoded using the Prosit alphabet.

    Returns
    -------
    list_mods : list of tuples
        A list of the modifications stored as tuples containing location and identity.
    """
    list_mods = []
    for mod in [PROSIT_ALPHABET['M(ox)'], PROSIT_ALPHABET['C']]:
        for position in np.where(sequence_integer == mod)[0]:
            if mod == PROSIT_ALPHABET['C']:
                list_mods.append((position + 1, "C", "Carbamidomethyl"))
            elif mod == PROSIT_ALPHABET['M(ox)']:
                list_mods.append((position + 1, "M", "Oxidation"))
            else:
                raise ValueError(f'Modification ID {mod} not allowed.')
    list_mods.sort(key=lambda tup: tup[0])

    return list_mods


def generate_mod_strings(sequence_integer):
    """
    >>> x = np.array([1,2,3,1,2,21,0])
    >>> y, z = generate_mod_strings(x)
    >>> y
    '3/1,C,Carbamidomethyl/4,C,Carbamidomethyl/5,M,Oxidation'
    >>> z
    'Carbamidomethyl@C2; Carbamidomethyl@C5; Oxidation@M6'
    """
    list_mods = generate_mods_string_tuples(sequence_integer)
    if len(list_mods) == 0:
        return '0', ''
    indexed_mods_string = ''
    at_mods_string = ''
    indexed_mods_string += str(len(list_mods))
    for i, mod_tuple in enumerate(list_mods):
        indexed_mods_string += (
            '/' + str(mod_tuple[0] + 1) + ',' + mod_tuple[1] + ',' + mod_tuple[2]
        )
        if i == 0:
            at_mods_string += (
                mod_tuple[2] + '@' + mod_tuple[1] + str(mod_tuple[0] + 1)
            )
        else:
            at_mods_string += (
                '; ' + mod_tuple[2] + '@' + mod_tuple[1] + str(mod_tuple[0] + 1)
            )

    return indexed_mods_string, at_mods_string


def format_msp_spectrum(
        pred_intes,
        collision_energy,
        pred_irt,
        precursor_charge,
        sequence_integer,
        pred_ion_names,
    ):
    """ Function to correctly format an MSP spectrum from Prosit predictions.

    Parameters
    ----------
    pred_intes : np.array
        The predicted intensities from Prosit.
    collision_energy : int
        The collision energy predicted for.
    pred_irt : float
        The predicted iRT value from Prosit.
    precursor_charge : int
        The charge of the peptide.
    sequence_integer : np.array
        The encoded peptide sequence.
    pred_ion_names : np.array
        The name of the ions matching the intensities.
    """
    mod, mod_string = generate_mod_strings(sequence_integer)
    unmod_seq = ''.join(
        [PROSIT_UNMOD_ALPHA_S[i] if i in PROSIT_UNMOD_ALPHA_S else '' for i in sequence_integer]
    )

    print_string = (
        f'Name: {unmod_seq}/{precursor_charge}\nMW: 0.0\n' +
        f'Comment: Parent=0.0 Collision_energy={np.round(collision_energy, 0)} ' +
        f'Mods={mod} ModString={unmod_seq}//{mod_string}/{precursor_charge}' +
        f' iRT={pred_irt}' +
        f'\nNum peaks: {len(pred_intes)}'
    )

    for intensity, ion in zip(pred_intes, pred_ion_names):
        print_string += '\n0.0\t' + str(intensity) + '\t"'
        print_string += ion + '/0.0ppm"'

    return print_string


def write_msp_spectrum(df_row, out_file):
    """ Function to write a single spectrum in MSP format.

    Parameters
    ----------
    df_row : pd.Series
        A single row of the DataFrame containing prediction data.
    out_file : file
        The File where the spectrum will be written.
    """
    first_spec = df_row.name == 0
    pred_intes = df_row['intensities_pred']
    sel = np.where(pred_intes > 0)
    pred_intes = pred_intes[sel]
    collision_energy = df_row['collision_energy']
    pred_irt = df_row['iRT']
    precursor_charge = df_row['precursor_charge']
    sequence_integer = df_row['sequence_integer']
    pred_ion_names = PROSIT_IONS[sel]
    spec = format_msp_spectrum(
        pred_intes,
        collision_energy,
        pred_irt,
        precursor_charge,
        sequence_integer,
        pred_ion_names,
    )
    if not first_spec:
        out_file.write("\n")
    first_spec = False
    out_file.write(str(spec))


def write_msp_file(peptide_df, prediction_data, out_path):
    """ Function to write MSP output from Prosit predictions.

    Parameters
    ----------
    peptide_df : pd.DataFrame
        The input DataFrame used  from prosit prediction.
    prediction_data : dict
        Dictionary of the Prosit output predictions.
    out_path : str
        The location where results should be written.
    """
    peptide_df['iRT'] = prediction_data['iRT']
    peptide_df['intensities_pred'] = pd.Series(list(prediction_data['intensities_pred']))
    peptide_df['sequence_integer'] = pd.Series(list(prediction_data['sequence_integer']))
    with open(out_path, mode='w', encoding='UTF-8') as out_file:
        peptide_df.apply(lambda df_row : write_msp_spectrum(df_row, out_file), axis=1)
