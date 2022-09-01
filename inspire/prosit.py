""" Code for running Prosit models on CPU.
    Large sections based on https://github.com/kusterlab/prosit.
"""

import keras
import numpy as np
import pandas as pd
from pyteomics.mass import Unimod, std_aa_comp
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers, constraints, initializers
from tensorflow.keras.layers import Layer
import yaml

from inspire.constants import (
    ION_TYPES,
    MAX_CHARGE,
    MAX_FRAG_CHARGE,
    MAX_ION_IDX,
    MAX_SEQ_LEN,
    N_ION_TYPES,
    N_LOSSES,
    PROSIT_ALPHABET,
    PROSIT_ALPHABET_S,
    PROSIT_MASK_VALUE,
    PROSIT_PRED_BATCH_SIZE,
)

class PrositAttention(Layer):
    """ The modified attention layer used by Prosit.
    """
    def __init__(
        self,
        context=False,
        W_regularizer=None,
        b_regularizer=None,
        u_regularizer=None,
        W_constraint=None,
        b_constraint=None,
        u_constraint=None,
        bias=True,
        **kwargs
    ):
        self.supports_masking = True
        self.init = initializers.get("glorot_uniform")
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.bias = bias
        self.context = context
        super(PrositAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(
            shape=(input_shape[-1],),
            initializer=self.init,
            name="{}_W".format(self.name),
            regularizer=self.W_regularizer,
            constraint=self.W_constraint,
        )
        if self.bias:
            self.b = self.add_weight(
                shape=(input_shape[1],),
                initializer="zero",
                name="{}_b".format(self.name),
                regularizer=self.b_regularizer,
                constraint=self.b_constraint,
            )
        else:
            self.b = None
        if self.context:
            self.u = self.add_weight(
                shape=(input_shape[-1],),
                initializer=self.init,
                name="{}_u".format(self.name),
                regularizer=self.u_regularizer,
                constraint=self.u_constraint,
            )

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        a = K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1)
        if self.bias:
            a += self.b
        a = K.tanh(a)
        if self.context:
            a = K.squeeze(K.dot(x, K.expand_dims(self.u)), axis=-1)
        a = K.exp(a)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = {
            "bias": self.bias,
            "context": self.context,
            "W_regularizer": regularizers.serialize(self.W_regularizer),
            "b_regularizer": regularizers.serialize(self.b_regularizer),
            "u_regularizer": regularizers.serialize(self.u_regularizer),
            "W_constraint": constraints.serialize(self.W_constraint),
            "b_constraint": constraints.serialize(self.b_constraint),
            "u_constraint": constraints.serialize(self.u_constraint),
        }
        base_config = super(PrositAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def load_model(config_loc, model_loc, weights_loc):
    with open(config_loc, 'r', encoding='UTF-8') as stream:
        config_dict = yaml.safe_load(stream)

    with open(model_loc, "r") as f:
        model = tf.keras.models.model_from_yaml(
            f.read(), custom_objects={"Attention": PrositAttention}
        )
    model.load_weights(weights_loc)

    d_irt = {}
    d_irt["graph"] = tf.Graph()

    d_irt["session"] = tf.compat.v1.Session()
    with d_irt["session"].as_default():
        d_irt["model"], d_irt["config"] = model, config_dict
        d_irt["model"].compile(optimizer="adam", loss="mse")

    return d_irt

def get_precursor_charge_onehot(charges):
    array = np.zeros([len(charges), MAX_CHARGE], dtype=int)
    for i, precursor_charge in enumerate(charges):
        array[i, precursor_charge - 1] = 1
    return array


def peptide_parser(p):
    p = p.replace("_", "")
    if p[0] == "(":
        raise ValueError("sequence starts with '('")
    n = len(p)
    i = 0
    while i < n:
        if i < n - 3 and p[i + 1] == "(":
            j = p[i + 2 :].index(")")
            offset = i + j + 3
            yield p[i:offset]
            i = offset
        else:
            yield p[i]
            i += 1

def get_sequence_integer(sequences):
    array = np.zeros([len(sequences), MAX_SEQ_LEN], dtype=int)
    for i, sequence in enumerate(sequences):
        for j, s in enumerate(peptide_parser(sequence)):
            array[i, j] = PROSIT_ALPHABET[s]
    return array



def process_csv_file(input_loc):
    input_df = pd.read_csv(input_loc)
    input_df.reset_index(drop=True, inplace=True)
    n_seqs = input_df.shape[0]


    data = {
        "collision_energy_aligned_normed": (
            np.array(input_df.collision_energy).astype(float).reshape([n_seqs, 1]) / 100.0
        ),
        "sequence_integer": get_sequence_integer(input_df.modified_sequence),
        "precursor_charge_onehot": get_precursor_charge_onehot(input_df.precursor_charge),
    }

    return data

def sanitize(data):

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
    # check for mandatory keys
    x = [data[key] for key in d_model["config"]["x"]]

    keras.backend.set_session(d_model["session"])

    prediction = d_model["model"].predict(
        x, verbose=True, batch_size=PROSIT_PRED_BATCH_SIZE
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

def get_ions():
    x = np.empty(
        [MAX_ION_IDX, N_ION_TYPES, MAX_FRAG_CHARGE],
        dtype="|S6",
    )
    for fz in range(MAX_FRAG_CHARGE):
        for fty_i, fty in enumerate(ION_TYPES):
            for fi in range(MAX_ION_IDX):
                ion = fty + str(fi + 1)
                if fz > 0:
                    ion += "({}+)".format(fz + 1)
                x[fi, fty_i, fz] = ion
    x.flatten()
    return x

# IONS = np.array(
#     ,
#     dtype='|S6'
# )


class Converter():
    def __init__(self, data, out_path):
        self.out_path = out_path
        self.data = data
        self.aa_comp = generate_aa_comp()

    def convert(self, redux=False):
        IONS = get_ions().reshape(174, -1).flatten()

        with open(self.out_path, mode="w", encoding="utf-8") as out_file:
            first_spec = True
            for i in range(self.data["iRT"].shape[0]):
                aIntensity = self.data["intensities_pred"][i]
                sel = np.where(aIntensity > 0)
                aIntensity = aIntensity[sel]
                collision_energy = self.data["collision_energy_aligned_normed"][i] * 100
                iRT = self.data["iRT"][i]
                precursor_charge = self.data["precursor_charge_onehot"][i].argmax() + 1
                sequence_integer = self.data["sequence_integer"][i]
                aIons = IONS[sel]
                spec = Spectrum(
                    aIntensity,
                    collision_energy,
                    iRT,
                    precursor_charge,
                    sequence_integer,
                    aIons,
                )
                if not first_spec:
                    out_file.write("\n")
                first_spec = False
                out_file.write(str(spec))
        return spec

def generate_aa_comp():
    """
    >>> aa_comp = generate_aa_comp()
    >>> aa_comp["M"]
    Composition({'H': 9, 'C': 5, 'S': 1, 'O': 1, 'N': 1})
    >>> aa_comp["Z"]
    Composition({'H': 9, 'C': 5, 'S': 1, 'O': 2, 'N': 1})
    """
    db = Unimod()
    aa_comp = dict(std_aa_comp)
    s = db.by_title("Oxidation")["composition"]
    aa_comp["Z"] = aa_comp["M"] + s
    s = db.by_title("Carbamidomethyl")["composition"]
    aa_comp["C"] = aa_comp["C"] + s
    return aa_comp

def generate_mods_string_tuples(sequence_integer):
    list_mods = []
    for mod in [PROSIT_ALPHABET['M(ox)'], PROSIT_ALPHABET['C']]:
        for position in np.where(sequence_integer == mod)[0]:
            if mod == PROSIT_ALPHABET['C']:
                list_mods.append((position, "C", "Carbamidomethyl"))
            elif mod == PROSIT_ALPHABET['M(ox)']:
                list_mods.append((position, "M", "Oxidation"))
            else:
                raise ValueError("cant be true")
    list_mods.sort(key=lambda tup: tup[0])  # inplace
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
        return "0", ""
    else:
        returnString_mods = ""
        returnString_modString = ""
        returnString_mods += str(len(list_mods))
        for i, mod_tuple in enumerate(list_mods):
            returnString_mods += (
                "/" + str(mod_tuple[0]) + "," + mod_tuple[1] + "," + mod_tuple[2]
            )
            if i == 0:
                returnString_modString += (
                    mod_tuple[2] + "@" + mod_tuple[1] + str(mod_tuple[0] + 1)
                )
            else:
                returnString_modString += (
                    "; " + mod_tuple[2] + "@" + mod_tuple[1] + str(mod_tuple[0] + 1)
                )

    return returnString_mods, returnString_modString

class Spectrum(object):
    """ Spectrum class for all the information that must be outputed in the msp file.
    """
    def __init__(
        self,
        aIntensity,
        collision_energy,
        iRT,
        precursor_charge,
        sequence_integer,
        aIons,
    ):
        self.aIntensity = aIntensity
        self.collision_energy = collision_energy
        self.iRT = iRT
        self.aIons = aIons
        self.precursor_charge = precursor_charge
        self.mod, self.mod_string = generate_mod_strings(sequence_integer)
        self.sequence = ''.join(
            [PROSIT_ALPHABET_S[i] if i in PROSIT_ALPHABET_S else '' for i in sequence_integer]
        )

    def __str__(self):
        print_string = "Name: {sequence}/{charge}\nMW: 0.0\n"
        print_string += "Comment: Parent=0.0 Collision_energy={collision_energy} "
        print_string += "Mods={mod} ModString={sequence}//{mod_string}/{charge}"
        print_string += " iRT={iRT}"
        print_string += "\nNum peaks: {num_peaks}"
        num_peaks = len(self.aIntensity)
        print_string = print_string.format(
            sequence=self.sequence.replace("M(ox)", "M"),
            charge=self.precursor_charge,
            collision_energy=np.round(self.collision_energy[0], 0),
            mod=self.mod,
            mod_string=self.mod_string,
            num_peaks=num_peaks,
            iRT=self.iRT[0],
        )
        for intensity, ion in zip(self.aIntensity, self.aIons):
            print_string += "\n0.0\t" + str(intensity) + '\t"'
            print_string += ion.decode("UTF-8").replace("(", "^").replace("+", "") + '/0.0ppm"'
        return print_string
