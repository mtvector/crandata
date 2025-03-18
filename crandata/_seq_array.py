import numpy as np
import pandas as pd
import xarray as xr

# Create a hot encoding table for DNA nucleotides (A, C, G, T)
def get_hot_encoding_table(
    alphabet: str = "ACGT",
    neutral_alphabet: str = "N",
    neutral_value: int = 0,
    dtype=np.uint8,
) -> np.ndarray:
    """Return a (256 x len(alphabet)) table mapping ASCII byte values to one-hot encodings."""
    def str_to_uint8(string: str) -> np.ndarray:
        return np.frombuffer(string.encode("ascii"), dtype=np.uint8)
    
    table = np.zeros((256, len(alphabet)), dtype=dtype)
    eye = np.eye(len(alphabet), dtype=dtype)
    # set uppercase and lowercase for the nucleotides
    table[str_to_uint8(alphabet.upper())] = eye
    table[str_to_uint8(alphabet.lower())] = eye
    # assign neutral bases (if any) to be all zeros (or neutral_value)
    table[str_to_uint8(neutral_alphabet.upper())] = neutral_value
    table[str_to_uint8(neutral_alphabet.lower())] = neutral_value
    return table

HOT_ENCODING_TABLE = get_hot_encoding_table()


def one_hot_encode_sequence(sequence: str) -> np.ndarray:
    """
    One-hot encode a DNA sequence using the pre-computed HOT_ENCODING_TABLE.
    
    Parameters
    ----------
    sequence : str
        The DNA sequence to encode.
        
    Returns
    -------
    np.ndarray
        A (seq_length, 4) numpy array of type np.uint8.
    """
    # Convert sequence to its byte representation and look up the encoding for each character.
    return HOT_ENCODING_TABLE[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]


def create_one_hot_encoded_array(
    ranges_df: pd.DataFrame, genome, seq_length: int = None
) -> xr.DataArray:
    """
    Create a one-hot encoded xarray DataArray from genomic ranges.
    
    Parameters
    ----------
    ranges_df : pd.DataFrame
        DataFrame with genomic ranges. Expected columns: 'chrom', 'start', 'end', and optionally 'strand'.
    genome : Genome
        A Genome object (from _genome.py) that provides a fetch(chrom, start, end, strand) method.
    seq_length : int, optional
        The sequence length to fetch. If None, it is inferred as (end - start) from the first row.
    
    Returns
    -------
    xr.DataArray
        One-hot encoded sequences with dimensions [var, seq_len, 4].
    """
    sequences = []
    
    # Infer sequence length if not provided.
    if seq_length is None:
        seq_length = int(ranges_df.iloc[0]['end']) - int(ranges_df.iloc[0]['start'])
    
    for idx, row in ranges_df.iterrows():
        chrom = row['chrom']
        start = int(row['start'])
        end = int(row['end'])
        if (end - start) != seq_length:
            raise ValueError(f"Row {idx} has length {(end - start)} which differs from expected {seq_length}.")
        # Default strand is '+' if not provided.
        strand = row.get('strand', '+') if 'strand' in row else '+'
        
        # Fetch the sequence using the Genome object. The fetch method should accept chrom, start, end, and strand.
        seq = genome.fetch(chrom, start, end, strand=strand)
        
        # Ensure the sequence is of the expected length; pad with 'N' if needed.
        if len(seq) < seq_length:
            seq = seq.ljust(seq_length, 'N')
        elif len(seq) > seq_length:
            seq = seq[:seq_length]
        
        # One-hot encode the sequence.
        one_hot = one_hot_encode_sequence(seq)  # shape (seq_length, 4)
        sequences.append(one_hot)
    
    # Stack all sequences into a numpy array of shape (n_ranges, seq_length, 4)
    encoded_array = np.stack(sequences, axis=0)
    
    # Create an xarray DataArray; we use dims ["var", "seq_len", "nuc"] where "nuc" has length 4.
    da = xr.DataArray(encoded_array, dims=["var", "seq_len", "nuc"])
    return da

#Reverse complement
#DataDarray.isel(y=slice(None, None, -1))to reverse index
#ACGT -> TGCA [0,1,2,3]>[3,2,1,0]