"""seq_io.py contains sequence loading and transform utilities"""
import numpy as np
import pandas as pd
import xarray as xr
import json

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

def add_genome_sequences_to_crandata(adata: xr.Dataset, ranges_df: pd.DataFrame, genome, key: str = "sequences", seq_length: int = None) -> xr.Dataset:
    """
    Create a one-hot encoded array of genomic sequences using the provided genome and ranges,
    add it to the CrAnData (an xarray.Dataset) under the specified key (default 'sequences'),
    and store genome metadata in the Dataset's attributes for later retrieval.
    
    Parameters
    ----------
    adata : xr.Dataset
        CrAnData object represented as an xarray.Dataset.
    ranges_df : pd.DataFrame
        DataFrame with genomic ranges. Expected columns: 'chrom', 'start', 'end', and optionally 'strand'.
    genome
        Genome object (from _genome.py) with a fetch method to extract sequences.
    key : str, optional
        Key under which to store the one-hot encoded array in adata. Default is 'sequences'.
    seq_length : int, optional
        The sequence length to fetch. If None, it is inferred from ranges_df.
    
    Returns
    -------
    xr.Dataset
        Updated CrAnData object with the one-hot encoded array added as a data variable and genome metadata stored in attrs.
    """
    # Create the one-hot encoded DataArray.
    da = create_one_hot_encoded_array(ranges_df, genome, seq_length=seq_length)
    
    # Add the one-hot encoded array as a new data variable.
    adata[key] = da
    
    # Store key genome metadata in the dataset's attributes.
    adata.attrs["genome_name"] = genome.name
    adata.attrs["genome_fasta"] = str(genome._fasta) if hasattr(genome, "_fasta") else None
    adata.attrs["genome_chrom_sizes"] = json.dumps(genome.chrom_sizes) if len(genome.chrom_sizes.keys())<100 else None
    
    return adata

def hot_encoding_to_sequence(one_hot_encoded_sequence: np.ndarray) -> str:
    """
    Decode a one hot encoded sequence to a DNA sequence string. Directly from CREsted

    Parameters
    ----------
    one_hot_encoded_sequence
        A numpy array with shape (x, 4) with dtype=np.float32.

    Returns
    -------
    The DNA sequence string of length x.
    """
    # Convert hot encoded seqeuence from:
    #   (x, 4) with dtype=np.float32
    # to:
    #   (x, 4) with dtype=np.uint8
    # and finally combine ACGT dimensions to:
    #   (x, 1) with dtype=np.uint32
    hes_u32 = one_hot_encoded_sequence.astype(np.uint8).view(np.uint32)

    # Do some bitshifting magic to decode uint32 to DNA sequence string.
    sequence = (
        HOT_DECODING_TABLE[
            (
                (
                    hes_u32 << 31 >> 31
                )  # A: 2^0  : 1        -> 1 = A in HOT_DECODING_TABLE
                | (
                    hes_u32 << 23 >> 30
                )  # C: 2^8  : 256      -> 2 = C in HOT_DECODING_TABLE
                | (
                    hes_u32 << 15 >> 29
                )  # G: 2^16 : 65536    -> 4 = G in HOT_DECODING_TABLE
                | (
                    hes_u32 << 7 >> 28
                )  # T: 2^24 : 16777216 -> 8 = T in HOT_DECODING_TABLE
            ).astype(np.uint8)
        ]
        .tobytes()
        .decode("ascii")
    )

    return sequence

def reverse_complement(sequence: str | list[str] | np.ndarray) -> str | np.ndarray:
    """
    Perform reverse complement on either a one-hot encoded array or a (list of) DNA sequence string(s).

    Parameters
    ----------
    sequence
        The DNA sequence string(s) or one-hot encoded array to reverse complement.

    Returns
    -------
    The reverse complemented DNA sequence string or one-hot encoded array.
    """

    def complement_str(seq: str) -> str:
        complement = str.maketrans("ACGTacgt", "TGCAtgca")
        return seq.translate(complement)[::-1]

    if isinstance(sequence, str):
        return complement_str(sequence)
    elif isinstance(sequence, list):
        return [complement_str(seq) for seq in sequence]
    elif isinstance(sequence, np.ndarray):
        if sequence.ndim == 2:
            if sequence.shape[1] == 4:
                return sequence[::-1, ::-1]
            elif sequence.shape[0] == 4:
                return sequence[:, ::-1][:, ::-1]
            else:
                raise ValueError(
                    "One-hot encoded array must have shape (W, 4) or (4, W)"
                )
        elif sequence.ndim == 3:
            if sequence.shape[1] == 4:
                return sequence[:, ::-1, ::-1]
            elif sequence.shape[2] == 4:
                return sequence[:, ::-1, ::-1]
            else:
                raise ValueError(
                    "One-hot encoded array must have shape (B, 4, W) or (B, W, 4)"
                )
        else:
            raise ValueError("One-hot encoded array must have 2 or 3 dimensions")
    else:
        raise TypeError(
            "Input must be either a DNA sequence string or a one-hot encoded array"
        )

class DNATransform:
    def __init__(self, out_len: int, random_rc: bool = False, max_shift: int = None):
        """
        Initialize a DNATransform.
        
        Parameters
        ----------
        out_len : int
            The desired output window length. Must be <= seq_len.
        random_rc : bool, default False
            If True, each sample has a 50% chance to be reverse complemented.
        max_shift : int, optional
            The maximum number of bases to randomly shift the window center away from the sequence midpoint.
            Defaults to the maximum allowed value: (seq_len - out_len) // 2.
            If provided, its absolute value is clipped to (seq_len - out_len) // 2.
        """
        self.out_len = out_len
        self.random_rc = random_rc
        self.max_shift_param = max_shift

    def __call__(self, da: xr.DataArray) -> xr.DataArray:
        """
        Transform a one-hot encoded DataArray by extracting shifted windows and, optionally,
        reverse complementing them.
        
        Each window is extracted from the original sequence such that its center is randomly
        shifted from the midpoint by an integer in [-max_shift, max_shift]. Then, if random_rc
        is enabled, a 50% coin flip determines whether the window is reverse complemented.
        
        The reverse complement is implemented by reversing the sequence order along the
        seq_len dimension and swapping the nucleotide channels (ACGT → TGCA).
        
        Parameters
        ----------
        da : xarray.DataArray
            One-hot encoded array with dimensions ["var", "seq_len", "nuc"].
            
        Returns
        -------
        xr.DataArray
            Transformed DataArray with dimensions ["var", "seq_len", "nuc"] where seq_len equals out_len.
        """
        # Get the original sequence length from the DataArray.
        seq_len = da.sizes["seq_len"]
        if self.out_len > seq_len:
            raise ValueError(f"out_len ({self.out_len}) cannot be larger than seq_len ({seq_len}).")
        
        # Compute allowed maximum shift so that window stays in bounds.
        allowed_max_shift = (seq_len - self.out_len) // 2
        # Determine effective max_shift from parameter (or default)
        if self.max_shift_param is None:
            effective_max_shift = allowed_max_shift
        else:
            effective_max_shift = min(abs(self.max_shift_param), allowed_max_shift)
        
        # Precompute the midpoint (using integer division).
        mid = seq_len // 2

        transformed_windows = []
        # Loop over each sample (var) to apply independent transformations.
        for i in range(da.sizes["var"]):
            # Sample a random shift from -effective_max_shift to effective_max_shift.
            shift = np.random.randint(-effective_max_shift, effective_max_shift + 1) if effective_max_shift > 0 else 0
            new_center = mid + shift
            start_index = new_center - (self.out_len // 2)
            end_index = start_index + self.out_len
            
            # Use isel for lazy slicing; extract the window from the current sample.
            window = da.isel(var=i, seq_len=slice(start_index, end_index))
            
            # Optionally apply reverse complement:
            if self.random_rc and np.random.rand() < 0.5:
                # Reverse the order along the sequence dimension and swap nucleotide channels.
                # This is done by slicing seq_len in reverse and nuc in reverse.
                window = window.isel(seq_len=slice(None, None, -1)).isel(nuc=slice(None, None, -1))
            
            transformed_windows.append(window)
        
        # Concatenate all transformed windows along the "var" dimension.
        transformed_da = xr.concat(transformed_windows, dim="var")
        
        # Optionally, reassign a coordinate for seq_len to reflect the new window length.
        transformed_da = transformed_da.assign_coords(seq_len=np.arange(self.out_len))
        return transformed_da
