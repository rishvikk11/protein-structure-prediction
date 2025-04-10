from Bio.PDB import PDBParser, PPBuilder, PDBList
import os
import pickle
import numpy as np
import random

# Download PDB files
def download_pdb_files(pdb_ids, save_dir="pdb_files"):
    pdbl = PDBList()
    for pdb_id in pdb_ids:
        pdbl.retrieve_pdb_file(pdb_id, pdir=save_dir, file_format="pdb")

# Normalize coordinates by centering them around the mean
def normalize_coords(coords):
    coords = np.array(coords)
    center = np.mean(coords, axis=0)  # Calculate the center of the coordinates
    return coords - center  # Subtract the center to center the coordinates

# Parse PDB structure and extract sequence
PDB_DIR = "pdb_files/"
protein_data = []

def encode_sequence(seq, max_len):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_index = {aa: i + 1 for i, aa in enumerate(amino_acids)}  # 0 = padding
    encoded = [aa_to_index.get(aa, 0) for aa in seq]
    if len(encoded) < max_len:
        # Pad with zeros
        encoded += [0] * (max_len - len(encoded))
    else:
        # Truncate if too long
        encoded = encoded[:max_len]
    return encoded

def pad_coordinates(coords, max_len):
    coords = np.array(coords)
    if len(coords) < max_len:
        padding = np.zeros((max_len - len(coords), 3))
        coords = np.vstack([coords, padding])
    else:
        coords = coords[:max_len]
    return coords

def parse_structure_sequence(filename, max_len):
    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure("protein", filename)
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return None
    
    ppb = PPBuilder()
    peptides = ppb.build_peptides(structure)

    if not peptides:
        print(f"No peptide sequence found in {filename}")
        return None
    
    sequence = str(peptides[0].get_sequence())

    

    # Only use the first model
    model = next(structure.get_models())

    for chain in model:
        # Extract C-alpha (CA) coordinates for protein structure
        ca_coords = []
        chain_sequence = ""
        for residue in chain:
            if "CA" in residue:
                ca_coords.append(residue["CA"].get_coord())
                # Process this chain's data

    min_length = min(len(sequence), len(ca_coords))
    sequence = sequence[:min_length]
    ca_coords = ca_coords[:min_length]

    # Encode the sequence and pad the coordinates
    encoded_sequence = encode_sequence(sequence, max_len)
    padded_coords = pad_coordinates(ca_coords, max_len)

    # Normalize the coordinates
    normalized_coords = normalize_coords(padded_coords)

    return (encoded_sequence, normalized_coords)

def load_cleaned_data():
    protein_data = []
    total_files = 0
    success_files = 0
    for filename in os.listdir(PDB_DIR):
        if filename.endswith(".ent") or filename.endswith(".pdb"):
            total_files += 1
            filepath = os.path.join(PDB_DIR, filename)
            try:
                result = parse_structure_sequence(filepath, 256)
                if result:
                    protein_data.append(result)
                    success_files += 1
                    print(f"Successfully parsed {filename}")
            except Exception as e:
                print(f"Failed to parse {filename}: {str(e)}")
    
    print(f"\nSummary: {success_files}/{total_files} files processed successfully")

    # Save the parsed and processed data to disk
    with open("cleaned_protein_data.pkl", "wb") as f:
        pickle.dump(protein_data, f)