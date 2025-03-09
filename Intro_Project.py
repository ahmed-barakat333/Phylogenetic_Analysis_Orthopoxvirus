# CIT-654 - Introduction to Bioinformatics
# Spring 2024
# Project
# Student name: Ahmed Barakat
'''
Orthopoxvirus virus family contains several strains.

Strains included in this analysis are

# Variola virus (taxid 10255), Vaccinia virus (taxid 10245),
# Monkeypox virus (taxid 10244), Cowpox virus (taxid 10243),
# Camelpox virus (taxid 28873), Ectromelia virus (taxid 12643),
# Volepox virus (taxid 28874), Taterapox virus (taxid 28871),
# Orthopoxvirus Abatino (taxid 2478919), Raccoonpox virus (taxid 10256),
# Horsepox virus (taxid 397342), Akhmeta virus (taxid 2200830),
# Skunkpox virus (taxid 160796)

https://www.ncbi.nlm.nih.gov/genomes/GenomesGroup.cgi?taxid=10242
'''

# import packages
import numpy as np
import subprocess
from Bio.Seq import Seq
from Bio import AlignIO, SeqIO, SeqRecord, Entrez, Phylo
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from pymsaviz import MsaViz
from collections import Counter
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


############################################################################################################
############################################################################################################
''''
read genome sequences
'''

Entrez.email = "ahmedbarakat@aun.edu.eg"

def fetch_genome_sequences(accessions):
    sequences = []
    for acc in accessions:
        handle = Entrez.efetch(db="nucleotide", id=acc, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "gb")
        record.id = record.description.split(',')[0].replace(" ", "_")
        handle.close()
        sequences.append(record)
        print("The virus name is %s and its genome length is %d." % (record.id,
                                                                   len(record.seq)))
    return sequences

genome_accessions = ["NC_006998", "NC_001611", "NC_003391", "NC_003663", "NC_004105", "NC_066642",
              "NC_063383", "NC_055231", "NC_055230","NC_027213", "NC_031038", "NC_008291",
              "NC_031033"]
genomes = fetch_genome_sequences(genome_accessions)
SeqIO.write(sequences=genomes, handle="genomes.fasta", format='fasta')

############################################################################################################
############################################################################################################
'''
run MAFFT multiple sequence alignment in the command line using subprocess module
URL: https://mafft.cbrc.jp/alignment/server/index.html
'''
def run_mafft(input_file):
    mafft_cmd = f"mafft --preservecase --auto --quiet {input_file}"
    try:
        result = subprocess.run(mafft_cmd, shell=True, check=True, capture_output=True, text=True)
        with open("alignment.fasta", "w") as temp_file:
            temp_file.write(result.stdout)
        alignment = AlignIO.read("alignment.fasta", "fasta")
        return alignment
    except subprocess.CalledProcessError as e:
        print(f"Error running MAFFT: {e}")
        return None

alignment = run_mafft('genomes.fasta')


############################################################################################################
############################################################################################################
'''
visualize multiple sequence alignment using pymsaviz package
URL: https://github.com/moshi4/pyMSAviz
'''
msa_file = alignment
mv = MsaViz(msa_file, format='fasta', wrap_length=60, show_count=True, start=117983, end=118102,
            show_grid=True, show_consensus=True)
mv.set_plot_params(ticks_interval=5, x_unit_size=0.20, grid_color="black")

mv.savefig("alignment_vis.png")

############################################################################################################
############################################################################################################
'''
get consensus sequence from multiple sequence alignment
'''
def create_consensus_sequence(alignment):
    consensus = ""
    alignment_array = np.array([list(rec.seq) for rec in alignment])
    num_seqs, alignment_length = alignment_array.shape

    for i in range(alignment_length):
        column = alignment_array[:, i]
        unique, counts = np.unique(column, return_counts=True)
        most_common = unique[np.argmax(counts)]
        consensus += most_common

    return SeqRecord(Seq(consensus), id="Consensus", description="Consensus genome")

consensus_genome = create_consensus_sequence(alignment)

SeqIO.write(consensus_genome, "consensus_genome.fasta", format='fasta')

############################################################################################################
############################################################################################################

# calculate and plot shannon entropy of the multiple sequence alignment
# shannon entropy is a measure of the amount of variability through a column in an alignment.
# URL: https://www.biostars.org/p/3856/

def calculate_shannon_entropy(alignment):
    alignment_array = np.array([list(rec.seq) for rec in alignment])
    num_seqs, alignment_length = alignment_array.shape
    entropy_scores = np.zeros(alignment_length)

    for i in range(alignment_length):
        column = alignment_array[:, i]
        unique, counts = np.unique(column, return_counts=True)
        frequencies = counts / num_seqs
        entropy = -np.sum(frequencies * np.log2(frequencies))
        entropy_scores[i] = entropy

    return entropy_scores

entropy_scores = calculate_shannon_entropy(alignment)

def plot_entropy_profile(entropy_scores, title='Entropy Profile'):
    positions = list(range(1, len(entropy_scores) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(positions, entropy_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('Position')
    plt.ylabel('Shannon Entropy')
#    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_entropy_profile(entropy_scores, title='Conservation profile of included genomes')

############################################################################################################
############################################################################################################
'''
identify and plot conserved regions based on entropy score and accepted threshold
'''

def identify_conserved_regions(entropy_scores, threshold, min_region_length):
    conserved_regions = []
    start = None

    for i, score in enumerate(entropy_scores):
        if score <= threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_region_length:
                    conserved_regions.append((start, i))
                start = None

    if start is not None and len(entropy_scores) - start >= min_region_length:
        conserved_regions.append((start, len(entropy_scores)))

    return conserved_regions

threshold=0.1
min_region_length=10
conserved_regions = identify_conserved_regions(entropy_scores,threshold, min_region_length)
print(f'Number of conserved regions is {len(conserved_regions)} at entropy threshold {threshold} and miniumum region length {min_region_length}')
# print("Conserved Regions:", conserved_regions)

def plot_conserved_regions( conserved_regions, sequence_length, title='Conserved Regions'):
    plt.figure(figsize=(12, 2))

    # Draw a horizontal line representing the sequence
    plt.plot([0, sequence_length], [0, 0], color='black', linewidth=2)

    # Highlight conserved regions on the line
    for start, end in conserved_regions:
        plt.plot([start + 1, end], [0, 0], color='green', linewidth=5, alpha=0.6)

    plt.xlabel('Position')
#    plt.title(title)
    plt.yticks([])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

sequence_length = len(consensus_genome.seq)
plot_conserved_regions(conserved_regions, sequence_length, title='Conserved regions along consensus genome')

############################################################################################################
############################################################################################################
'''
identify and plot mutations for each strain sequence against the reference i.e., consensus sequence
'''
def choose_reference_genome(alignment, reference_id=None):
    if reference_id:
        for record in alignment:
            if record.id == reference_id:
                return record
        raise ValueError(f"Reference ID {reference_id} not found in alignment.")
    else:
        # Choose the first sequence as the reference by default
        return alignment[0]

def remove_reference_from_alignment(alignment, reference_genome):
    """Remove the reference sequence from the alignment."""
    new_alignment = MultipleSeqAlignment([record for record in alignment if record.id != reference_genome.id])
    return new_alignment

def identify_mutations(alignment, reference_seq):
    """Identify mutations relative to the reference sequence."""
    mutations = []
    ref_seq_array = np.array(list(reference_seq.seq))

    for record in alignment:
        seq_id = record.id
        seq_array = np.array(list(record.seq))

        differences = (seq_array != ref_seq_array) & (ref_seq_array != '-') & (seq_array != '-')
        mutation_positions = np.where(differences)[0]

        mutation_details = []
        for pos in mutation_positions:
            mutation_details.append({
                'position': pos + 1,  # 1-based position
                'ref_base': ref_seq_array[pos],
                'alt_base': seq_array[pos],
                'seq_id': seq_id
            })

        mutations.append({'seq_id': seq_id, 'mutations': mutation_details})

    return mutations


# extract mutations positions
def store_mutations_in_dict(mutations):
    mutations_dict = {}

    for mutation in mutations:
        seq_id = mutation['seq_id']
        positions = [m['position'] + 1 for m in mutation['mutations']]  # Convert to 1-based positions

        mutations_dict[seq_id] = positions

    return mutations_dict

def plot_mutations_from_alignment(alignment, mutations_data):
    fig, ax = plt.subplots(figsize=(10, len(alignment) * 0.4))

    # Plot each sequence in the alignment
    for idx, record in enumerate(alignment):
        seq_id = record.id
        sequence = str(record.seq)

        # Plot the sequence line
        ax.plot(range(len(sequence)), [idx] * len(sequence), color='black', linewidth=0.5)

        # Plot mutations if available for the current sequence
        if seq_id in mutations_data:
            mutation_positions = mutations_data[seq_id]
            for pos in mutation_positions:
                if pos < len(sequence):
                    ax.plot(pos, idx, marker='|', color='red', markersize=3)

        # Annotate the sequence ID
        ax.text(-20, idx, seq_id, ha='right', va='center', fontsize=10)

    ax.set_yticks([])
    ax.set_xlabel('Position')
#    plt.title('Mutations location across genomes')
    plt.show()

alignment= AlignIO.read('alignment.fasta', 'fasta')

reference_id = "Cowpox_virus" #Cowpox genome accession id
reference_genome = choose_reference_genome(alignment, reference_id)


alignment_without_reference = remove_reference_from_alignment(alignment, reference_genome)

mutations = identify_mutations(alignment_without_reference, reference_genome)

# print("Mutations and Variations:")
# for mutation in mutations:
#     print(mutation)
mutations_dict = store_mutations_in_dict(mutations)

plot_mutations_from_alignment(alignment_without_reference, mutations_dict)

############################################################################################################
############################################################################################################
'''
construct a phylogenetic tree using distance Matrix method and UPGMA algorithm
URL: https://medium.com/geekculture/phylogenetic-trees-implement-in-python-3f9df96c0c32
'''
calculator = DistanceCalculator('pam250')
distMatrix = calculator.get_distance(alignment)
print(distMatrix)

constructor = DistanceTreeConstructor()
UGMATree = constructor.upgma(distMatrix)
Phylo.draw(UGMATree)

############################################################################################################
############################################################################################################

'''
extract functional regions of genes from viral genomes based on genomic locations provided in NCBI database
Orthologs were identified based on a previous study (URL: https://www.sciencedirect.com/science/article/pii/S0168170222003045#sec0008)
six strains were included in this analysis with available ortholog gene names 
# Variola virus (taxid 10255), Vaccinia virus (taxid 10245),
# Monkeypox virus (taxid 10244), Cowpox virus (taxid 10243),
# Horsepox virus (taxid 397342)
'''

# prepare dictionary of each vairola virus gene orthologs
strains_A1L_mapping = {
    "NC_001611": {"gene_name": "A1L"}, # Variolla
    "NC_006998": {"gene_name": "A1L"}, # Vaccinia
    "NC_003663": {"gene_name": "CPXV132 CDS"}, # Cowpox
    "NC_063383": {"gene_name": "OPG126"}, # Monkeypox
    "NC_004105": {"gene_name": "EVM103"}, # Ectromelia
    "NC_066642": {"gene_name": "OPG126"}, # Horsepox
}

strains_C12L_mapping = {
    "NC_001611": {"gene_name": "C12L"}, # Variolla
    "NC_006998": {"gene_name": "C12L"}, # Vaccinia
    "NC_003663": {"gene_name": "CPXV217 CDS"}, # Cowpox
    "NC_063383": {"gene_name": "OPG208"}, # Monkeypox
    "NC_004105": {"gene_name": "C14R"}, # Ectromelia
    "NC_066642": {"gene_name": "OPG208"}, # Horsepox
}

strains_K2L_mapping = {
    "NC_001611": {"gene_name": "K2L"}, # Variolla
    "NC_006998": {"gene_name": "K2L"}, # Vaccinia
    "NC_003663": {"gene_name": "SPI3"}, # Cowpox
    "NC_063383": {"gene_name": "OPG040"}, # Monkeypox
    "NC_004105": {"gene_name": "H14-B"}, # Ectromelia
    "NC_066642": {"gene_name": "OPG040"}, # Horsepox
}

strains_A10L_mapping = {
    "NC_001611": {"gene_name": "A10L"}, # Variolla
    "NC_006998": {"gene_name": "A10L"}, # Vaccinia
    "NC_003663": {"gene_name": "CPXV142 CDS"}, # Cowpox
    "NC_063383": {"gene_name": "OPG136"}, # Monkeypox
    "NC_004105": {"gene_name": "EVM113"}, # Ectromelia
    "NC_066642": {"gene_name": "OPG136"}, # Horsepox
}

# retrieve genome sequences
def fetch_genome_sequence(accession):
    handle = Entrez.efetch(db="nuccore", id=accession, rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    handle.close()
    return record

# retrieve genome annotations
def fetch_annotation_data(accession):
    handle = Entrez.efetch(db="nuccore", id=accession, rettype="gb", retmode="text")
    record = SeqIO.read(handle, "genbank")
    handle.close()
    return record

# retrieve gene location based on gene name
def find_gene_location(genome_record, gene_name):
    locations = []
    for feature in genome_record.features:
        if feature.type == "gene" and gene_name in feature.qualifiers.get("gene", []):
            start = int(feature.location.start)
            end = int(feature.location.end)
            locations.append((start, end))
    return locations

# retrieve gene sequence based on gene location
def extract_functional_regions(genome_record, locations):
    functional_regions = []
    for start, end in locations:
        functional_region = genome_record.seq[start:end]
        functional_regions.append((start, end, functional_region))
    return functional_regions

# save extracted regions from all strains
def save_regions_to_fasta(regions, output_file):
    seq_records = [SeqRecord(region, id=f"{start}-{end}", description="") for start, end, region in regions]
    SeqIO.write(seq_records, output_file, "fasta")

# assess degree of conservation by dividing common residues by list of residues for each alignment column
def assess_conservation(alignment):
    alignment_array = [list(record.seq) for record in alignment]
    alignment_length = len(alignment[0].seq)

    conservation_scores = []

    for i in range(alignment_length):
        column = [row[i] for row in alignment_array]
        counter = Counter(column)
        most_common_residue_count = counter.most_common(1)[0][1]
        conservation_score = most_common_residue_count / len(alignment_array)
        conservation_scores.append(conservation_score)

    return conservation_scores

# combine above functions into one to use for several Vairola virus gene dictionaries
conservation_heatmap = {}

def process_species_gene_mapping(species_gene_mapping, file_suffix):
    """Process a dictionary of species-gene mappings and perform extraction, alignment, and conservation assessment."""
    all_regions = []

    for accession, gene_info in species_gene_mapping.items():
        gene_name = gene_info["gene_name"]

        # Fetch the genome sequence
        genome_record = fetch_genome_sequence(accession)

        # Fetch the annotation data
        annotation_data = fetch_annotation_data(accession)

        # Find gene locations based on the gene name
        locations = find_gene_location(annotation_data, gene_name)

        # Extract functional regions based on locations
        extracted_regions = extract_functional_regions(genome_record, locations)
        all_regions.extend(extracted_regions)

        # Save extracted regions to FASTA file
        fasta_file = f"extracted_regions_{file_suffix}.fasta"
        save_regions_to_fasta(all_regions, fasta_file)

        # Align sequences using MUSCLE
        alignment = run_mafft(fasta_file)

    # Assess the degree of conservation
    conservation_scores = assess_conservation(alignment)
    average_conservation = sum(conservation_scores) / len(conservation_scores)
    conservation_heatmap[file_suffix] = average_conservation

    # print conservation scores
    print(f"Average conservation score of {file_suffix}:")
    print(average_conservation)


# list of genome accessions for included strains
# Variola virus (taxid 10255), Vaccinia virus (taxid 10245),
# Monkeypox virus (taxid 10244), Cowpox virus (taxid 10243),
# Horsepox virus (taxid 397342)
genome_accessions = ["NC_001611", "NC_006998", "NC_003663", "NC_063383", "NC_004105", "NC_066642"]

# calculate conservation score for each gene across included genomes
A1L_conservation = process_species_gene_mapping(strains_A1L_mapping, "A1L")
A10L_conservation = process_species_gene_mapping(strains_A10L_mapping, "A10L")

C12L_conservation = process_species_gene_mapping(strains_C12L_mapping, "C12L")
K2L_conservation = process_species_gene_mapping(strains_K2L_mapping, "K2L")


def plot_conservation_heatmap(gene_conservation):
    """Plot a heatmap of average conservation scores."""
    df = pd.DataFrame(list(gene_conservation.items()), columns=["Gene", "Average conservation score"])
    df.set_index("Gene", inplace=True)

    plt.figure(figsize=(5, 8))
    sns.heatmap(df, annot=True, cmap="viridis", cbar_kws={'label': 'Score'})
#    plt.title("Heatmap of Average Conservation Scores for Each Gene")
    plt.ylabel("Gene")
    # plt.xlabel("Conservation Score")
    plt.show()


plot_conservation_heatmap(conservation_heatmap)


############################################################################################################
############################################################################################################

# plot Vairola virus gene locations
def extract_gene_location(genome_record, gene_name):
    """Searches for a gene by name in the genome annotation and returns its location."""
    for feature in genome_record.features:
        if feature.type == "gene" and gene_name in feature.qualifiers.get("gene", []):
            start = int(feature.location.start)
            end = int(feature.location.end)
            return (gene_name, start, end)
    return None

def plot_gene_locations(genome_record, gene_locations, output_file="gene_locations.png"):
    """Plots the gene locations on the genome sequence."""
    fig, ax = plt.subplots(figsize=(15, 3))

    genome_length = len(genome_record.seq)
    ax.plot([0, genome_length], [0, 0], color="black", lw=2)

    for gene in gene_locations:
        if gene:
            gene_name, start, end = gene
            rect = patches.Rectangle((start, -0.1), end - start, 0.2, edgecolor="black", facecolor="orange")
            ax.add_patch(rect)
            ax.text((start + end) / 2, 0.15, gene_name, ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_xlim(0, genome_length)
    ax.set_ylim(-0.2, 0.5)
    ax.set_xlabel("Genomic Position")
    ax.set_ylabel("Genes")
    ax.set_yticks([])
    ax.set_title("Gene Locations on Variola Virus Genome")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


# variola virus genome accession
variola_accession = "NC_001611"

# retrieve variola virus genome sequence and annotation data
genome_record = fetch_genome_sequence(variola_accession)
annotation_data = fetch_annotation_data(variola_accession)

# list of variola virus gene names to find
gene_names = ["A1L", "A10L", "C12L", "K2L"]

# extract gene locations
gene_locations = []
for gene_name in gene_names:
    location = extract_gene_location(annotation_data, gene_name)
    if location:
        gene_locations.append(location)
    else:
        print(f"Gene {gene_name} not found in the annotation data.")

# plot gene locations
plot_gene_locations(genome_record, gene_locations)