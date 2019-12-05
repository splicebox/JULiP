from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, gzip, subprocess
from collections import defaultdict
from functools import partial
from utils.io_utils import *
from utils.regions import *


def get_read_count_from(bam_file, seq_name=None):
    counts_dict = dict()
    strings = pysam.idxstats(bam_file).rstrip('\n').split('\n')
    for _str in strings:
        seq_name, length, num_mapped_reads, num_unmapped_reads = _str.split('\t')
        counts_dict[seq_name] = int(num_mapped_reads)
    if seq_name is None:
        return reduce(lambda x, y: x + y, counts_dict.values())
    else:
        return counts_dict[seq_name]


def generate_files(bam_file_list, _out_dir, file_type, threads=1, compress=True, save_list=True, junc_str='--nh 5'):
    from os.path import splitext, dirname, basename, isdir, join

    def splice_file_func(out_dir, file_name, bam_file):
        splice_suffix = '.splice' if not compress else '.splice.gz'
        splice_file_name = join(out_dir, file_name + splice_suffix)
        splice_file = open(splice_file_name, 'w') if not compress else gzip.open(splice_file_name, 'wb')
        splice_file.write(subprocess.Popen('bin/junc %s %s' % (bam_file, junc_str), shell=True,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read())
        splice_file.close()
        return splice_file_name

    def region_file_func(out_dir, file_name, bam_file):
        empty_file = os.path.join(_out_dir, '.empty.txt')
        if not os.path.exists(empty_file):
            open(empty_file, 'a').close()

        region_suffix = '.region' if not compress else '.region.gz'
        region_file_name = join(out_dir, file_name + region_suffix)
        region_file = open(region_file_name, 'w') if not compress else gzip.open(region_file_name, 'wb')
        region_file.write(subprocess.Popen('bin/region_depth %s %s' % (bam_file, empty_file), shell=True,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.read())
        region_file.close()
        return region_file_name

    def process_func(line):
        _list = line.split()
        bam_file = _list[0]
        condition = _list[1] if len(_list) > 2 else None
        basename = _list[-1]
        out_dir = join(_out_dir, 'data')
        if not os.path.isdir(out_dir):
            try:
                os.mkdir(out_dir)
            except:
                pass
        if file_type == 'region_file':
            file_name = region_file_func(out_dir, basename, bam_file)
        elif file_type == 'splice_file':
            file_name = splice_file_func(out_dir, basename, bam_file)
        return [(file_name, condition)]

    def save_list(results):
        if file_type == 'region_file':
            file_list = os.path.join(_out_dir, "regionFileList.txt")
        elif file_type == 'splice_file':
            file_list = os.path.join(_out_dir, "spliceFileList.txt")
        with open(file_list, 'w') as f:
            for file, condition in results.items():
                if condition:
                    f.write("{}\t{}\n".format(file, condition))
                else:
                    f.write("{}\n".format(file))
        return file_list

    def preprocess_file_list(_list):
        if '.bam' not in _list[0]:  # skip header
            _list = _list[1:]
        for i in range(len(_list)):
            _list[i] += '\tsample_{}'.format(i + 1)
        return _list

    with open(bam_file_list, 'r') as f:
        _list = f.read().splitlines()
        _list = preprocess_file_list(_list)
    mapreduce = get_mapreduce_by_threads(threads)
    results = mapreduce(_list, process_func)
    file_list = save_list(results)
    return file_list


#################################### get intron info for nb model ##########################################
def get_introns_from_intron_files(file_list, threads=1):
    def process_func(line, gene_region_introns):
        scale = 1E6
        (seq_name, start, end, count, strand, gene_region_name, num_seq_reads) = line.strip().split('\t')
        num_reads = float(count) * scale / float(num_seq_reads)
        intron = Intron(seq_name, start, end, strand, num_reads)
        gene_region_introns[region_name].append(intron)
    return read_files_from(file_list, process_func, threads=threads)


def save_selection_results(out_file, gene_region_names, betas_dict, coordinates_dict, eps):
    with open(out_file, 'w') as f:
        for gene_region_name in gene_region_names:
            selections = [1 if b > eps else 0 for b in betas_dict[gene_region_name]]
            seq_name = gene_region_name.split("_")[0]
            for selection, coordinate in zip(selections, coordinates_dict[gene_region_name]):
                start, end = coordinate
                f.write("%s\t%s\t%s\t%s\t%s\n" % (seq_name, start, end, selection, gene_region_name))

######################################## for selection results analysis #########################################
# def get_splices_from_splice_files(file_list, threads=1):
#     def process_func(line, intron_dict, *args):
#         (seq_name, start, end, count, strand, num_uniq_mapped_reads,
#             num_multi_mapped_reads) = line.strip().split(' ')[:7]
#         if int(num_uniq_mapped_reads) > 0.05*(int(num_uniq_mapped_reads)+int(num_multi_mapped_reads)):
#             intron = Intron(seq_name, start, end, strand)
#             intron_dict[intron] = 1
#     def merge_func(item):
#         intron, occurances = item
#         return (intron, sum(occurances))
#     return read_files_from(file_list, process_func, threads=threads, merge_func=merge_func)


def read_fasta_files(file_list, ref_trans_dict, threads=1):
    def _process_func(line, intron_dict, file, ref_trans_dict, *args):
        if line.startswith('>'):
            parts = line.strip('\n').split(';')
            transcript_id = parts[0].split('/')[1]
            mate = int(os.path.splitext(os.path.basename(file))[0].split('_')[-1])
            # coordinate of the read
            coordinate = tuple(parts[mate].split(':')[1].split('-'))
            if len(coordinate) == 2:
                start, end = coordinate
                transcript = ref_trans_dict[transcript_id]
                for intron, intron_location in zip(transcript.introns, transcript.intron_locations):
                    if int(start) <= intron_location <= int(end):
                        intron.count += 1
                        #intron_dict[intron] +=1

    def merge_func(item):
        intron, occurances = item
        return (intron, sum(occurances))
    process_func = partial(_process_func, ref_trans_dict=ref_trans_dict)
    return read_files_from(file_list, process_func, threads=threads)


# file = 'true.introns.txt'
def write_sampled_introns_to_file(file, ref_trans_dict):
    with open(file, 'w') as f:
        for trans_id, transcript in ref_trans_dict.items():
            for intron in transcript.introns:
                if intron.count > 0:
                    f.write("%s\t%d\t%d\t%d\n" % (intron.seq_name, intron.start, intron.end, intron.count))


def get_gtf_info(file):
    # identify genes, transcripts, introns, exons from gtf file
    lines = list()
    with open(file, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                lines.append(line.strip().split("\t"))

    gene_id_gene_dict = dict()
    genes = list()
    for tokens in lines:
        seq_name, source, feature, start, end, score, strand, frame, attribute = tokens
        if feature == 'gene':
            gene_id = attribute.split('"')[1]
            gene = Gene(seq_name, start, end, strand, gene_id=gene_id)
            gene_id_gene_dict[gene_id] = gene
            genes.append(gene)

    transcript_id_transcript_dict = dict()
    transcripts = list()
    for tokens in lines:
        seq_name, source, feature, start, end, score, strand, frame, attribute = tokens
        if feature == 'transcript':
            gene_id = attribute.split('"')[1]
            transcript_id = attribute.split('"')[3]
            transcript = Transcript(seq_name, start, end, strand, transcript_id=transcript_id)
            transcript.gene = gene_id_gene_dict[gene_id]
            gene_id_gene_dict[gene_id].transcripts.append(transcript)
            transcript_id_transcript_dict[transcript_id] = transcript
            transcripts.append(transcript)

    exon_id_exon_dict = defaultdict(lambda: None)
    exons = list()
    for tokens in lines:
        seq_name, source, feature, start, end, score, strand, frame, attribute = tokens
        if feature == 'exon':
            transcript_id = attribute.split('"')[3]
            transcript = transcript_id_transcript_dict[transcript_id]
            exon_id = "%s_%s_%s" % (seq_name, start, end)
            if exon_id_exon_dict[exon_id] is None:
                exon = Exon(seq_name, start, end, strand)
                exon_id_exon_dict[exon_id] = exon
                exons.append(exon)
            else:
                exon = exon_id_exon_dict[exon_id]
            transcript.exons.append(exon)
            exon.transcripts.append(transcript)

    intron_id_intron_dict = defaultdict(lambda: None)
    introns = list()
    for transcript in transcript_id_transcript_dict.values():
        exons = transcript.exons
        intron_locations = list()
        location = 0
        if len(exons) < 2:
            continue
        exons.sort(key=lambda exon: exon.coordinate)
        for cur_exon, next_exon in zip(exons[:-1], exons[1:]):
            seq_name = cur_exon.seq_name
            strand = cur_exon.strand
            cur_start, cur_end = cur_exon.coordinate
            next_start, next_end = next_exon.coordinate
            intron_id = "%s_%s_%s" % (seq_name, cur_end, next_start)
            if intron_id_intron_dict[intron_id] is None:
                intron = Intron(seq_name, cur_end, next_start, strand)
                intron_id_intron_dict[intron_id] = intron
                introns.append(intron)
            else:
                intron = intron_id_intron_dict[intron_id]
            transcript.introns.append(intron)
            intron.transcripts.append(transcript)
            location += cur_end - cur_start + 1
            intron_locations.append(location)
        transcript.intron_locations = intron_locations
    return genes, transcripts, exons, introns, gene_id_gene_dict, transcript_id_transcript_dict

