from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import heapq, re, gzip
from collections import defaultdict, OrderedDict, deque, Counter

import numpy as np

from utils.io_utils import read_files_from, get_mapreduce_by_threads
from utils.regions import *


##############################################  region utils  ##############################################
def merge(regions, condition, action):
    if regions:
        regions.sort(key=lambda region: (region.coordinate, region.strand))
        results = [regions[0]]
        for i in xrange(1, len(regions)):
            if condition(results[-1], regions[i]):
                action(results[-1], regions[i])
            else:
                results.append(regions[i])
        return results
    else:
        return list()


def merge_when_overlap(regions, action):
    def condition(dst, src):
        return (src.start <= dst.end)
    return merge(regions, condition, action)


def merge_when_matched(regions, action):
    def condition(dst, src):
        return (dst.start == src.start and dst.end == src.end and dst.strand == src.strand)
    return merge(regions, condition, action)


def get_regions_from(file_list, _seq_name=None, threads=1):
    def process_func(line, seq_regions, *args, **kwargs):
        seq_name, start, end, count = line.strip().split(' ')[:4]
        if not _seq_name or seq_name == _seq_name:
            if int(count) / (int(end) - int(start) + 1.0) >= 4.0:  # region's mean coverage >= 4
                seq_regions[seq_name].append(Region(seq_name, start, end, None, count))

    def action(dst, src):
        dst.end = max(src.end, dst.end)
        dst.coordinate = (dst.start, dst.end)  # for SimpleRegion
        # del src
        # dst.count += src.count

    def merge_func(seq_name, regions1, regions2=None):
        if regions2:
            regions = regions1 + regions2
            regions = merge_when_overlap(regions, action)
            return (seq_name, regions)
        else:
            return (seq_name, regions1)

    return read_files_from(file_list, process_func, threads=threads, merge_func=merge_func)


# assume the regions and boundaries are sorted
def identify_exon_regions(seq_regions, seq_boundaries):
    def action(dst, src):
        dst.intron_boudaries += src.intron_boudaries

    def get_introns(exon_region, boundary, attribute, strand):
        if boundary.attribute == attribute and (boundary.strand == strand or boundary.strand == '?'):
            if attribute == 'istart':
                for intron in boundary.start_introns:
                    intron.start_exon_region = exon_region
                return boundary.start_introns
            else:
                for intron in boundary.end_introns:
                    intron.end_exon_region = exon_region
                return boundary.end_introns
        return list()

    seq_exon_regions = OrderedDict()
    seq_names = seq_regions.keys()
    for seq_name in seq_names:
        exon_regions = list()
        regions = seq_regions[seq_name]
        if seq_name in seq_boundaries:
            boundaries = seq_boundaries[seq_name]
        else:
            continue
        index = 0
        for boundary in boundaries:
            for i in xrange(index, len(regions)):
                region = regions[i]
                if region.start <= boundary.position <= region.end:
                    # save time/space by ignoring duplicate exon_regions
                    if ((not exon_regions) or (exon_regions[-1].coordinate != region.coordinate)):
                        exon_regions.append(ExonRegion.from_region(region))
                    exon_regions[-1].intron_boudaries.append(boundary)
                elif (region.end < boundary.position):
                    index = i + 1
                else:
                    break
        exon_regions = merge_when_matched(exon_regions, action)
        for exon_region in exon_regions:
            map(lambda b: exon_region.plus_start_introns.extend(
                get_introns(exon_region, b, 'istart', '+')), exon_region.intron_boudaries)
            map(lambda b: exon_region.minus_start_introns.extend(
                get_introns(exon_region, b, 'istart', '-')), exon_region.intron_boudaries)
            map(lambda b: exon_region.plus_end_introns.extend(
                get_introns(exon_region, b, 'iend', '+')), exon_region.intron_boudaries)
            map(lambda b: exon_region.minus_end_introns.extend(
                get_introns(exon_region, b, 'iend', '-')), exon_region.intron_boudaries)
        seq_exon_regions[seq_name] = exon_regions
    return seq_exon_regions


def write_seq_regions(file, seq_regions):
    func = lambda r: "%s\t%s\t%s\t%s\n" % (r.seq_name, r.start, r.end, r.count)
    write(file, seq_regions, func)


##############################################  intron utils  ##############################################
def get_introns_from(splice_file_list, _seq_name=None, threads=1, _filter=3):
    def process_func_1(line):
        tokens = line.split('\t')
        if len(tokens) == 2:
            file, condition = tokens
        else:
            file = tokens[0]
            condition = 0
        compress = file.endswith('.gz')
        f = open(file, 'r') if not compress else gzip.open(file, 'rb')
        num_lines = 0
        for line in f:
            num_lines += 1
        return [(file, (num_lines, condition))]

    def process_func_2(file_dict, threads):
        thread_dict = defaultdict(dict)
        for file, (num_lines, condition) in file_dict.items():
            block = num_lines // threads
            for i in range(threads):
                if i < threads - 1:
                    thread_dict[i][file] = ((i * block, (i + 1) * block - 1), condition)
                else:
                    thread_dict[i][file] = ((i * block, num_lines), condition)
        return thread_dict.items()

    # read file and process splice junction in given range
    def process_func_3(value):
        seq_intron_dict = defaultdict(dict)
        thread, file_range_cond_dict = value
        for file, (_range, condition) in file_range_cond_dict.items():
            line_start, line_end = _range
            compress = file.endswith('.gz')
            f = open(file, 'r') if not compress else gzip.open(file, 'rb')
            for i, line in enumerate(f):
                if i >= line_start:
                    (seq_name, start, end, _num_reads, strand, num_uniq_mapped_reads,
                    num_multi_mapped_reads, _, _ , primary_mapped_reads, multi_hits_reads) = line.strip().split(' ')
                    if int(primary_mapped_reads) > 0:
                        file_obj = File(file, int(primary_mapped_reads), condition)
                        intron = Intron(seq_name, start, end, strand, int(primary_mapped_reads))
                        intron.file_objs.append(file_obj)
                        if intron in seq_intron_dict[seq_name]:
                            seq_intron_dict[seq_name][intron].count += max(seq_intron_dict[seq_name][intron].count, intron.count)
                            seq_intron_dict[seq_name][intron].file_objs += intron.file_objs
                            seq_intron_dict[seq_name][intron].sum += intron.sum
                        else:
                            seq_intron_dict[seq_name][intron] = intron
                if i == line_end:
                    break
            f.close()
        return [(str(thread), seq_intron_dict)]

    # merge seq_inton_dict
    def process_func_4(seq_introns_dicts, n):
        seq_introns_dict = defaultdict(dict)
        for _seq_introns_dict in seq_introns_dicts:
            for seq_name, intron_dict in _seq_introns_dict.items():
                for intron in intron_dict.values():
                    if intron in seq_introns_dict[seq_name]:
                        seq_introns_dict[seq_name][intron].count = max(seq_introns_dict[seq_name][intron].count, intron.count)
                        seq_introns_dict[seq_name][intron].file_objs += intron.file_objs
                    else:
                        seq_introns_dict[seq_name][intron] = intron
        seq_introns = defaultdict(list)
        for seq_name, intron_dict in seq_introns_dict.items():
            for intron in intron_dict.values():
                # filter intron
                if intron.sum > max(int(n * 0.04), 2):
                    seq_introns[seq_name].append(intron)
        return seq_introns

    with open(splice_file_list, 'r') as f:
        _list = f.read().splitlines()
        if '.bam' not in _list[0]:  # skip header
            _list = _list[1:]

    threads = 10
    mapreduce = get_mapreduce_by_threads(threads)
    file_dict = mapreduce(_list, process_func_1)
    thread_list = process_func_2(file_dict, threads)
    mapreduce = get_mapreduce_by_threads(threads)
    seq_intron_dict_dict = mapreduce(thread_list, process_func_3)
    n = len(file_dict)
    seq_introns = process_func_4(seq_intron_dict_dict.values(), n)
    return seq_introns


# this function not support non-canonical introns
def group_introns(seq_gene_regions, condition_counter):
    for gene_regions in seq_gene_regions.values():
        plus_postion_group_dict = defaultdict(set)
        minus_postion_group_dict = defaultdict(set)
        for gene_region in gene_regions:
            for intron in gene_region.introns:
                if intron.selected:
                    intron.cond_psi_dict = {}
                    start = intron.start
                    end = intron.end
                    postion_group_dict = plus_postion_group_dict
                    if intron.strand == '-':
                        postion_group_dict = minus_postion_group_dict
                    group1 = postion_group_dict[start]
                    group2 = postion_group_dict[end]
                    group = group1.union(group2).union([intron])
                    postion_group_dict[start] = group
                    postion_group_dict[end] = group

        for postion_group_dict in [plus_postion_group_dict, minus_postion_group_dict]:
            for introns in postion_group_dict.values():
                for cond, count in condition_counter.items():
                    _sum = sum(i.cond_count_dict[cond] for i in introns)
                    for intron in introns:
                        psi = 0 if _sum == 0 else intron.cond_count_dict[cond] / _sum
                        intron.cond_psi_dict[cond] = psi



##############################################  Boundary(splice side) utils  ##############################################
def get_seq_boundaries(seq_introns):
    def generate_boundary(intron, boundaries):
        seq_name, start, end, strand = intron.seq_name, intron.start, intron.end, intron.strand
        boundaries.append(Boundary(seq_name, start, 'istart', strand))
        boundaries[-1].start_introns = [intron]
        boundaries.append(Boundary(seq_name, end, 'iend', strand))
        boundaries[-1].end_introns = [intron]

    def merge_duplicate_boundaries(boundaries):
        if boundaries:
            boundaries.sort(key=lambda boundary: (boundary.position, boundary.attribute, boundary.strand))
            results = [boundaries[0]]
            for i in xrange(1, len(boundaries)):
                if (boundaries[i].position != results[-1].position
                        or boundaries[i].attribute != results[-1].attribute
                        or boundaries[i].strand != results[-1].strand):
                    results.append(boundaries[i])
                else:
                    if boundaries[i].attribute is 'istart':
                        results[-1].start_introns += boundaries[i].start_introns
                    else:
                        results[-1].end_introns += boundaries[i].end_introns
            return results
        else:
            return list()

    seq_boundaries = OrderedDict()
    for seq_name, introns in seq_introns.items():
        boundaries = list()
        for intron in introns:
            generate_boundary(intron, boundaries)
        seq_boundaries[seq_name] = merge_duplicate_boundaries(boundaries)
    return seq_boundaries


##############################################  gene region utils  ##############################################
def generate_gene_regions(seq_exon_regions):
    def generate_strand_specific_exon_regions(exon_regions):
        plus_exon_regions = list()
        minus_exon_regions = list()
        for exon_region in exon_regions:
            if exon_region.plus_start_introns or exon_region.plus_end_introns:
                plus_exon_regions.append(exon_region)
            if exon_region.minus_start_introns or exon_region.minus_end_introns:
                minus_exon_regions.append(exon_region)
        return plus_exon_regions, minus_exon_regions

    def generate_gene_region(exon_regions, strand):
        mask = strand + "gene"
        queue = deque([exon_regions[0]])
        exon_regions[0].mask = mask  # masked when the exon regions have been visited
        _exon_regions = list()
        introns_set = set()
        seq_name = exon_regions[0].seq_name
        start = np.inf
        end = 0
        while queue:
            current = queue.popleft()
            _exon_regions.append(current)
            start = start if start < current.start else current.start
            end = end if end > current.end else current.end
            if current not in exon_regions:
                raise Exception('%s not in exon_regions', current)
            exon_regions.remove(current)
            if strand == '+':
                start_introns = current.plus_start_introns
                end_introns = current.plus_end_introns
            else:
                start_introns = current.minus_start_introns
                end_introns = current.minus_end_introns
            introns_set.update(start_introns)
            introns_set.update(end_introns)
            for intron in start_introns:
                if intron.end_exon_region and intron.end_exon_region.mask != mask:
                    queue.append(intron.end_exon_region)
                    intron.end_exon_region.mask = mask
            for intron in end_introns:
                if intron.start_exon_region and intron.start_exon_region.mask != mask:
                    queue.append(intron.start_exon_region)
                    intron.start_exon_region.mask = mask

        gene_region = GeneRegion(seq_name, start, end, strand)
        gene_region.exon_regions = _exon_regions
        for exon_region in gene_region.exon_regions:
            if strand == '+':
                exon_region.plus_gene_region = gene_region
            else:
                exon_region.minus_gene_region = gene_region
        gene_region.introns = list(introns_set)
        for intron in gene_region.introns:
            intron.gene_region = gene_region
        return gene_region

    seq_gene_regions = OrderedDict()
    for seq_name, exon_regions in seq_exon_regions.items():
        plus_exon_regions, minus_exon_regions = generate_strand_specific_exon_regions(exon_regions)
        gene_regions = list()
        while plus_exon_regions:
            gene_regions.append(generate_gene_region(plus_exon_regions, '+'))
        while minus_exon_regions:
            gene_regions.append(generate_gene_region(minus_exon_regions, '-'))
        seq_gene_regions[seq_name] = sorted(gene_regions, key=lambda region: (region.coordinate, region.strand))
    return seq_gene_regions


##
# Warning: this function doesn't update the info in intron, exon_region
#
def cut_gene_regions(seq_gene_regions, n=3, fold=2.):
    def set_represent_introns(seq_gene_regions):
        for seq_name, gene_regions in seq_gene_regions.items():
            for gene_region in gene_regions:
                file_intron_dict = defaultdict(list)
                file_sum_dict = defaultdict(lambda: 0.)
                for intron in gene_region.introns:
                    for file_obj in intron.file_objs:
                        file_intron_dict[file_obj.file].append((intron, file_obj.count))
                        file_sum_dict[file_obj.file] += file_obj.count
                        intron.count = 0.
                _file = None
                max_sum = -1
                for file, _sum in file_sum_dict.items():
                    if _sum > max_sum:
                        max_sum = _sum
                        _file = file
                for intron, count in file_intron_dict[_file]:
                    intron.count = count

    def get_cut_point(mv_list):
        # get intron that the cut point right after this intron
        _list = sorted(mv_list, key=lambda i: i[1].coordinate)
        maxdiff = 0
        index = 1
        for i in range(1, len(_list) - 1):
            drop1 = abs(_list[i - 1][0] - _list[i][0])
            drop2 = abs(_list[i][0] - _list[i + 1][0])
            diff = abs(drop1 - drop2)
            if diff > maxdiff:
                maxdiff = diff
                if drop1 > drop2:
                    index = i - 1
                else:
                    index = i
        return _list[index][1].end + 1

    set_represent_introns(seq_gene_regions)
    for seq_name, gene_regions in seq_gene_regions.items():
        new_gene_regions = list()
        for gene_region in gene_regions:
            introns = sorted(gene_region.introns, key=lambda i: i.coordinate)
            value = 0.
            _heap = list()
            cut_points = list()
            # moving average parameters
            mv_list = [(1, None)] * n
            fold1 = fold
            fold2 = 1. / fold1
            k = 0
            for i in range(len(introns)):
                while len(_heap) > 0:
                    if _heap[0].end <= introns[i].start:
                        old_intron = heapq.heappop(_heap)
                        # ext_intron = -1
                        # if len(_heap) > 0 :
                        #     next_intron = _heap[0]
                        value -= old_intron.count
                    else:
                        break
                if introns[i].count > 0:
                    heapq.heappush(_heap, introns[i])
                    value += introns[i].count
                    sum1 = sum([l[0] for l in mv_list]) / len(mv_list)
                    mv_list[k % n] = (value, introns[i])
                    sum2 = sum([l[0] for l in mv_list]) / len(mv_list)
                    tmp = sum1 / sum2
                    k += 1
                    if k > 3 and (tmp >= fold1 or tmp <= fold2):
                        cut_points.append(get_cut_point(mv_list))

            if cut_points:
                cut_points = [introns[0].start - 1] + cut_points + [introns[-1].end + 1]
                for j in range(1, len(cut_points)):
                    _list = filter(lambda i: cut_points[j - 1] <= i.start and i.end < cut_points[j], introns)
                    new_gene_region = GeneRegion(seq_name, cut_points[j - 1], cut_points[j], gene_region.strand)
                    new_gene_region.introns = _list
                    # for intron in new_gene_region.introns:
                    #     intron.gene_region = new_gene_region
                    ###
                    # add codes here for the exon_regions
                    ###
                    new_gene_regions.append(new_gene_region)
            else:
                new_gene_regions.append(gene_region)

        seq_gene_regions[seq_name] = sorted(new_gene_regions, key=lambda region: (region.coordinate, region.strand))


def map_to_annotated_genes(annotation_file, _seq_gene_regions, seq_introns, target_seq=None):
    seq_gene_regions = defaultdict(list)
    id_gene_dict = {}
    plus_coordinate_dict = defaultdict(lambda: defaultdict(set))
    minus_coordinate_dict = defaultdict(lambda: defaultdict(set))

    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue

            seq_name, _, _type, start, end, _, strand, _, strings = line.split('\t', 8)
            if ((not target_seq) or (target_seq == seq_name)):
                start, end = int(start), int(end)
                if _type == 'gene':
                    gene_region = GeneRegion(seq_name, start, end, strand)
                    for token in strings.split(';'):
                        if token.strip().startswith('gene_id'):
                            gene_id = token.strip().split(' ')[1][1:-1]
                            gene_region.ensembl_id = gene_id
                            gene_name = ''
                        elif token.strip().startswith('gene_name'):
                            gene_name = token.strip().split(' ')[1][1:-1]
                    gene_region.gene_name = gene_name
                    _id = '-'.join((gene_id, gene_name))
                    id_gene_dict[_id] = gene_region
                    seq_gene_regions[seq_name].append(gene_region)
                if _type == 'exon':
                    for token in strings.split(';'):
                        if token.strip().startswith('gene_id'):
                            gene_id = token.strip().split(' ')[1][1:-1]
                            gene_name = ''
                        elif token.strip().startswith('gene_name'):
                            gene_name = token.strip().split(' ')[1][1:-1]
                    _id = '-'.join((gene_id, gene_name))
                    if _id not in id_gene_dict:
                        gene_region = GeneRegion(seq_name, start, end, strand)
                        gene_region.ensembl_id = gene_id
                        gene_region.gene_name = gene_name
                        id_gene_dict[_id] = gene_region
                        seq_gene_regions[seq_name].append(gene_region)
                    else:
                        id_gene_dict[_id].start = min(gene_region.start, start)
                        id_gene_dict[_id].end = max(gene_region.end, end)

                    if strand == '+':
                        plus_coordinate_dict[seq_name][start].add(id_gene_dict[_id])
                        plus_coordinate_dict[seq_name][end].add(id_gene_dict[_id])
                    else:
                        minus_coordinate_dict[seq_name][start].add(id_gene_dict[_id])
                        minus_coordinate_dict[seq_name][end].add(id_gene_dict[_id])

    for seq_name, gene_regions in _seq_gene_regions.items():
        if seq_name not in seq_gene_regions:
            continue
        for gene_region in gene_regions:
            _gene_regions = set()
            introns = []
            for intron in gene_region.introns:
                if (intron.strand == '+' or intron.strand == '?'):
                    coordinate_dict = plus_coordinate_dict[seq_name]

                if (intron.strand == '-' or intron.strand == '?'):
                    coordinate_dict = minus_coordinate_dict[seq_name]

                start, end = intron.coordinate
                gene_regions1 = coordinate_dict[start]
                gene_regions2 = coordinate_dict[end]
                gene_regions3 = gene_regions1.union(gene_regions2)
                if gene_regions3:
                    for _gene_region in gene_regions3:
                        _gene_region.introns.append(intron)
                        intron.gene_region = _gene_region
                        _gene_regions.add(_gene_region)
                else:
                    introns.append(intron)
            if _gene_regions and introns:
                for intron in introns:
                    for _gene_region in _gene_regions:
                        _gene_region.introns.append(intron)
                        intron.gene_region = _gene_region
            if not(_gene_regions) and introns:
                if seq_name in seq_gene_regions:
                    seq_gene_regions[seq_name].append(gene_region)
    return seq_gene_regions


def get_genes_from_annotation(annotation_file, seq_introns, target_seq=None, n=0):
    seq_gene_regions = defaultdict(list)
    id_gene_dict = {}
    plus_coordinate_dict = defaultdict(lambda: defaultdict(set))
    minus_coordinate_dict = defaultdict(lambda: defaultdict(set))

    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue

            seq_name, _, _type, start, end, _, strand, _, strings = line.split('\t', 8)
            if ((not target_seq) or (target_seq == seq_name)):
                start, end = int(start), int(end)
                if _type == 'gene':
                    gene_region = GeneRegion(seq_name, start, end, strand)
                    for token in strings.split(';'):
                        if token.strip().startswith('gene_id'):
                            gene_id = token.strip().split(' ')[1][1:-1]
                            gene_region.ensembl_id = gene_id
                            gene_name = ''
                        elif token.strip().startswith('gene_name'):
                            gene_name = token.strip().split(' ')[1][1:-1]
                    gene_region.gene_name = gene_name
                    _id = '-'.join((gene_id, gene_name))
                    id_gene_dict[_id] = gene_region
                    seq_gene_regions[seq_name].append(gene_region)
                if _type == 'exon':
                    for token in strings.split(';'):
                        if token.strip().startswith('gene_id'):
                            gene_id = token.strip().split(' ')[1][1:-1]
                            gene_name = ''
                        elif token.strip().startswith('gene_name'):
                            gene_name = token.strip().split(' ')[1][1:-1]
                    _id = '-'.join((gene_id, gene_name))
                    if _id not in id_gene_dict:
                        gene_region = GeneRegion(seq_name, start, end, strand)
                        gene_region.ensembl_id = gene_id
                        gene_region.gene_name = gene_name
                        id_gene_dict[_id] = gene_region
                        seq_gene_regions[seq_name].append(gene_region)
                    else:
                        id_gene_dict[_id].start = min(gene_region.start, start)
                        id_gene_dict[_id].end = max(gene_region.end, end)

                    if strand == '+':
                        for i in range(end - n, end + n + 1):
                            plus_coordinate_dict[seq_name][i].add(id_gene_dict[_id])
                        for i in range(start - n, start + n + 1):
                            plus_coordinate_dict[seq_name][i].add(id_gene_dict[_id])
                    else:
                        for i in range(end - n, end + n + 1):
                            minus_coordinate_dict[seq_name][i].add(id_gene_dict[_id])
                        for i in range(start - n, start + n + 1):
                            minus_coordinate_dict[seq_name][i].add(id_gene_dict[_id])

    def assign_intron_to_genes(intron, coordinate_dict):
        start, end = intron.coordinate
        gene_regions1 = coordinate_dict[start]
        gene_regions2 = coordinate_dict[end]
        for gene_region in gene_regions1.union(gene_regions2):
            gene_region.introns.append(intron)

    for seq_name, introns in seq_introns.items():
        for intron in introns:
            if (intron.strand == '+' or intron.strand == '?'):
                assign_intron_to_genes(intron, plus_coordinate_dict[seq_name],
                                      plus_coordinate_dict[seq_name])
            if (intron.strand == '-' or intron.strand == '?'):
                assign_intron_to_genes(intron, minus_coordinate_dict[seq_name],
                                      minus_coordinate_dict[seq_name])

    for gene_regions in seq_gene_regions.values():
        gene_regions.sort(key=lambda region: (region.coordinate, region.strand))
    return seq_gene_regions


##############################################  dataset and coordinate  ##############################################
def get_file_seq_info(seq_introns, target_seq=None):
    file_counts_dict = defaultdict(lambda: defaultdict(lambda: 0.))
    scale_dict = defaultdict(lambda: 0.)
    for seq_name, introns in seq_introns.items():
        if (not target_seq) or (target_seq == seq_name):
            for intron in introns:
                for file_obj in intron.file_objs:
                    # file_counts_dict: dict to store read counts for sequences/chromosomes
                    file_counts_dict[file_obj.file][seq_name] += file_obj.count
            scale_dict[seq_name] = sum([file_counts_dict[file][seq_name] for file in file_counts_dict.keys()])
            scale_dict[seq_name] /= len(file_counts_dict.keys())
    return scale_dict, file_counts_dict


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def get_dataset_for_differential_analysis(seq_gene_regions, scale_dict, file_counts_dict, target_seq=None):
    dataset = defaultdict(list)
    coordinate_dict = defaultdict(list)
    id_gene_dict = OrderedDict()
    files = file_counts_dict.keys()
    seq_names = sorted(seq_gene_regions.keys(), key=natural_keys)
    file_condition_dict = dict.fromkeys(files, 0.)
    for seq_name in seq_names:
        gene_regions = seq_gene_regions[seq_name]
        if (not target_seq) or (target_seq == seq_name):
            for gene_region in gene_regions:
                _id = gene_region.gene_id
                id_gene_dict[_id] = gene_region
                introns = []
                for intron in gene_region.introns:
                    if intron.selected:
                        introns.append(intron)
                        intron_count_dict = dict.fromkeys(files, 0.)
                        cond_count_dict = defaultdict(lambda: 0.)
                        for file_obj in intron.file_objs:
                            normalized_count = file_obj.count * scale_dict[seq_name] / file_counts_dict[file_obj.file][seq_name]
                            intron_count_dict[file_obj.file] = normalized_count
                            file_obj.normalized_count = normalized_count
                            cond_count_dict[file_obj.condition] += normalized_count
                            file_condition_dict[file_obj.file] = file_obj.condition
                        intron.cond_count_dict = cond_count_dict
                        dataset[_id].append(intron_count_dict.values())
                        coordinate_dict[_id].append(intron.coordinate)
                gene_region.introns = introns

    _conditions = file_condition_dict.values()
    condition_counter = Counter(_conditions)
    # ex: 'control', 'test' -> 0, 1
    condition_list = list(set(_conditions))
    conditions = []
    for cond in _conditions:
        conditions.append(condition_list.index(cond))

    length = len(conditions)
    conditions = np.asarray(conditions, dtype=np.float32).reshape((length, 1))

    for gene_id, matrix in dataset.items():
        dataset[gene_id] = (np.asarray(matrix).T, conditions)

    return id_gene_dict, coordinate_dict, dataset, condition_counter


def get_dataset_for_intron_detection(seq_gene_regions, scale_dict, file_counts_dict, target_seq=None):
    dataset = defaultdict(list)
    id_gene_dict = OrderedDict()
    files = file_counts_dict.keys()
    seq_names = sorted(seq_gene_regions.keys(), key=natural_keys)
    for seq_name in seq_names:
        gene_regions = seq_gene_regions[seq_name]
        if (not target_seq) or (target_seq == seq_name):
            for gene_region in gene_regions:
                _id = gene_region.gene_id
                id_gene_dict[_id] = gene_region
                for intron in gene_region.introns:
                    intron_count_dict = dict.fromkeys(files, 0.)
                    for file_obj in intron.file_objs:
                        normalized_count = file_obj.count * scale_dict[seq_name] / file_counts_dict[file_obj.file][seq_name]
                        intron_count_dict[file_obj.file] = normalized_count
                    dataset[_id].append(intron_count_dict.values())

    for gene_id, matrix in dataset.items():
        dataset[gene_id] = np.asarray(matrix).T

    return id_gene_dict, dataset

