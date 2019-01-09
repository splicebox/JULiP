from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import optparse
import sys
import os
import time
import logging

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from utils.data_utils import generate_files
from utils.region_utils import *

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d-%y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


def get_arguments():
    parser = optparse.OptionParser(usage="usage: python %prog [options] --bam-file-list bam_file_list.txt",
                        version="%prog 1.0")
    parser.add_option("--bam-file-list", dest="bam_file_list", type=str, help="bam file list")
    parser.add_option("--annotation", dest="annotation_file", type=str, default=None,
                        help="path of annotation file (.gtf)")
    parser.add_option("--out-dir", dest="out_dir", type=str, default="out",
                        help="output directory (default: %default)")
    parser.add_option("--seq-name", dest="seq_name", type=str, default=None,
                        help="specify sequence or chromosome name, None for whole sequences.")
    parser.add_option("--cut-regions", action="store_false", dest="cut_regions", default=False,
                        help=optparse.SUPPRESS_HELP)
    parser.add_option("--test-method", dest="test_method", type="choice", default="pearson",
                        choices=["log-likelihood", "pearson", "freeman-tukey", "mod-log-likelihood", "neyman", "cressie-read"],
                        help=optparse.SUPPRESS_HELP)
    parser.add_option("--mode", dest="mode", type="choice", choices=['differential-analysis', 'intron-detection'],
                        help='JULiP processing mode ("differential-analysis" or "intron-detection").')
    parser.add_option("--compress", action="store_true", dest="compress", default=True,
                        help=optparse.SUPPRESS_HELP)
    parser.add_option("--threads", dest="threads", type=int, default="4",
                        help="number of data processing threads, at least 4 threads is required. (default: %default)")
    parser.add_option("--splice-file-list", dest="splice_file_list", type=str, default=None,
                        help="manually provide splice file list")
    parser.add_option("--region-file-list", dest="region_file_list", type=str, default=None,
                        help="manually provide region file list")
    parser.add_option("--debug", action="store_false", dest="debug", default=False,
                        help="add normalized read counts and raw read counts to differential analysis output file")

    (options, args) = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    check_options(options)
    return options


def check_options(options):
    if options.threads < 4:
        options.threads = 4


def differential_analysis(seq_gene_regions, seq_introns, splice_file_list, out_dir, threads=4,
                            ds_threshold=0.2, de_threshold=0.05, lambda_="pearson", debug=False):
    def init_process():
        global nb_model
        nb_model = NB1.NegativeBinomialModel(lo_beta=-6, hi_beta=10, lo_alpha=1E-7, hi_alpha=30,
                                            lr=1., decay=0.0003, alpha_threshold=0.005)

    def train_model_star(values):
        gene_id, (data, conditions) = values
        indices0 = np.where(conditions == 0)[0]
        indices1 = np.where(conditions == 1)[0]
        sum0 = np.sum(np.mean(data[indices0, :], axis=0))
        sum1 = np.sum(np.mean(data[indices1, :], axis=0))
        if sum0 > sum1:
            conditions = 1 - conditions

        vector0 = np.mean(data[indices0, :], axis=0)
        vector1 = np.mean(data[indices1, :], axis=0)

        global nb_model1
        # global nb_model2
        likelihoods = nb_model.train_model(data, data, conditions, 1000)
        a = np.sum(likelihoods[0], axis=0)
        b = np.sum(likelihoods[1], axis=0)

        p_values = 1 - stats.chi2.cdf(2 * (a - b), 2)

        rejects, q_values, _, _ = multipletests(p_values, method='fdr_bh')
        gene_vector0 = np.mean(data[indices0, :], axis=1)
        gene_vector1 = np.mean(data[indices1, :], axis=1)
        gene_vector = np.asarray([np.mean(gene_vector0), np.mean(gene_vector1)])
        distance0 = np.mean(gene_vector0)
        distance1 = np.mean(gene_vector1)
        diff_expr = diff_expr = np.log2(distance1 / distance0)
        return gene_id, (diff_expr, p_values, q_values, rejects,
                         vector0, vector1, distance0, distance1, gene_vector)

    from pathos import multiprocessing
    import models.differential_analysis_model as NB1

    logger.info("Preparing training data ...")
    scale_dict, file_counts_dict = get_file_seq_info(seq_introns)
    id_gene_dict, coordinate_dict, dataset, condition_counter = \
                        get_dataset_for_differential_analysis(seq_gene_regions, scale_dict, file_counts_dict)

    group_introns(seq_gene_regions, condition_counter)

    sig_genes_fh = open(os.path.join(out_dir, 'all_genes.txt'), 'w')
    diff_splicing_fh = open(os.path.join(out_dir, 'diff_splicing.txt'), 'w')
    sig_gene_str = "\t".join(['ID', 'ensembl_ID', 'gene_symbol', 'type', 'chromosome', 'start', 'end',
                            'strand'])
    splicing_str = "\t".join(['ID', 'gene_id', 'gene_name', 'type', 'chromosome', 'start', 'end', 'strand',
                            'value1', 'value2', 'log2(fold_change)', 'state', 'p_value', 'q_value',
                            'psi1', 'psi2', 'delta_psi'])
    if debug:
        splicing_str = "\t".join([splicing_str, 'normalized_read_counts', 'raw_read_counts'])

    sig_genes_fh.write(sig_gene_str + '\n')
    diff_splicing_fh.write(splicing_str + '\n')

    pool = multiprocessing.Pool(processes=threads, initializer=init_process)
    logger.info("Training differential analysis models ...")
    start = time.time()
    results = pool.map(train_model_star, dataset.items())
    end = time.time()
    logger.info("Models took %f seconds to process all data" % int(end - start))

    logger.info("Writing results to files ...")
    results = dict(results)

    g = 0
    for i, (_id, gene_region) in enumerate(id_gene_dict.items()):
        if _id not in results:
            continue

        diff_expr, p_values, q_values, rejects, \
            vector0, vector1, distance0, distance1, \
            gene_vector = results[_id]

        coordinates = coordinate_dict[_id]

        sig_splicing = 'no'
        if (rejects.size > 0 and any(rejects)):
            sig_splicing = 'yes'

        g += 1
        gene_id = 'G{0:06}'.format(g)
        gene_name = gene_region.gene_name if gene_region.gene_name else '-'
        ensembl_id = gene_region.ensembl_id if gene_region.ensembl_id else '-'

        i = 1
        for intron, p_value, q_value, coord, v1, v2, flag in \
                zip(gene_region.introns, p_values, q_values, coordinates, vector0, vector1, rejects):
            psi1, psi2 = intron.cond_psi_dict.values()
            delta_psi = abs(psi1 - psi2)
            log2_ratio = np.log2(v2 / v1)
            intron_id = gene_id + ':I{0:03}'.format(i)
            intron_state = 'NOTEST' if p_value == '-' else 'TESTED'
            splicing_str = "\t".join([intron_id, gene_id, gene_name, 'intron', gene_region.seq_name,
                            str(coord[0]), str(coord[1]), gene_region.strand,
                            str(v1), str(v2),
                            str(log2_ratio), str(intron_state),
                            str(p_value), str(q_value),
                            str(psi1), str(psi2), str(delta_psi)])
            diff_splicing_fh.write(splicing_str + '\n')
            i += 1

        if sig_splicing == 'yes':
            sig_gene_str = "\t".join([gene_id, ensembl_id, gene_name, 'gene', gene_region.seq_name,
                                     str(gene_region.start), str(gene_region.end), gene_region.strand])
            sig_genes_fh.write(sig_gene_str + '\n')

        diff_splicing_fh.flush()
        sig_genes_fh.flush()

    sig_genes_fh.close()
    diff_splicing_fh.close()


def intron_detection(seq_gene_regions, seq_introns, splice_file_list, out_dir, threads=4):
    def init_process():
        global nb_model
        nb_model = NegativeBinomialModel(lo_beta=1e-4, lo_alpha=1e-4, hi_alpha=3, lr=1,
                                        reg_beta=2, reg_alpha=0.1, decay=0.1)

    def train_model_star(values):
        gene_id, data = values
        global nb_model
        betas = nb_model.train_model(data, 1000)
        time.sleep(0.002)
        return gene_id, betas

    from pathos import multiprocessing
    from models.intron_detection_model import NegativeBinomialModel
    logger.info("Preparing training data ...")
    scale_dict, file_counts_dict = get_file_seq_info(seq_introns)
    id_gene_dict, dataset = get_dataset_for_intron_detection(seq_gene_regions, scale_dict, file_counts_dict)
    print("hello worlds in get_dataset_for_intron_detection")

    diff_splicing_fh = open(os.path.join(out_dir, 'introns.results.txt'), 'w')
    splicing_str = "\t".join(['ID', 'gene_id', 'gene_name', 'type', 'chromosome',
                            'start', 'end', 'strand', 'beta_value'])
    diff_splicing_fh.write(splicing_str + '\n')

    pool = multiprocessing.Pool(processes=threads, initializer=init_process)
    logger.info("Training intron detection models ...")
    start = time.time()
    results = pool.map(train_model_star, dataset.items())
    end = time.time()
    logger.info("Models took %f seconds to process all data" % (end - start))

    logger.info("Writing results to files ...")
    results = dict(results)

    epsilon = 1. / 10
    g = 0
    for _id, gene_region in id_gene_dict.items():
        if _id not in results:
            continue
        betas = results[_id]
        g += 1
        gene_id = 'G{0:06}'.format(g)
        gene_name = gene_region.gene_name if gene_region.gene_name else '-'
        i = 1
        for beta, intron in zip(betas, gene_region.introns):
            intron_id = gene_id + ':I{0:03}'.format(i)
            if beta < epsilon:
                intron.selected = False
            coord = intron.coordinate
            splicing_str = "\t".join([intron_id, gene_id, gene_name, 'intron', gene_region.seq_name,
                                str(coord[0]), str(coord[1]), gene_region.strand,
                                "{:.4f}".format(beta)])
            diff_splicing_fh.write(splicing_str + '\n')
            i += 1

        diff_splicing_fh.flush()
    diff_splicing_fh.close()


def run(options):
    if not os.path.isdir(options.out_dir):
        logger.info("Creating output directory: {}".format(options.out_dir))
        os.mkdir(options.out_dir)
    # with_annotation = True if options.annotation_file else False
    if options.splice_file_list is None:
        logger.info("Detecting splice junctions ...")
        splice_file_list = generate_files(options.bam_file_list, options.out_dir, 'splice_file',
                                        threads=options.threads)
    else:
        splice_file_list = options.splice_file_list

    logger.info("Processing and merging splice junctions ...")
    seq_introns = get_introns_from(splice_file_list, options.seq_name, threads=options.threads)

    if options.region_file_list is None:
        logger.info("Detecting mapping regions ...")
        region_file_list = generate_files(options.bam_file_list, options.out_dir, 'region_file',
                                        threads=options.threads)
    else:
        region_file_list = options.region_file_list

    logger.info("Processing and merging mapping regions ...")
    seq_regions = get_regions_from(region_file_list, options.seq_name, threads=options.threads)
    logger.info("Identifying boundaries ...")
    seq_boundaries = get_seq_boundaries(seq_introns)
    logger.info("Identifying exon regions ...")
    seq_exon_regions = identify_exon_regions(seq_regions, seq_boundaries)
    logger.info("Generating gene regions ...")

    # these gene regions can also be named as gene segments
    seq_gene_regions = generate_gene_regions(seq_exon_regions)
    if options.cut_regions:
        cut_gene_regions(seq_gene_regions)

    if options.annotation_file is not None:
        # these are the genes from annotated file
        logger.info("Applying reference annotation ...")
        seq_gene_regions = map_to_annotated_genes(options.annotation_file, seq_gene_regions, seq_introns, options.seq_name)

    # for gene_regions in seq_gene_regions.values():
    #     for gene_region in gene_regions:
    #         print(len(gene_region.introns))

    if options.mode == 'differential-analysis':
        logger.info("Starting intron detection models")
        intron_detection(seq_gene_regions, seq_introns, splice_file_list, options.out_dir, threads=options.threads)
        logger.info("Starting differential analysis models")
        differential_analysis(seq_gene_regions, seq_introns, splice_file_list, options.out_dir,
                                            threads=options.threads, lambda_=options.test_method, debug=options.debug)

    if options.mode == 'intron-detection':
        logger.info("Starting intron detection models")
        intron_detection(seq_gene_regions, seq_introns, splice_file_list, options.out_dir, threads=options.threads)


def main():
    options = get_arguments()
    start = time.time()
    logger.info("Beginning JULiP run")
    run(options)
    logger.info("End program")
    end = time.time()
    logger.info("Julip took %f seconds to run" % (end - start))


if __name__ == '__main__':
    main()

