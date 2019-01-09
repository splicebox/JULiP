## JULiP2 ##

Copyright (C) 2017-2018, and GNU GPL v3.0, by Guangyu Yang, Liliana Florea

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.  

### <a name="table-of-contents"></a> Table of contents
- [What is JULiP2?](#what-is-julip)
- [How JULiP2 works?](#how-julip2-works)
- [Installation](#installation)
- [Usage](#usage)
- [Input/Output](#inputoutput)
- [Example](#example)
- [Support](#support)

### <a name="what-is-julip"></a> What is JULiP?
JULiP is a Python program design to select (novel) introns and detect alternative splicing from multiple samples.

### <a name="how-julip2-works"></a> How JULiP2 works?
JULiP works on aligned RNA sequencing reads (generated by Tophat, STAR, etc) using statistical methods to select introns and detect alternative splicing from multiple samples.

#### Features  
Estimates of differential isoform expression for single-end or paired-end RNA-Seq data;  
Expression estimates at the alternative splicing event (intron) level or at the whole mRNA gene level;  
Confidence intervals for expression estimates and quantitative measures of differential expression;  
Basic functionality for use on cluster / distributed computing system.

### <a name="installation"></a> Installation
JULiP is written in Python. We use the Theano library at the backend. You can install the latest version from our GitHub repository. See below for detailed installation instructions.  

#### Installation

To download the codes, you can clone this repository by,
```
git clone https://github.com/Guangyu-Yang/julip2.git
```

#### System requirement
* Linux or Mac  
* Python 2.7   

#### Required Python modules:
* [Theano](http://deeplearning.net/software/theano/), a Python library that define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays.  
* numpy  
* scipy  
* [statsmodels](http://www.statsmodels.org/stable/index.html), a Python module for the estimation of different statistical models, conducting statistical tests and data exploration.  
* [pysam](https://github.com/pysam-developers/pysam), a Python library for working with SAM/BAM files through samtools.   
* [pathos](https://pypi.python.org/pypi/pathos), a framework for heterogenous computing.  

If you are using pip, install the packages with commands:  
pip install --user theano numpy scipy statsmodels pysam pathos

#### Other required software:  
* [samtools](http://samtools.sourceforge.net/), for accessing SAM/BAM files

### <a name="usage"></a> Usage
```
Usage: python run.py [options] --bam-file-list bam_file_list.txt

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit
  --bam-file-list=BAM_FILE_LIST
                        bam file list
  --annotation=ANNOTATION_FILE
                        path of annotation file (.gtf)
  --out-dir=OUT_DIR     output directory (default: out)
  --seq-name=SEQ_NAME   specify sequence or chromosome name, None for whole
                        sequences.
  --mode=MODE           JULiP processing mode ("differential-analysis" or
                        "intron-detection").
  --threads=THREADS     number of data processing thread. (default: 1)
```

### <a name="inputoutput"></a> Input/Output
The main input of JULiP is a list of BAM files with RNA-Seq read mappings.  
As an option, the BAM file can be sorted by their genomic location and be indexed for random access.  
```
samtools sort -o accepted_hits.sorted.bam accepted_hits.bam
samtools index accepted_hits.sorted.bam
```

### <a name="example"></a> Example
#### Example: run differential analysis model:  
```
REF="path_to_gtf_file"
BAM_LIST="path_to_bam_file_list"
python run.py --bam-file-list $BAM_LIST \              
              --mode 'differential-analysis' \
              --threads 10 \              
              --annotation $REF
```
#### Example: run intron detection model:  
```
python run.py --bam-file-list $BAM_LIST \              
              --mode 'intron-detection' \
              --threads 10 \              
              --annotation $REF
```

### <a name="support"></a> Support
Contact: gyang22@jhu.edu, florea@jhu.edu  

#### License information
See the file LICENSE for information on the history of this software, terms
& conditions for usage, and a DISCLAIMER OF ALL WARRANTIES.