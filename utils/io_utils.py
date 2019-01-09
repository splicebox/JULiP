from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict, OrderedDict
import os, uuid, logging, gzip, zlib, shutil
from functools import partial
from pathos import multiprocessing
import cPickle as pickle

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d-%y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


class MapReduce(object):
    def __init__(self):
        pass

    def start(self):
        pass

    def mapper(self, map_func, inputs, chunksize=1):
        self.map_responses = map(map_func, inputs)

    def partitioner(self):
        self.partitioned_data = defaultdict(list)
        for mapped_values in self.map_responses:
            for key, value in mapped_values:
                self.partitioned_data[key].append(value)

    def reducer(self, reduce_func):
        def func(reduce_func, inputs):
            return map(reduce_func, inputs)

        if reduce_func:
            reduced_values = func(reduce_func, list(self.partitioned_data.items()))
            output_values = OrderedDict()
            for key, value in reduced_values:
                output_values[key] = value
            self.output_values = output_values
        else:
            self.output_values = self.partitioned_data

    def finish(self):
        pass

    def __call__(self, inputs, map_func, reduce_func=None):
        self.start()
        self.mapper(map_func, inputs)
        self.partitioner()
        self.reducer(reduce_func)
        self.finish()
        return self.output_values


class MultiprocessingMapReduce(MapReduce):
    def __init__(self, num_workers=None):
        self.num_workers = num_workers
        self.tmp_dir = '.' + str(uuid.uuid4())
        # if os.path.exists(self.tmp_dir):
        #     shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir)

    def start(self):
        self.pool = multiprocessing.Pool(self.num_workers)

    def mapper(self, map_func, inputs, chunksize=None):
        logger.debug('In mapper')

        def func(input):
            results = map_func(input)
            files = []
            for result in results:
                key, value = result
                file = str(zlib.crc32(key)) + '.' + str(uuid.uuid4())
                file = os.path.join(self.tmp_dir, file)
                files.append(file)
                with open(file, 'wb') as f:
                    pickle.dump((key, value), f)
            return files

        self.map_responses_files = self.pool.map(func, inputs, chunksize=chunksize)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def partitioner(self):
        logger.debug('In partitioner')
        self.partitioned_dict = defaultdict(list)
        self.partitioned_files = []
        for files in self.map_responses_files:
            for file in files:
                key = os.path.basename(file).split('.')[0]
                self.partitioned_dict[key].append(file)
                self.partitioned_files.append(file)

    def reducer(self, reduce_func):
        logger.debug('In reducer')

        def func1(key_files_tuple):
            _, file_list = key_files_tuple
            value2 = None
            for file in file_list:
                with open(file, 'rb') as f:
                    key, value1 = pickle.load(f)
                    key, value2 = reduce_func(key, value1, value2)
                os.remove(file)
            file = str(uuid.uuid4()) + '.tmp'
            file = os.path.join(self.tmp_dir, file)
            with open(file, 'wb') as f:
                pickle.dump((key, value2), f)
            return file

        if reduce_func:
            file_list = self.pool.map(func1, list(self.partitioned_dict.items()))
        else:
            file_list = self.partitioned_files

        self.output_values = OrderedDict()
        for file in file_list:
            with open(file, 'rb') as f:
                key, value = pickle.load(f)
                self.output_values[key] = value

    def finish(self):
        self.pool.close()
        self.pool.join()
        shutil.rmtree(self.tmp_dir)


class SparkMapReduce(MapReduce):
    def start(self):
        conf = SparkConf().setAppName("Spark MapReduce")
        self.sc = SparkContext(conf=conf)

    def mapper(self, map_func, inputs):
        distData = self.sc.parallelize(inputs)
        self.map_responses = distData.map(map_func)

    def partitioner(self):
        pass

    def reducer(self, reduce_func):
        if reduce_func:
            output_values = self.map_responses.reduceByKey(reduce_func).collect()
        else:
            output_values = self.map_responses.groupByKey().collect()
        self.output_values = OrderedDict()
        for key, value in output_values:
            self.output_values[key] = value


def get_mapreduce_by_threads(threads):
    if threads == 1:
        return MapReduce()
    if threads > 1:
        return MultiprocessingMapReduce(threads)
    if threads == 0:
        return SparkMapReduce()


def read_files_from(file_list, process_func, threads=1, end_process_func=None, merge_func=None):
    if isinstance(file_list, str) and os.path.isfile(file_list):
        with open(file_list, 'r') as f:
            _list = f.read().splitlines()
    elif isinstance(file_list, list):
        _list = file_list
    else:
        raise Exception('Unkown input type %s or input file not exist!' % str(file_list))

    map_func = partial(_read, process_func=process_func, end_process_func=end_process_func)
    mapreduce = get_mapreduce_by_threads(threads)
    return mapreduce(_list, map_func, merge_func)


def _read(line, process_func, end_process_func=None):
    strings = line.split('\t')
    _dict = defaultdict(list)
    file = strings[0]
    compress = file.endswith('.gz')
    f = open(file, 'r') if not compress else gzip.open(file, 'rb')
    for line in f:
        line = line.strip()
        other = strings[1:] if len(strings) > 1 else [None]
        process_func(line, _dict, file, other=other)
    f.close()
    if end_process_func is not None:
        _dict = end_process_func(_dict)
    return list(_dict.items())

