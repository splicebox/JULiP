from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Region(object):
    def __init__(self, seq_name, start, end, strand=None, count=0):
        self.seq_name   = seq_name
        self._start     = int(start)
        self._end       = int(end)
        self.coordinate = (self._start, self._end)
        self.strand     = strand
        self.count      = float(count)
        self.file_objs  = []

    @classmethod
    def from_region(cls, other):
        seq_name   = other.seq_name
        start      = other._start
        end        = other._end
        coordinate = other.coordinate
        strand     = other.strand
        count      = other.count
        return cls(seq_name, start, end, strand, count)

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        assert value >= 0
        self._start = value
        self.coordinate = (self._start, self._end)

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        assert value >= 0
        self._end = value
        self.coordinate = (self._start, self._end)

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return ((self.coordinate == other.coordinate) and (self.seq_name == other.seq_name) and
                (self.strand == other.strand))
        return NotImplemented

    def __ne__(self, other):
        """Define a non-equality test"""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        """Override the default hash behavior (that returns the id or the object)"""
        return hash((self.seq_name, self.coordinate, self.strand))


class File(object):
    def __init__(self, file, count, condition=None):
        self.file = file
        self.count = count
        # condition: test or control
        self.condition = condition


class Intron(Region):
    def __init__(self, seq_name, start, end, strand=None, count=0,
            transcripts=list(), start_exon_region=None, end_exon_region=None):
        super(Intron, self).__init__(seq_name, start, end, strand, count)
        self.transcripts = transcripts
        self.start_exon_region = start_exon_region
        self.end_exon_region = end_exon_region
        self.gene_region = None
        self.selected = True
        self.sum = count
        self.cond_count_dict = None


class ExonRegion(Region):
    def __init__(self, seq_name, start, end, strand=None, count=0):
        super(ExonRegion, self).__init__(seq_name, start, end, strand, count)
        self.plus_start_introns  = list()
        self.minus_start_introns = list()
        self.plus_end_introns    = list()
        self.minus_end_introns   = list()
        self.intron_boudaries    = list()
        self.mask                = 'none'
        self.plus_gene_region    = None
        self.minus_gene_region   = None


class GeneRegion(Region):
    def __init__(self, seq_name, start, end, strand=None, count=0):
        super(GeneRegion, self).__init__(seq_name, start, end, strand, count)
        self.gene_id = '_'.join((seq_name, str(start), str(end), strand))
        self.exon_regions = list()
        self.introns = list()
        self.gene_name = None
        self.ensembl_id = None

    @property
    def start(self):
        return super(GeneRegion, self).start

    @start.setter
    def start(self, value):
        assert value >= 0
        self._start = value
        self.coordinate = (self._start, self._end)
        self.gene_id = '_'.join((self.seq_name, str(self._start), str(self._end), self.strand))

    @property
    def end(self):
        return super(GeneRegion, self).end

    @end.setter
    def end(self, value):
        # super(GeneRegion, self).end = value
        assert value >= 0
        self._end = value
        self.coordinate = (self._start, self._end)
        self.gene_id = '_'.join((self.seq_name, str(self._start), str(self._end), self.strand))


class Boundary(object):
    def __init__(self, seq_name, position, attribute, strand=None, start_introns=list(), end_introns=list()):
        self.seq_name = seq_name
        self.position = int(position)
        self.attribute = attribute
        self.strand = strand
        self.start_introns = start_introns
        self.end_introns = end_introns

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return ((self.position == other.position) and (self.seq_name == other.seq_name) and
                (self.strand == other.strand))
        return NotImplemented

    def __ne__(self, other):
        """Define a non-equality test"""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        """Override the default hash behavior (that returns the id or the object)"""
        return hash((self.seq_name, self.position, self.strand))

