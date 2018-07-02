from __future__ import absolute_import, division, print_function, unicode_literals

import h5py
import numpy as np
from collections import OrderedDict


def save_madminer_file(filename,
                       parameters,
                       benchmarks,
                       morphing_components=None,
                       morphing_matrix=None,
                       overwrite_existing_files=True):
    """
    Saves all MadMiner settings into an HDF5 file.

    :param filename:
    :param parameters:
    :param benchmarks:
    :param morphing_components:
    :param morphing_matrix:
    :param overwrite_existing_files:
    """

    io_tag = 'w' if overwrite_existing_files else 'x'

    with h5py.File(filename, io_tag) as f:

        # Prepare parameters
        parameter_names = [pname for pname in parameters]
        n_parameters = len(parameter_names)
        parameter_names_ascii = [pname.encode("ascii", "ignore") for pname in parameter_names]
        parameter_ranges = np.array(
            [parameters[key][3] for key in parameter_names],
            dtype=np.float
        )
        parameter_lha_blocks = [parameters[key][0].encode("ascii", "ignore") for key in parameter_names]
        parameter_lha_ids = np.array(
            [parameters[key][1] for key in parameter_names],
            dtype=np.int
        )

        # Store parameters
        f.create_dataset('parameters/names', (n_parameters,), dtype='S256', data=parameter_names_ascii)
        f.create_dataset('parameters/ranges', data=parameter_ranges)
        f.create_dataset('parameters/lha_blocks', (n_parameters,), dtype='S256', data=parameter_lha_blocks)
        f.create_dataset("parameters/lha_ids", data=parameter_lha_ids)

        # Prepare benchmarks
        benchmark_names = [bname for bname in benchmarks]
        n_benchmarks = len(benchmark_names)
        benchmark_names_ascii = [bname.encode("ascii", "ignore") for bname in benchmark_names]
        benchmark_values = np.array(
            [
                [benchmarks[bname][pname] for pname in parameter_names]
                for bname in benchmark_names
            ]
        )

        # Store benchmarks
        f.create_dataset('benchmarks/names', (n_benchmarks,), dtype='S256', data=benchmark_names_ascii)
        f.create_dataset('benchmarks/values', data=benchmark_values)

        # Store morphing info
        if morphing_components is not None:
            f.create_dataset("morphing/components", data=morphing_components.astype(np.int))
        if morphing_matrix is not None:
            f.create_dataset("morphing/morphing_matrix", data=morphing_matrix.astype(np.float))


def load_madminer_file(filename):
    """ Loads MadMiner settings, observables, and weights from a HDF5 file. """

    with h5py.File(filename, 'r') as f:

        # Parameters
        try:
            parameter_names = f['parameters/names'][()]
            parameter_ranges = f['parameters/ranges'][()]
            parameter_lha_blocks = f['parameters/lha_blocks'][()]
            parameter_lha_ids = f['parameters/lha_ids'][()]

            parameters = OrderedDict()

            for pname, prange, pblock, pid in zip(parameter_names, parameter_ranges, parameter_lha_blocks,
                                                  parameter_lha_ids):
                parameters[pname] = (
                    str(pblock),
                    int(pid),
                    tuple(prange)
                )

        except:
            raise IOError('Cannot read parameters from HDF5 file')

        # Benchmarks
        try:
            benchmark_names = f['benchmarks/names'][()]
            benchmark_values = f['benchmarks/values'][()]

            benchmarks = OrderedDict()

            for bname, bvalue_matrix in zip(benchmark_names, benchmark_values):
                bvalues = OrderedDict()
                for pname, pvalue in zip(parameter_names, bvalue_matrix):
                    bvalues[pname] = pvalue

                benchmarks[bname] = bvalues

        except:
            raise IOError('Cannot read benchmarks from HDF5 file')

        # Morphing
        try:
            morphing_components = np.asarray(f['morphing/components'][()], dtype=np.int)
            morphing_matrix = np.asarray(f['morphing/components'][()], dtype=np.int)

        except:
            morphing_components = None
            morphing_matrix = None

        # Observables
        try:
            observables = OrderedDict()

            observable_names = f['observables/names'][()]
            observable_names = [str(oname) for oname in observable_names]
            observable_definitions = f['observables/definitions'][()]
            observable_definitions = [str(odef) for odef in observable_definitions]

            for oname, odef in zip(observable_names, observable_definitions):
                observables[oname] = odef
        except:
            observables = None

        # Observations
        try:
            observations = np.asarray(f['samples/observations'][()], dtype=np.float)
            weights = np.asarray(f['samples/weights'][()], dtype=np.float64)
        except:
            observations = None
            weights = None

        return parameters, benchmarks, morphing_components, morphing_matrix, observables, observations, weights