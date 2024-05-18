from loguru import logger

import phlash
from phlash.data import RawContig
from phlash.size_history import DemographicModel


def psmc(
    psmcfa_files: list[str], window_size: int = 100, hold_out: bool = True, **options
) -> list[DemographicModel]:
    """Run phlash on PSMC-formatted data.

    Args:
        psmcfa_files: A list of input files in .psmcfa format.
        hold_out: If True, hold out the first chromosome to assess convergence.
        options: Additional options to pass to the phlash MCMC sampler.

    Returns:
        A list of posterior DemographicModel samples.
    """

    logger.info("Reading PSMC data")
    contigs = [
        c for f in psmcfa_files for c in RawContig.from_psmcfa_iter(f, window_size)
    ]
    test_data = None
    if hold_out and len(contigs) > 1:
        test_data = contigs.pop(0)
    return phlash.fit(contigs, test_data=test_data, **options)
