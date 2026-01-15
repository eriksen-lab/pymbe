__all__ = ["MBE"]

from pymbe.pymbe import MBE


def _install_mpi_exception_hook():
    try:
        from mpi4py import MPI
    except Exception:
        return

    # only activate in real MPI runs
    if MPI.COMM_WORLD.Get_size() <= 1:
        return

    import sys
    from pymbe.logger import logger

    sys_excepthook = sys.excepthook

    def global_except_hook(exctype, value, traceback):
        DIVIDER = "*" * 93
        try:
            logger.error("\n" + DIVIDER + "\n")
            logger.error(f"Uncaught exception on rank {MPI.COMM_WORLD.Get_rank()}\n")
            sys_excepthook(exctype, value, traceback)
            logger.error("\nShutting down MPI processes...\n")
            logger.error(DIVIDER + "\n")
        finally:
            try:
                MPI.COMM_WORLD.Abort(1)
            except Exception as exception:
                logger.error("MPI failed to stop, this process will hang.\n")
                logger.error(DIVIDER + "\n")
                raise exception

    sys.excepthook = global_except_hook


_install_mpi_exception_hook()
del _install_mpi_exception_hook
