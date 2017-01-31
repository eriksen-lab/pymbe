#!/usr/bin/env python

#
# python driver for inc.-corr. calculations using CFOUR as backend program.
# written by Janus J. Eriksen (jeriksen@uni-mainz.de), Fall/Winter 2016 + Winter/Spring 2017, Mainz, Germnay.
#
# Requires the path of the cfour basis GENBAS file ($CFOURBASIS) and bin directory ($CFOURBIN) in inc_corr_utils.py
#

import inc_corr_mpi
import inc_corr_utils
import inc_corr_orb_rout
import inc_corr_gen_rout
import inc_corr_e_rout
import inc_corr_plot

__author__ = 'Dr. Janus Juul Eriksen, JGU Mainz'

def main():
   #
   #  ---  init molecule dictionary... ---
   #
   molecule = {}
   #
   #  ---  init MPI info
   #
   inc_corr_mpi.init_mpi_info(molecule)
   #
   #  ---  master only
   #
   if (molecule['master']):
      #
      #  ---  redirect stdout to output.out - if present in wrk dir (alongside plotting output), delete these files before proceeding...  ---
      #
      inc_corr_utils.redirect_stdout(molecule)
      #
      #  ---  initialize the calculation...  ---
      #
      inc_corr_utils.init_calc(molecule)
      #
      #  ---  setup of scratch directory...  ---
      #
      inc_corr_utils.setup_calc(molecule)
      #
      #  ---  run HF calc to determine problem size parameters...  ---
      #
      inc_corr_gen_rout.run_calc_hf(molecule)
      #
      #  ---  run a few sanity checks...  ---
      #
      inc_corr_utils.sanity_chk(molecule)
      #
      #  ---  print program header
      #
      inc_corr_utils.print_header(molecule)
      #
      #  ---  initialization done - start the calculation...  ---
      #
      inc_corr_e_rout.inc_corr_main(molecule)
      #
      #  ---  start (potential) reference calculation...  ---
      #
      if (molecule['ref'] and (not molecule['error'][0][-1])):
         #
         inc_corr_gen_rout.ref_calc(molecule)
      #
      #  ---  print summary of the calculation  ---
      #
      inc_corr_utils.inc_corr_summary(molecule)
      #
      #  ---  plot the results of the calculation  ---
      #
      if (not molecule['error'][0][-1]):
         #
         inc_corr_plot.ic_plot(molecule)
      #
      #  ---  terminate calculation and clean up...  ---
      #
      inc_corr_utils.term_calc(molecule)
      #
      inc_corr_mpi.finalize_mpi(molecule)

if __name__ == '__main__':
   #
   main()

