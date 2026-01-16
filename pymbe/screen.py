#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
screening module
"""

from __future__ import annotations

__author__ = "Dr. Janus Juul Eriksen, University of Bristol, UK"
__license__ = "MIT"
__version__ = "0.9"
__maintainer__ = "Dr. Janus Juul Eriksen"
__email__ = "janus.eriksen@bristol.ac.uk"
__status__ = "Development"

import numpy as np
from math import floor, log10
from scipy import stats
from typing import TYPE_CHECKING

from pymbe.logger import logger

from pymbe.tools import n_tuples, n_tuples_predictors
from pymbe.clustering import SEED


if TYPE_CHECKING:
    from typing import List, Dict, Tuple, Optional


def fixed_screen(
    screen_func: str,
    screen_start: int,
    screen_perc: float,
    screen_lst: List[Dict[str, np.ndarray]],
    exp_clusters: List[np.ndarray],
    exp_space: np.ndarray,
    order: int,
) -> List[int]:
    """
    this function returns indices to be removed from fixed screening
    """
    # get indices for screening list from maximum cluster size
    screen_idx = min(order, max(cluster.size for cluster in exp_clusters[-1]))

    # get screening contributions
    screen_arr = np.vstack(
        [screen_dict[screen_func] for screen_dict in screen_lst[-screen_idx:]]
    )
    screen = (
        np.max(screen_arr, axis=0)
        if screen_func == "max"
        else np.sum(screen_arr, axis=0)
    )

    thres = 1.0 if order < screen_start else screen_perc
    nscreen = exp_space.size - int(thres * exp_space.size)
    if screen_func == "rnd":
        rng = np.random.default_rng()
        norb = 0
        remove_idx = []
        while norb < nscreen:
            idx = rng.integers(len(exp_clusters))
            norb += exp_clusters[idx].size
            remove_idx.append(idx.item())

    else:
        cluster_screen = [np.abs(screen[cluster[0]]) for cluster in exp_clusters]
        # stable sorting algorithm is important to ensure that the same
        # clusters are screened away every time in case of equality
        cluster_significance = np.argsort(cluster_screen, kind="stable")

        norb = 0
        remove_idx = []
        for idx in cluster_significance:
            if norb >= nscreen:
                break
            norb += exp_clusters[idx].size
            remove_idx.append(idx.item())

    return remove_idx


def adaptive_screen(
    mbe_tot_energy: List[float],
    mbe_tot_error: float,
    screen_thres: float,
    screen: Dict[str, np.ndarray],
    exp_clusters: List[np.ndarray],
    exp_space: np.ndarray,
    exp_single_orbs: bool,
    incs: List[List[np.ndarray]],
    nocc: int,
    ref_nelec: np.ndarray,
    ref_nhole: np.ndarray,
    vanish_exc: int,
    curr_order: int,
    conv_tol: float,
) -> Tuple[Optional[int], float]:
    # define maximum possible order
    max_order = exp_space.size

    # get total number of increments
    tot_nincs = sum(
        np.count_nonzero(tup_nocc_incs)
        for order_incs in incs
        for tup_nocc_incs in order_incs
    )

    # check if no increments were calculated at current order or expansion has ended
    if (
        sum(np.count_nonzero(tup_nocc_incs) for tup_nocc_incs in incs[-1]) == 0
        or tot_nincs <= 1
        or curr_order >= max_order
    ):
        return None, mbe_tot_error

    # define maximum cluster size
    max_cluster_size = max(cluster.size for cluster in exp_clusters)

    # define allowed error
    error_thresh = screen_thres - mbe_tot_error

    # determine required confidence level
    conv_orders = 0
    for idx in range(max_cluster_size + 1, len(mbe_tot_energy) + 1, max_cluster_size):
        cum_inc = np.abs(mbe_tot_energy[-1] - mbe_tot_energy[-idx])
        if cum_inc > error_thresh:
            break
        conv_orders += 1
    conf_level = 0.95
    if conv_orders == 1:
        conf_level = 0.8
    elif conv_orders == 2:
        conf_level = 0.5
    elif conv_orders > 2:
        conf_level = 0.05

    # initialize array for estimated quantities
    est_error = np.zeros((len(exp_clusters), max_order - curr_order), dtype=np.float64)
    est_mean_inc = np.zeros_like(est_error)

    # initialize array for cluster contribution errors
    tot_error = np.zeros(len(exp_clusters), dtype=np.float64)

    # initialize array for error difference to threshold
    error_diff = np.zeros_like(tot_error)

    # create random number generator
    rng = np.random.default_rng(seed=SEED)

    if tot_nincs <= 10000:
        # get sample increments
        sample_incs = np.concatenate(
            [
                tup_nocc_incs[tup_nocc_incs.nonzero()]
                for order_incs in incs
                for tup_nocc_incs in order_incs
            ],
            axis=0,
        )
    else:
        # maximum number of increments to sample from
        nsample_inc = 10000

        # initialize sample incs
        sample_incs = np.empty(nsample_inc, dtype=np.float64)

        # draw random integers
        sample_indices = rng.integers(tot_nincs, size=nsample_inc)
        sample_indices.sort()

        # get sample increments
        prev_sample_idx = 0
        prev_tup_nocc_idx = 0
        for order_incs in incs:
            for tup_nocc_incs in order_incs:
                sample_idx = sample_indices.searchsorted(
                    prev_tup_nocc_idx + np.count_nonzero(tup_nocc_incs)
                ).item()
                sample_incs[prev_sample_idx:sample_idx] = tup_nocc_incs[
                    tup_nocc_incs.nonzero()
                ][sample_indices[prev_sample_idx:sample_idx] - prev_tup_nocc_idx]
                prev_sample_idx = sample_idx
                prev_tup_nocc_idx += np.count_nonzero(tup_nocc_incs)

    # get kernel density estimate for previous-order increment distribution
    kde_kernel = stats.gaussian_kde(np.log(np.abs(sample_incs)))

    # evaluate kde pdf
    prev_inc_probs = kde_kernel.evaluate(np.log(np.abs(sample_incs)))

    # initialize screening print boolean
    print_screen = True

    # loop over clusters
    for cluster_idx, cluster_orbs in enumerate(exp_clusters):
        # get remaining clusters
        remain_clusters = exp_clusters[:cluster_idx] + exp_clusters[cluster_idx + 1 :]

        # get mask number of clusters with more than single sample
        mask = screen["inc_count"][:, cluster_orbs[0]] > 1

        # get corresponding number of clusters
        nclusters = np.arange(1, screen["inc_count"].shape[0] + 1, dtype=np.float64)[
            mask
        ]

        # require at least 3 points to fit and ensure all
        # (screening exponent + 1)-orbital correlations have been calculated
        if (
            nclusters.size >= 3
            and curr_order
            >= sum(
                cluster.size
                for cluster in remain_clusters[floor(log10(screen_thres)) :]
            )
            + cluster_orbs.size
        ):
            # log confidence level
            if print_screen:
                logger.info3(f" Confidence level: {conf_level}\n")
                print_screen = False

            # calculate mean logarithm increment magnitude
            mean = (
                screen["log_inc_sum"][mask, cluster_orbs[0]]
                / screen["inc_count"][mask, cluster_orbs[0]]
            )

            # calculate variance of the logarithm increment magnitude
            variance = (
                screen["log_inc_sum2"][mask, cluster_orbs[0]]
                / screen["inc_count"][mask, cluster_orbs[0]]
                - mean**2
            )
            variance[(0.0 > variance) & (variance > -1e-11)] = 0.0

            # log fit information
            if cluster_orbs.size == 1:
                logger.info3(
                    f" Screening information for orbital {cluster_orbs.item()}:"
                )
            else:
                cluster_str = np.array2string(cluster_orbs, separator=", ")
                logger.info3(
                    f" Screening information for orbital cluster {cluster_str}:"
                )
            logger.info3(f" {np.array2string(nclusters, separator=', ')}")
            logger.info3(f" {np.array2string(mean, separator=', ')}")
            logger.info3(f" {np.array2string(variance, separator=', ')}")

            # calculate variance of the mean logarithm increment magnitude
            mean_variance = variance / screen["inc_count"][mask, cluster_orbs[0]]

            # calculate weights as reciprocal of standard error of the mean
            weights = np.divide(
                1,
                mean_variance,
                out=np.zeros_like(mean_variance),
                where=mean_variance != 0,
            )

            # fit logarithmic mean increment magnitude
            fit = np.polynomial.polynomial.Polynomial(
                np.polynomial.polynomial.polyfit(nclusters, mean, 1, w=np.sqrt(weights))
            )

            # get t-statistic for confidence level
            t_stat = stats.t.ppf(conf_level, nclusters.size - 2)

            # get residuals of model
            residuals = fit(nclusters) - mean

            # get standard error of model
            s_err = np.sqrt(np.sum(weights * residuals**2) / (nclusters.size - 2))

            # initialize number of tuples for cluster for remaining
            # orders
            ntup_cluster = []

            # initialize number of total tuples for remaining orders
            ntup_total = []

            # loop over remaining orders
            for order_idx, order in enumerate(range(curr_order + 1, max_order + 1)):
                # get total number of tuples at this order
                ntup_total.append(
                    n_tuples(
                        exp_space,
                        exp_clusters if not exp_single_orbs else None,
                        nocc,
                        ref_nelec,
                        ref_nhole,
                        vanish_exc,
                        order,
                    )
                )

                # initialize number of tuples for cluster for this order
                ntup_cluster.append(0)

                # intitialize list for distribution information
                dist_info: List[Tuple[int, float, float]] = []

                # initialize print order boolean
                print_order = True

                # loop over number of tuples and predictors
                for ntup, ncluster, ncontrib in n_tuples_predictors(
                    exp_space,
                    exp_clusters if not exp_single_orbs else None,
                    cluster_idx,
                    nocc,
                    ref_nelec,
                    ref_nhole,
                    vanish_exc,
                    order,
                ):
                    # check if any tuples for this cluster at this order contribute
                    if ntup > 0:
                        # log order
                        if print_order:
                            logger.info3(f" Prediction for order {order}:")
                            print_order = False

                        # add to number of tuples for cluster for this order
                        ntup_cluster[-1] += ntup

                        # variance estimate from maximum observed variance
                        var_est = variance[-1]
                        if screen["inc_count"][mask, cluster_orbs[0]][-1] < 1000:
                            var_est = max(var_est, 10)

                        # mean variance estimate
                        mean_var_est = var_est / ntup

                        # weighted average of predictors
                        average_nclusters = np.average(nclusters, weights=weights)

                        # error estimate
                        err = (
                            t_stat
                            * s_err
                            * np.sqrt(
                                mean_var_est
                                + 1 / np.sum(weights)
                                + (ncluster - average_nclusters) ** 2
                                / np.sum(weights * (nclusters - average_nclusters) ** 2)
                            )
                        )

                        # estimate logarithmic mean increment magnitude prediction
                        # interval for given predictors, add to mean increment
                        # magnitude and check whether this is lower than previous point
                        # for which all contributions have been calculated, this should
                        # be worst-case behaviour
                        full_ncluster = int(nclusters[0]) - 1
                        size = cluster_orbs.size + sum(
                            cluster.size
                            for cluster in remain_clusters[::-1][:full_ncluster]
                        )
                        while size <= curr_order:
                            size += remain_clusters[::-1][full_ncluster].size
                            full_ncluster += 1
                        log_mean_abs_inc = min(
                            fit(ncluster) + err, mean[nclusters == full_ncluster][0]
                        )

                        # save information from distribution
                        dist_info.append(
                            (
                                ntup,
                                log_mean_abs_inc + np.log(ncontrib),
                                np.sqrt(var_est),
                            )
                        )
                        logger.info3(
                            f" Log-transformed normalized mean: {log_mean_abs_inc}"
                        )
                        logger.info3(
                            f" Distribution of {ntup} tuples with {ncontrib} "
                            f"{ncluster}-orbital contributions:"
                        )
                        logger.info3(
                            f" Log-transformed mean: "
                            f"{log_mean_abs_inc + np.log(ncontrib)}"
                        )
                        logger.info3(f" Variance of log-transformed mean: {var_est}")

                # check if any tuples for this cluster at this order contribute
                if ntup_cluster[-1] > 0:
                    # starting number of samples
                    curr_nsamples = 100

                    # starting number of tuples per distribution, a smaller number
                    # will only overestimate the actual error
                    max_ntups = 1000

                    # initialize samples
                    samples: List[np.ndarray] = []

                    # initialize convergence booleans
                    conv = False

                    # initialize errors
                    prev_error = prev_error2 = 0.0

                    # simulate order until convergence
                    while not conv:
                        # loop over distributions
                        for dist_idx, (dist_ntup, dist_mean, dist_std) in enumerate(
                            dist_info
                        ):
                            # generate distribution probabilities and importance weights
                            # for increments
                            importance = stats.norm.pdf(
                                np.log(np.abs(sample_incs)),
                                loc=dist_mean,
                                scale=dist_std,
                            )
                            importance /= prev_inc_probs
                            if np.sum(importance) > 0.0:
                                # draw random samples
                                importance /= np.sum(importance)
                                try:
                                    samples[dist_idx] = np.concatenate(
                                        (
                                            samples[dist_idx],
                                            rng.choice(
                                                sample_incs,
                                                (
                                                    curr_nsamples // 2,
                                                    samples[dist_idx].shape[1],
                                                ),
                                                replace=True,
                                                p=importance,
                                            ),
                                        ),
                                        axis=0,
                                    )
                                except IndexError:
                                    samples.append(
                                        rng.choice(
                                            sample_incs,
                                            (curr_nsamples, min(dist_ntup, max_ntups)),
                                            replace=True,
                                            p=importance,
                                        )
                                    )
                            else:
                                # mean is too small
                                try:
                                    samples[dist_idx] = np.atleast_2d(0.0)
                                except IndexError:
                                    samples.append(np.atleast_2d(0.0))

                        # calculate error
                        error = np.quantile(
                            np.abs(
                                sum(
                                    [
                                        dist[0] * np.mean(sample, axis=1)
                                        for sample, dist in zip(samples, dist_info)
                                    ]
                                )
                            ),
                            conf_level,
                        )

                        # determine if simulation has converged
                        conv = error == 0.0 or (
                            abs(error - prev_error) / error < 0.05
                            and abs(prev_error - prev_error2) / error < 0.05
                        )

                        if not conv:
                            # prepare errors for next step
                            prev_error2 = prev_error
                            prev_error = error

                            # double number of samples
                            curr_nsamples *= 2

                    # set final error
                    est_error[cluster_idx, order_idx] = error

                    # estimate the mean increment for all increments at this order
                    est_mean_inc[cluster_idx, order_idx] = (
                        est_error[cluster_idx, order_idx] / ntup_cluster[-1]
                    )

                    # add to total error
                    tot_error[cluster_idx] += est_error[cluster_idx, order_idx]

                    logger.info3(
                        f" Estimated mean increment: "
                        f"{est_mean_inc[cluster_idx, order_idx]}"
                    )
                    logger.info3(
                        f" Estimated error: {est_error[cluster_idx, order_idx]}\n"
                    )

                    # stop if the last few orders contribute less than 1% or if
                    # accumulated error is larger than threshold
                    if order_idx >= max_cluster_size - 1 and (
                        (
                            np.sum(
                                est_error[
                                    cluster_idx,
                                    max(0, order_idx - max_cluster_size + 1) : order_idx
                                    + 1,
                                ]
                            )
                            / tot_error[cluster_idx]
                            < 0.01
                        )
                        or tot_error[cluster_idx]
                        > (
                            (
                                np.sum(ntup_cluster[:max_cluster_size])
                                / np.sum(ntup_total[:max_cluster_size])
                            )
                            * error_thresh
                        )
                    ):
                        break

            logger.info3("\n")

            # calculate difference to allowed error
            error_diff[cluster_idx] = (
                np.sum(ntup_cluster[:max_cluster_size])
                / np.sum(ntup_total[:max_cluster_size])
            ) * error_thresh - tot_error[cluster_idx]

    # get index in expansion space for minimum cluster contribution
    min_idx = int(np.argmax(error_diff))

    # screen cluster away if contribution is smaller than threshold
    if error_diff[min_idx] > 0.0:
        # log screening
        if exp_clusters[min_idx].size == 1:
            logger.info2(
                f" Orbital {exp_clusters[min_idx].item()} is screened away (Error "
                f"= {tot_error[min_idx]:>10.4e})" + "\n"
            )
        else:
            cluster_str = np.array2string(exp_clusters[min_idx], separator=", ")
            logger.info2(
                f" Orbital cluster {cluster_str} is screened away (Error = "
                f"{tot_error[min_idx]:>10.4e})" + "\n"
            )
        logger.info2(" " + 42 * "-")
        logger.info2("  Order | Est. mean increment | Est. error")
        logger.info2(" " + 42 * "-")
        for order, (mean, error) in enumerate(
            zip(est_mean_inc[min_idx], est_error[min_idx]), start=curr_order + 1
        ):
            if error > 0.0:
                logger.info2(f"  {order:5} |      {mean:>10.4e}     | {error:>10.4e}")
        logger.info2(" " + 42 * "-" + "\n\n")

        # add screened cluster contribution to error
        mbe_tot_error += tot_error[min_idx]

        return min_idx, mbe_tot_error

    # check if geometric mean absolute increment contribution for minimum error cluster
    # comes close to convergence threshold
    elif np.sum(screen["inc_count"][:, min_idx]) > 0 and (
        0.0
        < np.sum(screen["log_inc_sum"][:, min_idx])
        / np.sum(screen["inc_count"][:, min_idx])
        < np.log(1e1 * conv_tol)
    ):
        # log screening
        if exp_clusters[min_idx].size == 1:
            logger.info2(
                f" Orbital {exp_clusters[min_idx].item()} is screened away due to the "
                "majority of increments getting close to convergence\n criterium"
            )
        else:
            cluster_str = np.array2string(exp_clusters[min_idx], separator=", ")
            logger.info2(
                f" Orbital cluster {cluster_str} is screened away due to the majority "
                "of increments getting close to convergence\n criterium"
            )

        return min_idx, mbe_tot_error

    # cluster with minimum contribution is not screened away
    else:
        return None, mbe_tot_error
