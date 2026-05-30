"""Alarm-flow MAP EM Multivariate Hawkes Process.

Parallel to alarm_flow_brunch but uses a windowed sparse MAP EM algorithm
(`mhp/`) for joint learning of edges + triggering kernel + immigrant rate,
rather than BRUNCH's MCMC over (B, C) with frozen Θ.
"""
