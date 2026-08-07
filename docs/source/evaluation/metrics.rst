Metrics
=======

Virne records both per-event data and run-level summaries. This page connects
the benchmark definitions to the fields that users will find in
``summary.csv``.

Core Metrics
------------

Request Acceptance Rate (RAC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RAC is the fraction of arriving VN requests that are embedded successfully:

.. math::

   RAC =
   \frac{\sum_{t=0}^{\mathcal{T}} |\tilde{\mathcal{I}}(t)|}
        {\sum_{t=0}^{\mathcal{T}} |\mathcal{I}(t)|}.

Here, :math:`\mathcal{I}(t)` is the set of requests arriving at time
:math:`t`, and :math:`\tilde{\mathcal{I}}(t)` is the accepted subset. Virne
stores RAC as ``acceptance_rate`` in the range :math:`[0, 1]`; multiply it by
100 to report a percentage.

Long-term Revenue-to-Cost Ratio (LRC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LRC compares the lifetime-weighted revenue of accepted requests with their
lifetime-weighted physical resource cost:

.. math::

   LRC =
   \frac{\sum_t \sum_{I \in \tilde{\mathcal{I}}(t)} REV(S_I)\,\varpi_I}
        {\sum_t \sum_{I \in \tilde{\mathcal{I}}(t)} COST(S_I)\,\varpi_I}.

The corresponding summary field is ``long_term_time_r2c_ratio``. Virne also
exports ``long_term_r2c_ratio``, which omits request lifetime weighting, and
``avg_r2c_ratio``, which averages the per-request ratios. These fields answer
different questions and should not be used interchangeably. All three ratio
fields are stored as fractions; multiply by 100 when reporting a percentage.

Long-term Average Revenue (LAR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LAR is lifetime-weighted revenue per unit of simulated time:

.. math::

   LAR = \frac{1}{\mathcal{T}}
   \sum_t \sum_{I \in \tilde{\mathcal{I}}(t)} REV(S_I)\,\varpi_I.

This is exported as ``long_term_avg_time_revenue``. The additional field
``long_term_avg_revenue`` uses unweighted revenue divided by simulated time.

Average Solving Time (AST)
~~~~~~~~~~~~~~~~~~~~~~~~~~

AST measures the mean time spent by a solver on one VN request. The current
``summary.csv`` does **not** export a solver-only AST field. It does export
``clock_running_time``, which measures wall-clock time for the complete run and
therefore includes framework, recording, and data-generation overhead. Treat
it as end-to-end runtime, not as AST.

Summary Field Reference
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 32 38

   * - Quantity
     - ``summary.csv`` field
     - Interpretation
   * - Request acceptance rate
     - ``acceptance_rate``
     - Accepted requests divided by arrived requests; stored as a fraction.
   * - Lifetime-weighted LRC
     - ``long_term_time_r2c_ratio``
     - Lifetime-weighted revenue divided by lifetime-weighted cost.
   * - Unweighted long-term R2C
     - ``long_term_r2c_ratio``
     - Total revenue divided by total cost.
   * - Mean instance R2C
     - ``avg_r2c_ratio``
     - Mean of the per-request ``v_net_r2c_ratio`` values.
   * - Lifetime-weighted LAR
     - ``long_term_avg_time_revenue``
     - Lifetime-weighted revenue per unit of simulated time.
   * - Unweighted average revenue
     - ``long_term_avg_revenue``
     - Total unweighted revenue per unit of simulated time.
   * - End-to-end runtime
     - ``clock_running_time``
     - Wall-clock seconds for the complete run; not solver-only AST.

Output Files
------------

Each run normally contains:

* ``config.yaml``: the resolved configuration needed to interpret or reproduce
  the run.
* ``summary.csv``: one row of run-level metrics and experiment metadata.
* ``records/*.csv``: per-event state, solution, feasibility, resource, and
  reward fields.

The :doc:`Quickstart <../start/running>` shows the default output
layout. For fair comparisons, keep the PN/VN settings and seeds fixed, report
whether ratios are lifetime weighted, and distinguish simulated time from
wall-clock runtime.
