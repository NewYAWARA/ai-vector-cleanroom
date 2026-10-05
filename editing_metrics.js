/* Shared by the offline editing page and Node regression tests. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.EditingMetrics = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const PRIMARY_TASK = 'complete_deliverable';

  function seconds(value) {
    if (typeof value !== 'number' && typeof value !== 'string') return null;
    if (typeof value === 'string' && value.trim() === '') return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
  }

  function evaluateTask(task) {
    const vector = seconds(task.vector_seconds);
    const redraw = seconds(task.redraw_seconds);
    const reasons = [];
    if (task.applicable !== true) reasons.push('not_applicable');
    if (task.vector_status !== 'completed') reasons.push('vector_not_completed');
    if (task.redraw_status !== 'completed') reasons.push('redraw_not_completed');
    if (task.vector_evidence !== 'actual') reasons.push('vector_not_actually_timed');
    if (task.redraw_evidence !== 'actual') reasons.push('redraw_not_actually_timed');
    if (!(vector > 0)) reasons.push('vector_time_missing_invalid_or_zero');
    if (!(redraw > 0)) reasons.push('redraw_time_missing_invalid_or_zero');
    const saving = vector > 0 && redraw > 0 ? (redraw - vector) / redraw * 100 : null;
    if (saving !== null && !Number.isFinite(saving)) reasons.push('nonfinite_time_ratio');
    const eligible = reasons.length === 0;
    const attempted = vector > 0 || redraw > 0 ||
      [task.vector_status, task.redraw_status].some(status =>
        ['in_progress', 'partial', 'unable', 'completed'].includes(status));
    return {
      id: task.id,
      applicable: task.applicable === true,
      status: task.vector_status || 'not_started',
      vector_status: task.vector_status || 'not_started',
      redraw_status: task.redraw_status || 'not_started',
      vector_seconds: vector,
      redraw_seconds: redraw,
      vector_evidence: task.vector_evidence || 'none',
      redraw_evidence: task.redraw_evidence || 'none',
      attempted,
      comparison_eligible: eligible,
      exclusion_reasons: reasons,
      time_saving_percent: eligible ? saving : null,
      note: task.note || ''
    };
  }

  function summarize(rawTasks) {
    const tasks = rawTasks.map(evaluateTask);
    const primary = tasks.find(task => task.id === PRIMARY_TASK);
    const comparable = !!primary && primary.comparison_eligible;
    const primarySaving = comparable ? primary.time_saving_percent : null;
    return {
      tasks,
      summary: {
        metric_scope: 'complete_deliverable_only',
        primary_task_id: PRIMARY_TASK,
        primary_comparison_eligible: comparable,
        primary_time_saving_percent: primarySaving,
        // Legacy summary names retain only the full-deliverable measurement.
        // Diagnostic tasks can overlap and must never be added to its time.
        actual_timed_comparable_tasks: comparable ? 1 : 0,
        vector_seconds_sum: comparable ? primary.vector_seconds : null,
        redraw_seconds_sum: comparable ? primary.redraw_seconds : null,
        actual_timed_weighted_saving_percent: primarySaving,
        observed_ge_80_percent_for_this_session: comparable && primarySaving >= 80,
        diagnostic_comparable_tasks: tasks.filter(task =>
          task.id !== PRIMARY_TASK && task.comparison_eligible).length,
        applicable_tasks: tasks.filter(task => task.applicable).length,
        attempted_tasks: tasks.filter(task => task.attempted).length,
        incomplete_vector_tasks: tasks.filter(task => task.attempted &&
          task.vector_status !== 'completed').length,
        incomplete_redraw_tasks: tasks.filter(task => task.attempted &&
          task.redraw_status !== 'completed').length,
        unable_tasks: tasks.filter(task =>
          task.vector_status === 'unable' || task.redraw_status === 'unable').length,
        uncomparable_attempts: tasks.filter(task =>
          task.attempted && !task.comparison_eligible).length,
        product_claim_validated: false
      }
    };
  }
  return {PRIMARY_TASK, seconds, evaluateTask, summarize};
});
