"""ST harness for the golden-candidate NPU integration test (tests/st/harness/).

Split by concern: eval_run (调用 kernel_eval) · golden_mock (golden 候选;未来 baseline_mock 同级)
· report (解读结果 + 组织产物) · select_from_changes (PR 改动 → -k)。Re-exported here so callers
`from harness import ...`.
"""
from .eval_run import (
    REPO_ROOT, KERNEL_EVAL_SRC, TASKS, RUN_EVALUATION_SH, LEVELS,
    has_npu, kernel_eval_env, build_eval_cmd, run_eval_cli,
)
from .golden_mock import build_golden_candidate
from .report import (
    latest_report_json, load_report, iter_cases, case_num,
    case_has_accuracy, case_has_perf, case_acc_passed,
    TOP_FIELDS, SUMMARY_FIELDS, OPERATOR_FIELDS, CASE_FIELDS, ACCURACY_FIELDS,
    schema_diff, has_drift,
    VALID_ACTIONS, VALID_TARGETS, load_known_issues, for_target,
    hang_cases, xfail_set, xfail_all_ops, skip_ops,
    build_trimmed_tasktree, collect_artifacts,
)
# select_from_changes 作为脚本经 `python -m harness.select_from_changes` 调用(run_st.sh),
# 不在此 re-export —— 避免 __init__ 导入它后 `-m` 再当 __main__ 跑触发 RuntimeWarning。
# 需要其逻辑时:from harness.select_from_changes import selector
