import scripts.run_kv3d_budget_selection as budget_script


def test_budget_selection_default_max_new_tokens_is_64(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_kv3d_budget_selection.py",
            "--profile-dir",
            "outputs/profile",
            "--output-dir",
            "outputs/budget",
        ],
    )

    args = budget_script.parse_args()

    assert args.max_new_tokens == 64
