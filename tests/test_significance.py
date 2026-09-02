from suschain.eval.significance import compare_models, mean_std, wilcoxon_pairwise


def test_mean_std_hand_computed():
    mean, std = mean_std([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean == 3.0
    assert abs(std - 1.5811388300841898) < 1e-9  # sample std (ddof=1)


def test_wilcoxon_identical_values_not_significant():
    result = wilcoxon_pairwise([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert result["significant_at_0.05"] is False


def test_wilcoxon_max_consistent_n5_hits_the_power_ceiling_not_0_05():
    # model A consistently beats model B on all 5 seeds -- the most extreme
    # signal possible with n=5. The EXACT two-sided Wilcoxon test's minimum
    # achievable p-value at n=5 is 2*(1/32)=0.0625, which can NEVER reach
    # the conventional p<0.05 threshold no matter how large or consistent
    # the effect is. This is a real power limitation of 5-seed evaluation
    # (worth stating explicitly in the paper, not just a quirk of this
    # toy test) -- confirmed here rather than assumed.
    a = [0.50, 0.52, 0.51, 0.49, 0.53]
    b = [0.10, 0.11, 0.09, 0.12, 0.10]
    result = wilcoxon_pairwise(a, b)
    assert result["p_value"] == 0.0625
    assert result["significant_at_0.05"] is False


def test_compare_models_structure_with_7_seeds_reaches_significance():
    # 7 seeds (not 5) specifically so p<0.05 is mathematically reachable at
    # all with a consistent effect (min two-sided exact p at n=7 is
    # 2*(1/128)=0.015625) -- see the n=5 power-ceiling test above for why
    # this project's real multi-seed runs use 7 seeds, not the plan's
    # literal "5", for any comparison meant to support a significance claim.
    results = {
        "sage": [0.30, 0.31, 0.29, 0.32, 0.30, 0.31, 0.29],
        "gat": [0.05, 0.06, 0.04, 0.05, 0.06, 0.05, 0.04],
    }
    out = compare_models(results)
    assert set(out["summary"].keys()) == {"sage", "gat"}
    assert "sage_vs_gat" in out["pairwise"]
    assert out["pairwise"]["sage_vs_gat"]["significant_at_0.05"] is True
    assert abs(out["summary"]["sage"]["mean"] - 0.30285714285714283) < 1e-9
