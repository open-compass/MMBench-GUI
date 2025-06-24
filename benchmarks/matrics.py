import pandas as pd
from collections import defaultdict
import numpy as np
from utils.misc import insert_at, deep_nested, revert_defaultdict
from vlmeval.smp import dump


def level1_calculate_scores_normal(df):
    """
    In this implementation, we follow the conventions of other mqa benchmarks.
    """
    res = defaultdict(deep_nested)
    splits = list(set(df["difficulty"]))

    for split in splits:
        data = df[df["difficulty"] == split]
        platforms = list(set(df["platform"]))
        for platform in platforms:
            new_data = data[data["platform"] == platform]
            total_num_per_platform_per_level = int(len(new_data))
            score_hit = np.mean([row.hit for row in new_data.itertuples(index=False)])
            res[split][platform]["score"] = float(score_hit)
            res[split][platform]["sample_num"] = total_num_per_platform_per_level
        res[split]["level_sample_num"] = int(
            np.sum([x["sample_num"] for x in res[split].values()])
        )
        level_score_list = [
            x["score"] * x["sample_num"]
            for k, x in res[split].items()
            if k != "level_sample_num"
        ]
        res[split]["level_score"] = float(
            np.sum(level_score_list) / res[split]["level_sample_num"]
        )
    res["total_sample_num"] = int(np.sum([x["level_sample_num"] for x in res.values()]))

    final_score_dict = revert_defaultdict(res)
    return final_score_dict


def level1_calculate_scores_weighted(df):
    """
    In this implementation, we consider the number of options when making the score.
    This means that:
        if the agent answer one question correctly and this question has 5 options, the score is 4/5=0.8 rather then 1.0
        if the agent answer one question correctly and this question has 4 options, the score is 3/4=0.75 rather then 1.0
    """
    res = defaultdict(deep_nested)
    splits = list(set(df["difficulty"]))

    for split in splits:
        data = df[df["difficulty"] == split]
        platforms = list(set(df["platform"]))
        for platform in platforms:
            new_data = data[data["platform"] == platform]
            total_num_per_platform_per_level = int(len(new_data))
            score_weighted = np.mean(
                [
                    row.hit * (row.options_num - 1) / row.options_num
                    for row in new_data.itertuples(index=False)
                ]
            )
            score_total = np.mean(
                [
                    (row.options_num - 1) / row.options_num
                    for row in new_data.itertuples(index=False)
                ]
            )
            score_norm = score_weighted / score_total
            res[split][platform]["score"] = float(score_norm)
            res[split][platform]["score_weighted"] = float(score_weighted)
            res[split][platform]["score_base"] = float(score_total)
            res[split][platform]["sample_num"] = total_num_per_platform_per_level
        res[split]["level_sample_num"] = int(
            np.sum([x["sample_num"] for x in res[split].values()])
        )
        level_score_list = [
            x["score"] * x["sample_num"]
            for k, x in res[split].items()
            if k != "level_sample_num"
        ]
        res[split]["level_score"] = float(
            np.sum(level_score_list) / res[split]["level_sample_num"]
        )
    res["total_sample_num"] = int(np.sum([x["level_sample_num"] for x in res.values()]))

    final_score_dict = revert_defaultdict(res)
    return final_score_dict


def level2_calculate_scores(stats: defaultdict):
    final_score_dict = {}
    # we first process `basic` and `adcanced`
    for level, level_value in stats.items():
        final_score_dict[level] = {}
        for platform, platform_value in level_value.items():
            level_platform_total_num = sum(
                [len(platform_value[t]) for t in platform_value]
            )
            level_platform_icon_num = len(platform_value["icon"])
            level_platform_text_num = len(platform_value["text"])
            level_platform_icon_correct = sum(np.array(platform_value["icon"]) == 1)
            level_platform_text_correct = sum(np.array(platform_value["text"]) == 1)
            level_platform_icon_incorrect = sum(np.array(platform_value["icon"]) == 0)
            level_platform_text_incorrect = sum(np.array(platform_value["text"]) == 0)
            level_platform_icon_wrong = sum(np.array(platform_value["icon"]) == -1)
            level_platform_text_wrong = sum(np.array(platform_value["text"]) == -1)

            icon_acc = level_platform_icon_correct / level_platform_icon_num * 100
            text_acc = level_platform_text_correct / level_platform_text_num * 100
            total_acc = (
                (level_platform_icon_correct + level_platform_text_correct)
                / level_platform_total_num
                * 100
            )

            final_score_dict[level][platform] = {
                "Total num": int(level_platform_total_num),
                "Icon num": int(level_platform_icon_num),
                "Text num": int(level_platform_text_num),
                "Total accuracy": total_acc,
                "Icon accuracy": icon_acc,
                "Text accuracy": text_acc,
                "Correct num": int(
                    level_platform_icon_correct + level_platform_text_correct
                ),
                "Wrong num": int(
                    level_platform_icon_incorrect + level_platform_text_incorrect
                ),
                "Error format num": int(
                    level_platform_icon_wrong + level_platform_text_wrong
                ),
            }
    basic_summary_item_num = [
        x["Total num"] for _, x in final_score_dict["basic"].items()
    ]
    basic_summary_item_score = [
        x["Total accuracy"] for _, x in final_score_dict["basic"].items()
    ]
    basic_summary_item_factor = np.array(basic_summary_item_num) / sum(
        basic_summary_item_num
    )
    basic_summary_score = sum(
        [x * y for x, y in zip(basic_summary_item_score, basic_summary_item_factor)]
    )
    advanced_summary_item_num = [
        x["Total num"] for _, x in final_score_dict["advanced"].items()
    ]
    advanced_summary_item_score = [
        x["Total accuracy"] for _, x in final_score_dict["advanced"].items()
    ]
    advanced_summary_item_factor = np.array(advanced_summary_item_num) / sum(
        advanced_summary_item_num
    )
    advanced_summary_score = sum(
        [
            x * y
            for x, y in zip(advanced_summary_item_score, advanced_summary_item_factor)
        ]
    )
    summary_info = {
        "Average accuracy": np.mean([basic_summary_score, advanced_summary_score]),
        "Basic accuracy": basic_summary_score,
        "Advanced accuracy": advanced_summary_score,
    }
    summary_info.update(final_score_dict)

    return summary_info
