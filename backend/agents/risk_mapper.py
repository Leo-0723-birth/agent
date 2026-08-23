# -*- coding: utf-8 -*-

"""
RiskMapper Agent

作用：
将公告研读Agent输出的自然语言风险因素
映射到45类监管关注点体系。

输入：
ctx["risk_factors"]

输出：
ctx["risk_labels"]
"""


from .base import AgentBase

from .label_keywords_v2 import (
    LABEL_KEYWORDS,
    expand_labels,
    TAXONOMY_NAMES,
    ANNOUNCEMENT_CATEGORY_MAP,
)


class RiskMapperAgent(AgentBase):

    name = "RiskMapperAgent"


    def execute(self, company, ctx):

        risk_factors = getattr(
            ctx.semantic,
            "risk_factors",
            []
        )

        results = []


        for item in risk_factors:

            labels = set()


            # ------------------------
            # 1. 风险描述关键词匹配
            # ------------------------

            text = " ".join([
                str(item.get("category", "")),
                str(item.get("description", "")),
                str(item.get("evidence", ""))
            ])


            for code, keywords in LABEL_KEYWORDS.items():

                for kw in keywords:

                    if kw in text:
                        labels.add(code)
                        break


            # ------------------------
            # 2. 上游category映射
            # ------------------------

            category = item.get(
                "category",
                ""
            )


            labels.update(
                ANNOUNCEMENT_CATEGORY_MAP.get(
                    category,
                    []
                )
            )


            # ------------------------
            # 3. 统一展开
            # ------------------------

            labels = expand_labels(
                labels
            )


            taxonomy_labels = [
                x
                for x in labels
                if x in TAXONOMY_NAMES
            ]


            results.append({

                "risk_factor":
                    item,

                "taxonomy_labels":
                    sorted(
                        taxonomy_labels
                    ),

                "label_names":
                    [
                        TAXONOMY_NAMES[x]
                        for x in taxonomy_labels
                    ]

            })


        ctx.semantic.risk_labels = results


        return ctx