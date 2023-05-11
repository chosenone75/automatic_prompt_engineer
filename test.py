import openai
from collections import defaultdict
openai.api_key = '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'

records = []
with open("/Users/lebronran/Desktop/AIGC_Detector/prompt_input.tsv", "r") as f:
    index2column = {}
    column2text = {
        'addtime':'**时间**',
        'shopname':"**商户名**",
        'shopaddress':"**商户地址**",
        'landmark':"**地标**",
        'subway':"**地铁**",
        'regionname':"**商圈**",
        "branchname":"**分店名**",
        "dishname":"**菜品名称**",
        "tastestar":"**口味分，满分五分**",
        "envstar":"**环境分，满分五分**",
        "servicestar":"**服务分，满分五分**",
        "avgprice":"**人均价格**"
    }
    right_count = 0
    for i, line in enumerate(f):
        line = line.strip()
        if i == 0:
            for j, column in enumerate(line.split('\t')):
                index2column[j]=column.strip()
            right_count = len(line.split('\t'))

        else:
            infos = line.strip().split("\t")
            record = ""
            review = ""
            if len(infos) == right_count:
                for j,info in enumerate(infos):
                    info = info.strip(",")
                    # input
                    if index2column[j] in column2text:
                        record += (column2text[index2column[j]]+ ":"+ info+",")
                    # output
                    if index2column[j] == 'review':
                        review = info
                        review = review.replace("\\\\n", "").replace("\n", "")
            records.append((record.strip(","), review))

records_debug = records[:10]

# demo
# First, let's define a simple dataset consisting of words and their antonyms.
words = ["sane", "direct", "informally", "unpopular", "subtractive", "nonresidential",
    "inexact", "uptown", "incomparable", "powerful", "gaseous", "evenly", "formality",
    "deliberately", "off"]
antonyms = ["insane", "indirect", "formally", "popular", "additive", "residential",
    "exact", "downtown", "comparable", "powerless", "solid", "unevenly", "informality",
    "accidentally", "on"]

# dp data
# First, let's define a simple dataset consisting of words and their antonyms.
hints = [record[0] for record in records_debug]
reviews = [record[1] for record in records_debug]

# Now, we need to define the format of the prompt that we are using.

eval_template = \
"""Instruction: [PROMPT]
Input: [INPUT]
Output: [OUTPUT]"""

# Now, let's use APE to find prompts that generate antonyms for each word.
from automatic_prompt_engineer import ape

result = ape.simple_estimate_cost(
    dataset=(hints, reviews),
    eval_template=eval_template,
)
print(result)

result, demo_fn = ape.simple_ape(
    dataset=(hints, reviews),
    eval_template=eval_template,
)

print(result)

from automatic_prompt_engineer import ape

manual_prompt = "用给定的信息写一条评价："

human_result = ape.simple_eval(
    dataset=(words, antonyms),
    eval_template=eval_template,
    prompts=[manual_prompt],
)

print(human_result)