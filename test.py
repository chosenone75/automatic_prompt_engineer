import openai
from collections import defaultdict
import pickle
openai.api_key = ''


def save(tf, data):
    with open(tf, "wb") as f:
        pickle.dump(data, f)

def load(tf):
    with open(tf, "rb") as f:
        data = pickle.load(f)
    return data

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
print(records_debug)
raise ""

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

# prompt_gen_templates = [
#     '''我给朋友说了个根据输入信息写商户评价的任务，根据那个具体的任务要求他们给出来如下这些输入和输出文本:\n\n[full_DEMO]\n\n你觉得具体的任务要求是什么呢，尽可能详细，请给出你推测的任务要求：[APE]''',
#     '''我需要给几家商户写大众点评平台的评价，其中会有一些具体的任务要求，根据那个具体的任务要求我写了如下的这些输入和输出文本:\n\n[full_DEMO]\n\n你觉得具体的任务要求是什么呢，尽可能详细，请给出你推测的任务要求：[APE]''',
#     '''现在有一个给大众点评写评价的任务，这个任务有一些详细具体的任务要求，比如要考虑哪些方面、要用什么行文风格去写等等，根据那个具体的任务要求不同的工作人员给出来如下这些输入和输出文本:\n\n[full_DEMO]\n\n你觉得具体的任务要求是什么呢，尽可能详细，请给出你推测的任务要求：[APE]''',
#     '''我给朋友说了个根据输入信息写商户评价的任务，根据那个具体的任务要求他们给出来如下这些输入和输出文本:\n\n[full_DEMO]\n\n你觉得具体的任务要求是什么呢，尽可能详细，尽可能的口语化描述，请给出你推测的任务要求：[APE]''',
#     '''有这样一些输入和输出文本:\n\n[full_DEMO]\n\n，你觉得从输入到输出的任务要求是什么呢，提示一下：是给商户写评价的任务，请你推测下任务要求，尽可能详细，请给出你推测的任务要求：[APE]'''
# ]

# with open("result_all.txt", "w") as f:
#     for prompt_gen_template in prompt_gen_templates:
#         result, demo_fn = ape.simple_ape(
#             dataset=(hints, reviews),
#             eval_template=eval_template,
#             num_prompts=50,
#             prompt_gen_template=prompt_gen_template
#         )

#         print(result)

#         f.write(str(result)+"\n")
from automatic_prompt_engineer import ape

manual_prompt = "请根据以下信息写一条商户的评价，需要考虑口味、环境、服务、菜品等信息，"

human_result = ape.simple_eval(
    dataset=(hints, reviews),
    eval_template=eval_template,
    prompts=[manual_prompt],
)

print(human_result)