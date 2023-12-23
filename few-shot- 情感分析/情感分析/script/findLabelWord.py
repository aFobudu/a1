from transformers import AutoTokenizer, AutoModelForMaskedLM, FillMaskPipeline

tokenizer = AutoTokenizer.from_pretrained("../bert-base-chinese")

model = AutoModelForMaskedLM.from_pretrained("../bert-base-chinese")

pipeline = FillMaskPipeline(model=model, tokenizer=tokenizer)

prompt = "，它的情感是[MASK]极的。"

support_len = 512 - 2 - len(prompt)

import pandas as pd

df = pd.read_csv('../data/ChnSentiCorp_htl_ba_16.csv')

label_word = {}


# 每轮5个候选词beam search，最后通过greedy 算法，计算每个词出现的次数*权重，累加求和，最后最大的两个字可以作为label词
def calculate_token_str_sum_score(result: dict):
    if label_word.get(result['token_str']):
        label_word[result['token_str']] += result['score']
    else:
        label_word[result['token_str']] = result['score']


for context in df['review']:
    # {'sequence': 'context', 'score': 0.9549766182899475, 'token': 4916, 'token_str': '积'}
    # {'sequence': 'context', 'score': 0.01617138460278511, 'token': 5303, 'token_str': '终'}
    # {'sequence': 'context', 'score': 0.00806897971779108, 'token': 697, 'token_str': '两'}
    # {'sequence': 'context', 'score': 0.006713161710649729, 'token': 3867, 'token_str': '消'}
    # {'sequence': 'context', 'score': 0.0032030229922384024, 'token': 5635, 'token_str': '至'}
    res = pipeline(context[:support_len] + prompt)
    print(res[0]['score'], res[0]['token_str'])
    for i in res:
        calculate_token_str_sum_score(i)

# 积 10.472238317131996
# 终 0.11165991047164425
# 两 0.569754971191287
# 消 4.528043618425727
# 至 0.006657441379502416
# 積 0.049101658165454865
# 善 0.011103423312306404
# 恶 0.024828742840327322
# 多 0.0019735475070774555
for label, score in label_word.items():
    print(label, score)
