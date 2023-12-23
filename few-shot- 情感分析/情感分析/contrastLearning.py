# encoding=utf-8
from transformers import AutoTokenizer, AutoModelForMaskedLM, FillMaskPipeline

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

pipeline = FillMaskPipeline(model=model, tokenizer=tokenizer)

prompt = "，它的情感是{}极的。"

positive = '相当好的酒店，很意外在山西能有这种水准的4星酒店，基本完美，没想出有什么不好的地方，住了2天，很满意！'

negative = '中庸的酒店,本来冲着环保房去的,可是,哎,有些失望啊'

# [{'sequence': '相 当 好 的 酒 店 ， 很 意 外 在 山 西 能 有 这 种 水 准 的 4 星 酒 店 ， 基 本 完 美 ， 没 想 出 有 什 么 不 好 的 地 方 ， 住 了 2 天 ， 很 满 意 ！ ， 它 的 情 感 是 积 极 的 。', 'score': 0.9549766182899475, 'token': 4916, 'token_str': '积'}, {'sequence': '相 当 好 的 酒 店 ， 很 意 外 在 山 西 能 有 这 种 水 准 的 4 星 酒 店 ， 基 本 完 美 ， 没 想 出 有 什 么 不 好 的 地 方 ， 住 了 2 天 ， 很 满 意 ！ ， 它 的 情 感 是 终 极 的 。', 'score': 0.01617138460278511, 'token': 5303, 'token_str': '终'}, {'sequence': '相 当 好 的 酒 店 ， 很 意 外 在 山 西 能 有 这 种 水 准 的 4 星 酒 店 ， 基 本 完 美 ， 没 想 出 有 什 么 不 好 的 地 方 ， 住 了 2 天 ， 很 满 意 ！ ， 它 的 情 感 是 两 极 的 。', 'score': 0.00806897971779108, 'token': 697, 'token_str': '两'}, {'sequence': '相 当 好 的 酒 店 ， 很 意 外 在 山 西 能 有 这 种 水 准 的 4 星 酒 店 ， 基 本 完 美 ， 没 想 出 有 什 么 不 好 的 地 方 ， 住 了 2 天 ， 很 满 意 ！ ， 它 的 情 感 是 消 极 的 。', 'score': 0.006713161710649729, 'token': 3867, 'token_str': '消'}, {'sequence': '相 当 好 的 酒 店 ， 很 意 外 在 山 西 能 有 这 种 水 准 的 4 星 酒 店 ， 基 本 完 美 ， 没 想 出 有 什 么 不 好 的 地 方 ， 住 了 2 天 ， 很 满 意 ！ ， 它 的 情 感 是 至 极 的 。', 'score': 0.0032030229922384024, 'token': 5635, 'token_str': '至'}]
# [{'sequence': '中 庸 的 酒 店, 本 来 冲 着 环 保 房 去 的, 可 是, 哎, 有 些 失 望 啊 ， 它 的 情 感 是 消 极 的 。', 'score': 0.6481347680091858, 'token': 3867, 'token_str': '消'}, {'sequence': '中 庸 的 酒 店, 本 来 冲 着 环 保 房 去 的, 可 是, 哎, 有 些 失 望 啊 ， 它 的 情 感 是 积 极 的 。', 'score': 0.14473390579223633, 'token': 4916, 'token_str': '积'}, {'sequence': '中 庸 的 酒 店, 本 来 冲 着 环 保 房 去 的, 可 是, 哎, 有 些 失 望 啊 ， 它 的 情 感 是 两 极 的 。', 'score': 0.10825502127408981, 'token': 697, 'token_str': '两'}, {'sequence': '中 庸 的 酒 店, 本 来 冲 着 环 保 房 去 的, 可 是, 哎, 有 些 失 望 啊 ， 它 的 情 感 是 终 极 的 。', 'score': 0.02846483513712883, 'token': 5303, 'token_str': '终'}, {'sequence': '中 庸 的 酒 店, 本 来 冲 着 环 保 房 去 的, 可 是, 哎, 有 些 失 望 啊 ， 它 的 情 感 是 恶 极 的 。', 'score': 0.012999629601836205, 'token': 2626, 'token_str': '恶'}]

res = pipeline(positive + prompt.format('[MASK]'))
print(res)

res = pipeline(negative + prompt.format('[MASK]'))
print(res)

reference = '[SEP]' + positive + prompt.format('积') + '[SEP]' + negative + prompt.format("消")

test = '四星级酒店，符合标准的地方我就不提了，有以下缺点：1.标间不提供免费瓶装水（吧台内的瓶装水8元一瓶）2.距离市区远，根本无法打到车，且不提供叫车服务3.其它餐不知，早餐巨差免费注册网站导航宾馆索引服务说明关于携程诚聘英才代理合作'

input_str = test + prompt.format('[MASK]') + reference

res = pipeline(input_str)

print(res)

# 再把刚刚的那些放进来看看效果
import pandas as pd

df = pd.read_csv('data/ChnSentiCorp_htl_ba_16.csv')

support_len = 512 - 2 - len(prompt) - len(reference)

for context in df['review']:
    res = pipeline(context[:support_len] + prompt.format('[MASK]') + reference)
    print(res[0]['score'], res[0]['token_str'])
