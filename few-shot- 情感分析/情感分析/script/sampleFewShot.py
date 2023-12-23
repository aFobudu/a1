import pandas as pd

pd_all = pd.read_csv('../data/ChnSentiCorp_htl_all.csv', index_col=None)

pd_all = pd_all.dropna()

pd_positive = pd_all[pd_all.label == 1]
pd_negative = pd_all[pd_all.label == 0]


def get_balance_corpus(corpus_size, corpus_pos, corpus_neg):
    sample_size = corpus_size // 2
    pd_corpus_balance = pd.concat([corpus_pos.sample(sample_size, replace=corpus_pos.shape[0] < sample_size),
                                   corpus_neg.sample(sample_size, replace=corpus_neg.shape[0] < sample_size)])

    print('评论数目（总体）：%d' % pd_corpus_balance.shape[0])
    print('评论数目（正向）：%d' % pd_corpus_balance[pd_corpus_balance.label == 1].shape[0])
    print('评论数目（负向）：%d' % pd_corpus_balance[pd_corpus_balance.label == 0].shape[0])

    return pd_corpus_balance


# 先做一次全部负样本抽样，然后正样本取16个出来,负样本取16个出来
ChnSentiCorp_htl_ba_2444 = get_balance_corpus(16, pd_positive, pd_negative)

print(ChnSentiCorp_htl_ba_2444)

# 固定每次训练的样本，控制变量
ChnSentiCorp_htl_ba_2444.to_csv('../data/ChnSentiCorp_htl_ba_16.csv')
