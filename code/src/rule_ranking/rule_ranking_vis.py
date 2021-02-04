import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 4
# before_ranking = (23, 8, 3, 27, 13)
# after_ranking_i1 = (24, 11, 4, 34, 13)
# after_ranking_i2 = (27, 12, 4, 34, 13)
# after_ranking_i3 = (25, 11, 2, 36, 13)
# after_ranking_i4 = (26, 13, 3, 34, 14)
# after_ranking_i5 = (20, 11, 4, 38, 16)

before_ranking = (98, 40, 18, 6)
after_ranking_i1 = (98, 44, 19, 6)
after_ranking_i2 = (102, 40, 21, 6)
after_ranking_i3 = (98, 41, 19, 4)
after_ranking_i4 = (106, 41, 22, 6)
after_ranking_i5 = (100, 46, 21, 6)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
opacity = 0.8
# bar_width = 0.1
#
# rects1 = plt.bar(index, before_ranking, bar_width, alpha=opacity, color='b',
#                  label='without personalised ranking')
#
# rects2 = plt.bar(index + bar_width, after_ranking_i1, bar_width, alpha=opacity, color='c',
#                  label='with personalised ranking (x=1)')
#
# rects3 = plt.bar(index + bar_width * 2, after_ranking_i2, bar_width, alpha=opacity, color='g',
#                  label='with personalised ranking (x=2)')
#
# rects4 = plt.bar(index + bar_width * 3, after_ranking_i3, bar_width, alpha=opacity, color='y',
#                  label='with personalised ranking (x=3)')
#
# rects5 = plt.bar(index + bar_width *4, after_ranking_i4, bar_width, alpha=opacity, color='r',
#                  label='with personalised ranking (x=4)')
#
# rects6 = plt.bar(index + bar_width * 5, after_ranking_i5, bar_width, alpha=opacity, color='m',
#                  label='with personalised ranking (x=5)')
#
# plt.xlabel('Features')
# plt.ylabel('Number of rules contaning feature')
# plt.xticks(index + bar_width*2.5, ('KCTD3', 'RARA', 'STARD3', 'ERLIN2'))
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
#           fancybox=True, shadow=False, ncol=1, prop={'size': 6})
#
# plt.tight_layout()
# plt.savefig('src/rule_ranking/feature_freq_all.png')




index = np.arange(0, n_groups/4, 0.25)
bar_width = 0.05

plt.xlabel('Features')
plt.ylabel('Number of rules contaning feature')

rects1 = plt.bar(index/2, before_ranking, bar_width, alpha=opacity, color='b',
                 label='without personalised ranking')
rects6 = plt.bar(index/2 + bar_width, after_ranking_i5, bar_width, alpha=opacity, color='m',
                 label='with personalised ranking')

plt.xticks(index/2 + bar_width/2, ('KCTD3', 'RARA', 'STARD3', 'ERLIN2'))

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
          fancybox=True, shadow=False, ncol=1, prop={'size': 10})
plt.tight_layout()
plt.savefig('src/rule_ranking/feature_freq_2.png')