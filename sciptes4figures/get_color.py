import matplotlib.pyplot as plt

c_list = []
for i in range(10):
    line = plt.plot([i, i])
    c= line[0].get_color()
    plt.annotate(c, (0, i))
    c_list.append(c)
plt.show()
print('\''+'\',\n \''.join(c_list)+'\'')