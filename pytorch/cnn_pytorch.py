import matplotlib.pyplot as plt
v = 10
rs = 200
rl = [x for x in range(0, 1001)]
total_r = 0

result = []
for i in rl:
    total_r = rs + i
    result.append(((v*v)*i/(total_r*total_r))*1000)

result_a = [result[50], result[100], result[350],
            result[500], result[650], result[800], result[950]]

plt.plot(result)
plt.xlabel('RL')
plt.ylabel('PL')
plt.show()
