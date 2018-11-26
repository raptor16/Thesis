import math

n = 68
total = 0
for k in range(3, n):
    comb = math.factorial(n) / (math.factorial(k) * math.factorial(n-k))
    total = total + comb
print total