import os

import pandas as pd

f = open("log.txt")
lines = f.readlines()
cnt = 0
result = []
block = []
titles = []
record = 0
title = 0
pre_out = 0
block_size = 0
for line in lines:
    print(line)
    if "******************************************************" in line:
        record = record + 1
        title = title + 1
        pre_out = 0
        continue
    else:
        if ":" not in line:
            if "csv" in line:
                break
            else:
                continue

    if record > 0:
        index = line.find(":")

        # if title == 1:
        name = line[:index]
        titles.append(name)

        block.append(line.replace("\n", "")[index + 2 :])
    if "output:" in line:
        if len(block):
            if len(block) != block_size:
                title = 1
            if title == 1:
                block_size = len(block)
                result.append(titles)
            result.append(block)
            block = []
            titles = []
        pre_out = 1

name = ["index"]
test = pd.DataFrame(data=result)
test.to_csv("ops_0.csv", encoding="utf-8")
s = test.T
s.to_csv("ops.csv", encoding="utf-8")
