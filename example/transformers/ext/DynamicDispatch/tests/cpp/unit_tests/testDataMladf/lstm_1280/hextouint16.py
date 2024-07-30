import numpy as np
func = lambda x: int(x, base=16)

def unhex(fname):
    with open(fname, 'r') as f:
        x = list(f)
        y = [func(item.strip()) for item in x]
    return y

x = unhex('ifm32.txt')
y = np.uint32(x)
print(y.shape)
y.tofile("ifm.bin")

x = unhex('wts32.txt')
y = np.uint32(x)
print(y.shape)
y.tofile("wts.bin")
