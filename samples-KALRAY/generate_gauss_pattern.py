s='''P2
1920 1080
255
'''

def get_pix_value(x, y):
    if x == 0 and y == 0:
        return 255
    if 0 <= x <= 5 and 0 <= y <= 5:
        return 0
    if x < 300 and y < 300:
        return 255 if x%2 == y%2 else 0
    if x < 300 or y < 300:
        return 255 if (x/2)%2 == (y/2)%2 else 0
    return 255 if (x/3)%2 == (y/3)%2 else 0


for y in range(1080):
    for x in range(1920):
        s += str(get_pix_value(x, y)) + ' '
    s += '\n'
print(s)
