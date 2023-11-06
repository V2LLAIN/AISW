# https://www.acmicpc.net/source/68367943
string = input()

ans = 0

if string == string[::-1]:
    if string[:len(string) // 2 + 1] == string[:len(string) // 2 + 1][::-1]:
        ans = -1
    else:
        ans = len(string) - 1
else:
    ans = len(string)

print(ans)