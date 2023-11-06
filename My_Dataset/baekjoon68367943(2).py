# https://www.acmicpc.net/source/68367943
string = input()

def check_palindrome(s):
    return s == s[::-1]

def find_longest_palindrome(s):
    n = len(s)
    for i in range(n, 0, -1):
        for j in range(n - i + 1):
            substring = s[j:j+i]
            if check_palindrome(substring):
                return i - 1
    return n

ans = find_longest_palindrome(string)

print(ans)
