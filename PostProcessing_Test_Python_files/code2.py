# 유사 회문 판단 함수
def palindrome_like(string, left, right):
    while left < right:
        if string[left] == string[right]:
            left += 1
            right -= 1
        else:
            return False
    return True
# 회문 함수
def palindrome(string, left, right):
    while left < right:
        if string[left] == string[right]:
            left += 1
            right -= 1
        else:
            pseudo1 = palindrome_like(string, left+1, right)
            pseudo2 = palindrome_like(string, left, right-1)

            if pseudo1 or pseudo2:
                return 1
            else:
                return 2
    return 0

for string in list([input() for _ in range(int(input()))]):
    print(palindrome(string, 0, len(string) - 1))