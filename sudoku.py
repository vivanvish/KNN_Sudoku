import sys
import time

final_ans = ''

def same_row(i,j):
    return int(i/9) == int(j/9)

def same_col(i,j):
    return (i-j) % 9 == 0

def same_block(i,j):
    return int(i/27) == int(j/27) and int(i%9/3) == int(j%9/3)

def solve(a):
    i = a.find('0')
    if i == -1:
        global final_ans
        final_ans = a

    excluded_numbers = set()

    for j in range(81):
        if (same_row(i,j) or same_col(i,j) or same_block(i,j)):
            excluded_numbers.add(a[j])

    for m in '123456789':
        if m not in excluded_numbers:
            solve(a[:i]+m+a[i+1:])

def solve_sudoku(a):
    if len(a) == 81:
        solve(a)
        return final_ans

    else:
        print('Arg value error')
        return -1