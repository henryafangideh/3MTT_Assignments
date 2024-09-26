# Addition Function
def add(num1,num2):
    ans = num1+num2
    return ans

# Subtraction
def sub(num1,num2):
    ans = num1-num2
    return ans

# Multiplication
def mult(num1,num2):
    ans = num1*num2
    return ans

# Division
def div(num1,num2):
    ans = num1/num2
    return ans

num1 = int(input('Enter the first number:'))
num2 = int(input('Enter the second number:'))
operation = (input("Type 'add' for addition, 'sub' for subtraction,'mult' for multiplication and 'div' for division.")).lower()

if operation == 'add':
    result = add(num1,num2)
    print (f'Your result is {result}')
elif operation == 'sub':
    result = sub(num1,num2)
    print (f'Your result is {result}')
elif operation == 'mult':
    result = mult(num1,num2)
    print (f'Your result is {result}')
elif operation == 'div':
    result = div(num1,num2)
    print (f'Your result is {result}')


