from sympy import isprime
size = input("Enter Size:")
numberList = [int(input("Enter Numbers:")) for i in range(int(size))]
for i in numberList:
    if(isprime(i)):
        print(i, end=', ')