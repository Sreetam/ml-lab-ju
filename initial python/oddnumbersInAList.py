size = input("Enter Size:")
numberList = [int(input("Enter Numbers:")) for i in range(int(size))]
for i in numberList:
    if(i%2!=0):
        print(i, end=', ')