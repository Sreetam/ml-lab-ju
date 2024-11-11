size = int(input("Enter Size:"))
numberList = [int(input("Enter Numbers:")) for i in range(size)]
lindex = int(input("Enter Lower Index:"))
gindex = int(input("Enter Lower Index:"))
oldlist = numberList.copy()
for i  in range(lindex, gindex):
    numberList[i] = numberList[i] ** 2
print(oldlist)
print(numberList)
