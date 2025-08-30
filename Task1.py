# Finding Greater Numbers 

A = int(input('Enter the 1st number: '))
B = int(input('Enter the 2nd number: '))
C = int(input('Enter the 3rd number: '))

print("the Graeter numer is ",max(A,B,C))

A = int(input('Enter the 1st number: '))
B = int(input('Enter the 2ndst number: '))
C = int(input('Enter the 2ndst number: '))

if  A > B & A > C:
    print("A is greater ")
elif  A > B & A > C:
    print("A is greater ")
else:
    print(' C is greater no')

# Question 2  Reverse the string 
txt = input(print("Enter the txt "))
print(f'the reverse number is ',txt[::-1])

# Question 3 Counting the Vovels :
V=0
Vovels = input(print("Enter the char for volves  "))
for i in Vovels:
    if i in ('AEIOUaeiou'):
        V +=1
print(f'The numbers of vovles are :::{V}')

#  Checking for Palindrome
Palindrome =input(print("Enter the Palindrome "))
if Palindrome == Palindrome[::-1]:
    print("yes the string you entred is Palindrome")
else:
    print("noi its not a Palindrome")


# printing fabonochi series :
n =  int(input(print('Enter the length of fabonochi series you want to see::::')))
i= 0
for i in range (i,n):
    if i ==0:
        i=i+i
        print(i)
    else:
        i = i+(i-1)
        print(i)
    i = i+1
