

for i in range (1,1000):

    if i%3 ==0:
        print(f"Fuzz ",{i})
    elif i%5==0:

        print(f"Buzz",{i})
    else:
        print("Not multiple")

#question 2 

N= int(input(print('Enter the nth number you want to find the fabonachi series ::')))
a=0
b=1
for i in range(N):
    c=a+b
    a=b
    b=c
    c=a+b
    print(c)

#prime numbers 

P = input(int('Enter the number::'))

for i in range(1,10):
    if P%i==1:
        print("P is prime:")        
    else:
        print('It is not prime')
P = int(input("Enter a number: "))
i = 2
is_prime = True   

while i <= P // 2:   
    if P % i == 0:   
        is_prime = False
        break
    i += 1

if is_prime and P > 1:
    print(f"{P} is a prime number")
else:
    print(f"{P} is not a prime number")

#guess the number ::;
A=200

from itertools import(count)
for i in count(1):
    i+=1
    G = int(input(print('Enter a number ')))
    if G>A:
        print("Greater::")
    elif G<A:
        print("Smaller ::")
    elif G==A:
        print("YOU ARE RIGHT")
        break

txt = str(input(print('Enter a string::')))

if txt == txt[::-1]:
    print("yes Is plandroe")
else:print("No its not")

txt_1= str(input(print('Enter a string::')))
txt_2 = str(input(print('Enter a string::')))
A=0
for chr in (txt_1.lower):

    if chr not in (txt_2.lower):
        print("not anagrams.")
    else:
        A +=1
    
    
if A!=0:
    print("yes it is anagrams")

F =input(int('Enter the Tempatur in F::'))
C = (F - 32) * 5/9
print(f"The Tempatuiue in C is ",{C})

for_even_NUM=int(input('Enter the the number of Evn numbers you want to see::'))

for i in range (for_even_NUM):
    if i%2==0:
        print(i,"EVEN ")
    else:(i,"Not EVEN")