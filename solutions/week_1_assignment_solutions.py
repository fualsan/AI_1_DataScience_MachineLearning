# 1. Variables and Assignment
length = 5
width = 3
area = length * width
print('Area of rectangle:', area)

# 2. Data Types
pi = 3.14
radius = 2.5
circumference = 2 * pi * radius
print('Circumference of circle:', circumference)

# 3. Operator Precedence
x = 5
y = x * 2 + 10
print('Before:', y)
y = x * (2 + 10) # (2 + 10) is computed first
print('After:', y)

# 4. Booleans and Logic Operations
a = 10
b = 5
print('a > b:', a > b)
print('a == b:', a == b)

# 5. Conditionals
number = int(input('Enter a number: '))
if number % 2 == 0:
    print('The number is even.')
else:
    print('The number is odd.')

# 6. Loops
counter = 1
while counter <= 5:
    print(counter)
    counter += 1

# 7. Functions
def calculate_length(string):
    return len(string)

word1 = 'Hello'
word2 = 'Python'
print('Length of', word1, 'is', calculate_length(word1))
print('Length of', word2, 'is', calculate_length(word2))

# 8. Python Data Structures
# List
fruits = ['Apple', 'Banana', 'Orange']
fruits.append('Grapes')
print('Updated list of fruits:', fruits)

# Tuple
mixed_tuple = (1, 'two', 3.0)
for item in mixed_tuple:
    print(item)

# Dictionary
animals_legs = {'Dog': 4, 'Spider': 8, 'Bird': 2}
animals_legs['Fish'] = 0
print('Updated dictionary:', animals_legs)

# Set
set1 = {1, 2, 3}
set2 = {3, 4, 5}
intersection_set = set1.intersection(set2)
print('Intersection of sets:', intersection_set)

# 9. Slicing
text = 'PythonProgramming'
print('First three characters:', text[:3])
print('Last three characters:', text[-3:])
print('Reversed string:', text[::-1])

# 10. File I/O
user_input = input('Enter some text: ')
with open('saved_user_input.txt', 'w') as file:
    file.write(user_input)
    file.write('\n') # next line

# 11. Object-Oriented Programming (OOP)
class Car:
    def __init__(self, make, model):
        self.make = make
        self.model = model

car1 = Car('Toyota', 'Camry')
car2 = Car('Honda', 'Accord')

print('Car 1:', car1.make, car1.model)
print('Car 2:', car2.make, car2.model)

