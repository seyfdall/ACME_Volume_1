# standard_library.py
"""Python Essentials: The Standard Library.
<Name> Dallin Seyfried
<Class> Section 002
<Date> 8/16/22
"""

import random
import calculator as calc
from itertools import combinations
import sys
import time
import box


# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order, separated by a comma).
    """
    # Using built-in functions to calculate the min, max, and average
    return min(L), max(L), sum(L) / len(L)


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test integers, strings, lists, tuples, and sets. Print your results.
    """
    int1 = 4
    int2 = int1
    int2 += 3
    print("Integer mutable: ")
    print(int1 == int2)
    str1 = "Hello There"
    str2 = str1
    str2 += " General Kenobi"
    print("String mutable: ")
    print(str1 == str2)
    list1 = [1, 2, 3]
    list2 = list1
    list2[0] = 0
    print("List mutable: ")
    print(list1 == list2)
    tuple1 = (1, 2)
    tuple2 = tuple1
    tuple2 += (1,)
    print("Tuple mutable: ")
    print(tuple1 == tuple2)
    set1 = {1, 2, 3}
    set2 = set1
    set2.add(4)
    print("Set mutable: ")
    print(set1 == set2)


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt() that are
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    # Calculating the squares of both a and b using calculator module's product func
    a_sqr = calc.product(a, a)
    b_sqr = calc.product(b, b)
    # Summing the squares
    sum = calc.sum(a_sqr, b_sqr)
    # Returning the square root of sum
    return calc.sqrt(sum)


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    # Initialize empty array
    p_set = []
    # Iterate through the range of the length of the iterable
    for i in range(len(A) + 1):
        # Generate combinations according to the current length you're at
        curr_combinations = list(combinations(A, i))
        # Append each new combination generated to the p_set array
        for j in range(len(curr_combinations)):
            p_set.append(set(curr_combinations[j]))
    return p_set


# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""
    # Initializing remaining numbers and start time
    remaining_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    start_time = time.time()

    # Begin the game loop which continues till error or game ends
    while len(remaining_numbers) > 0:
        print("Numbers Left: " + str(remaining_numbers))

        # Roll a die (two if score is over 6)
        roll = random.randint(1, 6)
        if sum(remaining_numbers) > 6:
            roll += random.randint(1, 6)
        print("Roll: " + str(roll))

        # Check if roll is valid
        if not box.isvalid(roll, remaining_numbers):
            print("Game Over!")
            break

        # Print the time taken so far
        current_time = timelimit - (time.time() - start_time)
        print("Seconds left: " + str(current_time.__round__(2)))

        # Take in input and eliminate it from the remaining numbers
        eliminate_str = input("Numbers to eliminate: ")
        eliminate_numbers = box.parse_input(eliminate_str, remaining_numbers)
        remaining_numbers = [num for num in remaining_numbers if num not in eliminate_numbers]

        # If input is invalid end the game
        if len(eliminate_numbers) == 0:
            print("Invalid Input")

        # Check time taken so far, if it has exceeded the time limit end the game
        current_time = timelimit - (time.time() - start_time)
        if current_time <= 0:
            print("Game Over!")
            break
        print("")

    # Output the player's stats for the game
    time_played = time.time() - start_time
    print("Score for player " + player + ": " + str(sum(remaining_numbers)) + " points")
    print("Time played: " + str(time_played.__round__(2)) + " seconds")
    if len(remaining_numbers) > 0:
        print("Better luck next time >:)")
    else:
        print("Congratulations!! You shut the box!")


if __name__ == "__main__":
    # prob1List = [1, 2, 3, 4]
    # print(prob1(prob1List))
    # print(prob2())
    # print(hypot(3, 4))
    # print(power_set({'a', 'b', 'c'}))
    if len(sys.argv) == 3:
        shut_the_box(sys.argv[1], float(sys.argv[2]))
    else:
        print("Please enter your name and a time limit as the arguments")
