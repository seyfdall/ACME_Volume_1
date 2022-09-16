# object_oriented.py
"""Python Essentials: Object Oriented Programming.
<Name> Dallin Seyfried
<Class> Math 345 Volume 1
<Date> 9/8/22
"""
from math import sqrt
import cmath


class Backpack:
    """A Backpack object class. Has a name, a color, a max_size, and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        color (str): the color of the backpack.
        max_size (int): the max number of items in the backpack
        contents (list): the contents of the backpack.
    """

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size=5):
        """Set the name and initialize an empty list of contents.

        Parameters:
            name (str): the name of the backpack's owner.
            color (str): the color of the backpack.
            max_size (int): the max number of items in the backpack
        """
        self.name = name
        self.color = color
        self.max_size = max_size
        self.contents = []

    def put(self, item):
        """Add an item to the backpack's list of contents.
        If too many items, don't add them.
        """
        # If there are too many items in contents print "No Room!"
        # Otherwise, add the item to the backpack
        if len(self.contents) >= self.max_size:
            print("No Room!")
        else:
            self.contents.append(item)

    def take(self, item):
        """Remove an item from the backpack's list of contents."""
        self.contents.remove(item)

    def dump(self):
        """Resets the contents of the backpack to an empty list"""
        # Set contents to empty list
        self.contents.clear()

    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)

    def __eq__(self, other):
        """Determine if two Backpacks are equal if their attributes match"""
        # Check all conditions and return true if self and other match, otherwise return false
        if self.color == other.color and self.name == other.name and len(self.contents) == len(other.contents):
            return True
        return False

    def __str__(self):
        """Return a string describing the Backpack"""
        # Build a description string one line at a time
        description = ("Owner:\t\t" + str(self.name) + '\n' +
                       "Color:\t\t" + str(self.color) + '\n' +
                       "Size:\t\t" + str(len(self.contents)) + '\n' +
                       "Max Size:\t" + str(self.max_size) + '\n' +
                       "Contents:\t" + str(self.contents) + '\n')
        return description


# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):
    """A Jetpack object class. Inherits from the Backpack class.
        A Jetpack is smaller than a backpack and can be used to fly to the moon.

        Attributes:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
            contents (list): the contents of the backpack.
            closed (bool): whether or not the knapsack is tied shut.
    """

    def __init__(self, name, color, max_size=2, fuel=10):
        """Use the Backpack constructor to initialize the name, color,
        max_size, and fuel attributes. A Jetpack only holds 2 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
            fuel (int): the amount of fuel the Jetpack has
        """
        # Calling parent constructor and adding Jetpack attribute
        Backpack.__init__(self, name, color, max_size)
        self.fuel = fuel

    def fly(self, fuel):
        """Method to fly the Jetpack to the moon or elsewhere - to infinity
        and beyond you yayhoos.  Informs the Jetpack user if there is not enough
        fuel, otherwise it burns it and the user flies.

        Args:
            fuel: Takes in an amount of fuel to burn
        """
        # Check fuel and reduce if there's enough, otherwise print error
        if fuel <= self.fuel:
            self.fuel -= fuel
        else:
            print("Not enough fuel!")

    def dump(self):
        """Overrides the Backpack dump method to also empty the Jetpack of fuel"""
        # Adjust fuel and then call Backpack dump method
        self.fuel = 0
        Backpack.dump(self)


# Problem 4: Write a 'ComplexNumber' class.
class ComplexNumber:
    """A Complex Number object class.

        Attributes:
            real (int): The real number component of a complex number
            imag (int): The imaginary number component of a complex number
    """

    def __init__(self, real, imag):
        """Use the ComplexNumber constructor to initialize a complex number
        object containing an integer for the reals and an integer for the imaginary

        Args:
            real: (int) real part of complex number
            imag: (int) imaginary part of complex number
        """
        # Pass parameters to the object attributes
        self.real = real
        self.imag = imag

    def conjugate(self):
        """Returns a conjugate complex number object of the current one"""
        # Call the constructor, invert the imaginary and return the new object
        return ComplexNumber(self.real, -self.imag)

    def __str__(self):
        """Returns a string description of the ComplexNumber Object"""
        # Return a description based on the sign of the imaginary number
        if self.imag < 0:
            return "(" + str(self.real) + "-" + str(abs(self.imag)) + "j)"
        else:
            return "(" + str(self.real) + "+" + str(abs(self.imag)) + "j)"

    def __abs__(self):
        """Returns the magnitude of the complex number"""
        # Uses math.sqrt to compute the magnitude
        return sqrt(self.real**2 + self.imag**2)

    def __eq__(self, other):
        """Compares the imaginary and real parts of two different complex number
        objects and returns true if the match"""
        # Conditional check to see if both equal
        if self.real == self.real and self.imag == self.imag:
            return True
        return False

    def __add__(self, other):
        """Returns a ComplexNumber Object from adding two other complex numbers"""
        # Builds a ComplexNumber object by using the constructor
        return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        """Returns a ComplexNumber Object from subtracting two other complex numbers"""
        # Builds a ComplexNumber object by using the constructor
        return ComplexNumber(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        """Returns a ComplexNumber Object from multiplying two other complex numbers"""
        # Builds a ComplexNumber object by using the constructor
        return ComplexNumber(self.real * other.real - self.imag * other.imag,
                             self.imag * other.real + self.real * other.imag)

    def __truediv__(self, other):
        """Returns a ComplexNumber Object from dividing two other complex numbers"""
        # Builds a ComplexNumber object by using the constructor
        if other.real == 0 and other.imag == 0:
            print("Fool of a took, throw yourself in next time and rid us of your stupidity :)")
        else:
            return ComplexNumber((self.real * other.real + self.imag * other.imag) / (other.real**2 + other.imag**2),
                                 (self.imag * other.real - self.real * other.imag) / (other.real**2 + other.imag**2))


# Test methods
def test_backpack():
    testpack = Backpack("Barry", "black") # Instantiate the object.
    if testpack.name != "Barry": # Test an attribute.
        print("Backpack.name assigned incorrectly")
    for item in ["pencil", "pen", "paper", "computer", "body"]:
        testpack.put(item) # Test a method.
    print("Contents:", testpack.contents)
    print(str(testpack))


def test_Jetpack():
    jetpack = Jetpack("Zach", "blue", 50, 500)
    if jetpack.name != "Zach":
        print("Jetpack.name assigned incorrectly")
    for item in ["pencil", "pen", "paper", "computer", "shrinkray"]:
        jetpack.put(item)  # Test a method.
    print("Contents:", jetpack.contents)
    jetpack.fly(490)
    if jetpack.fuel != 10:
        print("Jetpack fly is not implemented correctly fuel:", jetpack.fuel)
    jetpack.dump()
    if len(jetpack.contents) != 0 or jetpack.fuel != 0:
        print("Jetpack dump not implemented correctly")


def test_ComplexNumber(a, b):
    py_cnum, my_cnum = complex(a, b), ComplexNumber(a, b)
    # Validate the constructor.
    if my_cnum.real != a or my_cnum.imag != b:
        print("__init__() set self.real and self.imag incorrectly")
    # Validate conjugate() by checking the new number's imag attribute.
    if py_cnum.conjugate().imag != my_cnum.conjugate().imag:
        print("conjugate() failed for", py_cnum)
    # Validate __str__().
    if str(py_cnum) != str(my_cnum):
        print("__str__() failed for", py_cnum)
    # Validate __eq__
    if py_cnum != my_cnum:
        print("__eq__() failed for", py_cnum)
    # Validate __add__
    if my_cnum + my_cnum != py_cnum + py_cnum:
        print("__add__() failed for", py_cnum)
    # Validate __sub__
    if my_cnum - my_cnum != py_cnum - py_cnum:
        print("__sub__() failed for", py_cnum)


if __name__ == "__main__":
    test_backpack()
    test_Jetpack()
    test_ComplexNumber(-2, -5)