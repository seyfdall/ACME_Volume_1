# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
<Name>
<Class>
<Date>
"""

from random import choice


# Problem 1
def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:

    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """

    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")

    # Raise error if input is not an integer or a 3 digit number
    if not step_1.isdigit() or len(step_1) != 3:
        raise ValueError("Input was not a 3 digit number")

    # Raise error if difference between first and last integer is less than 2
    if abs(int(step_1[0]) - int(step_1[2])) < 2:
        raise ValueError("First number's first and last digits differ by less than 2")

    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")

    # Raise error if the second number is not the reverse
    if step_2[::-1] != step_1:
        raise ValueError("The second number is not the reverse of the first number")

    step_3 = input("Enter the positive difference of these numbers: ")

    # Raise error if the third number is not the abs difference between the first two
    if abs(int(step_2) - int(step_1)) != int(step_3):
        raise ValueError("The third number is not the positive difference of the first two numbers")

    step_4 = input("Enter the reverse of the previous result: ")

    # Raise error if the fourth number is not the reverse of the third
    if step_4[::-1] != step_3:
        raise ValueError("The fourth number is not the reverse of the third number")

    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")


# Problem 2
def random_walk(max_iters=1e12):
    """
    If the user raises a KeyboardInterrupt by pressing ctrl+c while the
    program is running, the function should catch the exception and
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk.
    """

    walk = 0
    directions = [1, -1]
    i = 0

    # Try catch block with KeyboardInterrupt
    try:
        for i in range(int(max_iters)):
            walk += choice(directions)
    except KeyboardInterrupt:
        print("Process interrupted at iteration", i)
    else:
        print("Process completed")
    finally:
        return walk


# Problems 3 and 4: Write a 'ContentFilter' class.
class ContentFilter(object):
    """Class for reading in file

    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file

    """
    # Problem 3
    def __init__(self, filename):
        """ Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """
        open_successful = False
        contents = ''
        while not open_successful:
            # Attempt to open, read and store input from a file
            # Ask for input again if an exception is raised during the process
            try:
                with open(filename, 'r') as infile:
                    contents = infile.read()
                    open_successful = True
            except Exception as e:
                filename = input("Please enter a valid file name:")

        # Generate attributes to help with printing the string description later
        self.contents = contents
        self.filename = filename
        self.alphacount = len([c.lower() for c in self.contents if c.isalpha()])
        self.numcount = len([c for c in self.contents if c.isdigit()])
        self.spacecount = len([c for c in self.contents if c in [' ', '\t', '\n']])
        self.lines = 0 if len(self.contents) == 0 else len([c for c in self.contents.strip() if c == '\n']) + 1


 # Problem 4 ---------------------------------------------------------------
    def check_mode(self, mode):
        """ Raise a ValueError if the mode is invalid. """
        if mode != 'w' and mode != 'x' and mode != 'a':
            raise ValueError("The mode to use the file is not set correctly")

    def uniform(self, outfile, mode='w', case='upper'):
        """ Write the data to the outfile with uniform case. Include an additional
        keyword argument case that defaults to "upper". If case="upper", write
        the data in upper case. If case="lower", write the data in lower case.
        If case is not one of these two values, raise a ValueError. """

        self.check_mode(mode)

        with open(outfile, mode) as outputFile:
            if case == "upper":
                outputFile.write(self.contents.upper())
            elif case == "lower":
                outputFile.write(self.contents.lower())
            else:
                raise ValueError("Case not correctly input for uniform() method")



    def reverse(self, outfile, mode='w', unit='word'):
        """ Write the data to the outfile in reverse order. Include an additional
        keyword argument unit that defaults to "line". If unit="word", reverse
        the ordering of the words in each line, but write the lines in the same
        order as the original file. If units="line", reverse the ordering of the
        lines, but do not change the ordering of the words on each individual
        line. If unit is not one of these two values, raise a ValueError. """

        self.check_mode(mode)
        with open(outfile, mode) as outputFile:
            if unit == "word":
                output = self.contents.split('\n')
                for i in range(len(output)):
                    output[i] = ' '.join(output[i].split(' ')[::-1])
                output = '\n'.join(output)
                outputFile.write(output)
            elif unit == "Line":
                output = '\n'.join(self.contents.split('\n')[::-1])
                outputFile.write(output)
            else:
                raise ValueError("Case not correctly input for uniform() method")

    def transpose(self, outfile, mode='w'):
        """ Write a transposed version of the data to the outfile. That is, write
        the first word of each line of the data to the first line of the new file,
        the second word of each line of the data to the second line of the new
        file, and so on. Viewed as a matrix of words, the rows of the input file
        then become the columns of the output file, and viceversa. You may assume
        that there are an equal number of words on each line of the input file. """

        self.check_mode(mode)
        with open(outfile, mode) as outputFile:
            # Cycle through the contents splitting each line and word
            output = self.contents.split('\n')
            for i in range(len(output)):
                output[i] = output[i].split(' ')

            # Generate 2d array to be the transpose of the contents
            transpose_output = [[0 for i in range(len(output))] for j in range(len(output[0]))]
            for i in range(len(output)):
                for j in range(len(output[i])):
                    transpose_output[j][i] = output[i][j]

            # Recombine the transpose into one single string to write to file
            for i in range(len(transpose_output)):
                transpose_output[i] = ' '.join(transpose_output[i])
            transpose_output = '\n'.join(transpose_output)

            outputFile.write(transpose_output)



    def __str__(self):
        """ Printing a ContentFilter object yields the following output:

        Source file:            <filename>
        Total characters:       <The total number of characters in file>
        Alphabetic characters:  <The number of letters>
        Numerical characters:   <The number of digits>
        Whitespace characters:  <The number of spaces, tabs, and newlines>
        Number of lines:        <The number of lines>
        """

        # Using f strings to return a string for the content filter
        return (f"Source file:\t\t\t{self.filename}\n" +
                f"Total characters:\t\t{len(self.contents)}\n" +
                f"Alphabetic characters:\t{self.alphacount}\n" +
                f"Numerical characters:\t{self.numcount}\n" +
                f"Whitespace characters:\t{self.spacecount}\n" +
                f"Number of lines:\t\t{self.lines}")


if __name__ == "__main__":
    # arithmagic()
    # random_walk(100000000)
    content = ContentFilter("test")
    content.transpose("test1", 'w')
    print(str(content))
