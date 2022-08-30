#!/bin/bash

# remove any previously unzipped copies of Shell1/
if [ -d Shell1 ];
then
  echo "Removing old copies of Shell1/..."
  rm -r Shell1
  echo "Done"
fi

# unzip a fresh copy of Shell1/
echo "Unzipping Shell1.zip..."
unzip -q Shell1
echo "Done"

: ' Problem 1: In the space below, write commands to change into the
Shell1/ directory and print a string telling you the current working
directory. '
echo "Problem 1 - Changing into Shell 1 Directory"
cd Shell1
echo "Printing Directory using pwd"
pwd
echo "Done"


: ' Problem 2: Use ls with flags to print one list of the contents of
Shell1/, including hidden files and folders, listing contents in long
format, and sorting output by file size. '
echo "Problem 2 - Print contents of Shell1"
ls -alS
echo "Done"

: ' Problem 3: Inside the Shell1/ directory, delete the Audio/ folder
along with all its contents. Create Documents/, Photos/, and
Python/ directories. Rename the Random/ folder as Files/. '
echo "Problem 3"
echo "Delete Audio Dir and Contents"
rm -r Audio
echo "Create directories"
mkdir Documents
mkdir Photos
mkdir Python
echo "Rename Random to Files"
mv Random Files
ls
echo "Done"


: ' Problem 4: Using wildcards, move all the .jpg files to the Photos/
directory, all the .txt files to the Documents/ directory, and all the
.py files to the Python/ directory. '
echo "Problem 4"
echo "Move .jpg files to Photos"
mv *.jpg Photos/
echo "Move .txt files to Documents"
mv *.txt Documents/
echo "Move .py files to Python"
mv *.py Python/
ls
echo "Done"


: ' Problem 5: Move organize_photos.sh to Scripts/, add executable
permissions to the script, and run the script. '
echo "Problem 5"
ls Photos/
echo "Find organize_photos.sh"
find . -name "organize_photos.sh" -type f
echo "Move organize_photos.sh to Scripts"
mv Files/Feb/organize_photos.sh Scripts/
echo "Changing permissions on organize_photos.sh"
chmod u+x Scripts/organize_photos.sh
echo "Running Script"
./Scripts/organize_photos.sh
ls Photos/
ls -l Scripts/
echo "Done"

: ' Problem 6: Copy img_649.jpg from UnixShell1/ to Shell1/Photos, making
sure to leave a copy of the file in UnixShell1/.'
echo "Problem 6 - copy img"
cp ../img_649.jpg ../Shell1/Photos
echo "Done"


# remove any old copies of UnixShell1.tar.gz
if [ ! -d Shell1 ];
then
  cd ..
fi

if [ -f UnixShell1.tar.gz ];
then
  echo "Removing old copies of UnixShell1.tar.gz..."
  rm -v UnixShell1.tar.gz
  echo "Done"
fi

# archive and compress the Shell1/ directory
echo "Compressing Shell1/ Directory..."
tar -zcpf UnixShell1.tar.gz Shell1/*
echo "Done"
