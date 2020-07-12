#!/bin/bash

# Helper script for publishing to crates.io

# Check readme is up-to-date. If this changed the README, then the dirty files
# check later should pick it up.
cargo readme > README.md

# Clean .gitignore that hides files we need for the package
echo "Adding tensorflow download and gen to package..."
echo "" > submodules/tensorflow/tensorflow/lite/micro/tools/make/.gitignore
echo

# Check packaging
rm -rf target/package
echo "Attempting to package tfmicro..."
cargo publish --dry-run --allow-dirty --features="build"

if [ $? -eq 0 ]
then
    echo "Done"
    echo
else
    echo "Fail"
    exit
fi

# Check crate size
echo "Checking crate size..."
CRATE_BYTES=$(stat -c %s target/package/tfmicro-*.crate)
CRATE_MB=$(echo $CRATE_BYTES | awk '{print $1 / (1024 * 1024)}')
echo "Crate is $CRATE_MB MB..."

if [ $CRATE_BYTES -gt 5000000 ] # 5MB
then
    echo
    echo "You probably want to slim that down (crates.io hard limit is 10MB?)"
    echo
    read -p "Press enter to print a list of files..." -n 1 -r
    tar tvaf target/package/tfmicro-*.crate | awk '{print $3 "\t" $6}' | sort -n
    exit
else
    echo "Nice"
    echo
fi

echo "Checking for dirty files..."

# Run cargo publish without the --allow-dirty flag
#
# Count the number of lines that aren't from the downloads directory
DIRTY_N_LINES=$(cargo publish --dry-run 2>&1 >/dev/null | grep -cv submodules/tensorflow/tensorflow/lite/micro/tools/make/downloads)

# There should be five lines of human speak
if [ ! $DIRTY_N_LINES -eq "5" ]
then
    echo "Detected dirty files that outside the tensorflow/lite/micro/tools/make/downloads directory..."
    echo
    cargo publish --dry-run 2>&1 >/dev/null | grep -v submodules/tensorflow/tensorflow/lite/micro/tools/make/downloads
    echo
    echo Fail
    exit
else
    echo "Clean"
fi

# Final confirmation
echo
read -p "Ready to publish. Are you sure? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo
    cargo publish --allow-dirty
else
    echo "Crate not published"
fi


# Put back that .gitignore file we had to remove
cd submodules/tensorflow
git checkout -- tensorflow/lite/micro/tools/make/.gitignore
