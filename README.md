# Directory Requirement
- plots/
- weights/
- filtered_corpusid_input.csv (input file, can be specified)

# Train

`python -m citexformer_train -n 1000 -T abstract,introduction -N year,publicationDate -e 1 -b 2 -l 2e-5 -f 256 -F 128 -D 0.1`

# Test

`python -m citexformer_test -w "[YOUR_MODEL_PATH]" -b 4`
