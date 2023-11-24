# Directory Requirement
- plots/
- weights/
- filtered_corpusid_input.csv (input file, can be specified)

# Train

`python -m citexformer_train -n 1000 -T abstract,introduction -N year,publicationDate -e 1 -b 2 -l 2e-5 -f 256 -F 128 -D 0.1`

# Test

## New version (for models after this repo was created)

No need to specify any arguments other than the weight directory itself and the testing batch size.

`python -m citexformer_test -w "[YOUR_MODEL_PATH]" -b 4`

## Old version (for models before this repo was created)

Need to specify ALL args used for training (except epochs and learning rate), as well as weight directory and testing batch size.

`python -m citexformer_test -w "[YOUR_MODEL_PATH]" -n 1000 -T abstract,introduction -N year,publicationDate -b 4 -f 256 -F 128 -D 0.1`
