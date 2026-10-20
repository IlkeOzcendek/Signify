from pathlib import Path
from collections import defaultdict # To create a list for people automatically

DATASET_DIR = Path("dataset/cedar/signatures") # Main dataset folder

ORG_DIR = DATASET_DIR / "full_org" # True signatures
FORG_DIR = DATASET_DIR / "full_forg" # False signatures, Forgerier

def get_person_id(file_path): # Gets the person's IDs
# In python, the paremeter does not take the type, only the name of the variable. Python gets the type when it is used in the program

  name = file_path.stem # stem cuts off the file extension | (...).png --> ...
  parts = name.split("_") # Break into pieces whenever you see an underscore, for example: original_39_15 --> ["original", "39", "15"] and save it in "parts" that are reachable through indexes

  return int(parts[1]) # "parts" keeps them into Strings, therefore casting and return the person's ID

def group_by_person(directory): # Getting together everyones true signatures
  grouped = defaultdict(list) # When needed creat new objects using the "list" class

  for file_path in directory.glob("*.png"): # directory is the folder, ORG_DIR or FORD_DIR, glob: find all the files that are in that pattern
    person_id = get_person_id(file_path) # Get the ID and do the operations accordingly
    grouped[person_id].append(file_path) # Add them into a list

  return grouped # It should be usabe from the outside

originals = group_by_person(ORG_DIR)
frauds = group_by_person(FORG_DIR)

print("Number of the people who are original: ", len(originals)) # len: how many keys are there in dictionary
print("The number of the people who are fraud: ", len(frauds))

print("Total original signatures: ", sum(len(v) for v in originals.values())) # v = sign lists, get the values, sum them all up
print("Total fraud signatures ", sum(len(v) for v in frauds.values()))


print("Sample of person analysis: ")
for person_id in sorted(originals.keys())[:5]: # Get the first 5
    print(
        f"Person {person_id}: "
        f"{len(originals[person_id])} original, "
        f"{len(frauds[person_id])} forged"
    )
