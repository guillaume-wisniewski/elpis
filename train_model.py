import sys

import argparse

from elpis.engines import Interface, ENGINES

interface = Interface("./models")
ds = interface.new_dataset("pouet")

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True)
parser.add_argument("--tier_type", default="default-lt")
parser.add_argument("--tier_name", default="Phrase")

args = parser.parse_args()

# Load & prepare data
# -------------------
ds.add_directory(args.input_dir)
ds.select_importer("Elan")

importer = ds.importer
importer.set_setting("tier_type", args.tier_type)
importer.set_setting("tier_name", args.tier_name)
ds.validate()
ds.process()

def add_to_test(index, filename):
    return index % 2 == 0

# Create model
# ------------
interface.set_engine(ENGINES["espnet"])
m = interface.new_model("test_model")
m.link(ds, None)
m.build_structure(add_to_test=add_to_test)

def finish():
    print("victory!!!")

#m.train(finish)

#while not m.has_been_trained():
#    pass

#print(m.get_train_results())


